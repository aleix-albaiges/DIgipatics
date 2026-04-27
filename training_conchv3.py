"""
SICAPv2 Semantic Segmentation Training Pipeline
================================================
CONCH vision encoder (histopathology VLM, Nature Medicine 2024)
- Usa solo la torre visual: ViT-B/16 (~90M params), más ligera que UNI/UNI2 en GPU ~8GB.
- Mismo FPN que training_uni2.py; índices de bloque adaptados a profundidad 12 → [2,5,8,11].

Prerrequisitos:
    pip install git+https://github.com/mahmoodlab/CONCH.git
    pip install wandb && wandb login
    huggingface-cli login
Acceso gated: https://huggingface.co/MahmoodLab/CONCH

Usage:
    python training_conch.py
    python training_conch.py --dry-run
    python training_conch.py --no-wandb
    python training_conch.py --weights RUTA\\pytorch_model.bin
    python training_conch.py --hf-token $env:HF_TOKEN   # opcional si no hay login CLI
    python training_conch.py --output-dir RUTA\\checkpoints   # override carpeta de .pth
    python training_conch.py --seed 42 --fold Val1 --unfreeze-last 2

Pérdida train vs val (misma función GuidedLoss = CE ponderada + Dice):
    - Ambas usan los mismos class_weights; la pérdida no es "probabilidad" sino suma ponderada.
    - Train suele ser menor que val (augmentaciones + dropout implícito en batch norm train mode).
    - Ratio Val/Train > 1 es normal; si val sube y train baja, sospechar sobreajuste.
    - Si macro-F1 mejora solo al inicio (~pocas épocas) y luego se estanca, ReduceLROnPlateau
      (patience corto) reduce el LR para intentar otro descenso; el checkpoint guarda el mejor F1 de val.
"""

import os
import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
cv2.setNumThreads(0)  # CRITICAL: Prevents thread contention deadlocks with PyTorch DataLoader workers!
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

import wandb

warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_SEED = 42


def set_seed(seed: int) -> None:
    """Semilla fija para reproducibilidad (PyTorch, NumPy, Python, workers DataLoader).
    cudnn.deterministic=True y benchmark=False: más estable entre ejecuciones; algo más lento."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(base_seed: int):
    def _fn(worker_id: int):
        s = int((base_seed + worker_id) % (2**32))
        np.random.seed(s)
        random.seed(s)

    return _fn

# Proyecto W&B separado del entrenamiento previo (LUT antigua en máscaras JPEG)
WANDB_PROJECT_DEFAULT = "SICAPv2_CONCH_maskLUT"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
MASKS_DIR = BASE_DIR / "masks"
PARTITION_DIR = BASE_DIR / "partition"
# Checkpoints separados del run anterior (mapeo máscara: 43:85 / 85:160 / 160:)
OUTPUT_DIR = BASE_DIR / "checkpoints_conch_masklut"

NUM_CLASSES = 4
CLASS_NAMES = ["NC", "GG3", "GG4", "GG5"]
IMG_SIZE = 512  # múltiplo de patch_size=16 (CONCH visual = ViT-B/16)

# Fracción global de píxeles por clase con la LUT actual (solo patches en partition/;
# orden NC, GG3, GG4, GG5). Fuente: analyze_masks.py sin --all-masks.
_PIXEL_FRAC_PARTITION = np.array([0.7781, 0.0605, 0.1175, 0.0439], dtype=np.float64)
# CE: sqrt(1/f) frente a 1/f para no disparar pesos; NC normalizado a 1.0
_DEFAULT_CE_CLASS_WEIGHTS = (
    np.sqrt(1.0 / _PIXEL_FRAC_PARTITION) / np.sqrt(1.0 / _PIXEL_FRAC_PARTITION[0])
)
DEFAULT_CLASS_WEIGHTS = [float(round(x, 3)) for x in _DEFAULT_CE_CLASS_WEIGHTS]

# ViT-B tiene 12 bloques — cuatro escalas equidistantes
_FEATURE_BLOCKS = [2, 5, 8, 11]

# Igual que en el README oficial de CONCH (hf_hub:MahmoodLab/conch)
CONCH_HF_CHECKPOINT = "hf_hub:MahmoodLab/conch"

DEFAULT_CONFIG = {
    "num_classes"   : NUM_CLASSES,
    "num_workers"   : 4,
    "weight_decay"  : 1e-4,
    "max_epochs"    : 100,
    # Early stopping por macro-F1: si el mejor modelo sale ~época 4 y no mejora, esto corta antes que 30
    "patience"      : 18,
    # Un poco más de Dice en tareas muy desbalanceadas (F1 ≈ overlap por clase)
    "dice_weight"   : 0.55,
    "ce_weight"     : 0.45,
    "class_weights" : list(DEFAULT_CLASS_WEIGHTS),
    # ViT-B/16: cabe mejor en 8GB que ViT-L/H
    "batch_size"    : 6,
    "grad_accum_steps": 2,
    # Ligeramente más bajo que 6e-5 para reducir oscilación tras las primeras épocas
    "learning_rate" : 4e-5,
    # ReduceLROnPlateau: bajar LR antes cuando macro-F1 se estanca (p.ej. tras ~4 épocas)
    "lr_plateau_patience": 3,
    "fpn_channels"  : 256,
    "unfreeze_last" : 0,
    "use_weighted_sampler": True,
    "use_compile"   : None,
    "conch_checkpoint": None,   # None → CONCH_HF_CHECKPOINT
    "conch_hf_token" : None,
    "seed"            : DEFAULT_SEED,
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset  — LUT sobre valores crudos tras cv2.imdecode (JPEG comprime etiquetas)
# Histogramas en tiles: ~50→GG3, ~100→GG4, ~200→GG5 (no usar 213+ como en LUT antigua)
# ─────────────────────────────────────────────────────────────────────────────
_MASK_LUT = np.zeros(256, dtype=np.int64)
_MASK_LUT[43:85] = 1
_MASK_LUT[85:160] = 2
_MASK_LUT[160:] = 3


def compute_sample_weights(image_names: list, masks_dir: Path):
    """Sobremuestreo: tiles con GG5 → 2.5×; solo GG4 → 1.3×; resto 1×.
    (CE ya usa pesos por clase; aquí se evita duplicar demasiado el énfasis en GG5 vs 5×/2×.)"""
    weights = []
    for name in image_names:
        mask_path = masks_dir / name
        w = 1.0
        if mask_path.exists():
            buf_m = np.fromfile(str(mask_path), dtype=np.uint8)
            mask = cv2.imdecode(buf_m, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mapped = _MASK_LUT[mask]
                if np.any(mapped == 3):
                    w = 2.5
                elif np.any(mapped == 2):
                    w = 1.3
        weights.append(w)
    return weights


class SICAPv2Dataset(Dataset):
    def __init__(self, image_names: list, images_dir: Path, masks_dir: Path, transform=None):
        self.image_names = image_names
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

    def __len__(self): return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img_path = self.images_dir / name
        buf = np.fromfile(str(img_path), dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None: raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = self.masks_dir / name
        if mask_path.exists():
            buf_m = np.fromfile(str(mask_path), dtype=np.uint8)
            mask = cv2.imdecode(buf_m, cv2.IMREAD_GRAYSCALE)
        else:
            mask = None
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        mask = _MASK_LUT[mask]

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask

# ─────────────────────────────────────────────────────────────────────────────
# Augmentations — idénticas al original
# ─────────────────────────────────────────────────────────────────────────────
def get_train_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),  # múltiplo de patch_size=16
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),  # múltiplo de patch_size=16
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading — idéntico al original
# ─────────────────────────────────────────────────────────────────────────────
def get_fold_dataloaders(fold_name: str, config: dict):
    fold_dir = PARTITION_DIR / "Validation" / fold_name
    train_df = pd.read_excel(fold_dir / "Train.xlsx")
    val_df   = pd.read_excel(fold_dir / "Test.xlsx")
    train_names = train_df["image_name"].tolist()
    val_names   = val_df["image_name"].tolist()

    train_ds = SICAPv2Dataset(train_names, IMAGES_DIR, MASKS_DIR, transform=get_train_transforms())
    val_ds   = SICAPv2Dataset(val_names,   IMAGES_DIR, MASKS_DIR, transform=get_val_transforms())

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", config["num_workers"]))
    kwargs  = {"persistent_workers": True, "prefetch_factor": 4} if workers > 0 else {}

    seed = int(config.get("seed", DEFAULT_SEED))
    gen = torch.Generator()
    gen.manual_seed(seed)
    winit = _worker_init_fn(seed) if workers > 0 else None
    dl_common = dict(
        num_workers=workers,
        pin_memory=True,
        generator=gen,
        worker_init_fn=winit,
    )

    if config.get("use_weighted_sampler", False):
        sw = compute_sample_weights(train_names, MASKS_DIR)
        n5 = sum(1 for w in sw if w == 4.0)
        n4 = sum(1 for w in sw if w == 1.75)
        print(f"  [Sampler] train={len(train_names)} | GG5 imgs×4={n5} | GG4-only×1.75={n4}")
        sampler = WeightedRandomSampler(weights=sw, num_samples=len(train_names), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=config["batch_size"], sampler=sampler,
            drop_last=True, **dl_common, **kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=config["batch_size"], shuffle=True,
            drop_last=True, **dl_common, **kwargs,
        )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        **dl_common, **kwargs,
    )
    return train_loader, val_loader

# ─────────────────────────────────────────────────────────────────────────────
# CONCH visual trunk (timm ViT-B/16 dentro del modelo CoCa)
# ─────────────────────────────────────────────────────────────────────────────
def _extract_conch_trunk(weights_path=None, hf_token=None):
    try:
        from conch.open_clip_custom.factory import create_model
    except ImportError as e:
        raise ImportError(
            "Instala CONCH:\n  pip install git+https://github.com/mahmoodlab/CONCH.git"
        ) from e

    if weights_path:
        wp = Path(weights_path)
        if not wp.is_file():
            raise FileNotFoundError(f"No se encontró el checkpoint CONCH: {wp}")
        ckpt = str(wp.resolve())
    else:
        ckpt = CONCH_HF_CHECKPOINT

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    print(f"  [CONCH] Cargando checkpoint en CPU…")
    full = create_model(
        "conch_ViT-B-16",
        checkpoint_path=ckpt,
        device=torch.device("cpu"),
        hf_auth_token=token,
    )
    trunk = full.visual.trunk
    if hasattr(full, "text") and full.text is not None:
        del full.text
    if getattr(full, "text_decoder", None) is not None:
        del full.text_decoder
    del full.visual
    del full
    return trunk


class ConcHEncoder(nn.Module):
    """
    Solo la torre visual de CONCH (ViT-B/16, 768-D, 12 bloques).
    Entrada 512×512 → 32×32 tokens por escala.
    """
    def __init__(self, feature_blocks=_FEATURE_BLOCKS, unfreeze_last=0, weights_path=None, hf_token=None):
        super().__init__()
        self.trunk = _extract_conch_trunk(weights_path, hf_token)
        self.feature_blocks = set(feature_blocks)
        self.embed_dim = self.trunk.embed_dim
        self.num_prefix = int(getattr(self.trunk, "num_prefix_tokens", 1))

        for p in self.trunk.parameters():
            p.requires_grad = False

        if unfreeze_last > 0:
            total = len(self.trunk.blocks)
            for blk in self.trunk.blocks[total - unfreeze_last:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.trunk.norm.parameters():
                p.requires_grad = True
            print(f"  [ConcHEncoder] Descongelados los últimos {unfreeze_last}/{total} bloques")
        else:
            print("  [ConcHEncoder] Encoder congelado (solo entrena el decoder FPN)")

    def forward(self, x):
        B = x.shape[0]
        x_tok = self.trunk.patch_embed(x)
        x_tok = self.trunk._pos_embed(x_tok)
        if hasattr(self.trunk, "patch_drop"):
            x_tok = self.trunk.patch_drop(x_tok)
        if hasattr(self.trunk, "norm_pre"):
            x_tok = self.trunk.norm_pre(x_tok)

        num_spatial = x_tok.shape[1] - self.num_prefix
        H_p = W_p = int(num_spatial ** 0.5)
        if H_p * W_p != num_spatial:
            raise RuntimeError(f"Tokens espaciales no forman cuadrícula: N={num_spatial}")

        features = []
        for i, blk in enumerate(self.trunk.blocks):
            x_tok = blk(x_tok)
            if i in self.feature_blocks:
                spatial = x_tok[:, self.num_prefix : self.num_prefix + H_p * W_p, :]
                spatial = spatial.permute(0, 2, 1).reshape(B, self.embed_dim, H_p, W_p)
                features.append(spatial)

        return features



# ─────────────────────────────────────────────────────────────────────────────
# *** NUEVO *** FPN Decoder ligero
# ─────────────────────────────────────────────────────────────────────────────
class FPNDecoder(nn.Module):
    """
    Feature Pyramid Network: fusiona las 4 escalas del encoder
    y hace upsample progresivo hasta el tamaño original de la imagen.
    """
    def __init__(self, in_channels=768, fpn_channels=256, num_classes=4):
        super().__init__()

        # Proyecciones laterales embed_dim → fpn_channels en cada escala
        self.lat = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, fpn_channels, 1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
            ) for _ in range(4)
        ])

        # Convoluciones top-down (tras suma con feature de nivel inferior)
        self.merge = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
            ) for _ in range(3)
        ])

        # Cabeza de segmentación
        self.head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels // 2, num_classes, 1),
        )

    def forward(self, features, target_size):
        # Proyecciones laterales
        lats = [lat(f) for lat, f in zip(self.lat, features)]

        # Top-down: empezar por la más profunda (idx 3) y subir
        x = lats[3]
        for i in range(2, -1, -1):
            x = F.interpolate(x, size=lats[i].shape[-2:], mode="bilinear", align_corners=False)
            x = self.merge[i](x + lats[i])

        # Upsample final al tamaño de la imagen de entrada
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# CONCH + FPN
# ─────────────────────────────────────────────────────────────────────────────
class CONCHSegModel(nn.Module):
    def __init__(
        self,
        fpn_channels=256,
        num_classes=4,
        unfreeze_last=0,
        weights_path=None,
        hf_token=None,
    ):
        super().__init__()
        self.encoder = ConcHEncoder(
            unfreeze_last=unfreeze_last,
            weights_path=weights_path,
            hf_token=hf_token,
        )
        self.decoder = FPNDecoder(
            in_channels  = self.encoder.embed_dim,
            fpn_channels = fpn_channels,
            num_classes  = num_classes,
        )

    def forward(self, x):
        target_size = x.shape[-2:]
        features    = self.encoder(x)
        return self.decoder(features, target_size)


def build_model(config: dict):
    return CONCHSegModel(
        fpn_channels   = config["fpn_channels"],
        num_classes    = config["num_classes"],
        unfreeze_last  = config["unfreeze_last"],
        weights_path   = config.get("conch_checkpoint"),
        hf_token       = config.get("conch_hf_token"),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Guided Loss — idéntica al original
# ─────────────────────────────────────────────────────────────────────────────
class GuidedLoss(nn.Module):
    def __init__(self, class_weights: list, dice_weight=0.5, ce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight   = ce_weight
        self.dice_loss   = smp.losses.DiceLoss(mode="multiclass", classes=[0,1,2,3], smooth=smooth)
        self.register_buffer("weights_tensor", torch.tensor(class_weights).float())
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weights_tensor)

    def forward(self, logits, targets):
        logits = logits.float()
        ce   = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.dice_weight * dice + self.ce_weight * ce

# ─────────────────────────────────────────────────────────────────────────────
# Metrics — idénticas al original
# ─────────────────────────────────────────────────────────────────────────────
class SegmentationMetrics:
    def __init__(self, num_classes: int):
        self.num_classes      = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update_batch(self, logits: torch.Tensor, targets: torch.Tensor):
        preds = logits.argmax(dim=1).cpu().numpy()
        tgts  = targets.cpu().numpy()
        mask  = (tgts >= 0) & (tgts < self.num_classes)
        np.add.at(self.confusion_matrix, (tgts[mask], preds[mask]), 1)

    def compute(self):
        cm = self.confusion_matrix
        dice_per_class = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            if tp + fp + fn == 0:
                dice_per_class[c] = np.nan
            else:
                dice_per_class[c] = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)

        macro_f1       = np.nanmean(dice_per_class)
        dice_per_class = np.nan_to_num(dice_per_class, nan=0.0)

        return {
            "macro_f1"        : macro_f1,
            "f1_per_class"    : dice_per_class,
            "confusion_matrix": cm.copy(),
        }

# ─────────────────────────────────────────────────────────────────────────────
# Training & Val Epochs — idénticos al original
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_accum_steps: int = 1):
    model.train()
    total_loss, num_batches = 0.0, 0
    accum = 0

    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True).long()
        if accum == 0:
            optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            loss = criterion(model(images), masks)
        scaler.scale(loss / grad_accum_steps).backward()
        total_loss += loss.item()
        num_batches += 1
        accum += 1
        if accum >= grad_accum_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accum = 0
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    if accum > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return total_loss / max(num_batches, 1)

@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, num_batches = 0.0, 0
    metrics = SegmentationMetrics(NUM_CLASSES)

    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True).long()
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            logits = model(images)
            loss   = criterion(logits, masks)
        total_loss += loss.item()
        num_batches += 1
        metrics.update_batch(logits, masks)
    return total_loss / max(num_batches, 1), metrics.compute()

# ─────────────────────────────────────────────────────────────────────────────
# Train Flow — igual al original + LR diferencial encoder/decoder
# ─────────────────────────────────────────────────────────────────────────────
def train_fold(fold_name: str, config: dict, device: torch.device, dry_run: bool = False, use_wandb: bool = True):
    print(f"\n{'='*60}\n  FOLD: {fold_name}\n{'='*60}")
    train_loader, val_loader = get_fold_dataloaders(fold_name, config)

    model = build_model(config).to(device)

    use_compile = config.get("use_compile")
    if use_compile is None:
        import platform
        use_compile = platform.system() != "Windows"
    if use_compile and int(torch.__version__.split('.')[0]) >= 2:
        try:
            print("  ⏳ Compilando modelo con torch.compile...")
            import platform
            backend = "inductor" if platform.system() != "Windows" else "aot_eager"
            model = torch.compile(model, backend=backend)
            print("  ✅ torch.compile activado.")
        except Exception as e:
            print(f"  ⚠️ torch.compile no disponible: {e}")
    elif not use_compile:
        print("  ℹ️ torch.compile desactivado (recomendado en Windows / ahorrar VRAM).")

    criterion = GuidedLoss(config["class_weights"], config["dice_weight"], config["ce_weight"]).to(device)

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = list(model.decoder.parameters())
    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": config["learning_rate"] / 10})
    param_groups.append({"params": decoder_params, "lr": config["learning_rate"]})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config["weight_decay"])

    lr_plat = int(config.get("lr_plateau_patience", 3))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=lr_plat, min_lr=1e-7
    )
    scaler    = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_macro_f1, patience_counter, best_cm = 0.0, 0, None
    out_dir = Path(config.get("output_dir", OUTPUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)
    max_epochs = 1 if dry_run else config["max_epochs"]

    ga = int(config.get("grad_accum_steps", 1))
    for epoch in range(1, max_epochs + 1):
        lr_dec = optimizer.param_groups[-1]["lr"]
        print(f"\n  Epoch {epoch}/{max_epochs}  (decoder lr={lr_dec:.2e}, accum={ga})")
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            grad_accum_steps=ga,
        )
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        macro_f1 = val_metrics["macro_f1"]
        scheduler.step(macro_f1)
        ratio = val_loss / (train_loss + 1e-8)
        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss:   {val_loss:.4f}  |  Val/Train: {ratio:.3f}")
        print(f"  Macro F1:   {macro_f1:.4f}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"    {name} F1: {val_metrics['f1_per_class'][i]:.4f}")

        if use_wandb and wandb.run is not None:
            enc_lr = optimizer.param_groups[0]["lr"]
            dec_lr = optimizer.param_groups[-1]["lr"]
            metrics_dict = {
                f"{fold_name}/train_loss": train_loss,
                f"{fold_name}/val_loss": val_loss,
                f"{fold_name}/val_train_loss_ratio": ratio,
                f"{fold_name}/macro_f1": macro_f1,
                "epoch": epoch,
                f"{fold_name}/lr_encoder": enc_lr,
                f"{fold_name}/lr_decoder": dec_lr,
            }
            for i, name in enumerate(CLASS_NAMES):
                metrics_dict[f"{fold_name}/f1_{name}"] = float(val_metrics["f1_per_class"][i])
            wandb.log(metrics_dict)

        if macro_f1 > best_macro_f1:
            best_macro_f1, best_cm, patience_counter = macro_f1, val_metrics["confusion_matrix"], 0
            torch.save(model.state_dict(), out_dir / f"best_{fold_name}_{macro_f1:.4f}.pth")
            print(f"  ✓ Model saved (Macro F1={macro_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"  ⛔ Early stopping triggered at epoch {epoch}")
                break

        if dry_run: break

    return {"fold": fold_name, "best_macro_f1": best_macro_f1, "best_cm": best_cm}

# ─────────────────────────────────────────────────────────────────────────────
# Reporting — idéntico al original
# ─────────────────────────────────────────────────────────────────────────────
def print_aggregated_matrices(agg_cm):
    print(f"\n{'='*60}\n  AGGREGATED CONFUSION MATRICES (ALL 4 FOLDS)\n{'='*60}")

    print("\n[1] 4-Class Confusion Matrix (Rows: True, Cols: Pred)")
    df_4x4 = pd.DataFrame(agg_cm, index=[f"T_{c}" for c in CLASS_NAMES], columns=[f"P_{c}" for c in CLASS_NAMES])
    print(df_4x4.to_string())

    print("\n--- 4-Class Metrics ---")
    for i in range(4):
        tp = agg_cm[i, i]
        fp = agg_cm[:, i].sum() - tp
        fn = agg_cm[i, :].sum() - tp
        precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        print(f"  {CLASS_NAMES[i]:3s}: F1={f1:.4f}  Prec={precision:.4f}  Rec={recall:.4f}")

    print("\n[2] Binary Confusion Matrix (Cancer vs No Cancer)")
    nc_nc, nc_c = agg_cm[0, 0], agg_cm[0, 1:].sum()
    c_nc,  c_c  = agg_cm[1:, 0].sum(), agg_cm[1:, 1:].sum()

    df_2x2 = pd.DataFrame(np.array([[nc_nc, nc_c], [c_nc, c_c]]),
                          index=["T_NoCancer", "T_Cancer"],
                          columns=["P_NoCancer", "P_Cancer"])
    print(df_2x2.to_string())

    b_tp, b_fp, b_tn, b_fn = c_c, nc_c, nc_nc, c_nc
    b_prec, b_rec = b_tp / (b_tp + b_fp + 1e-8), b_tp / (b_tp + b_fn + 1e-8)
    b_f1  = 2 * (b_prec * b_rec) / (b_prec + b_rec + 1e-8)
    b_acc = (b_tp + b_tn) / (b_tp + b_tn + b_fp + b_fn + 1e-8)

    print("\n--- Binary Classification Metrics ---")
    print(f"  Cancer F1 (Macro) : {b_f1:.4f}")
    print(f"  Overall Accuracy  : {b_acc:.4f}\n")


def _aggregated_macro_f1_from_cm(agg_cm: np.ndarray) -> float:
    dice = []
    for c in range(NUM_CLASSES):
        tp = agg_cm[c, c]
        fp = agg_cm[:, c].sum() - tp
        fn = agg_cm[c, :].sum() - tp
        if tp + fp + fn == 0:
            dice.append(np.nan)
        else:
            dice.append((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8))
    return float(np.nanmean(dice))


# ─────────────────────────────────────────────────────────────────────────────
# Main — idéntico al original + argumento --unfreeze-last
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",       action="store_true",
                        help="Smoke test: 1 batch por fold")
    parser.add_argument("--unfreeze-last", type=int, default=0,
                        help="Descongelar los últimos N bloques del ViT-B CONCH (default: 0)")
    parser.add_argument("--weights", type=str, default=None, metavar="PATH",
                        help="pytorch_model.bin local (alternativa a descarga desde Hugging Face)")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="Token HF (opcional; si no, usa huggingface-cli login o HF_TOKEN)")
    parser.add_argument("--batch-size", type=int, default=None, help="Micro-batch.")
    parser.add_argument("--grad-accum", type=int, default=None, metavar="K",
                        help="Acumular K micro-batches (por defecto 2).")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Forzar torch.compile (Windows: puede fallar o usar más VRAM).")
    parser.add_argument("--no-compile", action="store_true", help="Desactivar torch.compile.")
    parser.add_argument("--no-wandb", action="store_true", help="No registrar en Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT_DEFAULT, help="Proyecto wandb.")
    parser.add_argument("--wandb-name", type=str, default=None, help="Nombre del run (opcional).")
    parser.add_argument("--fold", type=str, default=None, choices=["Val1", "Val2", "Val3", "Val4"], help="Ejecutar solo un fold en específico (ej. Val1). Si no se indica, se ejecutan los 4.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directorio para checkpoints .pth (por defecto: checkpoints_conch_masklut junto al script).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Semilla para reproducibilidad (default: {DEFAULT_SEED}).",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Seed: {args.seed} (cudnn deterministic=True, benchmark=False)")

    if os.name == 'nt':
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
            print("\n[System] Anti-sleep enabled.")
        except: pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = DEFAULT_CONFIG.copy()
    config["seed"] = args.seed
    config["unfreeze_last"] = args.unfreeze_last
    if args.weights:
        config["conch_checkpoint"] = args.weights
    if args.hf_token:
        config["conch_hf_token"] = args.hf_token
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        config["grad_accum_steps"] = max(1, args.grad_accum)
    if args.no_weighted_sampler:
        config["use_weighted_sampler"] = False
    if args.compile and args.no_compile:
        print("  [WARN] --compile y --no-compile a la vez; uso --no-compile.")
    if args.no_compile:
        config["use_compile"] = False
    elif args.compile:
        config["use_compile"] = True

    config["output_dir"] = Path(args.output_dir).resolve() if args.output_dir else OUTPUT_DIR
    print(f"Checkpoints -> {config['output_dir']}")

    eff = config["batch_size"] * config.get("grad_accum_steps", 1)
    print(
        f"CONCH train: micro_batch={config['batch_size']} × accum={config.get('grad_accum_steps', 1)} "
        f"≈ {eff} | weighted_sampler={config.get('use_weighted_sampler')} | lr={config['learning_rate']}"
    )
    print(
        "  class_weights (CE, sqrt-inv freq partition): "
        f"{config['class_weights']} | dice/ce: {config['dice_weight']}/{config['ce_weight']} | "
        f"lr_plateau_patience={config.get('lr_plateau_patience', 3)}"
    )

    use_wandb = not args.no_wandb
    if use_wandb and not args.dry_run:
        try:
            wandb_config = {
                "script": "training_conch",
                "img_size": IMG_SIZE,
                "encoder": "CONCH_ViT-B-16_visual",
                "fpn_channels": config["fpn_channels"],
                "batch_size": config["batch_size"],
                "grad_accum_steps": config.get("grad_accum_steps", 1),
                "effective_batch": eff,
                "learning_rate": config["learning_rate"],
                "weight_decay": config["weight_decay"],
                "max_epochs": config["max_epochs"],
                "patience": config["patience"],
                "dice_weight": config["dice_weight"],
                "ce_weight": config["ce_weight"],
                "class_weights": list(config["class_weights"]),
                "unfreeze_last": config["unfreeze_last"],
                "weighted_sampler": config.get("use_weighted_sampler", False),
                "conch_checkpoint": config.get("conch_checkpoint") or CONCH_HF_CHECKPOINT,
                "output_dir": str(config["output_dir"]),
                "mask_lut": "43:85->GG3, 85:160->GG4, 160:255->GG5, else->NC",
                "pixel_frac_partition": [float(x) for x in _PIXEL_FRAC_PARTITION],
                "lr_plateau_patience": config.get("lr_plateau_patience", 3),
                "seed": config.get("seed", DEFAULT_SEED),
                "fold": args.fold,
            }
            run_name = args.wandb_name or (
                f"CONCH_masklut_bs{config['batch_size']}_ga{config.get('grad_accum_steps', 1)}"
            )
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=wandb_config,
                tags=["mask_lut", "CONCH", "SICAPv2"],
            )
        except Exception as e:
            print(f"  [WARN] Weights & Biases no disponible: {e}")
            use_wandb = False

    if args.fold:
        fold_names = [args.fold]
    else:
        fold_names = ["Val1", "Val2", "Val3", "Val4"]
        
    aggregated_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for fold in fold_names:
        res = train_fold(fold, config, device, args.dry_run, use_wandb=use_wandb)
        if res["best_cm"] is not None:
            aggregated_cm += res["best_cm"]

    print_aggregated_matrices(aggregated_cm)
    if use_wandb and wandb.run is not None:
        agg_log = {"aggregated/macro_f1": _aggregated_macro_f1_from_cm(aggregated_cm)}
        for c, name in enumerate(CLASS_NAMES):
            tp = aggregated_cm[c, c]
            fp = aggregated_cm[:, c].sum() - tp
            fn = aggregated_cm[c, :].sum() - tp
            f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else 0.0
            agg_log[f"aggregated/f1_{name}"] = float(f1)
        wandb.log(agg_log)
    if args.dry_run:
        print("\n✅ Dry run completed successfully!")
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
