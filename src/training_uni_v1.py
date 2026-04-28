"""
SICAPv2 Semantic Segmentation Training Pipeline
================================================
UNI v1 Backbone (H&E foundation, lighter than UNI2-h)
- Architecture: UNI = ViT-L/16 (~304M), pretrained H&E (Nature Med 2024)
                + mismo FPN decoder (4 escalas, bloques 5/11/17/23)
- Patch grid a 512×512: 32×32 tokens (patch_size=16)
- Pensado para GPUs ~8GB (p. ej. RTX 4070 laptop): menos VRAM que UNI2-h (ViT-H/14, 1536-D)

Misma licencia/acceso que UNI2 en Hugging Face (repo gated + email institucional).
Alternativa sin descargar del hub: descarga `pytorch_model.bin` y usa --weights.

Prerequisites:
    pip install timm huggingface_hub
    huggingface-cli login   # o login() en Python, con acceso aceptado a MahmoodLab/uni

Usage:
    python training_uni_v1.py                     # 4 folds, encoder congelado
    python training_uni_v1.py --dry-run
    python training_uni_v1.py --weights PATH/pytorch_model.bin
    python training_uni_v1.py --batch-size 4 --grad-accum 2 --unfreeze-last 2
"""

import os
import argparse
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
import timm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
from paths import IMAGES_DIR, MASKS_DIR, PARTITION_DIR, default_checkpoint_dir

OUTPUT_DIR = default_checkpoint_dir("checkpoints_uni_v1")

NUM_CLASSES = 4
CLASS_NAMES = ["NC", "GG3", "GG4", "GG5"]
IMG_SIZE = 512  # multiple of patch_size=16 (UNI v1 = ViT-L/16)

# ViT-L blocks (24) from which we extract features — same relative indices que UNI2
_FEATURE_BLOCKS = [5, 11, 17, 23]

DEFAULT_CONFIG = {
    # ── identical to the original ──────────────────────────────────────
    "num_classes"   : NUM_CLASSES,
    "num_workers"   : 4,
    "weight_decay"  : 1e-4,
    "max_epochs"    : 100,
    "patience"      : 30,
    "dice_weight"   : 0.5,
    "ce_weight"     : 0.5,
    "class_weights" : [1.0, 3.0, 4.5, 18.0],
    # ── UNI v1: lighter than UNI2-h; portable profile ~8GB ─────────
    "batch_size"    : 4,
    "grad_accum_steps": 2,
    "learning_rate" : 6e-5,
    "fpn_channels"  : 256,
    "unfreeze_last" : 0,
    "use_weighted_sampler": True,
    "use_compile"   : None,
    "uni_weights_path": None,  # o --weights path/to/pytorch_model.bin
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset  — identical to the original (including LUT)
# ─────────────────────────────────────────────────────────────────────────────
_MASK_LUT = np.zeros(256, dtype=np.int64)
_MASK_LUT[43:85] = 1
_MASK_LUT[85:160] = 2
_MASK_LUT[160:] = 3


def compute_sample_weights(image_names: list, masks_dir: Path):
    """Imágenes con GG5 → 5×; only GG4 → 2×; rest 1×."""
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
                    w = 5.0
                elif np.any(mapped == 2):
                    w = 2.0
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
# Augmentations — identicals al original
# ─────────────────────────────────────────────────────────────────────────────
def get_train_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),  # multiple of patch_size=16
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
        A.Resize(IMG_SIZE, IMG_SIZE),  # multiple of patch_size=16
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading — identical to the original
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

    if config.get("use_weighted_sampler", False):
        sw = compute_sample_weights(train_names, MASKS_DIR)
        n5 = sum(1 for w in sw if w == 5.0)
        n4 = sum(1 for w in sw if w == 2.0)
        print(f"  [Sampler] train={len(train_names)} | GG5 imgs×5={n5} | GG4-only×2={n4}")
        sampler = WeightedRandomSampler(weights=sw, num_samples=len(train_names), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=config["batch_size"], sampler=sampler,
            num_workers=workers, pin_memory=True, drop_last=True, **kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=config["batch_size"], shuffle=True,
            num_workers=workers, pin_memory=True, drop_last=True, **kwargs,
        )
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False,
                              num_workers=workers, pin_memory=True, **kwargs)
    return train_loader, val_loader

# ─────────────────────────────────────────────────────────────────────────────
# UNI v1 Encoder (ViT-L/16, H&E)
# ─────────────────────────────────────────────────────────────────────────────
class UNIEncoder(nn.Module):
    """
    UNI (MahmoodLab): ViT-L/16, 1024-D, 24 bloques — mismos índices multi-escala que UNI2.
    Entrada típica 512×512 → grid 32×32 tokens en cada mapa intermedio.
    """
    def __init__(self, feature_blocks=_FEATURE_BLOCKS, unfreeze_last=0, weights_path=None):
        super().__init__()
        if weights_path is not None:
            wp = Path(weights_path)
            if not wp.is_file():
                raise FileNotFoundError(f"Pesos UNI not founds: {wp}")
            print(f"  [UNIEncoder] Cargando arquitectura timm + pesos locales: {wp}")
            self.vit = timm.create_model(
                "vit_large_patch16_224",
                pretrained=False,
                num_classes=0,
                init_values=1e-5,
                dynamic_img_size=True,
            )
            state = torch.load(wp, map_location="cpu", weights_only=True)
            self.vit.load_state_dict(state, strict=True)
        else:
            print("  [UNIEncoder] Cargando hf-hub:MahmoodLab/uni (requiere acceso HF + login)...")
            self.vit = timm.create_model(
                "hf-hub:MahmoodLab/uni",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True,
                num_classes=0,
            )

        self.feature_blocks = set(feature_blocks)
        self.embed_dim = self.vit.embed_dim
        self.num_prefix = int(getattr(self.vit, "num_prefix_tokens", 1))

        for p in self.vit.parameters():
            p.requires_grad = False

        if unfreeze_last > 0:
            total = len(self.vit.blocks)
            for blk in self.vit.blocks[total - unfreeze_last:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.vit.norm.parameters():
                p.requires_grad = True
            print(f"  [UNIEncoder] Unfrozen the last {unfreeze_last}/{total} bloques")
        else:
            print("  [UNIEncoder] Frozen encoder (only trains el decoder FPN)")

    def forward(self, x):
        B = x.shape[0]

        x_tok = self.vit.patch_embed(x)
        x_tok = self.vit._pos_embed(x_tok)
        if hasattr(self.vit, "patch_drop"):
            x_tok = self.vit.patch_drop(x_tok)
        if hasattr(self.vit, "norm_pre"):
            x_tok = self.vit.norm_pre(x_tok)

        num_spatial = x_tok.shape[1] - self.num_prefix
        H_p = W_p = int(num_spatial ** 0.5)
        if H_p * W_p != num_spatial:
            raise RuntimeError(f"Tokens espaciales no forman grid: N={num_spatial}")

        features = []
        for i, blk in enumerate(self.vit.blocks):
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
    def __init__(self, in_channels=1024, fpn_channels=256, num_classes=4):
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

        # Segmentation head
        self.head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels // 2, num_classes, 1),
        )

    def forward(self, features, target_size):
        # Proyecciones laterales
        lats = [lat(f) for lat, f in zip(self.lat, features)]

        # Top-down: empezar por la deepest (idx 3) y subir
        x = lats[3]
        for i in range(2, -1, -1):
            x = F.interpolate(x, size=lats[i].shape[-2:], mode="bilinear", align_corners=False)
            x = self.merge[i](x + lats[i])

        # Upsample final al input image size
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Modelo completo UNI v1 + FPN
# ─────────────────────────────────────────────────────────────────────────────
class UNIV1SegModel(nn.Module):
    def __init__(self, fpn_channels=256, num_classes=4, unfreeze_last=0, weights_path=None):
        super().__init__()
        self.encoder = UNIEncoder(
            unfreeze_last=unfreeze_last,
            weights_path=weights_path,
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
    return UNIV1SegModel(
        fpn_channels  = config["fpn_channels"],
        num_classes   = config["num_classes"],
        unfreeze_last = config["unfreeze_last"],
        weights_path  = config.get("uni_weights_path"),
    )

# ─────────────────────────────────────────────────────────────────────────────
# Guided Loss — identical al original
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
# Metrics — identicals al original
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
# Training & Val Epochs — identical to the original
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
def train_fold(fold_name: str, config: dict, device: torch.device, dry_run: bool = False):
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
        print("  ℹ️ torch.compile disabledo (recomendado en Windows / ahorrar VRAM).")

    criterion = GuidedLoss(config["class_weights"], config["dice_weight"], config["ce_weight"]).to(device)

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = list(model.decoder.parameters())
    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": config["learning_rate"] / 10})
    param_groups.append({"params": decoder_params, "lr": config["learning_rate"]})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config["weight_decay"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=6, min_lr=1e-7)
    scaler    = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_macro_f1, patience_counter, best_cm = 0.0, 0, None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss:   {val_loss:.4f}")
        print(f"  Macro F1:   {macro_f1:.4f}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"    {name} F1: {val_metrics['f1_per_class'][i]:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1, best_cm, patience_counter = macro_f1, val_metrics["confusion_matrix"], 0
            torch.save(model.state_dict(), OUTPUT_DIR / f"best_{fold_name}_{macro_f1:.4f}.pth")
            print(f"  ✓ Model saved (Macro F1={macro_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"  ⛔ Early stopping triggered at epoch {epoch}")
                break

        if dry_run: break

    return {"fold": fold_name, "best_macro_f1": best_macro_f1, "best_cm": best_cm}

# ─────────────────────────────────────────────────────────────────────────────
# Reporting — identical to the original
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

# ─────────────────────────────────────────────────────────────────────────────
# Main — identical to the original + argumento --unfreeze-last
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",       action="store_true",
                        help="Smoke test: 1 batch por fold")
    parser.add_argument("--unfreeze-last", type=int, default=0,
                        help="Descongelar los last N bloques del ViT UNI (default: 0 = frozen)")
    parser.add_argument("--weights", type=str, default=None, metavar="PATH",
                        help="pytorch_model.bin de UNI (si no usas descarga hf-hub:MahmoodLab/uni)")
    parser.add_argument("--batch-size", type=int, default=None, help="Micro-batch (by default 2).")
    parser.add_argument("--grad-accum", type=int, default=None, metavar="K",
                        help="Acumular K micro-batches (by default 2).")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Force torch.compile (Windows: can fail or use more VRAM).")
    parser.add_argument("--no-compile", action="store_true", help="Desactivar torch.compile.")
    args = parser.parse_args()

    if os.name == 'nt':
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
            print("\n[System] Anti-sleep enabled.")
        except: pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    config = DEFAULT_CONFIG.copy()
    config["unfreeze_last"] = args.unfreeze_last
    if args.weights:
        config["uni_weights_path"] = args.weights
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

    eff = config["batch_size"] * config.get("grad_accum_steps", 1)
    print(
        f"UNI v1 train: micro_batch={config['batch_size']} × accum={config.get('grad_accum_steps', 1)} "
        f"≈ {eff} | weighted_sampler={config.get('use_weighted_sampler')} | lr={config['learning_rate']}"
    )

    fold_names    = ["Val1", "Val2", "Val3", "Val4"]
    aggregated_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for fold in fold_names:
        res = train_fold(fold, config, device, args.dry_run)
        if res["best_cm"] is not None:
            aggregated_cm += res["best_cm"]

    print_aggregated_matrices(aggregated_cm)
    if args.dry_run:
        print("\n✅ Dry run completed successfully!")

if __name__ == "__main__":
    main()