"""
SICAPv2 Semantic Segmentation — CONCH + Hybrid U-Net (v5)
==========================================================
Encoder : CONCH ViT-B/16 visual trunk (768-D, 12 bloques, 32×32 tokens a 512px)
Decoder : Hybrid U-Net eficiente
          - Proyecciones laterales embed→decoder_channels
          - Fusión top-down con concat+conv (más expresivo que sum)
          - Attention gates ligeros en cada skip (tipo Attention U-Net)
          - 2 cabezas auxiliares (deep supervision ligera)
          - Cabeza final 2-conv + dropout

Ventajas sobre v2 (FPN):  atención en skips, concat-fusion, deep supervision
Ventajas sobre v4 (UPerNet): sin PPM, sin multi-scale concat → ~30% más rápido

LoRA opcional (--lora-rank N; default 0=desactivado) para no complicar el flujo base.

Usage:
    python training_conch_hybrid_unet.py --no-wandb --dry-run
    python training_conch_hybrid_unet.py --fold Val1 --unfreeze-last 2
    python training_conch_hybrid_unet.py --lora-rank 4 --aux-weight-1 0.3 --aux-weight-2 0.15
    python training_conch_hybrid_unet.py --decoder-channels 256 --dropout 0.15
"""

import os
import math
import argparse
import random
import warnings
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import cv2
cv2.setNumThreads(0)  # CRITICAL: evita deadlocks con DataLoader workers en Windows
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

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibilidad
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int, base_seed: int):
    s = int((base_seed + worker_id) % (2**32))
    np.random.seed(s)
    random.seed(s)


WANDB_PROJECT_DEFAULT = "SICAPv2_CONCH_HybridUNet"

# ─────────────────────────────────────────────────────────────────────────────
# Configuración global
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
IMAGES_DIR    = BASE_DIR / "images"
MASKS_DIR     = BASE_DIR / "masks"
PARTITION_DIR = BASE_DIR / "partition"
OUTPUT_DIR    = BASE_DIR / "checkpoints_conch_hybridunet"

NUM_CLASSES  = 4
CLASS_NAMES  = ["NC", "GG3", "GG4", "GG5"]
IMG_SIZE     = 512   # múltiplo de patch_size=16; NO subir a 1024 por defecto

_PIXEL_FRAC_PARTITION = np.array([0.7781, 0.0605, 0.1175, 0.0439], dtype=np.float64)
_DEFAULT_CE_CLASS_WEIGHTS = (
    np.sqrt(1.0 / _PIXEL_FRAC_PARTITION) / np.sqrt(1.0 / _PIXEL_FRAC_PARTITION[0])
)
DEFAULT_CLASS_WEIGHTS = [float(round(x, 3)) for x in _DEFAULT_CE_CLASS_WEIGHTS]

# ViT-B tiene 12 bloques → 4 escalas equidistantes
_FEATURE_BLOCKS   = [2, 5, 8, 11]
CONCH_HF_CHECKPOINT = "hf_hub:MahmoodLab/conch"

DEFAULT_CONFIG = {
    "num_classes"        : NUM_CLASSES,
    "num_workers"        : 4,
    "weight_decay"       : 1e-4,
    "max_epochs"         : 100,
    "patience"           : 18,
    "dice_weight"        : 0.55,
    "ce_weight"          : 0.45,
    "class_weights"      : list(DEFAULT_CLASS_WEIGHTS),
    "batch_size"         : 6,
    "grad_accum_steps"   : 2,
    "learning_rate"      : 4e-5,
    "lr_plateau_patience": 3,
    "decoder_channels"   : 256,  # canales internos del decoder (=fpn_channels en v2)
    "dropout"            : 0.1,
    "unfreeze_last"      : 0,
    # Deep supervision: loss = main + w1*aux1 + w2*aux2
    "aux_weight_1"       : 0.3,   # cabeza sobre features[2] (bloque 8)
    "aux_weight_2"       : 0.15,  # cabeza sobre features[1] (bloque 5)
    # LoRA (desactivado por defecto para mantener velocidad de v2)
    "lora_rank"          : 0,
    "lora_alpha"         : 1.0,
    "lora_blocks"        : 4,
    # Sampler
    "use_weighted_sampler": True,
    "sampler_weight_gg5" : 2.5,
    "sampler_weight_gg4" : 1.3,
    "sampler_weight_gg3" : 1.8,
    "use_compile"        : None,
    "conch_checkpoint"   : None,
    "conch_hf_token"     : None,
    "seed"               : DEFAULT_SEED,
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset — LUT idéntica a v2/v4
# ─────────────────────────────────────────────────────────────────────────────
_MASK_LUT = np.zeros(256, dtype=np.int64)
_MASK_LUT[25:75]  = 1  # GG3
_MASK_LUT[75:175] = 2  # GG4
_MASK_LUT[175:]   = 3  # GG5


def compute_sample_weights(image_names, masks_dir, weight_gg5, weight_gg4, weight_gg3):
    """Sobremuestreo por clase presente en el tile: GG5 > GG4 > GG3 > 1×."""
    weights = []
    for name in image_names:
        mask_path = masks_dir / name
        w = 1.0
        if mask_path.exists():
            buf_m = np.fromfile(str(mask_path), dtype=np.uint8)
            mask = cv2.imdecode(buf_m, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mapped = _MASK_LUT[mask]
                if np.any(mapped == 3):   w = weight_gg5
                elif np.any(mapped == 2): w = weight_gg4
                elif np.any(mapped == 1): w = weight_gg3
        weights.append(w)
    return weights


class SICAPv2Dataset(Dataset):
    def __init__(self, image_names, images_dir, masks_dir, transform=None):
        self.image_names = image_names
        self.images_dir  = images_dir
        self.masks_dir   = masks_dir
        self.transform   = transform

    def __len__(self): return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        buf  = np.fromfile(str(self.images_dir / name), dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None: raise FileNotFoundError(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = self.masks_dir / name
        if mask_path.exists():
            buf_m = np.fromfile(str(mask_path), dtype=np.uint8)
            mask  = cv2.imdecode(buf_m, cv2.IMREAD_GRAYSCALE)
        else:
            mask = None
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        mask = _MASK_LUT[mask]

        if self.transform:
            t = self.transform(image=image, mask=mask)
            image, mask = t["image"], t["mask"]
        return image, mask


# ─────────────────────────────────────────────────────────────────────────────
# Augmentations — moderadas (+ stain aug de v4, razonable y rápida)
# ─────────────────────────────────────────────────────────────────────────────
def get_train_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        # Quitada la augmentación de color lenta (HSV) porque atascaba el DataLoader.
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — idéntico a v2
# ─────────────────────────────────────────────────────────────────────────────
def get_fold_dataloaders(fold_name: str, config: dict):
    fold_dir    = PARTITION_DIR / "Validation" / fold_name
    train_df    = pd.read_excel(fold_dir / "Train.xlsx")
    val_df      = pd.read_excel(fold_dir / "Test.xlsx")
    train_names = train_df["image_name"].tolist()
    val_names   = val_df["image_name"].tolist()

    train_ds = SICAPv2Dataset(train_names, IMAGES_DIR, MASKS_DIR, get_train_transforms())
    val_ds   = SICAPv2Dataset(val_names,   IMAGES_DIR, MASKS_DIR, get_val_transforms())

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", config["num_workers"]))
    kwargs  = {"persistent_workers": True, "prefetch_factor": 4} if workers > 0 else {}

    seed  = int(config.get("seed", DEFAULT_SEED))
    gen   = torch.Generator(); gen.manual_seed(seed)
    winit = partial(_worker_init_fn, base_seed=seed) if workers > 0 else None
    dl_common = dict(num_workers=workers, pin_memory=True, generator=gen, worker_init_fn=winit)

    if config.get("use_weighted_sampler", False):
        w5 = float(config["sampler_weight_gg5"])
        w4 = float(config["sampler_weight_gg4"])
        w3 = float(config["sampler_weight_gg3"])
        sw = compute_sample_weights(train_names, MASKS_DIR, w5, w4, w3)
        n5 = sum(1 for x in sw if x == w5)
        n4 = sum(1 for x in sw if x == w4)
        n3 = sum(1 for x in sw if x == w3)
        print(f"  [Sampler] train={len(train_names)} | GG5×{w5}={n5} | GG4×{w4}={n4} | GG3×{w3}={n3}")
        sampler = WeightedRandomSampler(weights=sw, num_samples=len(train_names), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], sampler=sampler,
                                  drop_last=True, **dl_common, **kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                                  drop_last=True, **dl_common, **kwargs)

    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                            **dl_common, **kwargs)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# CONCH visual trunk loader — idéntico a v2/v4
# ─────────────────────────────────────────────────────────────────────────────
def _extract_conch_trunk(weights_path=None, hf_token=None):
    try:
        from conch.open_clip_custom.factory import create_model
    except ImportError as e:
        raise ImportError("pip install git+https://github.com/mahmoodlab/CONCH.git") from e

    ckpt = str(Path(weights_path).resolve()) if weights_path else CONCH_HF_CHECKPOINT
    if weights_path and not Path(weights_path).is_file():
        raise FileNotFoundError(f"CONCH checkpoint no encontrado: {weights_path}")

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    print("  [CONCH] Cargando checkpoint en CPU...")
    full  = create_model("conch_ViT-B-16", checkpoint_path=ckpt,
                         device=torch.device("cpu"), hf_auth_token=token)
    trunk = full.visual.trunk
    if hasattr(full, "text") and full.text is not None: del full.text
    if getattr(full, "text_decoder", None) is not None: del full.text_decoder
    del full.visual; del full
    return trunk


# ─────────────────────────────────────────────────────────────────────────────
# LoRA — Low-Rank Adaptation (opcional, desactivado por defecto)
# Tomado de v4; se activa con --lora-rank N (N>0)
# ─────────────────────────────────────────────────────────────────────────────
class LoRAQKV(nn.Module):
    """Envuelve nn.Linear(d, 3d) congelado y añade LoRA en Q y V."""
    def __init__(self, original_qkv: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original_qkv = original_qkv
        for p in self.original_qkv.parameters():
            p.requires_grad = False
        d = original_qkv.in_features
        self.dim = d
        self.scaling = alpha / rank
        self.lora_a_q = nn.Linear(d, rank, bias=False)
        self.lora_b_q = nn.Linear(rank, d, bias=False)
        self.lora_a_v = nn.Linear(d, rank, bias=False)
        self.lora_b_v = nn.Linear(rank, d, bias=False)
        nn.init.kaiming_uniform_(self.lora_a_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_q.weight)
        nn.init.kaiming_uniform_(self.lora_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_v.weight)

    def forward(self, x):
        qkv     = self.original_qkv(x)
        d       = self.dim
        delta_q = self.lora_b_q(self.lora_a_q(x)) * self.scaling
        delta_v = self.lora_b_v(self.lora_a_v(x)) * self.scaling
        return torch.cat([qkv[..., :d] + delta_q, qkv[..., d:2*d], qkv[..., 2*d:] + delta_v], dim=-1)


def inject_lora(trunk, rank: int, alpha: float, lora_blocks: int = 0):
    blocks = trunk.blocks
    total  = len(blocks)
    start  = (total - lora_blocks) if (0 < lora_blocks < total) else 0
    print(f"  [LoRA] Inyectando en bloques [{start}..{total-1}] de {total} | rank={rank}")
    lora_params = []
    for i, blk in enumerate(blocks):
        if i < start: continue
        attn = blk.attn
        if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
            lora = LoRAQKV(attn.qkv, rank=rank, alpha=alpha)
            attn.qkv = lora
            for n, p in lora.named_parameters():
                if "lora_" in n: lora_params.append(p)
    print(f"  [LoRA] {len(lora_params)} param tensores entrenables")
    return lora_params


# ─────────────────────────────────────────────────────────────────────────────
# Encoder CONCH
# ─────────────────────────────────────────────────────────────────────────────
class CONCHEncoder(nn.Module):
    """
    CONCH ViT-B/16 visual trunk.
    Extrae features intermedias en los bloques indicados por feature_blocks.
    Con 512×512 input → tokens espaciales 32×32 en TODAS las escalas.
    Con LoRA opcional en los últimos N bloques.
    """
    def __init__(self, feature_blocks=_FEATURE_BLOCKS, unfreeze_last=0,
                 lora_rank=0, lora_alpha=1.0, lora_blocks=4,
                 weights_path=None, hf_token=None):
        super().__init__()
        self.trunk         = _extract_conch_trunk(weights_path, hf_token)
        self.feature_blocks = set(feature_blocks)
        self.embed_dim     = self.trunk.embed_dim
        self.num_prefix    = int(getattr(self.trunk, "num_prefix_tokens", 1))

        # Congelar todo
        for p in self.trunk.parameters():
            p.requires_grad = False

        # LoRA (opcional)
        self.lora_params = []
        if lora_rank > 0:
            self.lora_params = inject_lora(self.trunk, lora_rank, lora_alpha, lora_blocks)

        # Descongelar últimos bloques (complementario a LoRA)
        if unfreeze_last > 0:
            total = len(self.trunk.blocks)
            for blk in self.trunk.blocks[total - unfreeze_last:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.trunk.norm.parameters():
                p.requires_grad = True
            print(f"  [Encoder] Descongelados últimos {unfreeze_last}/{total} bloques")
        elif lora_rank == 0:
            print("  [Encoder] Completamente congelado (solo entrena decoder)")

    def forward(self, x):
        B = x.shape[0]
        x_tok = self.trunk.patch_embed(x)
        x_tok = self.trunk._pos_embed(x_tok)
        if hasattr(self.trunk, "patch_drop"): x_tok = self.trunk.patch_drop(x_tok)
        if hasattr(self.trunk, "norm_pre"):   x_tok = self.trunk.norm_pre(x_tok)

        num_spatial = x_tok.shape[1] - self.num_prefix
        H_p = W_p = int(num_spatial ** 0.5)
        if H_p * W_p != num_spatial:
            raise RuntimeError(f"Tokens espaciales no cuadráticos: N={num_spatial}")

        features = []
        for i, blk in enumerate(self.trunk.blocks):
            x_tok = blk(x_tok)
            if i in self.feature_blocks:
                sp = x_tok[:, self.num_prefix:self.num_prefix + H_p * W_p, :]
                features.append(sp.permute(0, 2, 1).reshape(B, self.embed_dim, H_p, W_p))
        return features   # lista de 4 tensors (B, 768, 32, 32)


# ─────────────────────────────────────────────────────────────────────────────
# Attention Gate — ligero, tipo Attention U-Net (Oktay et al., 2018)
# Selecciona qué información lateral conviene fusionar.
# Coste: 3 conv1×1 pequeñas → negligible vs encoder.
# ─────────────────────────────────────────────────────────────────────────────
class AttentionGate(nn.Module):
    """
    g  : señal guía (feature más profunda, ya proyectada a dec_ch)
    x  : skip connection (feature menos profunda, ya proyectada a dec_ch)
    Devuelve x * alpha, donde alpha ∈ [0,1] es el mapa de atención espacial.
    """
    def __init__(self, dec_ch: int):
        super().__init__()
        # Eficiencia extrema: reducir a dec_ch//4 internamente
        inter = max(dec_ch // 4, 16)
        self.W_g = nn.Conv2d(dec_ch, inter, 1, bias=False)
        self.W_x = nn.Conv2d(dec_ch, inter, 1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(inter, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, g, x):
        # g y x tienen la misma resolución espacial (ViT plain → todas 32×32)
        alpha = self.psi(F.relu(self.W_g(g) + self.W_x(x), inplace=True))
        return x * alpha


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid U-Net Decoder
# ─────────────────────────────────────────────────────────────────────────────
class HybridUNetDecoder(nn.Module):
    """
    Decoder tipo Hybrid U-Net para encoder ViT (todas las features con la misma
    resolución espacial, a diferencia de un CNN U-Net).

    Flujo:
      features = [f0(blk2), f1(blk5), f2(blk8), f3(blk11)]  ← (B, 768, 32, 32)

      1. Proyecciones laterales 1×1: embed_dim → dec_ch  (lat0..lat3)
      2. Fusión top-down con concat+conv (más expresivo que sum directo):
           x = lat3
           para i=2,1,0:
               skip_i = AttentionGate(guide=x, skip=lat_i)
               x = merge_conv( cat([x, skip_i]) )  ← concat de 2*dec_ch → dec_ch
      3. Upsample bilinear a target_size
      4. Cabeza final: Conv3×3 → BN → ReLU → Dropout → Conv1×1 → num_classes
      5. Aux heads en features[2] y features[1] para deep supervision

    Decisión de diseño: concat+conv frente a sum
      - Sum es más rápido pero asume que g y skip tienen semántica compatible.
      - En ViT puro todas las features son del mismo espacio de representación,
        pero sus 'semánticas' son diferentes (shallow vs deep). concat+conv
        deja que el modelo aprenda cómo combinarlas sin imposiciones.
      - Coste extra: 1 conv extra por nivel (negligible).
    """
    def __init__(self, in_channels: int = 768, dec_ch: int = 256,
                 num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dec_ch = dec_ch

        # --- Proyecciones laterales (1×1) ---
        self.lat = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, dec_ch, 1, bias=False),
                nn.BatchNorm2d(dec_ch),
                nn.ReLU(inplace=True),
            ) for _ in range(4)
        ])

        # --- Attention gates (un gate por skip, niveles 0-1-2) ---
        self.att = nn.ModuleList([AttentionGate(dec_ch) for _ in range(3)])

        # --- Merge: concat(dec_ch, dec_ch) → dec_ch (como en v5 normal) ---
        self.merge = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dec_ch * 2, dec_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(dec_ch),
                nn.ReLU(inplace=True),
            ) for _ in range(3)
        ])

        # --- Cabeza final (reducida a dec_ch//2 como en v2) ---
        self.head = nn.Sequential(
            nn.Conv2d(dec_ch, dec_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(dec_ch // 2, num_classes, 1),
        )

        # --- Aux heads ultraligeras (deep supervision 1x1) ---
        self.aux1 = nn.Conv2d(dec_ch, num_classes, 1)
        self.aux2 = nn.Conv2d(dec_ch, num_classes, 1)

    def forward(self, features, target_size):
        # features: [f0, f1, f2, f3] todas (B, embed_dim, 32, 32)

        # 1. Proyecciones laterales
        lats = [lat(f) for lat, f in zip(self.lat, features)]

        # 2. Top-down con concat+conv y attention gates
        x = lats[3]
        out_lvl2 = out_lvl1 = None
        for i in range(2, -1, -1):
            skip = self.att[i](g=x, x=lats[i])          # attention gate
            x    = self.merge[i](torch.cat([x, skip], dim=1))  # concat + conv
            if i == 2: out_lvl2 = x
            if i == 1: out_lvl1 = x

        # 3. Upsample a resolución de imagen y cabeza final
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        main_out = self.head(x)

        # 4. Salidas auxiliares
        aux1 = F.interpolate(self.aux1(out_lvl2), size=target_size, mode="bilinear", align_corners=False)
        aux2 = F.interpolate(self.aux2(out_lvl1), size=target_size, mode="bilinear", align_corners=False)

        return main_out, aux1, aux2


# ─────────────────────────────────────────────────────────────────────────────
# Modelo completo
# ─────────────────────────────────────────────────────────────────────────────
class CONCHHybridUNet(nn.Module):
    def __init__(self, dec_ch=256, num_classes=4, unfreeze_last=0,
                 lora_rank=0, lora_alpha=1.0, lora_blocks=4, dropout=0.1,
                 weights_path=None, hf_token=None):
        super().__init__()
        self.encoder = CONCHEncoder(
            unfreeze_last=unfreeze_last,
            lora_rank=lora_rank, lora_alpha=lora_alpha, lora_blocks=lora_blocks,
            weights_path=weights_path, hf_token=hf_token,
        )
        self.decoder = HybridUNetDecoder(
            in_channels=self.encoder.embed_dim,
            dec_ch=dec_ch,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x):
        target_size = x.shape[-2:]
        features    = self.encoder(x)
        # Devuelve (main_logits, aux1_logits, aux2_logits)
        return self.decoder(features, target_size)


def build_model(config: dict) -> CONCHHybridUNet:
    model = CONCHHybridUNet(
        dec_ch        = config.get("decoder_channels", config.get("fpn_channels", 256)),
        num_classes   = config["num_classes"],
        unfreeze_last = config["unfreeze_last"],
        lora_rank     = config.get("lora_rank", 0),
        lora_alpha    = config.get("lora_alpha", 1.0),
        lora_blocks   = config.get("lora_blocks", 4),
        dropout       = config.get("dropout", 0.1),
        weights_path  = config.get("conch_checkpoint"),
        hf_token      = config.get("conch_hf_token"),
    )
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [Params] Total: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.1f}%)")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Función de pérdida
# GuidedLoss = Dice + CE ponderada (misma base que v2/v4)
# Extensible: comentarios indican cómo cambiar a Focal o Tversky
# ─────────────────────────────────────────────────────────────────────────────
class GuidedLoss(nn.Module):
    """
    Pérdida base: Dice multiclass + CrossEntropy ponderada (sqrt-inv freq).

    Para cambiar a FocalCE: sustituir nn.CrossEntropyLoss por
        smp.losses.FocalLoss(mode="multiclass", ...)
    Para Tversky/FocalTversky: sustituir DiceLoss por
        smp.losses.TverskyLoss(mode="multiclass", alpha=0.3, beta=0.7)
    """
    def __init__(self, class_weights: list, dice_weight=0.55, ce_weight=0.45, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight   = ce_weight
        self.dice_loss   = smp.losses.DiceLoss(mode="multiclass", classes=[0,1,2,3], smooth=smooth)
        self.register_buffer("weights_tensor", torch.tensor(class_weights).float())
        self.ce_loss     = nn.CrossEntropyLoss(weight=self.weights_tensor)

    def forward(self, logits, targets):
        logits = logits.float()
        return self.dice_weight * self.dice_loss(logits, targets) + \
               self.ce_weight   * self.ce_loss(logits, targets)


# ─────────────────────────────────────────────────────────────────────────────
# Métricas — idénticas a v2/v4
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
        cm   = self.confusion_matrix
        dice = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
            dice[c] = np.nan if tp+fp+fn == 0 else (2.*tp)/(2.*tp+fp+fn+1e-8)
        macro_f1 = np.nanmean(dice)
        return {"macro_f1": macro_f1, "f1_per_class": np.nan_to_num(dice, nan=0.0),
                "confusion_matrix": cm.copy()}


# ─────────────────────────────────────────────────────────────────────────────
# Training & Validation — adaptados para salida (main, aux1, aux2)
# Grad accum y mixed precision iguales que v2
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    grad_accum_steps=1, aux_weight_1=0.3, aux_weight_2=0.15):
    model.train()
    total_loss, num_batches, accum = 0.0, 0, 0
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and
                                   torch.cuda.is_bf16_supported()) else torch.float16

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True).long()
        if accum == 0:
            optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            main_out, aux1_out, aux2_out = model(images)
            loss = criterion(main_out, masks)
            # Deep supervision — solo durante entrenamiento; pesos moderados
            if aux_weight_1 > 0:
                loss = loss + aux_weight_1 * criterion(aux1_out, masks)
            if aux_weight_2 > 0:
                loss = loss + aux_weight_2 * criterion(aux2_out, masks)

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

    # Flush de gradientes pendientes (último mini-batch incompleto)
    if accum > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    """Validación solo con salida principal (las aux heads no se usan en inferencia)."""
    model.eval()
    total_loss, num_batches = 0.0, 0
    metrics  = SegmentationMetrics(NUM_CLASSES)
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and
                                   torch.cuda.is_bf16_supported()) else torch.float16

    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True).long()
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            main_out, _, _ = model(images)   # aux ignoradas en val
            loss = criterion(main_out, masks)
        total_loss += loss.item()
        num_batches += 1
        metrics.update_batch(main_out, masks)
    return total_loss / max(num_batches, 1), metrics.compute()


# ─────────────────────────────────────────────────────────────────────────────
# Train fold — idéntico en estructura a v2, con mejoras de v4 (LR groups)
# ─────────────────────────────────────────────────────────────────────────────
def train_fold(fold_name: str, config: dict, device: torch.device,
               dry_run: bool = False, use_wandb: bool = True):
    print(f"\n{'='*60}\n  FOLD: {fold_name}\n{'='*60}")
    train_loader, val_loader = get_fold_dataloaders(fold_name, config)

    model = build_model(config).to(device)

    # torch.compile — desactivado en Windows por defecto (igual que v2)
    use_compile = config.get("use_compile")
    if use_compile is None:
        import platform; use_compile = platform.system() != "Windows"
    if use_compile and int(torch.__version__.split('.')[0]) >= 2:
        try:
            import platform
            backend = "inductor" if platform.system() != "Windows" else "aot_eager"
            model = torch.compile(model, backend=backend)
            print("  torch.compile activado.")
        except Exception as e:
            print(f"  torch.compile no disponible: {e}")
    elif not use_compile:
        print("  torch.compile desactivado (recomendado en Windows).")

    criterion = GuidedLoss(config["class_weights"], config["dice_weight"],
                           config["ce_weight"]).to(device)

    # ── Grupos de parámetros con LR diferencial (igual que v2/v4) ──
    lora_ids      = {id(p) for p in model.encoder.lora_params}
    lora_params   = [p for p in model.encoder.lora_params if p.requires_grad]
    encoder_trunk = [p for p in model.encoder.trunk.parameters()
                     if p.requires_grad and id(p) not in lora_ids]
    decoder_params = list(model.decoder.parameters())

    lr = config["learning_rate"]
    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params,   "lr": lr,      "label": "lora"})
    if encoder_trunk:
        param_groups.append({"params": encoder_trunk, "lr": lr / 10, "label": "encoder"})
    param_groups.append(    {"params": decoder_params, "lr": lr,     "label": "decoder"})

    for pg in param_groups:
        n = sum(p.numel() for p in pg["params"])
        print(f"  [Optim] {pg['label']}: {n:,} params, lr={pg['lr']:.2e}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=int(config.get("lr_plateau_patience", 3)), min_lr=1e-7)
    scaler    = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_macro_f1, patience_counter, best_cm = 0.0, 0, None
    out_dir    = Path(config.get("output_dir", OUTPUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)
    max_epochs = 1 if dry_run else config["max_epochs"]
    ga         = int(config.get("grad_accum_steps", 1))
    aw1        = float(config.get("aux_weight_1", 0.3))
    aw2        = float(config.get("aux_weight_2", 0.15))

    for epoch in range(1, max_epochs + 1):
        lr_dec = optimizer.param_groups[-1]["lr"]
        print(f"\n  Epoch {epoch}/{max_epochs}  (decoder lr={lr_dec:.2e}, accum={ga})")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            grad_accum_steps=ga, aux_weight_1=aw1, aux_weight_2=aw2)
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        macro_f1 = val_metrics["macro_f1"]
        scheduler.step(macro_f1)
        ratio = val_loss / (train_loss + 1e-8)
        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}  |  Val/Train: {ratio:.3f}")
        print(f"  Macro F1:   {macro_f1:.4f}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"    {name} F1: {val_metrics['f1_per_class'][i]:.4f}")

        # ── W&B logging completo ──
        if use_wandb and wandb.run is not None:
            lrs = {f"{fold_name}/lr_{pg['label']}": pg["lr"]
                   for pg in optimizer.param_groups}
            m = {
                f"{fold_name}/train_loss"      : train_loss,
                f"{fold_name}/val_loss"        : val_loss,
                f"{fold_name}/val_train_ratio" : ratio,
                f"{fold_name}/macro_f1"        : macro_f1,
                "epoch"                        : epoch,
                **lrs,
            }
            for i, name in enumerate(CLASS_NAMES):
                m[f"{fold_name}/f1_{name}"] = float(val_metrics["f1_per_class"][i])
            wandb.log(m)

        if macro_f1 > best_macro_f1:
            best_macro_f1, best_cm, patience_counter = macro_f1, val_metrics["confusion_matrix"], 0
            torch.save(model.state_dict(), out_dir / f"best_{fold_name}_{macro_f1:.4f}.pth")
            print(f"  ✓ Model saved (Macro F1={macro_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"  ⛔ Early stopping at epoch {epoch}")
                break
        if dry_run: break

    return {"fold": fold_name, "best_macro_f1": best_macro_f1, "best_cm": best_cm}


# ─────────────────────────────────────────────────────────────────────────────
# Reporting — idéntico a v2
# ─────────────────────────────────────────────────────────────────────────────
def print_aggregated_matrices(agg_cm):
    print(f"\n{'='*60}\n  AGGREGATED CONFUSION MATRICES (ALL FOLDS)\n{'='*60}")
    print("\n[1] 4-Class (Rows=True, Cols=Pred)")
    df = pd.DataFrame(agg_cm, index=[f"T_{c}" for c in CLASS_NAMES],
                      columns=[f"P_{c}" for c in CLASS_NAMES])
    print(df.to_string())
    print("\n--- Per-class ---")
    for i in range(4):
        tp = agg_cm[i, i]; fp = agg_cm[:, i].sum() - tp; fn = agg_cm[i, :].sum() - tp
        pr, rc = tp/(tp+fp+1e-8), tp/(tp+fn+1e-8)
        print(f"  {CLASS_NAMES[i]:3s}: F1={2*pr*rc/(pr+rc+1e-8):.4f}  P={pr:.4f}  R={rc:.4f}")

    print("\n[2] Binary (Cancer vs NC)")
    nc_nc, nc_c = agg_cm[0, 0], agg_cm[0, 1:].sum()
    c_nc,  c_c  = agg_cm[1:, 0].sum(), agg_cm[1:, 1:].sum()
    df2 = pd.DataFrame([[nc_nc, nc_c], [c_nc, c_c]],
                        index=["T_NC", "T_Cancer"], columns=["P_NC", "P_Cancer"])
    print(df2.to_string())
    b_pr = c_c/(c_c+nc_c+1e-8); b_rc = c_c/(c_c+c_nc+1e-8)
    print(f"\n  Cancer F1: {2*b_pr*b_rc/(b_pr+b_rc+1e-8):.4f}")
    print(f"  Accuracy:  {(c_c+nc_nc)/(c_c+nc_nc+nc_c+c_nc+1e-8):.4f}\n")


def _agg_macro_f1(agg_cm: np.ndarray) -> float:
    d = []
    for c in range(NUM_CLASSES):
        tp = agg_cm[c, c]; fp = agg_cm[:, c].sum()-tp; fn = agg_cm[c, :].sum()-tp
        d.append(np.nan if tp+fp+fn == 0 else (2.*tp)/(2.*tp+fp+fn+1e-8))
    return float(np.nanmean(d))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CONCH + Hybrid U-Net (v5)")
    parser.add_argument("--dry-run",          action="store_true")
    parser.add_argument("--unfreeze-last",    type=int, default=0,
                        help="Descongelar últimos N bloques ViT-B (0=congelado)")
    parser.add_argument("--lora-rank",        type=int, default=0,
                        help="Rango LoRA en Q,V (0=desactivado; recomendado 4 si se activa)")
    parser.add_argument("--lora-blocks",      type=int, default=4,
                        help="Nº de bloques finales con LoRA (0=todos)")
    parser.add_argument("--lora-alpha",       type=float, default=1.0)
    parser.add_argument("--aux-weight-1",     type=float, default=0.3,
                        help="Peso loss auxiliar features[2] (bloque 8). 0=desactivar.")
    parser.add_argument("--aux-weight-2",     type=float, default=0.15,
                        help="Peso loss auxiliar features[1] (bloque 5). 0=desactivar.")
    parser.add_argument("--decoder-channels", type=int, default=256,
                        help="Canales internos del decoder (default 256)")
    parser.add_argument("--dropout",          type=float, default=0.1,
                        help="Dropout2d en cabeza final del decoder (default 0.1)")
    parser.add_argument("--weights",          type=str, default=None, metavar="PATH")
    parser.add_argument("--hf-token",         type=str, default=None)
    parser.add_argument("--batch-size",       type=int, default=None)
    parser.add_argument("--grad-accum",       type=int, default=None, metavar="K")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--sampler-gg5",      type=float, default=None)
    parser.add_argument("--sampler-gg4",      type=float, default=None)
    parser.add_argument("--sampler-gg3",      type=float, default=None)
    parser.add_argument("--compile",          action="store_true")
    parser.add_argument("--no-compile",       action="store_true")
    parser.add_argument("--no-wandb",         action="store_true")
    parser.add_argument("--wandb-project",    type=str, default=WANDB_PROJECT_DEFAULT)
    parser.add_argument("--wandb-name",       type=str, default=None)
    parser.add_argument("--fold",             type=str, nargs="+", default=None,
                        choices=["Val1", "Val2", "Val3", "Val4"])
    parser.add_argument("--output-dir",       type=str, default=None)
    parser.add_argument("--seed",             type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Seed: {args.seed} (cudnn deterministic=True, benchmark=False)")

    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
            print("[System] Anti-sleep enabled.")
        except: pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = DEFAULT_CONFIG.copy()
    config["seed"]             = args.seed
    config["unfreeze_last"]    = args.unfreeze_last
    config["lora_rank"]        = args.lora_rank
    config["lora_alpha"]       = args.lora_alpha
    config["lora_blocks"]      = args.lora_blocks
    config["aux_weight_1"]     = args.aux_weight_1
    config["aux_weight_2"]     = args.aux_weight_2
    config["decoder_channels"] = args.decoder_channels
    config["dropout"]          = args.dropout
    if args.weights:            config["conch_checkpoint"]    = args.weights
    if args.hf_token:           config["conch_hf_token"]      = args.hf_token
    if args.batch_size:         config["batch_size"]           = args.batch_size
    if args.grad_accum:         config["grad_accum_steps"]     = max(1, args.grad_accum)
    if args.no_weighted_sampler: config["use_weighted_sampler"] = False
    if args.sampler_gg5 is not None: config["sampler_weight_gg5"] = args.sampler_gg5
    if args.sampler_gg4 is not None: config["sampler_weight_gg4"] = args.sampler_gg4
    if args.sampler_gg3 is not None: config["sampler_weight_gg3"] = args.sampler_gg3
    if args.no_compile:         config["use_compile"] = False
    elif args.compile:          config["use_compile"] = True
    config["output_dir"] = Path(args.output_dir).resolve() if args.output_dir else OUTPUT_DIR
    print(f"Checkpoints -> {config['output_dir']}")

    eff = config["batch_size"] * config.get("grad_accum_steps", 1)
    print(f"Config: bs={config['batch_size']} × ga={config.get('grad_accum_steps',1)} = {eff} eff | "
          f"lr={config['learning_rate']} | dec_ch={config['decoder_channels']} | "
          f"LoRA rank={config['lora_rank']} | aux=({config['aux_weight_1']},{config['aux_weight_2']})")
    print(f"  CE weights: {config['class_weights']} | dice/ce: {config['dice_weight']}/{config['ce_weight']}")

    use_wandb = not args.no_wandb
    if use_wandb and not args.dry_run:
        try:
            wc = {
                "script"           : "training_conch_hybrid_unet",
                "architecture"     : "CONCH_HybridUNet",
                "encoder"          : "CONCH_ViT-B-16_visual",
                "decoder"          : "HybridUNet_AttentionGates",
                "img_size"         : IMG_SIZE,
                "decoder_channels" : config["decoder_channels"],
                "dropout"          : config["dropout"],
                "lora_rank"        : config["lora_rank"],
                "lora_alpha"       : config["lora_alpha"],
                "lora_blocks"      : config["lora_blocks"],
                "unfreeze_last"    : config["unfreeze_last"],
                "aux_weight_1"     : config["aux_weight_1"],
                "aux_weight_2"     : config["aux_weight_2"],
                "batch_size"       : config["batch_size"],
                "grad_accum"       : config.get("grad_accum_steps", 1),
                "effective_batch"  : eff,
                "learning_rate"    : config["learning_rate"],
                "weight_decay"     : config["weight_decay"],
                "max_epochs"       : config["max_epochs"],
                "patience"         : config["patience"],
                "dice_weight"      : config["dice_weight"],
                "ce_weight"        : config["ce_weight"],
                "class_weights"    : list(config["class_weights"]),
                "weighted_sampler" : config.get("use_weighted_sampler", False),
                "sampler_gg5"      : config.get("sampler_weight_gg5"),
                "sampler_gg4"      : config.get("sampler_weight_gg4"),
                "sampler_gg3"      : config.get("sampler_weight_gg3"),
                "mask_lut"         : "25:75->GG3, 75:175->GG4, 175:255->GG5",
                "pixel_frac"       : [float(x) for x in _PIXEL_FRAC_PARTITION],
                "seed"             : config["seed"],
                "fold"             : args.fold,
                "stain_aug"        : "HueSaturationValue(10,20,10,p=0.4)",
                "attention_type"   : "AttentionGate_UNetStyle",
                "fusion_type"      : "concat+conv (top-down)",
                "deep_supervision" : "aux1(blk8)+aux2(blk5)",
            }
            rn = args.wandb_name or (
                f"HybridUNet_dec{config['decoder_channels']}"
                f"_lora{config['lora_rank']}"
                f"_bs{config['batch_size']}"
            )
            wandb.init(project=args.wandb_project, name=rn, config=wc,
                       tags=["HybridUNet", "AttentionGates", "CONCH", "SICAPv2", "v5"])
        except Exception as e:
            print(f"  [WARN] W&B no disponible: {e}")
            use_wandb = False

    fold_names = args.fold if args.fold else ["Val1", "Val2", "Val3", "Val4"]
    agg_cm     = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for fold in fold_names:
        res = train_fold(fold, config, device, args.dry_run, use_wandb)
        if res["best_cm"] is not None:
            agg_cm += res["best_cm"]

    print_aggregated_matrices(agg_cm)
    if use_wandb and wandb.run is not None:
        al = {"aggregated/macro_f1": _agg_macro_f1(agg_cm)}
        for c, name in enumerate(CLASS_NAMES):
            tp = agg_cm[c, c]; fp = agg_cm[:, c].sum()-tp; fn = agg_cm[c, :].sum()-tp
            al[f"aggregated/f1_{name}"] = (2.*tp)/(2.*tp+fp+fn+1e-8) if tp+fp+fn > 0 else 0.
        wandb.log(al)
    if args.dry_run:
        print("\n✅ Dry run completado!")
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
