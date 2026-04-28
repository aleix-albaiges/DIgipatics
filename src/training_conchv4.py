"""
SICAPv2 Semantic Segmentation Training Pipeline v4
===================================================
CONCH ViT-B/16 + UPerNet decoder + LoRA (PEFT)

Mejoras sobre v2:
  1. UPerNet decoder: PPM (contexto global) + FPN + cabeza multi-escala (concat P2-P5)
  2. LoRA: adaptación de bajo rango en proyecciones Q,V de atención (~150K params)
  3. Stain augmentation: HueSaturationValue para robustez ante variación H&E
  4. Auxiliary loss: supervisión intermedia en feature[2] (bloque 8)

Usage:
    python training_conchv4.py --dry-run --no-wandb
    python training_conchv4.py --fold Val1 --lora-rank 4
    python training_conchv4.py --lora-rank 0 --unfreeze-last 4   # sin LoRA, descongelar
    python training_conchv4.py --lora-rank 8 --aux-weight 0.4
    python training_conchv4.py --seed 42 --fold Val1 Val2
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
cv2.setNumThreads(0)
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


WANDB_PROJECT_DEFAULT = "SICAPv2_CONCH_UPerNet"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
from paths import IMAGES_DIR, MASKS_DIR, PARTITION_DIR, default_checkpoint_dir

OUTPUT_DIR = default_checkpoint_dir("checkpoints_conch_upernet")

NUM_CLASSES = 4
CLASS_NAMES = ["NC", "GG3", "GG4", "GG5"]
IMG_SIZE = 512

_PIXEL_FRAC_PARTITION = np.array([0.7781, 0.0605, 0.1175, 0.0439], dtype=np.float64)
_DEFAULT_CE_CLASS_WEIGHTS = (
    np.sqrt(1.0 / _PIXEL_FRAC_PARTITION) / np.sqrt(1.0 / _PIXEL_FRAC_PARTITION[0])
)
DEFAULT_CLASS_WEIGHTS = [float(round(x, 3)) for x in _DEFAULT_CE_CLASS_WEIGHTS]

_FEATURE_BLOCKS = [2, 5, 8, 11]

CONCH_HF_CHECKPOINT = "hf_hub:MahmoodLab/conch"

DEFAULT_CONFIG = {
    "num_classes"       : NUM_CLASSES,
    "num_workers"       : 4,
    "weight_decay"      : 1e-4,
    "max_epochs"        : 100,
    "patience"          : 18,
    "dice_weight"       : 0.55,
    "ce_weight"         : 0.45,
    "class_weights"     : list(DEFAULT_CLASS_WEIGHTS),
    "batch_size"        : 6,
    "grad_accum_steps"  : 2,
    "learning_rate"     : 4e-5,
    "lr_plateau_patience": 3,
    "fpn_channels"      : 256,
    "unfreeze_last"     : 0,
    # ── Nuevos v4 ──
    "lora_rank"         : 4,
    "lora_alpha"        : 1.0,
    "lora_blocks"       : 4,    # Nº de bloques finales donde inyectar LoRA (0=todos)
    "aux_weight"        : 0.4,
    # ────────────────
    "use_weighted_sampler": True,
    "sampler_weight_gg5": 2.5,
    "sampler_weight_gg4": 1.3,
    "sampler_weight_gg3": 1.8,
    "use_compile"       : None,
    "conch_checkpoint"  : None,
    "conch_hf_token"    : None,
    "seed"              : DEFAULT_SEED,
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset — LUT identical a v2
# ─────────────────────────────────────────────────────────────────────────────
_MASK_LUT = np.zeros(256, dtype=np.int64)
_MASK_LUT[25:75] = 1
_MASK_LUT[75:175] = 2
_MASK_LUT[175:] = 3


def compute_sample_weights(image_names, masks_dir, weight_gg5, weight_gg4, weight_gg3):
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
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

    def __len__(self): return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        buf = np.fromfile(str(self.images_dir / name), dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None: raise FileNotFoundError(name)
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
            t = self.transform(image=image, mask=mask)
            image, mask = t["image"], t["mask"]
        return image, mask

# ─────────────────────────────────────────────────────────────────────────────
# Augmentations — con stain augmentation
# ─────────────────────────────────────────────────────────────────────────────
def get_train_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        # ── Stain augmentation (NUEVO v4) ──
        A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=25, val_shift_limit=12, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
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
# Data Loading — idéntico a v2
# ─────────────────────────────────────────────────────────────────────────────
def get_fold_dataloaders(fold_name, config):
    fold_dir = PARTITION_DIR / "Validation" / fold_name
    train_df = pd.read_excel(fold_dir / "Train.xlsx")
    val_df   = pd.read_excel(fold_dir / "Test.xlsx")
    train_names = train_df["image_name"].tolist()
    val_names   = val_df["image_name"].tolist()

    train_ds = SICAPv2Dataset(train_names, IMAGES_DIR, MASKS_DIR, get_train_transforms())
    val_ds   = SICAPv2Dataset(val_names,   IMAGES_DIR, MASKS_DIR, get_val_transforms())

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", config["num_workers"]))
    kwargs  = {"persistent_workers": True, "prefetch_factor": 4} if workers > 0 else {}

    seed = int(config.get("seed", DEFAULT_SEED))
    gen = torch.Generator(); gen.manual_seed(seed)
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
# CONCH visual trunk
# ─────────────────────────────────────────────────────────────────────────────
def _extract_conch_trunk(weights_path=None, hf_token=None):
    try:
        from conch.open_clip_custom.factory import create_model
    except ImportError as e:
        raise ImportError("pip install git+https://github.com/mahmoodlab/CONCH.git") from e

    ckpt = str(Path(weights_path).resolve()) if weights_path else CONCH_HF_CHECKPOINT
    if weights_path and not Path(weights_path).is_file():
        raise FileNotFoundError(f"CONCH checkpoint not found: {weights_path}")

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    print("  [CONCH] Cargando checkpoint en CPU...")
    full = create_model("conch_ViT-B-16", checkpoint_path=ckpt,
                        device=torch.device("cpu"), hf_auth_token=token)
    trunk = full.visual.trunk
    if hasattr(full, "text") and full.text is not None: del full.text
    if getattr(full, "text_decoder", None) is not None: del full.text_decoder
    del full.visual; del full
    return trunk

# ─────────────────────────────────────────────────────────────────────────────
# LoRA — Low-Rank Adaptation en Q,V de self-attention (NUEVO v4)
# ─────────────────────────────────────────────────────────────────────────────
class LoRAQKV(nn.Module):
    """Envuelve nn.Linear(dim, 3*dim) congelado y añade LoRA a las porciones Q y V.
    Inicialización: A=kaiming, B=zeros → LoRA empieza como identidad (delta=0)."""

    def __init__(self, original_qkv: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original_qkv = original_qkv
        for p in self.original_qkv.parameters():
            p.requires_grad = False

        dim = original_qkv.in_features
        self.dim = dim
        self.scaling = alpha / rank

        self.lora_a_q = nn.Linear(dim, rank, bias=False)
        self.lora_b_q = nn.Linear(rank, dim, bias=False)
        self.lora_a_v = nn.Linear(dim, rank, bias=False)
        self.lora_b_v = nn.Linear(rank, dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_a_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_q.weight)
        nn.init.kaiming_uniform_(self.lora_a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_v.weight)

    def forward(self, x):
        qkv = self.original_qkv(x)
        d = self.dim
        delta_q = self.lora_b_q(self.lora_a_q(x)) * self.scaling
        delta_v = self.lora_b_v(self.lora_a_v(x)) * self.scaling
        return torch.cat([qkv[..., :d] + delta_q,
                          qkv[..., d:2*d],
                          qkv[..., 2*d:] + delta_v], dim=-1)


def inject_lora(trunk, rank: int, alpha: float, lora_blocks: int = 0):
    """Inyecta LoRA en attn.qkv de los last `lora_blocks` bloques del ViT.
    Si lora_blocks=0 → todos los bloques.
    Los bloques finales son los más semánticos y los que más se benefician de adaptación."""
    blocks = trunk.blocks
    total = len(blocks)
    start = (total - lora_blocks) if (lora_blocks > 0 and lora_blocks < total) else 0
    print(f"  [LoRA] Inyectando en bloques [{start}..{total-1}] de {total} total")

    lora_params = []
    injected = 0
    for i, blk in enumerate(blocks):
        if i < start:
            continue  # bloques congelados sin LoRA — coste nulo
        attn = blk.attn
        if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
            lora = LoRAQKV(attn.qkv, rank=rank, alpha=alpha)
            attn.qkv = lora
            for n, p in lora.named_parameters():
                if "lora_" in n:
                    lora_params.append(p)
            injected += 1
        else:
            print(f"  [LoRA] WARN: bloque {i} sin attn.qkv estándar, saltando")
    print(f"  [LoRA] rank={rank} en {injected} bloques → {len(lora_params)} param tensores")
    return lora_params

# ─────────────────────────────────────────────────────────────────────────────
# Encoder CONCH con LoRA
# ─────────────────────────────────────────────────────────────────────────────
class ConcHEncoder(nn.Module):
    """CONCH ViT-B/16 visual trunk con LoRA optional y unfreeze optional."""

    def __init__(self, feature_blocks=_FEATURE_BLOCKS, unfreeze_last=0,
                 lora_rank=4, lora_alpha=1.0, lora_blocks=4,
                 weights_path=None, hf_token=None):
        super().__init__()
        self.trunk = _extract_conch_trunk(weights_path, hf_token)
        self.feature_blocks = set(feature_blocks)
        self.embed_dim = self.trunk.embed_dim
        self.num_prefix = int(getattr(self.trunk, "num_prefix_tokens", 1))

        # Congelar todo
        for p in self.trunk.parameters():
            p.requires_grad = False

        # LoRA (PEFT) — only en los last lora_blocks bloques
        self.lora_params = []
        if lora_rank > 0:
            self.lora_params = inject_lora(self.trunk, lora_rank, lora_alpha, lora_blocks)

        # Descongelar last bloques (optional, complementario a LoRA)
        if unfreeze_last > 0:
            total = len(self.trunk.blocks)
            for blk in self.trunk.blocks[total - unfreeze_last:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.trunk.norm.parameters():
                p.requires_grad = True
            print(f"  [Encoder] Descongelados last {unfreeze_last}/{total} bloques")
        elif lora_rank == 0:
            print("  [Encoder] Completamente congelado (only trains decoder)")

    def forward(self, x):
        B = x.shape[0]
        x_tok = self.trunk.patch_embed(x)
        x_tok = self.trunk._pos_embed(x_tok)
        if hasattr(self.trunk, "patch_drop"):  x_tok = self.trunk.patch_drop(x_tok)
        if hasattr(self.trunk, "norm_pre"):    x_tok = self.trunk.norm_pre(x_tok)

        num_spatial = x_tok.shape[1] - self.num_prefix
        H_p = W_p = int(num_spatial ** 0.5)
        assert H_p * W_p == num_spatial, f"Tokens no square: {num_spatial}"

        features = []
        for i, blk in enumerate(self.trunk.blocks):
            x_tok = blk(x_tok)
            if i in self.feature_blocks:
                sp = x_tok[:, self.num_prefix:self.num_prefix + H_p * W_p, :]
                features.append(sp.permute(0, 2, 1).reshape(B, self.embed_dim, H_p, W_p))
        return features

# ─────────────────────────────────────────────────────────────────────────────
# PPM — Pyramid Pooling Module (PSPNet → UPerNet) (NUEVO v4)
# Contexto global multi-escala: lo que más falta para distinguir GG3/4/5
# ─────────────────────────────────────────────────────────────────────────────
class PPM(nn.Module):
    """Pyramid Pooling Module: captura contexto a escalas 1×1, 2×2, 3×3, 6×6."""

    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6), out_channels=256):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ) for ps in pool_sizes
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * out_channels, out_channels, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        out = [x]
        for stage in self.stages:
            p = stage(x)
            out.append(F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False))
        return self.bottleneck(torch.cat(out, dim=1))

# ─────────────────────────────────────────────────────────────────────────────
# UPerNet Decoder (NUEVO v4) — reemplaza FPNDecoder de v2
# PPM sobre feature profunda + FPN top-down (sum) + cabeza multi-escala
# ─────────────────────────────────────────────────────────────────────────────
class UPerNetDecoder(nn.Module):
    """
    UPerNet (Xiao et al., ECCV 2018) adaptado a plain ViT.
    Nota: en ViT las 4 features tienen la misma resolución (32×32 con 512 input).
    El FPN top-down funciona como fusión de profundidades (shallow→deep), no de resolución.
    La PPM y la cabeza multi-escala son los componentes clave.
    """

    def __init__(self, in_channels=768, fpn_channels=256, num_classes=4,
                 pool_sizes=(1, 2, 3, 6), dropout=0.1):
        super().__init__()
        # PPM sobre la feature deepest (bloque 11)
        self.ppm = PPM(in_channels, pool_sizes, fpn_channels)

        # Laterales: proyección 1×1 para features[0..2] (bloque 2, 5, 8)
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, fpn_channels, 1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
            ) for _ in range(3)
        ])

        # Smooth tras cada fusión top-down
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
            ) for _ in range(3)
        ])

        # Fusión multi-escala: concat P2+P3+P4+P5 → fpn_channels
        self.fpn_fuse = nn.Sequential(
            nn.Conv2d(fpn_channels * 4, fpn_channels, 1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )

        # Segmentation head
        self.head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(fpn_channels, num_classes, 1),
        )

    def forward(self, features, target_size):
        # features: [f0(blk2), f1(blk5), f2(blk8), f3(blk11)]  todas (B, 768, 32, 32)

        # PPM en la deepest → P5
        p5 = self.ppm(features[3])

        # Laterales → P2, P3, P4
        p4 = self.laterals[2](features[2])
        p3 = self.laterals[1](features[1])
        p2 = self.laterals[0](features[0])

        # Top-down: sum con interpolación (no-op si mismo tamaño, pero correcto generalmente)
        p4 = self.smooth[2](p4 + F.interpolate(p5, size=p4.shape[-2:],
                                                mode="bilinear", align_corners=False))
        p3 = self.smooth[1](p3 + F.interpolate(p4, size=p3.shape[-2:],
                                                mode="bilinear", align_corners=False))
        p2 = self.smooth[0](p2 + F.interpolate(p3, size=p2.shape[-2:],
                                                mode="bilinear", align_corners=False))

        # Fusión multi-escala: resize todas a P2 y concat
        sz = p2.shape[-2:]
        fused = torch.cat([
            p2,
            F.interpolate(p3, size=sz, mode="bilinear", align_corners=False),
            F.interpolate(p4, size=sz, mode="bilinear", align_corners=False),
            F.interpolate(p5, size=sz, mode="bilinear", align_corners=False),
        ], dim=1)
        fused = self.fpn_fuse(fused)

        # Upsample a resolución de imagen y clasificar
        fused = F.interpolate(fused, size=target_size, mode="bilinear", align_corners=False)
        return self.head(fused)

# ─────────────────────────────────────────────────────────────────────────────
# CONCH + UPerNet + Aux Head
# ─────────────────────────────────────────────────────────────────────────────
class CONCHSegModel(nn.Module):
    def __init__(self, fpn_channels=256, num_classes=4, unfreeze_last=0,
                 lora_rank=4, lora_alpha=1.0, lora_blocks=4, aux_weight=0.4,
                 weights_path=None, hf_token=None):
        super().__init__()
        self.aux_weight = aux_weight

        self.encoder = ConcHEncoder(
            unfreeze_last=unfreeze_last, lora_rank=lora_rank, lora_alpha=lora_alpha,
            lora_blocks=lora_blocks, weights_path=weights_path, hf_token=hf_token,
        )
        self.decoder = UPerNetDecoder(
            in_channels=self.encoder.embed_dim, fpn_channels=fpn_channels,
            num_classes=num_classes, pool_sizes=(1, 2, 3),  # 1,2,3 suficiente; 6×6 muy costoso
        )
        # Cabeza auxiliar sobre features[2] (bloque 8) — supervisión intermedia
        if aux_weight > 0:
            self.aux_head = nn.Sequential(
                nn.Conv2d(self.encoder.embed_dim, fpn_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(fpn_channels, num_classes, 1),
            )
        else:
            self.aux_head = None

    def forward(self, x):
        target_size = x.shape[-2:]
        features = self.encoder(x)
        main_out = self.decoder(features, target_size)

        aux_out = None
        if self.aux_head is not None:
            aux_out = self.aux_head(features[2])
            aux_out = F.interpolate(aux_out, size=target_size, mode="bilinear", align_corners=False)
        return main_out, aux_out


def build_model(config):
    model = CONCHSegModel(
        fpn_channels  = config["fpn_channels"],
        num_classes   = config["num_classes"],
        unfreeze_last = config["unfreeze_last"],
        lora_rank     = config.get("lora_rank", 4),
        lora_alpha    = config.get("lora_alpha", 1.0),
        lora_blocks   = config.get("lora_blocks", 4),
        aux_weight    = config.get("aux_weight", 0.4),
        weights_path  = config.get("conch_checkpoint"),
        hf_token      = config.get("conch_hf_token"),
    )
    # Resumen de parámetros
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_n = sum(p.numel() for p in model.encoder.lora_params)
    dec_n = sum(p.numel() for p in model.decoder.parameters())
    aux_n = sum(p.numel() for p in model.aux_head.parameters()) if model.aux_head else 0
    print(f"  [Params] Total: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.1f}%)")
    print(f"           LoRA: {lora_n:,} | Decoder: {dec_n:,} | Aux: {aux_n:,}")
    return model

# ─────────────────────────────────────────────────────────────────────────────
# Loss — GuidedLoss (CE + Dice), misma que v2
# ─────────────────────────────────────────────────────────────────────────────
class GuidedLoss(nn.Module):
    def __init__(self, class_weights, dice_weight=0.5, ce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight   = ce_weight
        self.dice_loss   = smp.losses.DiceLoss(mode="multiclass", classes=[0,1,2,3], smooth=smooth)
        self.register_buffer("weights_tensor", torch.tensor(class_weights).float())
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weights_tensor)

    def forward(self, logits, targets):
        logits = logits.float()
        return self.dice_weight * self.dice_loss(logits, targets) + \
               self.ce_weight * self.ce_loss(logits, targets)

# ─────────────────────────────────────────────────────────────────────────────
# Metrics — identicals a v2
# ─────────────────────────────────────────────────────────────────────────────
class SegmentationMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update_batch(self, logits, targets):
        preds = logits.argmax(dim=1).cpu().numpy()
        tgts  = targets.cpu().numpy()
        mask  = (tgts >= 0) & (tgts < self.num_classes)
        np.add.at(self.confusion_matrix, (tgts[mask], preds[mask]), 1)

    def compute(self):
        cm = self.confusion_matrix
        dice = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
            dice[c] = np.nan if tp + fp + fn == 0 else (2.*tp)/(2.*tp + fp + fn + 1e-8)
        macro_f1 = np.nanmean(dice)
        return {"macro_f1": macro_f1, "f1_per_class": np.nan_to_num(dice, nan=0.0),
                "confusion_matrix": cm.copy()}

# ─────────────────────────────────────────────────────────────────────────────
# Training & Validation — modificados para salida (main, aux)
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    grad_accum_steps=1, aux_weight=0.4):
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
            main_logits, aux_logits = model(images)
            loss = criterion(main_logits, masks)
            if aux_logits is not None and aux_weight > 0:
                loss = loss + aux_weight * criterion(aux_logits, masks)

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
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and
                                    torch.cuda.is_bf16_supported()) else torch.float16

    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True).long()
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            main_logits, _ = model(images)
            loss = criterion(main_logits, masks)
        total_loss += loss.item()
        num_batches += 1
        metrics.update_batch(main_logits, masks)
    return total_loss / max(num_batches, 1), metrics.compute()

# ─────────────────────────────────────────────────────────────────────────────
# Train Flow — con param groups diferenciados (LoRA / encoder / decoder+aux)
# ─────────────────────────────────────────────────────────────────────────────
def train_fold(fold_name, config, device, dry_run=False, use_wandb=True):
    print(f"\n{'='*60}\n  FOLD: {fold_name}\n{'='*60}")
    train_loader, val_loader = get_fold_dataloaders(fold_name, config)

    model = build_model(config).to(device)

    use_compile = config.get("use_compile")
    if use_compile is None:
        import platform; use_compile = platform.system() != "Windows"
    if use_compile and int(torch.__version__.split('.')[0]) >= 2:
        try:
            print("  Compilando modelo...")
            import platform
            backend = "inductor" if platform.system() != "Windows" else "aot_eager"
            model = torch.compile(model, backend=backend)
            print("  torch.compile activado.")
        except Exception as e:
            print(f"  torch.compile no disponible: {e}")
    elif not use_compile:
        print("  torch.compile disabledo.")

    criterion = GuidedLoss(config["class_weights"], config["dice_weight"],
                           config["ce_weight"]).to(device)

    # ── Grupos de parámetros diferenciados ──
    lora_ids = {id(p) for p in model.encoder.lora_params}
    lora_params = [p for p in model.encoder.lora_params if p.requires_grad]
    encoder_params = [p for p in model.encoder.trunk.parameters()
                      if p.requires_grad and id(p) not in lora_ids]
    decoder_params = list(model.decoder.parameters())
    aux_params = list(model.aux_head.parameters()) if model.aux_head else []

    lr = config["learning_rate"]
    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": lr, "label": "lora"})
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": lr / 10, "label": "encoder"})
    param_groups.append({"params": decoder_params + aux_params, "lr": lr, "label": "decoder+aux"})

    for pg in param_groups:
        n = sum(p.numel() for p in pg["params"])
        print(f"  [Optim] {pg.get('label','?')}: {n:,} params, lr={pg['lr']:.2e}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=int(config.get("lr_plateau_patience", 3)), min_lr=1e-7)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_macro_f1, patience_counter, best_cm = 0.0, 0, None
    out_dir = Path(config.get("output_dir", OUTPUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)
    max_epochs = 1 if dry_run else config["max_epochs"]
    ga = int(config.get("grad_accum_steps", 1))
    aux_w = float(config.get("aux_weight", 0.4))

    for epoch in range(1, max_epochs + 1):
        lr_dec = optimizer.param_groups[-1]["lr"]
        print(f"\n  Epoch {epoch}/{max_epochs}  (decoder lr={lr_dec:.2e}, accum={ga})")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler,
                                     device, grad_accum_steps=ga, aux_weight=aux_w)
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        macro_f1 = val_metrics["macro_f1"]
        scheduler.step(macro_f1)
        ratio = val_loss / (train_loss + 1e-8)
        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}  |  Ratio: {ratio:.3f}")
        print(f"  Macro F1:   {macro_f1:.4f}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"    {name} F1: {val_metrics['f1_per_class'][i]:.4f}")

        if use_wandb and wandb.run is not None:
            lrs = {f"{fold_name}/lr_{pg.get('label','g'+str(i))}": pg["lr"]
                   for i, pg in enumerate(optimizer.param_groups)}
            m = {f"{fold_name}/train_loss": train_loss, f"{fold_name}/val_loss": val_loss,
                 f"{fold_name}/val_train_ratio": ratio, f"{fold_name}/macro_f1": macro_f1,
                 "epoch": epoch, **lrs}
            for i, name in enumerate(CLASS_NAMES):
                m[f"{fold_name}/f1_{name}"] = float(val_metrics["f1_per_class"][i])
            wandb.log(m)

        if macro_f1 > best_macro_f1:
            best_macro_f1, best_cm, patience_counter = macro_f1, val_metrics["confusion_matrix"], 0
            torch.save(model.state_dict(), out_dir / f"best_{fold_name}_{macro_f1:.4f}.pth")
            print(f"  Model saved (F1={macro_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break
        if dry_run: break

    return {"fold": fold_name, "best_macro_f1": best_macro_f1, "best_cm": best_cm}

# ─────────────────────────────────────────────────────────────────────────────
# Reporting — idéntico a v2
# ─────────────────────────────────────────────────────────────────────────────
def print_aggregated_matrices(agg_cm):
    print(f"\n{'='*60}\n  AGGREGATED CONFUSION MATRICES\n{'='*60}")
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
    df2 = pd.DataFrame([[nc_nc, nc_c],[c_nc, c_c]],
                        index=["T_NC","T_Cancer"], columns=["P_NC","P_Cancer"])
    print(df2.to_string())
    b_pr = c_c/(c_c+nc_c+1e-8); b_rc = c_c/(c_c+c_nc+1e-8)
    print(f"\n  Cancer F1: {2*b_pr*b_rc/(b_pr+b_rc+1e-8):.4f}")
    print(f"  Accuracy:  {(c_c+nc_nc)/(c_c+nc_nc+nc_c+c_nc+1e-8):.4f}\n")


def _agg_macro_f1(agg_cm):
    d = []
    for c in range(NUM_CLASSES):
        tp = agg_cm[c,c]; fp = agg_cm[:,c].sum()-tp; fn = agg_cm[c,:].sum()-tp
        d.append(np.nan if tp+fp+fn==0 else (2.*tp)/(2.*tp+fp+fn+1e-8))
    return float(np.nanmean(d))

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CONCH + UPerNet + LoRA (v4)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--unfreeze-last", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=4,
                        help="Rango LoRA en Q,V (0=disabler)")
    parser.add_argument("--lora-blocks", type=int, default=4,
                        help="Nº de bloques finales con LoRA (0=todos; default=4 para velocidad)")
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--aux-weight", type=float, default=0.4,
                        help="Peso de la loss auxiliar (0=disabler)")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--sampler-gg5", type=float, default=None)
    parser.add_argument("--sampler-gg4", type=float, default=None)
    parser.add_argument("--sampler-gg3", type=float, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT_DEFAULT)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--fold", type=str, nargs="+", default=None,
                        choices=["Val1","Val2","Val3","Val4"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Seed: {args.seed}")

    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
            print("[System] Anti-sleep enabled.")
        except: pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = DEFAULT_CONFIG.copy()
    config["seed"] = args.seed
    config["unfreeze_last"] = args.unfreeze_last
    config["lora_rank"] = args.lora_rank
    config["lora_alpha"] = args.lora_alpha
    config["lora_blocks"] = args.lora_blocks
    config["aux_weight"] = args.aux_weight
    if args.weights:      config["conch_checkpoint"] = args.weights
    if args.hf_token:     config["conch_hf_token"] = args.hf_token
    if args.batch_size:   config["batch_size"] = args.batch_size
    if args.grad_accum:   config["grad_accum_steps"] = max(1, args.grad_accum)
    if args.no_weighted_sampler: config["use_weighted_sampler"] = False
    if args.sampler_gg5 is not None: config["sampler_weight_gg5"] = args.sampler_gg5
    if args.sampler_gg4 is not None: config["sampler_weight_gg4"] = args.sampler_gg4
    if args.sampler_gg3 is not None: config["sampler_weight_gg3"] = args.sampler_gg3
    if args.no_compile:   config["use_compile"] = False
    elif args.compile:    config["use_compile"] = True
    config["output_dir"] = Path(args.output_dir).resolve() if args.output_dir else OUTPUT_DIR
    print(f"Checkpoints -> {config['output_dir']}")

    eff = config["batch_size"] * config.get("grad_accum_steps", 1)
    print(f"Config: bs={config['batch_size']} × ga={config.get('grad_accum_steps',1)} = {eff} eff | "
          f"lr={config['learning_rate']} | LoRA rank={config['lora_rank']} | "
          f"aux_weight={config['aux_weight']}")
    print(f"  CE weights: {config['class_weights']} | dice/ce: {config['dice_weight']}/{config['ce_weight']}")

    use_wandb = not args.no_wandb
    if use_wandb and not args.dry_run:
        try:
            wc = {
                "script": "training_conchv4", "img_size": IMG_SIZE,
                "encoder": "CONCH_ViT-B-16", "decoder": "UPerNet",
                "fpn_channels": config["fpn_channels"],
                "lora_rank": config["lora_rank"], "lora_alpha": config["lora_alpha"],
                "aux_weight": config["aux_weight"],
                "batch_size": config["batch_size"],
                "grad_accum": config.get("grad_accum_steps", 1),
                "effective_batch": eff, "learning_rate": config["learning_rate"],
                "weight_decay": config["weight_decay"],
                "max_epochs": config["max_epochs"], "patience": config["patience"],
                "dice_weight": config["dice_weight"], "ce_weight": config["ce_weight"],
                "class_weights": list(config["class_weights"]),
                "unfreeze_last": config["unfreeze_last"],
                "weighted_sampler": config.get("use_weighted_sampler", False),
                "mask_lut": "25:75->GG3, 75:175->GG4, 175:255->GG5",
                "seed": config["seed"], "fold": args.fold,
                "stain_aug": "HueSaturationValue(12,25,12)",
            }
            rn = args.wandb_name or f"UPerNet_LoRA{config['lora_rank']}_bs{config['batch_size']}"
            wandb.init(project=args.wandb_project, name=rn, config=wc,
                       tags=["UPerNet", "LoRA", "CONCH", "SICAPv2", "v4"])
        except Exception as e:
            print(f"  [WARN] W&B no disponible: {e}")
            use_wandb = False

    fold_names = args.fold if args.fold else ["Val1", "Val2", "Val3", "Val4"]
    agg_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for fold in fold_names:
        res = train_fold(fold, config, device, args.dry_run, use_wandb)
        if res["best_cm"] is not None:
            agg_cm += res["best_cm"]

    print_aggregated_matrices(agg_cm)
    if use_wandb and wandb.run is not None:
        al = {"aggregated/macro_f1": _agg_macro_f1(agg_cm)}
        for c, name in enumerate(CLASS_NAMES):
            tp = agg_cm[c,c]; fp = agg_cm[:,c].sum()-tp; fn = agg_cm[c,:].sum()-tp
            al[f"aggregated/f1_{name}"] = (2.*tp)/(2.*tp+fp+fn+1e-8) if tp+fp+fn>0 else 0.
        wandb.log(al)
    if args.dry_run:
        print("\nDry run completado!")
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
