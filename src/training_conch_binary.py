"""
SICAPv2 Semantic Segmentation Training Pipeline
================================================
CONCH vision encoder (histopathology VLM, Nature Medicine 2024)
- Uses only the vision tower: ViT-B/16 (~90M params), lighter than UNI/UNI2 on ~8GB GPUs.
- Same FPN as training_uni2.py; feature block indices adapted to depth 12 → [2,5,8,11].

Prerequisites:
    pip install git+https://github.com/mahmoodlab/CONCH.git
    pip install wandb && wandb login
    huggingface-cli login
Gated access: https://huggingface.co/MahmoodLab/CONCH

Usage:
    python src/training_conchv2.py
    python src/training_conchv2.py --dry-run
    python src/training_conchv2.py --no-wandb
    python src/training_conchv2.py --weights PATH\\pytorch_model.bin
    python src/training_conchv2.py --hf-token $env:HF_TOKEN   # optional if CLI login is unavailable
    python src/training_conchv2.py --output-dir PATH\\checkpoints   # override .pth output folder
    python src/training_conchv2.py --seed 42 --fold Val1 --unfreeze-last 2

Train vs val loss (same GuidedLoss = weighted CE + Dice):
    - Both use the same class_weights; the loss is not a "probability" but a weighted sum.
    - Train loss is usually lower than val loss (augmentations + implicit dropout-like effects in train mode).
    - A Val/Train ratio > 1 is common; if val rises while train falls, suspect overfitting.
    - If macro-F1 improves only in early epochs and then plateaus, ReduceLROnPlateau
      (short patience) lowers LR earlier; the checkpoint keeps the best validation F1.
"""

import os
import argparse
import random
import warnings
from pathlib import Path
from functools import partial

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
    """Fixed seed for reproducibility (PyTorch, NumPy, Python, workers DataLoader).
    cudnn.deterministic=True and benchmark=False: more stable across runs; slightly slower."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int, base_seed: int):
    """Picklable worker init for Windows multiprocessing."""
    s = int((base_seed + worker_id) % (2**32))
    np.random.seed(s)
    random.seed(s)

# W&B project separated from previous training (legacy LUT in JPEG masks)
WANDB_PROJECT_DEFAULT = "SICAPv2_CONCH_binary"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
from paths import IMAGES_DIR, MASKS_DIR, PARTITION_DIR, default_checkpoint_dir

# Checkpoints separated from the previous run (mask mapping: 43:85 / 85:160 / 160:)
OUTPUT_DIR = default_checkpoint_dir("checkpoints_conch_binary")

NUM_CLASSES = 2
CLASS_NAMES = ["NC", "Cancer"]
IMG_SIZE = 512  # multiple of patch_size=16 (CONCH visual = ViT-B/16)

# Global pixel fraction per class with current LUT (partition patches only;
# class order: NC, GG3, GG4, GG5). Source: analyze_masks.py without --all-masks.
_PIXEL_FRAC_PARTITION_4 = np.array([0.7781, 0.0605, 0.1175, 0.0439], dtype=np.float64)
_PIXEL_FRAC_PARTITION = np.array([_PIXEL_FRAC_PARTITION_4[0], _PIXEL_FRAC_PARTITION_4[1:].sum()], dtype=np.float64)
_DEFAULT_CE_CLASS_WEIGHTS = (
    np.sqrt(1.0 / _PIXEL_FRAC_PARTITION) / np.sqrt(1.0 / _PIXEL_FRAC_PARTITION[0])
)
DEFAULT_CLASS_WEIGHTS = [float(round(x, 3)) for x in _DEFAULT_CE_CLASS_WEIGHTS]

# ViT-B has 12 blocks — four evenly spaced scales
_FEATURE_BLOCKS = [2, 5, 8, 11]

# Same as in the official CONCH README (hf_hub:MahmoodLab/conch)
CONCH_HF_CHECKPOINT = "hf_hub:MahmoodLab/conch"

DEFAULT_CONFIG = {
    "num_classes"   : NUM_CLASSES,
    "num_workers"   : 4,
    "weight_decay"  : 1e-4,
    "max_epochs"    : 100,
    # Early stopping by macro-F1: if the best model appears around epoch 4 and does not improve,
    # this stops earlier than patience=30.
    "patience"      : 18,
    # Slightly higher Dice weight for highly imbalanced tasks (F1 tracks class overlap).
    "dice_weight"   : 0.55,
    "ce_weight"     : 0.45,
    "class_weights" : list(DEFAULT_CLASS_WEIGHTS),
    # ViT-B/16: fits better in 8GB than ViT-L/H
    "batch_size"    : 6,
    "grad_accum_steps": 2,
    # Slightly lower than 6e-5 to reduce oscillation after early epochs.
    "learning_rate" : 4e-5,
    # ReduceLROnPlateau: lower LR sooner when macro-F1 stalls (e.g., after ~4 epochs).
    "lr_plateau_patience": 3,
    "fpn_channels"  : 256,
    "unfreeze_last" : 0,
    "use_weighted_sampler": True,
    # Binary oversampling:
    # - cancer-present tiles get a base boost
    # - additional boost proportional to tumor pixel fraction in the tile
    "sampler_weight_cancer_present": 2.0,
    "sampler_weight_cancer_absent": 1.0,
    "sampler_tumor_ratio_scale": 2.0,
    # Validation-time threshold tuning for Cancer probability
    "eval_threshold": 0.5,
    "tune_threshold": True,
    "threshold_search_min": 0.30,
    "threshold_search_max": 0.70,
    "threshold_search_step": 0.05,
    # For BCEWithLogits on cancer channel
    "pos_weight": float(_PIXEL_FRAC_PARTITION[0] / (_PIXEL_FRAC_PARTITION[1] + 1e-8)),
    "use_compile"   : None,
    "conch_checkpoint": None,   # None → CONCH_HF_CHECKPOINT
    "conch_hf_token" : None,
    "seed"            : DEFAULT_SEED,
}

# ─────────────────────────────────────────────────────────────────────────────
# Dataset — LUT over raw values after cv2.imdecode (JPEG compression alters label values)
# Typical tile histograms: ~50->GG3, ~100->GG4, ~200->GG5 (do not use 213+ from the old LUT)
# ─────────────────────────────────────────────────────────────────────────────
_MASK_LUT = np.zeros(256, dtype=np.int64)
_MASK_LUT[25:75] = 1
_MASK_LUT[75:175] = 2
_MASK_LUT[175:] = 3


def compute_sample_weights(
    image_names: list,
    masks_dir: Path,
    weight_cancer_present: float,
    weight_cancer_absent: float,
    tumor_ratio_scale: float,
):
    """Oversample tiles by binary tumor content.
    Weight = absent_weight for pure NC, else present_weight + scale * tumor_fraction."""
    weights = []
    for name in image_names:
        mask_path = masks_dir / name
        w = float(weight_cancer_absent)
        if mask_path.exists():
            buf_m = np.fromfile(str(mask_path), dtype=np.uint8)
            mask = cv2.imdecode(buf_m, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mapped = (_MASK_LUT[mask] > 0).astype(np.uint8)
                tumor_frac = float(mapped.mean())
                if tumor_frac > 0.0:
                    w = float(weight_cancer_present) + float(tumor_ratio_scale) * tumor_frac
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
        mask = (mask > 0).astype(np.uint8)

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

    seed = int(config.get("seed", DEFAULT_SEED))
    gen = torch.Generator()
    gen.manual_seed(seed)
    winit = partial(_worker_init_fn, base_seed=seed) if workers > 0 else None
    dl_common = dict(
        num_workers=workers,
        pin_memory=True,
        generator=gen,
        worker_init_fn=winit,
    )

    if config.get("use_weighted_sampler", False):
        wc = float(config["sampler_weight_cancer_present"])
        wn = float(config["sampler_weight_cancer_absent"])
        wr = float(config["sampler_tumor_ratio_scale"])
        sw = compute_sample_weights(train_names, MASKS_DIR, wc, wn, wr)
        n_c = sum(1 for x in sw if x > wn)
        print(
            f"  [Sampler] train={len(train_names)} | cancer_tiles={n_c} | "
            f"w_absent={wn:.2f} | w_present_base={wc:.2f} | ratio_scale={wr:.2f}"
        )
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
            "Install CONCH:\n  pip install git+https://github.com/mahmoodlab/CONCH.git"
        ) from e

    if weights_path:
        wp = Path(weights_path)
        if not wp.is_file():
            raise FileNotFoundError(f"Not found el checkpoint CONCH: {wp}")
        ckpt = str(wp.resolve())
    else:
        ckpt = CONCH_HF_CHECKPOINT

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    print("  [CONCH] Loading checkpoint on CPU...")
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
    CONCH visual tower only (ViT-B/16, 768-D, 12 blocks).
    Input 512x512 -> 32x32 tokens per scale.
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
            print(f"  [ConcHEncoder] Unfrozen the last {unfreeze_last}/{total} bloques")
        else:
            print("  [ConcHEncoder] Frozen encoder (only trains el decoder FPN)")

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
            raise RuntimeError(f"Tokens espaciales no forman grid: N={num_spatial}")

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
    and progressively upsamples to the original image size.
    """
    def __init__(self, in_channels=768, fpn_channels=256, num_classes=4):
        super().__init__()

        # Lateral projections: embed_dim -> fpn_channels at each scale
        self.lat = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, fpn_channels, 1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
            ) for _ in range(4)
        ])

        # Top-down convolutions (after summing with lower-level features)
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
        # Lateral projections
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
# Guided Loss — identical al original
# ─────────────────────────────────────────────────────────────────────────────
class GuidedLoss(nn.Module):
    def __init__(self, pos_weight: float, dice_weight=0.5, ce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight   = ce_weight
        self.dice_loss   = smp.losses.DiceLoss(mode="binary", from_logits=True, smooth=smooth)
        self.register_buffer("pos_weight_tensor", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits, targets):
        # Model still outputs 2 channels; use Cancer channel as positive logit.
        cancer_logits = logits[:, 1:2, ...].float()
        cancer_targets = targets.unsqueeze(1).float()
        ce = F.binary_cross_entropy_with_logits(
            cancer_logits,
            cancer_targets,
            pos_weight=self.pos_weight_tensor,
        )
        dice = self.dice_loss(cancer_logits, cancer_targets)
        return self.dice_weight * dice + self.ce_weight * ce

# ─────────────────────────────────────────────────────────────────────────────
# Metrics — identicals al original
# ─────────────────────────────────────────────────────────────────────────────
class SegmentationMetrics:
    def __init__(self, num_classes: int, threshold: float = 0.5):
        self.num_classes      = num_classes
        self.threshold        = float(threshold)
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update_batch(self, logits: torch.Tensor, targets: torch.Tensor):
        if self.num_classes == 2:
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= self.threshold).long().cpu().numpy()
        else:
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
def validate_one_epoch(model, loader, criterion, device, eval_threshold=0.5, threshold_grid=None):
    model.eval()
    total_loss, num_batches = 0.0, 0
    metrics = SegmentationMetrics(NUM_CLASSES, threshold=eval_threshold)
    threshold_cms = {}
    if threshold_grid is not None and NUM_CLASSES == 2:
        threshold_cms = {float(t): np.zeros((2, 2), dtype=np.int64) for t in threshold_grid}

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

        if threshold_cms:
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            tgts = masks.cpu().numpy()
            valid = (tgts >= 0) & (tgts < 2)
            for t, cm in threshold_cms.items():
                preds_t = (probs >= t).astype(np.int64)
                np.add.at(cm, (tgts[valid], preds_t[valid]), 1)

    best_threshold = float(eval_threshold)
    best_cancer_f1 = float(metrics.compute()["f1_per_class"][1])
    if threshold_cms:
        for t, cm in threshold_cms.items():
            tp = cm[1, 1]
            fp = cm[0, 1]
            fn = cm[1, 0]
            f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else 0.0
            if f1 > best_cancer_f1:
                best_cancer_f1 = float(f1)
                best_threshold = float(t)

        metrics = SegmentationMetrics(NUM_CLASSES, threshold=best_threshold)
        for images, masks in loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True).long()
            with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
                logits = model(images)
            metrics.update_batch(logits, masks)

    return total_loss / max(num_batches, 1), metrics.compute(), best_threshold, best_cancer_f1

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
            print("  ⏳ Compiling model with torch.compile...")
            import platform
            backend = "inductor" if platform.system() != "Windows" else "aot_eager"
            model = torch.compile(model, backend=backend)
            print("  ✅ torch.compile enabled.")
        except Exception as e:
            print(f"  ⚠️ torch.compile unavailable: {e}")
    elif not use_compile:
        print("  ℹ️ torch.compile disabled (recommended on Windows to save VRAM).")

    criterion = GuidedLoss(config["pos_weight"], config["dice_weight"], config["ce_weight"]).to(device)

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
    best_threshold_global = float(config.get("eval_threshold", 0.5))
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
        threshold_grid = None
        if config.get("tune_threshold", True):
            tmin = float(config.get("threshold_search_min", 0.30))
            tmax = float(config.get("threshold_search_max", 0.70))
            tstep = float(config.get("threshold_search_step", 0.05))
            threshold_grid = np.arange(tmin, tmax + 1e-12, tstep, dtype=np.float64).tolist()
        val_loss, val_metrics, best_thr_epoch, best_cancer_f1_epoch = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            eval_threshold=float(config.get("eval_threshold", 0.5)),
            threshold_grid=threshold_grid,
        )

        macro_f1 = val_metrics["macro_f1"]
        scheduler.step(macro_f1)
        ratio = val_loss / (train_loss + 1e-8)
        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss:   {val_loss:.4f}  |  Val/Train: {ratio:.3f}")
        print(f"  Macro F1:   {macro_f1:.4f}")
        print(f"  Best val threshold (Cancer): {best_thr_epoch:.2f} | Cancer F1@best_thr: {best_cancer_f1_epoch:.4f}")
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
                f"{fold_name}/best_threshold_cancer": float(best_thr_epoch),
                f"{fold_name}/best_cancer_f1": float(best_cancer_f1_epoch),
                "epoch": epoch,
                f"{fold_name}/lr_encoder": enc_lr,
                f"{fold_name}/lr_decoder": dec_lr,
            }
            for i, name in enumerate(CLASS_NAMES):
                metrics_dict[f"{fold_name}/f1_{name}"] = float(val_metrics["f1_per_class"][i])
            wandb.log(metrics_dict)

        if macro_f1 > best_macro_f1:
            best_macro_f1, best_cm, patience_counter = macro_f1, val_metrics["confusion_matrix"], 0
            best_threshold_global = float(best_thr_epoch)
            torch.save(model.state_dict(), out_dir / f"best_{fold_name}_{macro_f1:.4f}.pth")
            print(f"  ✓ Model saved (Macro F1={macro_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"  ⛔ Early stopping triggered at epoch {epoch}")
                break

        if dry_run: break

    return {
        "fold": fold_name,
        "best_macro_f1": best_macro_f1,
        "best_cm": best_cm,
        "best_threshold": best_threshold_global,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Reporting — identical to the original
# ─────────────────────────────────────────────────────────────────────────────
def print_aggregated_matrices(agg_cm):
    print(f"\n{'='*60}\n  AGGREGATED CONFUSION MATRICES (ALL 4 FOLDS)\n{'='*60}")
    print("\n[1] Binary Confusion Matrix (Cancer vs No Cancer)")
    import pandas as pd
    df_2x2 = pd.DataFrame(agg_cm, index=[f"T_{c}" for c in CLASS_NAMES], columns=[f"P_{c}" for c in CLASS_NAMES])
    print(df_2x2.to_string())
    print("\n--- Binary Classification Metrics ---")
    for i in range(2):
        tp = agg_cm[i, i]
        fp = agg_cm[:, i].sum() - tp
        fn = agg_cm[i, :].sum() - tp
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        print(f"  {CLASS_NAMES[i]:10s}: F1={f1:.4f}  Prec={precision:.4f}  Rec={recall:.4f}")
    b_acc = agg_cm.trace() / (agg_cm.sum() + 1e-8)
    print(f"\n  Overall Accuracy  : {b_acc:.4f}\n")

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
# Main — identical to the original + argumento --unfreeze-last
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",       action="store_true",
                        help="Smoke test: 1 batch por fold")
    parser.add_argument("--unfreeze-last", type=int, default=0,
                        help="Unfreeze the last N CONCH ViT-B blocks (default: 0)")
    parser.add_argument("--weights", type=str, default=None, metavar="PATH",
                        help="pytorch_model.bin local (alternativa a descarga desde Hugging Face)")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HF token (optional; otherwise use huggingface-cli login or HF_TOKEN)")
    parser.add_argument("--batch-size", type=int, default=None, help="Micro-batch.")
    parser.add_argument("--grad-accum", type=int, default=None, metavar="K",
                        help="Acumular K micro-batches (by default 2).")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--sampler-cancer", type=float, default=None, metavar="W",
                        help="Peso base sampler tiles con presencia de cancer.")
    parser.add_argument("--sampler-no-cancer", type=float, default=None, metavar="W",
                        help="Peso base sampler tiles sin cancer.")
    parser.add_argument("--sampler-ratio-scale", type=float, default=None, metavar="S",
                        help="Factor extra por fraccion tumoral del tile.")
    parser.add_argument("--eval-threshold", type=float, default=None, metavar="T",
                        help="Umbral fijo para Cancer (si no se tunea).")
    parser.add_argument("--no-threshold-tuning", action="store_true",
                        help="Desactiva busqueda de umbral en validacion.")
    parser.add_argument("--thr-min", type=float, default=None, help="Min threshold search.")
    parser.add_argument("--thr-max", type=float, default=None, help="Max threshold search.")
    parser.add_argument("--thr-step", type=float, default=None, help="Step threshold search.")
    parser.add_argument("--compile", action="store_true", help="Force torch.compile (Windows: can fail or use more VRAM).")
    parser.add_argument("--no-compile", action="store_true", help="Desactivar torch.compile.")
    parser.add_argument("--no-wandb", action="store_true", help="Do not log to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT_DEFAULT, help="Proyecto wandb.")
    parser.add_argument("--wandb-name", type=str, default=None, help="Nombre del run (optional).")
    parser.add_argument("--fold", type=str, nargs="+", default=None, choices=["Val1", "Val2", "Val3", "Val4"], help="Run one or multiple specific folds (e.g., Val1 Val2). If not set, all 4 folds run.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directorio para checkpoints .pth (by default: checkpoints_conch_masklut junto al script).",
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
    if args.sampler_cancer is not None:
        config["sampler_weight_cancer_present"] = args.sampler_cancer
    if args.sampler_no_cancer is not None:
        config["sampler_weight_cancer_absent"] = args.sampler_no_cancer
    if args.sampler_ratio_scale is not None:
        config["sampler_tumor_ratio_scale"] = args.sampler_ratio_scale
    if args.eval_threshold is not None:
        config["eval_threshold"] = float(args.eval_threshold)
    if args.no_threshold_tuning:
        config["tune_threshold"] = False
    if args.thr_min is not None:
        config["threshold_search_min"] = float(args.thr_min)
    if args.thr_max is not None:
        config["threshold_search_max"] = float(args.thr_max)
    if args.thr_step is not None:
        config["threshold_search_step"] = float(args.thr_step)
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
        f"{config['class_weights']} | pos_weight={config['pos_weight']:.3f} | "
        f"dice/ce: {config['dice_weight']}/{config['ce_weight']} | "
        f"thr_tuning={config.get('tune_threshold', True)}"
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
                "pos_weight": float(config["pos_weight"]),
                "unfreeze_last": config["unfreeze_last"],
                "weighted_sampler": config.get("use_weighted_sampler", False),
                "sampler_weight_cancer_present": config.get("sampler_weight_cancer_present"),
                "sampler_weight_cancer_absent": config.get("sampler_weight_cancer_absent"),
                "sampler_tumor_ratio_scale": config.get("sampler_tumor_ratio_scale"),
                "eval_threshold": config.get("eval_threshold"),
                "tune_threshold": config.get("tune_threshold"),
                "threshold_search_min": config.get("threshold_search_min"),
                "threshold_search_max": config.get("threshold_search_max"),
                "threshold_search_step": config.get("threshold_search_step"),
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
            print(f"  [WARN] Weights & Biases unavailable: {e}")
            use_wandb = False

    if args.fold:
        fold_names = args.fold if isinstance(args.fold, list) else [args.fold]
    else:
        fold_names = ["Val1", "Val2", "Val3", "Val4"]
        
    aggregated_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    best_thresholds = []
    for fold in fold_names:
        res = train_fold(fold, config, device, args.dry_run, use_wandb=use_wandb)
        if res["best_cm"] is not None:
            aggregated_cm += res["best_cm"]
        best_thresholds.append(float(res.get("best_threshold", config.get("eval_threshold", 0.5))))

    print_aggregated_matrices(aggregated_cm)
    if best_thresholds:
        print(f"Best thresholds by fold: {[round(x, 3) for x in best_thresholds]}")
        print(f"Mean best threshold: {float(np.mean(best_thresholds)):.3f}")
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
