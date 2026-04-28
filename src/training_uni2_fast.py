"""
SICAPv2 Segmentation — UNI2 Fast Training (decoder-only)
=========================================================
Requires running first:
    python precompute_features.py

This script loads pre-cached features de UNI2-h desde disco
and trains only the decoder FPN. Al no ejecutar el encoder en cada batch,
training is ~100x faster.

Usage:
    python src/training_uni2_fast.py             # by default: batch 4 × accum 4 (~8GB)
    python src/training_uni2_fast.py --dry-run   # smoke test
    python src/training_uni2_fast.py --batch-size 8 --grad-accum 2 # if you have spare VRAM
    python src/training_uni2_fast.py --no-weighted-sampler         # as before
    python src/training_uni2_fast.py --cache-dir "D:\\cache\\uni2_features"

Features cache: by default ./uni2_features. Si existe
    %LOCALAPPDATA%\\SICAPv2\\uni2_features with .pt files, it is used automatically (sin flags).
    Also: set UNI2_FEATURES_DIR=... o python src/training_uni2_fast.py --cache-dir PATH
    (in CMD do not use $env:...; usa %LOCALAPPDATA%\\SICAPv2\\uni2_features).
"""

import os
import time
import argparse
import warnings
from pathlib import Path

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

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Paths & Config
# ─────────────────────────────────────────────────────────────────────────────
from paths import (
    IMAGES_DIR,
    MASKS_DIR,
    PARTITION_DIR,
    default_checkpoint_dir,
    uni2_features_default_cache,
)


def _default_cache_dir() -> Path:
    env = os.environ.get("UNI2_FEATURES_DIR")
    if env:
        return Path(env).expanduser().resolve()
    # Typical local copy (robocopy a AppData): avoids OneDrive without CLI flags
    la = os.environ.get("LOCALAPPDATA")
    if la:
        local = Path(la) / "SICAPv2" / "uni2_features"
        if local.is_dir():
            try:
                with os.scandir(local) as it:
                    for e in it:
                        if e.is_file() and e.name.endswith(".pt"):
                            return local.resolve()
            except OSError:
                pass
    return uni2_features_default_cache().resolve()


CACHE_DIR     = _default_cache_dir()
OUTPUT_DIR    = default_checkpoint_dir("checkpoints_uni2_fast")

NUM_CLASSES = 4
CLASS_NAMES = ["NC", "GG3", "GG4", "GG5"]
IMG_SIZE    = 504
EMBED_DIM   = 1536

# 8GB VRAM profile: micro-batch bajo + accumulation (batch 8×4 tensores 1536C OOM en ~8GB).
# If you see: c10_cuda_check_implementation / CUDAEvent::record → it is usually OOM; close other Python GPU processes.
DEFAULT_CONFIG = {
    "num_classes"   : NUM_CLASSES,
    "num_workers"   : 4,
    "batch_size"    : 4,
    "grad_accum_steps": 4,   # effective ≈ 16; increase batch con --batch-size 6/8 if you have spare VRAM
    "learning_rate" : 6e-5,
    "weight_decay"  : 1e-4,
    "max_epochs"    : 100,
    "patience"      : 30,
    "dice_weight"   : 0.5,
    "ce_weight"     : 0.5,
    "class_weights" : [1.0, 3.0, 4.5, 18.0],
    "fpn_channels"  : 256,
    "use_weighted_sampler": True,
}

# ─────────────────────────────────────────────────────────────────────────────
# Mask LUT — identical to the original
# ─────────────────────────────────────────────────────────────────────────────
_MASK_LUT = np.zeros(256, dtype=np.int64)
_MASK_LUT[43:85] = 1
_MASK_LUT[85:160] = 2
_MASK_LUT[160:] = 3


def compute_sample_weights(image_names: list, masks_dir: Path):
    """Imágenes con pixeles GG5 → peso 5; only GG4 → 2; rest 1 (mismo criterio que training_pcam)."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — carga features pre-cacheadas + mask con augmentations
# ─────────────────────────────────────────────────────────────────────────────
class CachedFeaturesDataset(Dataset):
    """
    Devuelve (features_dict, mask) donde features_dict contiene
    los 4 tensors float16 pre-computados por UNI2-h.
    Las augmentations geométricas se aplican tanto a la mask
    como a los 4 feature maps para mantener consistencia espacial.
    """
    def __init__(self, image_names, masks_dir, cache_dir, augment=False):
        self.image_names = image_names
        self.masks_dir   = masks_dir
        self.cache_dir   = cache_dir
        self.augment     = augment

    def __len__(self): return len(self.image_names)

    def _load_mask(self, name):
        mask_path = self.masks_dir / name
        if mask_path.exists():
            buf  = np.fromfile(str(mask_path), dtype=np.uint8)
            mask = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        else:
            mask = None
        if mask is None:
            mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        # Resize mask al mismo tamaño que las features se computaron
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        return _MASK_LUT[mask]

    @staticmethod
    def _geom_augment_pair(mask_np, feat_list):
        """
        Misma geometría en mask (H,W) y en cada feature [1,C,h,w]:
        flip horizontal, flip vertical, rotation 90°×k — same probabilities
        que el previous pipeline only-mask.
        """
        flip_lr = np.random.rand() < 0.5
        flip_ud = np.random.rand() < 0.5
        do_rot = np.random.rand() < 0.7
        rot_k = int(np.random.choice([0, 1, 2, 3])) if do_rot else 0

        if flip_lr:
            mask_np = np.fliplr(mask_np).copy()
        if flip_ud:
            mask_np = np.flipud(mask_np).copy()
        if do_rot:
            mask_np = np.rot90(mask_np, rot_k).copy()

        out = []
        for x in feat_list:
            if flip_lr:
                x = torch.flip(x, dims=[3])
            if flip_ud:
                x = torch.flip(x, dims=[2])
            if do_rot:
                x = torch.rot90(x, rot_k, dims=[2, 3])
            out.append(x)
        return mask_np, out

    def __getitem__(self, idx):
        name = self.image_names[idx]

        # Load features pre-cacheadas
        feat_path = self.cache_dir / (name + ".pt")
        if not feat_path.exists():
            raise FileNotFoundError(
                f"Feature cache no encontrada: {feat_path}\n"
                f"Ejecuta primero: python precompute_features.py"
            )
        feats = torch.load(feat_path, weights_only=True)  # dict: f0,f1,f2,f3

        # Load mask
        mask = self._load_mask(name)

        feat_list = [feats["f0"].float(), feats["f1"].float(),
                     feats["f2"].float(), feats["f3"].float()]

        if self.augment:
            mask, feat_list = self._geom_augment_pair(mask, feat_list)

        mask_tensor = torch.from_numpy(mask).long()

        return feat_list, mask_tensor


def collate_fn(batch):
    """Agrupa lista de (feat_list, mask) en batches."""
    feat_lists, masks = zip(*batch)
    # Cada feat_lists[b][i] tiene shape [1, 1536, H, W] → squeeze dim 0 antes de apilar
    batched_feats = [
        torch.stack([feat_lists[b][i].squeeze(0) for b in range(len(feat_lists))], dim=0)
        for i in range(4)
    ]
    batched_masks = torch.stack(masks, dim=0)
    return batched_feats, batched_masks


def get_fold_dataloaders(fold_name, config):
    fold_dir    = PARTITION_DIR / "Validation" / fold_name
    train_names = pd.read_excel(fold_dir / "Train.xlsx")["image_name"].tolist()
    val_names   = pd.read_excel(fold_dir / "Test.xlsx")["image_name"].tolist()

    train_ds = CachedFeaturesDataset(train_names, MASKS_DIR, CACHE_DIR, augment=True)
    val_ds   = CachedFeaturesDataset(val_names,   MASKS_DIR, CACHE_DIR, augment=False)

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", config["num_workers"]))
    # En Windows, num_workers > 0 con tensors en shared memory causa crashes
    if os.name == 'nt':
        workers = 0
    kwargs  = {"persistent_workers": True, "prefetch_factor": 4} if workers > 0 else {}

    use_sampler = config.get("use_weighted_sampler", False)
    if use_sampler:
        sample_weights = compute_sample_weights(train_names, MASKS_DIR)
        n_gg5 = sum(1 for w in sample_weights if w == 5.0)
        n_gg4 = sum(1 for w in sample_weights if w == 2.0)
        print(f"  [Sampler] train={len(train_names)} | GG5 imgs×5={n_gg5} | GG4-only×2={n_gg4}")
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_names),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=config["batch_size"], sampler=sampler,
            num_workers=workers, pin_memory=True, drop_last=True,
            collate_fn=collate_fn, **kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=config["batch_size"], shuffle=True,
            num_workers=workers, pin_memory=True, drop_last=True,
            collate_fn=collate_fn, **kwargs,
        )
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False,
                              num_workers=workers, pin_memory=True,
                              collate_fn=collate_fn, **kwargs)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# FPN Decoder — idéntico a training_uni2.py
# ─────────────────────────────────────────────────────────────────────────────
class FPNDecoder(nn.Module):
    def __init__(self, in_channels=EMBED_DIM, fpn_channels=256, num_classes=4):
        super().__init__()
        self.lat = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, fpn_channels, 1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
            ) for _ in range(4)
        ])
        self.merge = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.ReLU(inplace=True),
            ) for _ in range(3)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels // 2, num_classes, 1),
        )

    def forward(self, features, target_size):
        lats = [lat(f) for lat, f in zip(self.lat, features)]
        x = lats[3]
        for i in range(2, -1, -1):
            x = F.interpolate(x, size=lats[i].shape[-2:], mode="bilinear", align_corners=False)
            x = self.merge[i](x + lats[i])
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.head(x)


def build_model(config):
    return FPNDecoder(
        in_channels  = EMBED_DIM,
        fpn_channels = config["fpn_channels"],
        num_classes  = config["num_classes"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Loss — identical al original
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
        return self.dice_weight * self.dice_loss(logits, targets) \
             + self.ce_weight   * self.ce_loss(logits, targets)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics — identicals al original
# ─────────────────────────────────────────────────────────────────────────────
class SegmentationMetrics:
    def __init__(self, num_classes):
        self.num_classes      = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update_batch(self, logits, targets):
        preds = logits.argmax(dim=1).cpu().numpy()
        tgts  = targets.cpu().numpy()
        mask  = (tgts >= 0) & (tgts < self.num_classes)
        np.add.at(self.confusion_matrix, (tgts[mask], preds[mask]), 1)

    def compute(self):
        cm = self.confusion_matrix
        dice_per_class = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
            if tp + fp + fn == 0:
                dice_per_class[c] = np.nan
            else:
                dice_per_class[c] = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
        macro_f1       = np.nanmean(dice_per_class)
        dice_per_class = np.nan_to_num(dice_per_class, nan=0.0)
        return {"macro_f1": macro_f1, "f1_per_class": dice_per_class,
                "confusion_matrix": cm.copy()}


# ─────────────────────────────────────────────────────────────────────────────
# Train / Val Epochs
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_accum_steps: int = 1):
    model.train()
    total_loss, n = 0.0, 0
    accum = 0
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    target_size = (IMG_SIZE, IMG_SIZE)
    print(
        "  [Train] Waiting for first batch (reading .pt from disk)…",
        flush=True,
    )
    t_wait = time.perf_counter()
    pbar = tqdm(loader, desc="  Train", leave=False, mininterval=0.5)
    for feats, masks in pbar:
        if n == 0:
            print(
                f"  [Train] First batch loaded after {time.perf_counter() - t_wait:.1f}s.",
                flush=True,
            )
            print(
                "  [Train] GPU step running — bar stays at 0% until forward+backward+opt finish "
                "(CUDA is async; first step can take minutes).",
                flush=True,
            )
            t_step0 = time.perf_counter()
        feats = [f.to(device, non_blocking=True) for f in feats]
        masks = masks.to(device, non_blocking=True).long()
        if accum == 0:
            optimizer.zero_grad(set_to_none=True)
        t_fwd = time.perf_counter()
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            logits = model(feats, target_size)
            loss   = criterion(logits, masks)
        if n == 0 and device.type == "cuda":
            torch.cuda.synchronize()
        if n == 0:
            print(f"  [Train] Forward+loss: {time.perf_counter() - t_fwd:.1f}s", flush=True)
        t_bwd = time.perf_counter()
        scaler.scale(loss / grad_accum_steps).backward()
        if n == 0 and device.type == "cuda":
            torch.cuda.synchronize()
        if n == 0:
            print(f"  [Train] Backward: {time.perf_counter() - t_bwd:.1f}s", flush=True)
        total_loss += loss.item()
        n += 1
        accum += 1
        stepped = False
        if accum >= grad_accum_steps:
            t_opt = time.perf_counter()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accum = 0
            stepped = True
            if n <= grad_accum_steps and device.type == "cuda":
                torch.cuda.synchronize()
            if n <= grad_accum_steps:
                print(f"  [Train] Optimizer (accum×{grad_accum_steps}): {time.perf_counter() - t_opt:.1f}s", flush=True)
        if n == 1 and stepped:
            print(
                f"  [Train] First optimizer step after {grad_accum_steps} micro-batches — "
                f"{time.perf_counter() - t_step0:.1f}s total.",
                flush=True,
            )
        elif n == 1 and not stepped:
            print("  [Train] Waiting for more micro-batches (grad accum)…", flush=True)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    if accum > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return total_loss / max(n, 1)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    metrics   = SegmentationMetrics(NUM_CLASSES)
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    target_size = (IMG_SIZE, IMG_SIZE)
    print("  [Val] Waiting for first batch…", flush=True)
    pbar = tqdm(loader, desc="  Val  ", leave=False, mininterval=0.5)
    for feats, masks in pbar:
        feats = [f.to(device, non_blocking=True) for f in feats]
        masks = masks.to(device, non_blocking=True).long()
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            logits = model(feats, target_size)
            loss   = criterion(logits, masks)
        total_loss += loss.item(); n += 1
        metrics.update_batch(logits, masks)
    return total_loss / max(n, 1), metrics.compute()


# ─────────────────────────────────────────────────────────────────────────────
# Train Fold
# ─────────────────────────────────────────────────────────────────────────────
def train_fold(fold_name, config, device, dry_run=False):
    print(f"\n{'='*60}\n  FOLD: {fold_name}\n{'='*60}")
    train_loader, val_loader = get_fold_dataloaders(fold_name, config)

    model     = build_model(config).to(device)
    criterion = GuidedLoss(config["class_weights"],
                           config["dice_weight"], config["ce_weight"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6, min_lr=1e-7)
    scaler    = GradScaler("cuda", enabled=(device.type == "cuda"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1, patience_ctr, best_cm = 0.0, 0, None
    max_epochs = 1 if dry_run else config["max_epochs"]

    for epoch in range(1, max_epochs + 1):
        print(f"\n  Epoch {epoch}/{max_epochs}  (lr={optimizer.param_groups[0]['lr']:.2e})")
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            grad_accum_steps=config.get("grad_accum_steps", 1),
        )
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        macro_f1 = val_metrics["macro_f1"]
        scheduler.step(macro_f1)
        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss:   {val_loss:.4f}")
        print(f"  Macro F1:   {macro_f1:.4f}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"    {name} F1: {val_metrics['f1_per_class'][i]:.4f}")

        if macro_f1 > best_f1:
            best_f1, best_cm, patience_ctr = macro_f1, val_metrics["confusion_matrix"], 0
            torch.save(model.state_dict(), OUTPUT_DIR / f"best_{fold_name}_{macro_f1:.4f}.pth")
            print(f"  ✓ Model saved (Macro F1={macro_f1:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= config["patience"]:
                print(f"  ⛔ Early stopping at epoch {epoch}")
                break

        if dry_run: break

    return {"fold": fold_name, "best_macro_f1": best_f1, "best_cm": best_cm}


# ─────────────────────────────────────────────────────────────────────────────
# Reporting — identical to the original
# ─────────────────────────────────────────────────────────────────────────────
def print_aggregated_matrices(agg_cm):
    print(f"\n{'='*60}\n  AGGREGATED CONFUSION MATRICES (ALL 4 FOLDS)\n{'='*60}")
    df = pd.DataFrame(agg_cm, index=[f"T_{c}" for c in CLASS_NAMES],
                      columns=[f"P_{c}" for c in CLASS_NAMES])
    print("\n[1] 4-Class Confusion Matrix"); print(df.to_string())
    print("\n--- 4-Class Metrics ---")
    for i in range(4):
        tp = agg_cm[i,i]; fp = agg_cm[:,i].sum()-tp; fn = agg_cm[i,:].sum()-tp
        p = tp/(tp+fp+1e-8); r = tp/(tp+fn+1e-8)
        print(f"  {CLASS_NAMES[i]:3s}: F1={2*p*r/(p+r+1e-8):.4f}  Prec={p:.4f}  Rec={r:.4f}")
    nc_nc,nc_c = agg_cm[0,0], agg_cm[0,1:].sum()
    c_nc, c_c  = agg_cm[1:,0].sum(), agg_cm[1:,1:].sum()
    df2 = pd.DataFrame([[nc_nc,nc_c],[c_nc,c_c]],
                       index=["T_NoCancer","T_Cancer"],
                       columns=["P_NoCancer","P_Cancer"])
    print("\n[2] Binary Confusion Matrix"); print(df2.to_string())
    b_tp,b_fp,b_tn,b_fn = c_c,nc_c,nc_nc,c_nc
    b_p = b_tp/(b_tp+b_fp+1e-8); b_r = b_tp/(b_tp+b_fn+1e-8)
    print(f"\n  Cancer F1   : {2*b_p*b_r/(b_p+b_r+1e-8):.4f}")
    print(f"  Accuracy    : {(b_tp+b_tn)/(b_tp+b_tn+b_fp+b_fn+1e-8):.4f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global CACHE_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Carpeta con los .pt de UNI2 (sustituye UNI2_FEATURES_DIR y ./uni2_features).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Micro-batch en GPU (by default 4). Si OOM: 2; si sobra VRAM: 6 u 8.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=None,
        metavar="K",
        help="Acumular K micro-batches antes de optimizer.step (by default 4).",
    )
    parser.add_argument(
        "--no-weighted-sampler",
        action="store_true",
        help="Desactiva oversampling de images con GG5/GG4.",
    )
    args = parser.parse_args()
    if args.cache_dir is not None:
        CACHE_DIR = args.cache_dir.expanduser().resolve()

    run_config = dict(DEFAULT_CONFIG)
    if args.batch_size is not None:
        run_config["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        run_config["grad_accum_steps"] = max(1, args.grad_accum)
    if args.no_weighted_sampler:
        run_config["use_weighted_sampler"] = False

    if os.name == 'nt':
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
            print("\n[System] Anti-sleep enabled.")
        except: pass

    # any(glob) para en el primer .pt — no contar ~10k archivos (OneDrive parece colgado).
    if not CACHE_DIR.exists() or not any(CACHE_DIR.glob("*.pt")):
        print("❌ Not found el cache de features.")
        print("   Ejecuta primero: python precompute_features.py")
        return

    print(f"✅ Feature cache: {CACHE_DIR}")
    if "OneDrive" in str(CACHE_DIR):
        print(
            "   Note: cache is in OneDrive (I/O lento). Copia a disco local o usa\n"
            "   %LOCALAPPDATA%\\SICAPv2\\uni2_features — se usa only si existe y hay .pt.",
            flush=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        free_b, total_b = torch.cuda.mem_get_info()
        print(
            f"VRAM libre al inicio: {free_b / 2**20:.0f} / {total_b / 2**20:.0f} MiB "
            f"(close other processes Python en GPU si free < ~800 MiB)",
            flush=True,
        )
    print(
        f"Train: batch={run_config['batch_size']} × accum={run_config.get('grad_accum_steps', 1)} "
        f"≈ effective {run_config['batch_size'] * run_config.get('grad_accum_steps', 1)} | "
        f"weighted_sampler={run_config.get('use_weighted_sampler', False)} | lr={run_config['learning_rate']}"
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    fold_names    = ["Val1", "Val2", "Val3", "Val4"]
    aggregated_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    for fold in fold_names:
        res = train_fold(fold, run_config, device, args.dry_run)
        if res["best_cm"] is not None:
            aggregated_cm += res["best_cm"]

    print_aggregated_matrices(aggregated_cm)
    if args.dry_run:
        print("\n✅ Dry run completed successfully!")

if __name__ == "__main__":
    main()