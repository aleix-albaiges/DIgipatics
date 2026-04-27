"""
SICAPv2 Semantic Segmentation Training Pipeline
================================================
PCam Transfer Learning: Attention-Enhanced Hybrid U-Net
- Architecture: ResNet50 (PCam pretrained) + MHSA Bottleneck + ASPP + SE-UNet Decoder
- Loss: Focal CE + Class-Weighted Dice Loss (aggressive GG5 weighting)
- Augmentations: Random rotations, flips, contrast, color jitter, elastic, Gaussian noise
- Tracked Metric: Macro F-score (Macro Dice)
- Confusion Matrices explicitly aggregated.

Changes vs original:
  1. Fixed PCam weight loading (proper key remapping + diagnostics)
  2. Differential learning rates (encoder 10x lower)
  3. Focal Loss + aggressive GG5 weighting (20.0)
  4. Reduced early stopping patience (15)
  5. CosineAnnealingWarmRestarts scheduler
  6. Stronger augmentations (color jitter, elastic, stain-like)
  7. GG5-aware oversampling via WeightedRandomSampler

Usage:
    python training_pcam.py                # full training (4 folds)
    python training_pcam.py --dry-run      # smoke test (1 batch, 1 fold)
"""

import os
import argparse
import warnings
from pathlib import Path
import wandb

import numpy as np
import pandas as pd
import cv2
cv2.setNumThreads(0) # CRITICAL: Prevents thread contention deadlocks with PyTorch DataLoader workers!
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
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
MASKS_DIR = BASE_DIR / "masks"
PARTITION_DIR = BASE_DIR / "partition"
OUTPUT_DIR = BASE_DIR / "checkpoints_nature_pcam"

NUM_CLASSES = 4
CLASS_NAMES = ["NC", "GG3", "GG4", "GG5"]
IMG_SIZE = 512

DEFAULT_CONFIG = {
    "encoder": "resnet50",
    "decoder": "Unet",
    "pretrained_weights": "pcam",
    "num_classes": NUM_CLASSES,
    "batch_size": 8,
    "num_workers": 4,
    "learning_rate": 1e-4,         # Decoder + bottleneck LR
    "encoder_lr": 1e-5,            # Encoder LR 10x lower to preserve PCam features
    "weight_decay": 1e-3,          # [V5] Increased from 1e-4 to combat overfitting seen in val_loss
    "max_epochs": 100,
    "patience": 20,
    "ce_weight": 0.5,
    "lovasz_weight": 0.5,          # [V5] Replaces Dice/OHEM/Tversky. Directly optimizes IoU.
    "class_weights": [1.0, 2.0, 5.0, 10.0], # [V5] Moderate weights since Lovasz handles the imbalance
    "label_smoothing": 0.05,
    "gg5_inference_threshold": 0.50, # [V5.1] Lowered from 0.95 to 0.50 to make it easier to predict GG5 while still requiring absolute majority confidence
}

# Easy-to-try encoder presets. All except "pcam_resnet50" use native SMP/timm pretrained weights.
ENCODER_PRESETS = {
    "pcam_resnet50": {"encoder": "resnet50", "pretrained_weights": "pcam"},
    "imagenet_resnet50": {"encoder": "resnet50", "pretrained_weights": "imagenet"},
    "imagenet_resnet101": {"encoder": "resnet101", "pretrained_weights": "imagenet"},
    "imagenet_efficientnetb3": {"encoder": "timm-efficientnet-b3", "pretrained_weights": "imagenet"},
    "imagenet_convnext_base": {"encoder": "timm-convnext_base", "pretrained_weights": "imagenet"},
}

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────────────────────────────────────────
# Dataset (Reused Logic)
# ─────────────────────────────────────────────────────────────────────────────
_MASK_LUT = np.zeros(256, dtype=np.int64)
_MASK_LUT[43:85] = 1
_MASK_LUT[85:160] = 2
_MASK_LUT[160:] = 3

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
# Augmentations — [CHANGE 6] Stronger augmentations for H&E histopathology
# ─────────────────────────────────────────────────────────────────────────────
def get_train_transforms():
    """
    [CHANGE 6] Added ColorJitter (simulates H&E stain variation across labs/scanners),
    ElasticTransform (simulates tissue deformation), and HueSaturationValue (stain 
    normalization robustness). These are standard in computational pathology and address
    the overfitting gap observed in training (train loss 0.25 vs val loss 0.43).
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=0.5),
        # NEW: Simulate H&E stain variation (hue/saturation shifts)
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.4),
        # NEW: Elastic deformation for tissue-level variation
        A.ElasticTransform(alpha=80, sigma=50, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading — [CHANGE 7] GG5-aware oversampling
# ─────────────────────────────────────────────────────────────────────────────
def compute_sample_weights(image_names: list, masks_dir: Path):
    """
    [CHANGE 7] Compute per-sample weights for WeightedRandomSampler.
    Images containing GG5 pixels get 5x higher sampling probability.
    Images containing GG4 get 2x. This ensures the model sees enough 
    rare-class examples per epoch to generate gradient signal for GG5.
    """
    weights = []
    for name in image_names:
        mask_path = masks_dir / name
        w = 1.0  # default weight
        if mask_path.exists():
            buf_m = np.fromfile(str(mask_path), dtype=np.uint8)
            mask = cv2.imdecode(buf_m, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mapped = _MASK_LUT[mask]
                if np.any(mapped == 3):       # Contains GG5
                    w = 5.0
                elif np.any(mapped == 2):     # Contains GG4
                    w = 2.0
        weights.append(w)
    return weights

def get_fold_dataloaders(fold_name: str, config: dict):
    fold_dir = PARTITION_DIR / "Validation" / fold_name
    train_df = pd.read_excel(fold_dir / "Train.xlsx")
    val_df = pd.read_excel(fold_dir / "Test.xlsx")
    train_names = train_df["image_name"].tolist()
    val_names = val_df["image_name"].tolist()

    train_ds = SICAPv2Dataset(train_names, IMAGES_DIR, MASKS_DIR, transform=get_train_transforms())
    val_ds = SICAPv2Dataset(val_names, IMAGES_DIR, MASKS_DIR, transform=get_val_transforms())

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", config["num_workers"]))
    kwargs = {"persistent_workers": True, "prefetch_factor": 4} if workers > 0 else {}

    # [CHANGE 7] Weighted sampling to oversample GG5/GG4 images
    print(f"  Computing sample weights for {len(train_names)} training images...")
    sample_weights = compute_sample_weights(train_names, MASKS_DIR)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_names),  # Same epoch length
        replacement=True               # Required for oversampling
    )
    
    # Count oversampling stats
    n_gg5 = sum(1 for w in sample_weights if w == 5.0)
    n_gg4 = sum(1 for w in sample_weights if w == 2.0)
    n_other = len(sample_weights) - n_gg5 - n_gg4
    print(f"  Sample weights: {n_gg5} GG5 images (5x), {n_gg4} GG4-only images (2x), {n_other} other (1x)")
    
    # NOTE: Using sampler instead of shuffle=True (they are mutually exclusive)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], sampler=sampler,
                              num_workers=workers, pin_memory=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, 
                            num_workers=workers, pin_memory=True, **kwargs)
    return train_loader, val_loader

# ─────────────────────────────────────────────────────────────────────────────
# TAH U-Net Architecture (Hybrid ViT-CNN Approximation)
# ─────────────────────────────────────────────────────────────────────────────
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[3, 6, 9]):
        super().__init__()
        mid_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        ))
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            ))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)  # [CHANGE 6] Reduced from 0.5 — was too aggressive and hurting learning
        )
        
    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        return self.project(torch.cat(res, dim=1))

class AttentionBottleneck(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(embed_dim=in_channels, num_heads=8, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(in_channels)  # Post-attention normalization for stability
        self.aspp = ASPP(in_channels, in_channels)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1) # B, Seq, Embedded
        attn_out, _ = self.mhsa(x_flat, x_flat, x_flat)
        attn_out = self.norm(attn_out)  # Stabilize attention output
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        
        # Add residual and pass to ASPP
        out = self.aspp(attn_out + x)
        return out

class TAH_UNet(nn.Module):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet", num_classes=4):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            decoder_attention_type="scse",
            in_channels=3,
            classes=num_classes
        )
        self.bottleneck = AttentionBottleneck(in_channels=2048)

    def forward(self, x):
        features = self.unet.encoder(x)
        c5 = features[-1]
        c5_attn = self.bottleneck(c5)
        
        features_list = list(features)
        features_list[-1] = c5_attn
        
        decoder_output = self.unet.decoder(features_list)
        masks = self.unet.segmentation_head(decoder_output)
        return masks

# ─────────────────────────────────────────────────────────────────────────────
# [CHANGE 1] Fixed PCam Weight Loading with proper key remapping + diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def build_model(config: dict):
    """
    [CHANGE 1] The original code used `strict=False` which silently ignored key 
    mismatches between timm's state_dict and smp's encoder. The smp `tu-resnet50` 
    encoder wraps timm under a `.model` prefix, so timm keys like `layer1.0.conv1.weight`
    don't match smp keys like `model.layer1.0.conv1.weight`. This fix:
    1. Loads the smp model with NO pretrained weights (None)
    2. Downloads the timm PCam model
    3. Remaps timm keys → smp encoder keys by adding `model.` prefix
    4. Prints diagnostic: how many keys matched vs skipped
    """
    use_pcam = config["pretrained_weights"] == "pcam"
    if use_pcam and config["encoder"] != "resnet50":
        print(f"  [WARN] PCam only supports resnet50. Got encoder={config['encoder']}. Falling back to imagenet weights.")
        use_pcam = False
        config["pretrained_weights"] = "imagenet"

    smp_weights = None if use_pcam else config["pretrained_weights"]
    encoder_name = "tu-resnet50" if (use_pcam and config["encoder"] == "resnet50") else config["encoder"]
    
    model = TAH_UNet(encoder_name=encoder_name, encoder_weights=smp_weights, num_classes=config["num_classes"])
    
    if use_pcam:
        print("  -> Loading ResNet50 weights pretrained on PatchCamelyon (PCam)...")
        import timm
        timm_model = timm.create_model("hf_hub:1aurent/resnet50.tiatoolbox-pcam", pretrained=True)
        timm_sd = timm_model.state_dict()
        
        # Get the smp encoder's expected keys
        encoder_sd = model.unet.encoder.state_dict()
        
        # Build remapped state dict: timm key → "model." + timm key (smp's tu- prefix)
        remapped_sd = {}
        matched, skipped = 0, 0
        
        for timm_key, timm_val in timm_sd.items():
            # Skip classifier head keys (fc.weight, fc.bias)
            if timm_key.startswith("fc.") or timm_key.startswith("classifier.") or timm_key.startswith("head."):
                skipped += 1
                continue
            
            # Try direct match first
            if timm_key in encoder_sd and encoder_sd[timm_key].shape == timm_val.shape:
                remapped_sd[timm_key] = timm_val
                matched += 1
                continue
                
            # Try with "model." prefix (smp's tu- wrapper)
            smp_key = f"model.{timm_key}"
            if smp_key in encoder_sd and encoder_sd[smp_key].shape == timm_val.shape:
                remapped_sd[smp_key] = timm_val
                matched += 1
                continue
            
            skipped += 1
        
        # Load the remapped weights
        load_result = model.unet.encoder.load_state_dict(remapped_sd, strict=False)
        
        total_encoder_keys = len(encoder_sd)
        print(f"  [DIAGNOSTIC] PCam weight loading:")
        print(f"    Timm keys total:    {len(timm_sd)}")
        print(f"    Matched & loaded:   {matched}/{total_encoder_keys} encoder keys")
        print(f"    Skipped (head/mismatch): {skipped}")
        print(f"    Missing in encoder: {len(load_result.missing_keys)}")
        
        if matched == 0:
            print("  [WARN] ZERO keys matched! PCam weights NOT loaded. Check key format.")
        elif matched < total_encoder_keys * 0.5:
            print(f"  [WARN] Only {matched}/{total_encoder_keys} keys matched. Partial load.")
        else:
            print(f"  [OK] PCam weights successfully injected ({matched}/{total_encoder_keys} keys).")
        
    return model


# ─────────────────────────────────────────────────────────────────────────────
# [V5] Lovasz-Softmax + Standard CE Loss
# ─────────────────────────────────────────────────────────────────────────────
class GuidedLoss(nn.Module):
    """
    [V5] Replaced OHEM and Tversky with Lovasz-Softmax.
    Lovasz directly and smoothly optimizes the Jaccard index (Intersection over Union).
    It is the state-of-the-art method for extreme class imbalance (like Cityscapes) 
    because it sorts pixel errors and penalizes them relative to the class size,
    without needing aggressive hard-mining or massive class weights.
    """
    def __init__(self, class_weights: list, ce_weight=0.5, lovasz_weight=0.5, label_smoothing=0.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        # mode="multiclass" natively handles the (B, C, H, W) logits cleanly
        self.lovasz_loss = smp.losses.LovaszLoss(mode="multiclass")
        self.register_buffer("weights_tensor", torch.tensor(class_weights).float())
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        logits = logits.float()  # Force float32 to prevent NaN from FP16
        
        ce = F.cross_entropy(
            logits, targets, weight=self.weights_tensor,
            label_smoothing=self.label_smoothing
        )
        
        lovasz = self.lovasz_loss(logits, targets)
        
        return self.ce_weight * ce + self.lovasz_weight * lovasz

# ─────────────────────────────────────────────────────────────────────────────
# Metrics & Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────
class SegmentationMetrics:
    def __init__(self, num_classes: int, gg5_threshold: float = 0.95):
        self.num_classes = num_classes
        self.gg5_threshold = gg5_threshold
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update_batch(self, logits: torch.Tensor, targets: torch.Tensor):
        # [V5] Asymmetric Thresholding Strategy
        probs = torch.softmax(logits, dim=1)
        
        # 1. Base prediction: argmax over NC, GG3, GG4 ONLY (classes 0, 1, 2)
        base_preds = probs[:, :self.num_classes-1].argmax(dim=1)
        
        # 2. Extract highly confident GG5 pixels
        gg5_conf_mask = probs[:, self.num_classes-1] > self.gg5_threshold
        
        # 3. Override base predictions with GG5 only where we are extremely certain
        preds = base_preds.clone()
        preds[gg5_conf_mask] = self.num_classes - 1

        mask = (targets >= 0) & (targets < self.num_classes)
        valid_targets = targets[mask]
        valid_preds = preds[mask]
        inds = self.num_classes * valid_targets + valid_preds
        hist = torch.bincount(inds, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += hist.cpu().numpy()

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
        macro_f1 = np.nanmean(dice_per_class)
        dice_per_class = np.nan_to_num(dice_per_class, nan=0.0)
        return {
            "macro_f1": macro_f1,
            "f1_per_class": dice_per_class,
            "confusion_matrix": cm.copy(),
        }

# ─────────────────────────────────────────────────────────────────────────────
# Training & Val Epochs
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, num_batches = 0.0, 0
    
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            loss = criterion(model(images), masks)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(num_batches, 1)

@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, gg5_threshold):
    model.eval()
    total_loss, num_batches = 0.0, 0
    metrics = SegmentationMetrics(NUM_CLASSES, gg5_threshold=gg5_threshold)
    
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True).long()
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits, masks)
        total_loss += loss.item()
        num_batches += 1
        metrics.update_batch(logits, masks)
    return total_loss / max(num_batches, 1), metrics.compute()

# ─────────────────────────────────────────────────────────────────────────────
# Train Flow
# ─────────────────────────────────────────────────────────────────────────────
def train_fold(fold_name: str, config: dict, device: torch.device, dry_run: bool = False):
    print(f"\n{'='*60}\n  FOLD: {fold_name}\n{'='*60}")
    train_loader, val_loader = get_fold_dataloaders(fold_name, config)
    
    model = build_model(config).to(device)
    
    # PyTorch 2.x JIT Compilation
    if int(torch.__version__.split('.')[0]) >= 2:
        try:
            print("  [WAIT] Compiling model (may take 1-2 mins to start)...")
            import platform
            backend = "inductor" if platform.system() != "Windows" else "aot_eager"
            model = torch.compile(model, backend=backend)
            print("  [OK] Compilation enabled.")
        except Exception as e:
            print(f"  [WARN] torch.compile not applicable: {e}")
            
    criterion = GuidedLoss(
        config["class_weights"], 
        ce_weight=config["ce_weight"],
        lovasz_weight=config["lovasz_weight"],
        label_smoothing=config["label_smoothing"]
    ).to(device)
    
    # ─────────────────────────────────────────────────────────────────────────
    # [CHANGE 2] Differential Learning Rates
    # ─────────────────────────────────────────────────────────────────────────
    # The pretrained PCam encoder should be fine-tuned with a much lower LR (1e-5)
    # to avoid catastrophic forgetting. The randomly-initialized decoder/bottleneck
    # needs the full LR (1e-4) to learn quickly. This is standard transfer learning practice.
    encoder_params = list(model.parameters())[:len(list(model.unet.encoder.parameters())) if not hasattr(model, '_orig_mod') else len(list(model._orig_mod.unet.encoder.parameters()))]
    
    # Safer approach: use named parameter groups
    encoder_param_ids = set()
    encoder_module = model._orig_mod.unet.encoder if hasattr(model, '_orig_mod') else model.unet.encoder
    decoder_module = model._orig_mod.unet.decoder if hasattr(model, '_orig_mod') else model.unet.decoder
    seghead_module = model._orig_mod.unet.segmentation_head if hasattr(model, '_orig_mod') else model.unet.segmentation_head
    bottleneck_module = model._orig_mod.bottleneck if hasattr(model, '_orig_mod') else model.bottleneck
    
    for p in encoder_module.parameters():
        encoder_param_ids.add(id(p))
    
    encoder_params = [p for p in model.parameters() if id(p) in encoder_param_ids]
    non_encoder_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
    
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": config["encoder_lr"]},        # PCam encoder: gentle LR
        {"params": non_encoder_params, "lr": config["learning_rate"]},  # Decoder + bottleneck: full LR
    ], weight_decay=config["weight_decay"])
    
    print(f"  Optimizer: Encoder LR={config['encoder_lr']:.1e}, Decoder LR={config['learning_rate']:.1e}")
    print(f"  Encoder params: {sum(p.numel() for p in encoder_params):,}, Other params: {sum(p.numel() for p in non_encoder_params):,}")
    
    # [V3] Reverted to ReduceLROnPlateau — CosineAnnealing warm restarts
    # at epoch 16 caused massive val loss spikes (0.77->0.96).
    # Plateau is conservative: only drops LR when macro_f1 stops improving.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
    )
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_macro_f1, patience_counter, best_cm = 0.0, 0, None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    max_epochs = 1 if dry_run else config["max_epochs"]

    for epoch in range(1, max_epochs + 1):
        # Display both param group LRs
        enc_lr = optimizer.param_groups[0]['lr']
        dec_lr = optimizer.param_groups[1]['lr']
        print(f"\n  Epoch {epoch}/{max_epochs}  (enc_lr={enc_lr:.2e}, dec_lr={dec_lr:.2e})")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, device, config["gg5_inference_threshold"])
        
        macro_f1 = val_metrics["macro_f1"]
        scheduler.step(macro_f1)  # [V3] Step on metric, not epoch count
        
        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss:   {val_loss:.4f}")
        print(f"  Macro F1:   {macro_f1:.4f}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"    {name} F1: {val_metrics['f1_per_class'][i]:.4f}")

        if wandb.run is not None:
            metrics_dict = {
                f"{fold_name}/train_loss": train_loss,
                f"{fold_name}/val_loss": val_loss,
                f"{fold_name}/macro_f1": macro_f1,
                f"epoch": epoch,
                f"encoder_lr": enc_lr,
                f"decoder_lr": dec_lr,
            }
            for i, name in enumerate(CLASS_NAMES):
                metrics_dict[f"{fold_name}/f1_{name}"] = val_metrics['f1_per_class'][i]
            wandb.log(metrics_dict)

        if macro_f1 > best_macro_f1:
            best_macro_f1, best_cm, patience_counter = macro_f1, val_metrics["confusion_matrix"], 0
            torch.save(model.state_dict(), OUTPUT_DIR / f"best_{fold_name}_{macro_f1:.4f}.pth")
            print(f"  [+] Model saved (Macro F1={macro_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"  [STOP] Early stopping triggered at epoch {epoch}")
                break
                
        if dry_run: break
        
    return {"fold": fold_name, "best_macro_f1": best_macro_f1, "best_cm": best_cm}

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
    c_nc, c_c = agg_cm[1:, 0].sum(), agg_cm[1:, 1:].sum()
    
    df_2x2 = pd.DataFrame(np.array([[nc_nc, nc_c], [c_nc, c_c]]), index=["T_NoCancer", "T_Cancer"], columns=["P_NoCancer", "P_Cancer"])
    print(df_2x2.to_string())
    
    b_tp, b_fp, b_tn, b_fn = c_c, nc_c, nc_nc, c_nc
    b_prec, b_rec = b_tp / (b_tp + b_fp + 1e-8), b_tp / (b_tp + b_fn + 1e-8)
    b_f1 = 2 * (b_prec * b_rec) / (b_prec + b_rec + 1e-8)
    b_acc = (b_tp + b_tn) / (b_tp + b_tn + b_fp + b_fn + 1e-8)
    
    print("\n--- Binary Classification Metrics ---")
    print(f"  Cancer F1 (Macro) : {b_f1:.4f}")
    print(f"  Overall Accuracy  : {b_acc:.4f}\n")

def main():
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fold", type=str, default="all", help="Run a specific fold e.g. 'Val1'. Default is 'all'.")
    parser.add_argument(
        "--preset",
        type=str,
        default="pcam_resnet50",
        choices=list(ENCODER_PRESETS.keys()),
        help="Encoder + pretrained pair to use."
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Optional manual encoder override (e.g., resnet101, timm-convnext_base)."
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Optional manual pretrained override (pcam, imagenet, None)."
    )
    args = parser.parse_args()

    if os.name == 'nt':
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
            print("\n[System] Anti-sleep enabled.")
        except: pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_config = dict(DEFAULT_CONFIG)
    run_config.update(ENCODER_PRESETS[args.preset])
    if args.encoder is not None:
        run_config["encoder"] = args.encoder
    if args.pretrained is not None:
        run_config["pretrained_weights"] = None if args.pretrained.lower() == "none" else args.pretrained

    print(f"Model setup -> encoder={run_config['encoder']}, pretrained_weights={run_config['pretrained_weights']}")
    
    if not args.dry_run:
        try:
            run_name = f"Encoder_{run_config['encoder']}_{run_config['pretrained_weights']}_v5"
            wandb.init(project="SICAPv2_Segmentation", name=run_name, config=run_config)
        except Exception as e:
            print(f"  [WARN] Error activating Weights & Biases: {e}")
    
    if args.fold.lower() != "all":
        fold_names = [args.fold.capitalize()]  # Ensures 'val1' becomes 'Val1'
    else:
        fold_names = ["Val1", "Val2", "Val3", "Val4"]
        
    aggregated_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    
    for fold in fold_names:
        res = train_fold(fold, run_config, device, args.dry_run)
        if res["best_cm"] is not None:
            aggregated_cm += res["best_cm"]
            
    print_aggregated_matrices(aggregated_cm)
    if args.dry_run:
        print("\n✅ Dry run completed successfully!")
        
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
