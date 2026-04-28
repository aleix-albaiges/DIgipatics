"""
SICAPv2 Semantic Segmentation Training Pipeline
================================================
Nature Paper Replication: Attention-Enhanced Hybrid U-Net
- Architecture: ResNet50 + MHSA Bottleneck + ASPP + SE-UNet Decoder
- Loss: CrossEntropy (BCE equivalent for multiclass) + Dice Loss
- Augmentations: Random rotations, flips, contrast (±20%), Gaussian noise
- Tracked Metric: Macro F-score (Macro Dice)
- Confusion Matrices explicitly aggregated.

Usage:
    python training_nature.py                # full training (4 folds)
    python training_nature.py --dry-run      # smoke test (1 batch, 1 fold)
"""

import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
cv2.setNumThreads(0) # CRITICAL: Prevents thread contention deadlocks with PyTorch DataLoader workers!
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
from paths import IMAGES_DIR, MASKS_DIR, PARTITION_DIR, default_checkpoint_dir

OUTPUT_DIR = default_checkpoint_dir("checkpoints_nature")

NUM_CLASSES = 4
CLASS_NAMES = ["NC", "GG3", "GG4", "GG5"]
IMG_SIZE = 512

DEFAULT_CONFIG = {
    "encoder": "resnet50",
    "decoder": "Unet",
    "pretrained_weights": "imagenet",
    "num_classes": NUM_CLASSES,
    "batch_size": 8,
    "num_workers": 4, # Changed from 0 to 4 to prevent CPU bottleneck and speed up training
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "max_epochs": 100,
    "patience": 30, # Increased so it doesn't give up too quickly
    "dice_weight": 0.5,
    "ce_weight": 0.5,
    "class_weights": [1.0, 3.0, 4.5, 10.0],
}

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Necesario para reproducibilidad 100% estricta en GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─────────────────────────────────────────────────────────────────────────────
# Dataset (Reused Logic)
# ─────────────────────────────────────────────────────────────────────────────
# Precomputer Look-Up Table (LUT) a nivel global para evitar bugs de multiprocessing (pickling) en Windows
_MASK_LUT = np.zeros(256, dtype=np.int64)
_MASK_LUT[25:75] = 1
_MASK_LUT[75:175] = 2
_MASK_LUT[175:] = 3


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

        # Mapeo ultrarrápido O(1) vía LUT (evita iteraciones sobre toda la red)
        mask = _MASK_LUT[mask]

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask

# ─────────────────────────────────────────────────────────────────────────────
# Augmentations (Matched to Nature Paper)
# ─────────────────────────────────────────────────────────────────────────────
def get_train_transforms():
    return A.Compose([
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
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────
def get_fold_dataloaders(fold_name: str, config: dict):
    fold_dir = PARTITION_DIR / "Validation" / fold_name
    train_df = pd.read_excel(fold_dir / "Train.xlsx")
    val_df = pd.read_excel(fold_dir / "Test.xlsx")
    train_names = train_df["image_name"].tolist()
    val_names = val_df["image_name"].tolist()

    train_ds = SICAPv2Dataset(train_names, IMAGES_DIR, MASKS_DIR, transform=get_train_transforms())
    val_ds = SICAPv2Dataset(val_names, IMAGES_DIR, MASKS_DIR, transform=get_val_transforms())

    # Dynamically scale workers based on SLURM allocation to maximize IO speed, default to config
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", config["num_workers"]))
    
    # Enable persistent workers and prefetching to hide cluster NFS disk latency
    kwargs = {"persistent_workers": True, "prefetch_factor": 4} if workers > 0 else {}
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, 
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
        # Standard ASPP mathematically requires reducing channels beforehand.
        # Direct 2048->2048 in 3x3 atrous creates 170M+ parameters and chokeholds the GPU!
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
            nn.Dropout(0.5)
        )
        
    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        return self.project(torch.cat(res, dim=1))

class AttentionBottleneck(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mhsa = nn.MultiheadAttention(embed_dim=in_channels, num_heads=8, batch_first=True)
        self.aspp = ASPP(in_channels, in_channels)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1) # B, Seq, Embedded
        attn_out, _ = self.mhsa(x_flat, x_flat, x_flat)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        
        # Add residual and pass to ASPP
        out = self.aspp(attn_out + x)
        return out

class TAH_UNet(nn.Module):
    def __init__(self, encoder_name="resnet50", num_classes=4):
        super().__init__()
        # U-Net with ResNet50 encoder and Squeeze-and-Excitation (SE) attention in decoder
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            decoder_attention_type="scse",
            in_channels=3,
            classes=num_classes
        )
        # ResNet50 deepest feature map (stage 4) outputs 2048 channels
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

def build_model(config: dict):
    return TAH_UNet(encoder_name=config["encoder"], num_classes=config["num_classes"])


# ─────────────────────────────────────────────────────────────────────────────
# Guided Loss Mechanism (BCE/CE + Dice)
# ─────────────────────────────────────────────────────────────────────────────
class GuidedLoss(nn.Module):
    def __init__(self, class_weights: list, dice_weight=0.5, ce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", classes=[0,1,2,3], smooth=smooth)
        
        # CRITICAL FIX: The previous code completely ignored class_weights!
        # We use standard CrossEntropyLoss with the provided weights array to learn GG5
        self.register_buffer("weights_tensor", torch.tensor(class_weights).float())
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weights_tensor)

    def forward(self, logits, targets):
        # Force logits into float32 to prevent NaN overflow from FP16 large magnitudes
        logits = logits.float()
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.dice_weight * dice + self.ce_weight * ce

# ─────────────────────────────────────────────────────────────────────────────
# Metrics & Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────
class SegmentationMetrics:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update_batch(self, logits: torch.Tensor, targets: torch.Tensor):
        # Mantener los tensores en la GPU para el cálculo masivo de pixeles
        preds = logits.argmax(dim=1)
        mask = (targets >= 0) & (targets < self.num_classes)
        
        # Filtrar valores usando la mask en la GPU
        valid_targets = targets[mask]
        valid_preds = preds[mask]
        
        # Calcular el histograma de coincidencias (bincount es súper rápido en GPU)
        inds = self.num_classes * valid_targets + valid_preds
        hist = torch.bincount(inds, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        
        # Transferir only la matriz 4x4 resultante a la CPU y sumarla a la acumulada
        self.confusion_matrix += hist.cpu().numpy()

    def compute(self):
        cm = self.confusion_matrix
        dice_per_class = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            # If a class is not present in true targets AND not predicted, defined as 1.0 (or ignored).
            # To match rigorous Macro F1 standards, if it's missing in targets AND predictions, we don't penalize it.
            if tp + fp + fn == 0:
                dice_per_class[c] = np.nan # Use NaN so it's ignored in the macro mean
            else:
                dice_per_class[c] = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)

        # Macro F-score expressly ignoring NaNs (classes completely missing from batch)
        macro_f1 = np.nanmean(dice_per_class)
        # For display, convert NaNs to 0.0
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
    
    # Verificar soporte AMP fuera del bucle para ahorrar micro-sobrecargas
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        # bfloat16 avoids the massive FP16 (NaN) gradient explosions associated with attention modules
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
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, num_batches = 0.0, 0
    metrics = SegmentationMetrics(NUM_CLASSES)
    
    # Verificar soporte AMP como en el trainsmiento
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True).long()
        
        # FIX: Removed expensive Test-Time Augmentation (TTA) from the regular 
        # validation loop. TTA multiplies validation time by 3x causing SLURM Timeouts!
        # Mixed Precision añadido explícitamente para aceleración y ahorro de memoria
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
    
    # ⚡ PyTorch 2.x JIT Compilation para exprimir el rendimiento puro de la GPU
    # En servidores Linux, 'inductor' usará Triton para optimizar masivamente el Attention y bloques ASPP
    if int(torch.__version__.split('.')[0]) >= 2:
        try:
            print("  ⏳ Optimizando y compilando el modelo (puede tardar 1-2 mins en arrancar)...")
            import platform
            backend = "inductor" if platform.system() != "Windows" else "aot_eager"
            model = torch.compile(model, backend=backend)
            print("  ✅ Compilación activada con éxito (motor JIT súper veloz).")
        except Exception as e:
            print(f"  ⚠️ torch.compile no aplicable, usando modo normal: {e}")
            
    criterion = GuidedLoss(config["class_weights"], config["dice_weight"], config["ce_weight"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    
    # FIX: ReduceLROnPlateau automatically lowers the learning rate ONLY when the model stops improving. 
    # This prevents the model from violently fluctuating at high LR while patience runs out.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=6, min_lr=1e-6)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_macro_f1, patience_counter, best_cm = 0.0, 0, None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    max_epochs = 1 if dry_run else config["max_epochs"]

    for epoch in range(1, max_epochs + 1):
        print(f"\n  Epoch {epoch}/{max_epochs}  (lr={optimizer.param_groups[0]['lr']:.2e})")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, device)
        
        macro_f1 = val_metrics["macro_f1"]
        scheduler.step(macro_f1) # Step mathematically depends on the metric now
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
    set_seed(42)  # <-- Añadido para hacer el trainsmiento 100% reproducible
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if os.name == 'nt':
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
            print("\n[System] Anti-sleep enabled.")
        except: pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # NOTA: cudnn.benchmark está disabledo por set_seed() para priorizar la reproducibilidad.
    # Si en algún momento prefieres sacrificar la reproducibilidad por un ~5% extra de velocidad,
    # vuelve a reactivarlo aquí: # torch.backends.cudnn.benchmark = True
    
    fold_names = ["Val1", "Val2", "Val3", "Val4"]
    aggregated_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    
    for fold in fold_names:
        res = train_fold(fold, DEFAULT_CONFIG, device, args.dry_run)
        if res["best_cm"] is not None:
            aggregated_cm += res["best_cm"]
            
    # Print metrics even if dry_run just to test the display
    print_aggregated_matrices(aggregated_cm)
    if args.dry_run:
        print("\n✅ Dry run completed successfully!")

if __name__ == "__main__":
    main()
