"""
SICAPv2 Hierarchical Segmentation Training Pipeline (CONCH)
===========================================================
Stage 1: binary segmentation (NC vs Cancer={GG3,GG4,GG5}) optimized for macro-F1.
Stage 2: 4-class segmentation (NC, GG3, GG4, GG5).
Final prediction: gate with Stage 1, refine cancer pixels with Stage 2.
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
WANDB_PROJECT_DEFAULT = "SICAPv2_CONCH_hierarchical"

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
MASKS_DIR = BASE_DIR / "masks"
PARTITION_DIR = BASE_DIR / "partition"
OUTPUT_DIR = BASE_DIR / "checkpoints_conch_hierarchical"

IMG_SIZE = 512
CLASS_NAMES_4 = ["NC", "GG3", "GG4", "GG5"]
CLASS_NAMES_2 = ["NC", "Cancer"]
NUM_CLASSES_STAGE1 = 2
NUM_CLASSES_STAGE2 = 4

_PIXEL_FRAC_PARTITION = np.array([0.7781, 0.0605, 0.1175, 0.0439], dtype=np.float64)
_PIXEL_FRAC_BINARY = np.array([
    float(_PIXEL_FRAC_PARTITION[0]),
    float(_PIXEL_FRAC_PARTITION[1:].sum()),
], dtype=np.float64)

def _sqrt_inv_weights(freqs: np.ndarray) -> list:
    w = np.sqrt(1.0 / np.asarray(freqs, dtype=np.float64))
    w = w / w[0]
    return [float(round(x, 3)) for x in w]

DEFAULT_CLASS_WEIGHTS_STAGE2 = _sqrt_inv_weights(_PIXEL_FRAC_PARTITION)
DEFAULT_CLASS_WEIGHTS_STAGE1 = _sqrt_inv_weights(_PIXEL_FRAC_BINARY)

_FEATURE_BLOCKS = [2, 5, 8, 11]
CONCH_HF_CHECKPOINT = "hf_hub:MahmoodLab/conch"

DEFAULT_CONFIG = {
    "num_workers": 4,
    "max_epochs_stage1": 60,
    "max_epochs_stage2": 100,
    "patience_stage1": 15,
    "patience_stage2": 18,
    "weight_decay": 3e-4,
    "batch_size": 6,
    "grad_accum_steps": 2,
    "learning_rate": 4e-5,
    "lr_plateau_patience": 3,
    "fpn_channels": 256,
    "unfreeze_last": 0,
    "encoder_lr_ratio": 10,
    "use_weighted_sampler": True,
    "sampler_weight_gg5": 2.5,
    "sampler_weight_gg4": 1.3,
    "sampler_weight_gg3": 1.8,
    "decoder_dropout": 0.1,
    "use_compile": None,
    "conch_checkpoint": None,
    "conch_hf_token": None,
    "seed": DEFAULT_SEED,
    "class_weights_stage1": list(DEFAULT_CLASS_WEIGHTS_STAGE1),
    "class_weights_stage2": list(DEFAULT_CLASS_WEIGHTS_STAGE2),
    "dice_weight_stage1": 0.6,
    "ce_weight_stage1": 0.4,
    "dice_weight_stage2": 0.55,
    "ce_weight_stage2": 0.45,
    "stage2_ignore_index": 255,
    # Stage 1 should prioritize sensitivity to avoid irreversible misses in a hard cascade.
    "stage1_min_cancer_recall": 0.95,
    "stage1_threshold_policy": "macro_f1",  # macro_f1 | cancer_recall_constrained | cancer_f1
    "stage2_train_threshold": None,  # None -> max(0.05, stage1_threshold - 0.15)
    "stage2_gate_dilation": 0,
    "stage1_infer_gate_dilation": 0,
    "enable_stage2_rescue": False,
    "stage2_rescue_threshold": 0.70,
    "stage2_keep_tumor_outside_gate": False,
    "binary_threshold_candidates": [round(x, 2) for x in np.linspace(0.1, 0.9, 17)],
}

_MASK_LUT = np.zeros(256, dtype=np.int64)
_MASK_LUT[25:75] = 1
_MASK_LUT[75:175] = 2
_MASK_LUT[175:] = 3


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
    s = int((base_seed + worker_id) % (2 ** 32))
    np.random.seed(s)
    random.seed(s)


def compute_sample_weights(
    image_names: list,
    masks_dir: Path,
    weight_gg5: float,
    weight_gg4: float,
    weight_gg3: float,
):
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
                    w = weight_gg5
                elif np.any(mapped == 2):
                    w = weight_gg4
                elif np.any(mapped == 1):
                    w = weight_gg3
        weights.append(w)
    return weights


class SICAPv2Dataset(Dataset):
    def __init__(self, image_names: list, images_dir: Path, masks_dir: Path, transform=None, binary: bool = False):
        self.image_names = image_names
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.binary = bool(binary)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img_path = self.images_dir / name
        buf = np.fromfile(str(img_path), dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
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
        if self.binary:
            mask = (mask > 0).astype(np.uint8)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask


def get_train_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
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
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_fold_dataloaders(fold_name: str, config: dict, binary: bool):
    fold_dir = PARTITION_DIR / "Validation" / fold_name
    train_df = pd.read_excel(fold_dir / "Train.xlsx")
    val_df = pd.read_excel(fold_dir / "Test.xlsx")
    train_names = train_df["image_name"].tolist()
    val_names = val_df["image_name"].tolist()

    train_ds = SICAPv2Dataset(train_names, IMAGES_DIR, MASKS_DIR, transform=get_train_transforms(), binary=binary)
    val_ds = SICAPv2Dataset(val_names, IMAGES_DIR, MASKS_DIR, transform=get_val_transforms(), binary=binary)

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", config["num_workers"]))
    kwargs = {"persistent_workers": True, "prefetch_factor": 4} if workers > 0 else {}
    seed = int(config.get("seed", DEFAULT_SEED))
    gen = torch.Generator()
    gen.manual_seed(seed)
    winit = partial(_worker_init_fn, base_seed=seed) if workers > 0 else None
    dl_common = dict(num_workers=workers, pin_memory=True, generator=gen, worker_init_fn=winit)

    if config.get("use_weighted_sampler", False):
        w5 = float(config["sampler_weight_gg5"])
        w4 = float(config["sampler_weight_gg4"])
        w3 = float(config["sampler_weight_gg3"])
        sw = compute_sample_weights(train_names, MASKS_DIR, w5, w4, w3)
        sampler = WeightedRandomSampler(weights=sw, num_samples=len(train_names), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], sampler=sampler, drop_last=True, **dl_common, **kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, drop_last=True, **dl_common, **kwargs)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, **dl_common, **kwargs)
    return train_loader, val_loader


def _extract_conch_trunk(weights_path=None, hf_token=None):
    try:
        from conch.open_clip_custom.factory import create_model
    except ImportError as e:
        raise ImportError("Instala CONCH: pip install git+https://github.com/mahmoodlab/CONCH.git") from e

    if weights_path:
        wp = Path(weights_path)
        if not wp.is_file():
            raise FileNotFoundError(f"No se encontro el checkpoint CONCH: {wp}")
        ckpt = str(wp.resolve())
    else:
        ckpt = CONCH_HF_CHECKPOINT

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    print("  [CONCH] Cargando checkpoint en CPU...")
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
            print(f"  [ConcHEncoder] Descongelados {unfreeze_last}/{total} bloques")
        else:
            print("  [ConcHEncoder] Encoder congelado")

    def forward(self, x):
        bsz = x.shape[0]
        x_tok = self.trunk.patch_embed(x)
        x_tok = self.trunk._pos_embed(x_tok)
        if hasattr(self.trunk, "patch_drop"):
            x_tok = self.trunk.patch_drop(x_tok)
        if hasattr(self.trunk, "norm_pre"):
            x_tok = self.trunk.norm_pre(x_tok)

        num_spatial = x_tok.shape[1] - self.num_prefix
        h_p = w_p = int(num_spatial ** 0.5)
        if h_p * w_p != num_spatial:
            raise RuntimeError(f"Tokens espaciales invalidos: N={num_spatial}")

        feats = []
        for i, blk in enumerate(self.trunk.blocks):
            x_tok = blk(x_tok)
            if i in self.feature_blocks:
                spatial = x_tok[:, self.num_prefix:self.num_prefix + h_p * w_p, :]
                spatial = spatial.permute(0, 2, 1).reshape(bsz, self.embed_dim, h_p, w_p)
                feats.append(spatial)
        return feats


class FPNDecoder(nn.Module):
    def __init__(self, in_channels=768, fpn_channels=256, num_classes=4, dropout: float = 0.0):
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
        head_layers = [
            nn.Conv2d(fpn_channels, fpn_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels // 2),
            nn.ReLU(inplace=True),
        ]
        if dropout and dropout > 0:
            head_layers.append(nn.Dropout2d(p=float(dropout)))
        head_layers.append(nn.Conv2d(fpn_channels // 2, num_classes, 1))
        self.head = nn.Sequential(*head_layers)

    def forward(self, features, target_size):
        lats = [lat(f) for lat, f in zip(self.lat, features)]
        x = lats[3]
        for i in range(2, -1, -1):
            x = F.interpolate(x, size=lats[i].shape[-2:], mode="bilinear", align_corners=False)
            x = self.merge[i](x + lats[i])
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.head(x)


class CONCHSegModel(nn.Module):
    def __init__(self, fpn_channels=256, num_classes=4, unfreeze_last=0, weights_path=None, hf_token=None, decoder_dropout: float = 0.0):
        super().__init__()
        self.encoder = ConcHEncoder(unfreeze_last=unfreeze_last, weights_path=weights_path, hf_token=hf_token)
        self.decoder = FPNDecoder(in_channels=self.encoder.embed_dim, fpn_channels=fpn_channels, num_classes=num_classes, dropout=decoder_dropout)

    def forward(self, x):
        target_size = x.shape[-2:]
        features = self.encoder(x)
        return self.decoder(features, target_size)


def build_model(config: dict, num_classes: int):
    return CONCHSegModel(
        fpn_channels=config["fpn_channels"],
        num_classes=num_classes,
        unfreeze_last=config["unfreeze_last"],
        weights_path=config.get("conch_checkpoint"),
        hf_token=config.get("conch_hf_token"),
        decoder_dropout=float(config.get("decoder_dropout", 0.0)),
    )


class GuidedLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        class_weights: list,
        dice_weight=0.5,
        ce_weight=0.5,
        smooth=1e-6,
        ignore_index=None,
    ):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.dice_loss = smp.losses.DiceLoss(
            mode="multiclass",
            classes=list(range(num_classes)),
            smooth=smooth,
            ignore_index=ignore_index,
        )
        self.register_buffer("weights_tensor", torch.tensor(class_weights).float())
        ce_kwargs = {"weight": self.weights_tensor}
        if ignore_index is not None:
            ce_kwargs["ignore_index"] = int(ignore_index)
        self.ce_loss = nn.CrossEntropyLoss(**ce_kwargs)

    def forward(self, logits, targets):
        logits = logits.float()
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.dice_weight * dice + self.ce_weight * ce


def _trainable_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


class SegmentationMetrics:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update_batch(self, preds: np.ndarray, tgts: np.ndarray):
        mask = (tgts >= 0) & (tgts < self.num_classes)
        np.add.at(self.confusion_matrix, (tgts[mask], preds[mask]), 1)

    def compute(self):
        cm = self.confusion_matrix
        f1_per_class = np.zeros(self.num_classes, dtype=np.float64)
        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            if tp + fp + fn == 0:
                f1_per_class[c] = np.nan
            else:
                f1_per_class[c] = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
        macro_f1 = float(np.nanmean(f1_per_class))
        return {"macro_f1": macro_f1, "f1_per_class": np.nan_to_num(f1_per_class, nan=0.0), "confusion_matrix": cm.copy()}


def _binary_macro_f1_from_cm(cm: np.ndarray) -> float:
    f1s = []
    for c in range(2):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        if tp + fp + fn == 0:
            f1s.append(np.nan)
        else:
            f1s.append((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8))
    return float(np.nanmean(np.array(f1s, dtype=np.float64)))


def compute_binary_stats_from_cm(cm: np.ndarray) -> dict:
    tn = float(cm[0, 0])
    fp = float(cm[0, 1])
    fn = float(cm[1, 0])
    tp = float(cm[1, 1])
    cancer_precision = tp / (tp + fp + 1e-8)
    cancer_recall = tp / (tp + fn + 1e-8)
    cancer_f1 = (2.0 * cancer_precision * cancer_recall) / (cancer_precision + cancer_recall + 1e-8)
    nc_precision = tn / (tn + fn + 1e-8)
    nc_recall = tn / (tn + fp + 1e-8)
    nc_f1 = (2.0 * nc_precision * nc_recall) / (nc_precision + nc_recall + 1e-8)
    macro_f1 = 0.5 * (nc_f1 + cancer_f1)
    return {
        "macro_f1": float(macro_f1),
        "cancer_precision": float(cancer_precision),
        "cancer_recall": float(cancer_recall),
        "cancer_f1": float(cancer_f1),
        "nc_f1": float(nc_f1),
    }


def select_stage1_threshold(cms: dict, policy: str, min_cancer_recall: float) -> tuple[float, dict]:
    items = []
    for t, cm in cms.items():
        stats = compute_binary_stats_from_cm(cm)
        items.append((float(t), cm, stats))

    pol = str(policy).strip().lower()
    if pol == "cancer_recall_constrained":
        feasible = [x for x in items if x[2]["cancer_recall"] >= float(min_cancer_recall)]
        if feasible:
            # Prefer highest macro-F1 among high-recall thresholds.
            best = max(feasible, key=lambda x: (x[2]["macro_f1"], x[2]["cancer_f1"], -x[0]))
        else:
            # Fallback: maximize recall first, then macro-F1.
            best = max(items, key=lambda x: (x[2]["cancer_recall"], x[2]["macro_f1"], -x[0]))
    elif pol == "cancer_f1":
        best = max(items, key=lambda x: (x[2]["cancer_f1"], x[2]["macro_f1"], -x[0]))
    else:
        best = max(items, key=lambda x: (x[2]["macro_f1"], x[2]["cancer_f1"], -x[0]))
    return float(best[0]), {"confusion_matrix": best[1].copy(), **best[2]}


def dilate_gate_tensor(gate: torch.Tensor, kernel_size: int) -> torch.Tensor:
    k = int(kernel_size)
    if k <= 1:
        return gate.bool()
    x = gate.float().unsqueeze(1)
    y = F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    return (y.squeeze(1) > 0.5)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_accum_steps: int = 1):
    model.train()
    total_loss, num_batches = 0.0, 0
    accum = 0
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        if accum == 0:
            optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            loss = criterion(model(images), masks)
        scaler.scale(loss / grad_accum_steps).backward()
        total_loss += float(loss.item())
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


def train_one_epoch_stage2_gated(
    model,
    stage1_model,
    stage1_threshold: float,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    ignore_index: int,
    gate_threshold: float,
    gate_dilation: int,
    keep_tumor_outside_gate: bool,
    grad_accum_steps: int = 1,
):
    model.train()
    stage1_model.eval()
    total_loss, num_batches = 0.0, 0
    accum = 0
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pbar = tqdm(loader, desc="  TrainS2G", leave=False)
    for images, masks4 in pbar:
        images = images.to(device, non_blocking=True)
        masks4 = masks4.to(device, non_blocking=True).long()
        if accum == 0:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
                logits1 = stage1_model(images)
            p_cancer = torch.softmax(logits1.float(), dim=1)[:, 1]
            gate_pred = p_cancer >= float(gate_threshold)
            gate_pred = dilate_gate_tensor(gate_pred, gate_dilation)

        valid_pixels = int(gate_pred.sum().item())
        if valid_pixels == 0:
            pbar.set_postfix(loss="skip(empty-gate)")
            continue

        targets = masks4.clone()
        valid_mask = gate_pred
        if keep_tumor_outside_gate:
            valid_mask = valid_mask | (masks4 > 0)
        targets[~valid_mask] = int(ignore_index)
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            logits2 = model(images)
            loss = criterion(logits2, targets)

        scaler.scale(loss / grad_accum_steps).backward()
        total_loss += float(loss.item())
        num_batches += 1
        accum += 1
        if accum >= grad_accum_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accum = 0
        gate_ratio = float(valid_mask.float().mean().item())
        pbar.set_postfix(loss=f"{loss.item():.4f}", gate_px=valid_pixels, gate_pct=f"{gate_ratio*100:.1f}")
    if accum > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate_multiclass(model, loader, criterion, device, num_classes: int):
    model.eval()
    total_loss, num_batches = 0.0, 0
    metrics = SegmentationMetrics(num_classes)
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits, masks)
        total_loss += float(loss.item())
        num_batches += 1
        preds = logits.argmax(dim=1).cpu().numpy()
        tgts = masks.cpu().numpy()
        metrics.update_batch(preds, tgts)
    return total_loss / max(num_batches, 1), metrics.compute()


@torch.no_grad()
def validate_multiclass_stage2_gated(
    model,
    stage1_model,
    stage1_threshold: float,
    gate_threshold: float,
    gate_dilation: int,
    keep_tumor_outside_gate: bool,
    loader,
    criterion,
    device,
    num_classes: int,
):
    model.eval()
    stage1_model.eval()
    total_loss, num_batches = 0.0, 0
    metrics = SegmentationMetrics(num_classes)
    gate_ratio_sum, gate_batches = 0.0, 0
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pbar = tqdm(loader, desc="  ValS2G", leave=False)
    for images, masks4 in pbar:
        images = images.to(device, non_blocking=True)
        masks4 = masks4.to(device, non_blocking=True).long()
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            logits1 = stage1_model(images)
        p_cancer = torch.softmax(logits1.float(), dim=1)[:, 1]
        gate_pred = p_cancer >= float(gate_threshold)
        gate_pred = dilate_gate_tensor(gate_pred, gate_dilation)
        if int(gate_pred.sum().item()) == 0:
            continue

        targets = masks4.clone()
        valid = gate_pred
        if keep_tumor_outside_gate:
            valid = valid | (masks4 > 0)
        targets[~valid] = int(getattr(criterion.ce_loss, "ignore_index", 255))
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            logits2 = model(images)
            loss = criterion(logits2, targets)
        total_loss += float(loss.item())
        num_batches += 1
        gate_ratio_sum += float(valid.float().mean().item())
        gate_batches += 1

        pred4 = logits2.argmax(dim=1)
        metrics.update_batch(pred4[valid].cpu().numpy(), masks4[valid].cpu().numpy())
    out = metrics.compute()
    out["gate_ratio"] = gate_ratio_sum / max(gate_batches, 1)
    return total_loss / max(num_batches, 1), out


@torch.no_grad()
def validate_binary_with_threshold_search(model, loader, criterion, device, thresholds: list):
    model.eval()
    total_loss, num_batches = 0.0, 0
    cms = {float(t): np.zeros((2, 2), dtype=np.int64) for t in thresholds}
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pbar = tqdm(loader, desc="  ValS1", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits, masks)
        total_loss += float(loss.item())
        num_batches += 1
        probs = torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu().numpy()
        tgts = masks.detach().cpu().numpy()
        flat_p = probs.reshape(-1)
        flat_t = tgts.reshape(-1)
        for t in thresholds:
            pred = (flat_p >= float(t)).astype(np.int64)
            np.add.at(cms[float(t)], (flat_t, pred), 1)

    return total_loss / max(num_batches, 1), {"cms_by_threshold": cms}


@torch.no_grad()
def validate_cascade(
    stage1_model,
    stage2_model,
    val_loader_stage2,
    device,
    stage1_threshold: float,
    infer_gate_dilation: int = 0,
    enable_stage2_rescue: bool = False,
    stage2_rescue_threshold: float = 0.70,
):
    stage1_model.eval()
    stage2_model.eval()
    metrics = SegmentationMetrics(NUM_CLASSES_STAGE2)
    rescued_pixels = 0
    total_pixels = 0
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pbar = tqdm(val_loader_stage2, desc="  ValH ", leave=False)
    for images, masks4 in pbar:
        images = images.to(device, non_blocking=True)
        masks4 = masks4.to(device, non_blocking=True).long()
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            logits1 = stage1_model(images)
            logits2 = stage2_model(images)
        p_cancer = torch.softmax(logits1.float(), dim=1)[:, 1]
        gate = p_cancer >= float(stage1_threshold)
        gate = dilate_gate_tensor(gate, int(infer_gate_dilation))
        pred4 = logits2.argmax(dim=1)
        prob4 = torch.softmax(logits2.float(), dim=1)
        conf4 = prob4.max(dim=1).values
        if enable_stage2_rescue:
            rescue = (~gate) & (pred4 > 0) & (conf4 >= float(stage2_rescue_threshold))
        else:
            rescue = torch.zeros_like(gate, dtype=torch.bool)
        final_pred = torch.where(gate | rescue, pred4, torch.zeros_like(pred4))
        rescued_pixels += int(rescue.sum().item())
        total_pixels += int(rescue.numel())
        metrics.update_batch(final_pred.cpu().numpy(), masks4.cpu().numpy())
    out = metrics.compute()
    out["rescued_ratio"] = float(rescued_pixels / max(total_pixels, 1))
    return out


def train_fold_stage1(
    fold_name: str,
    config: dict,
    device: torch.device,
    dry_run: bool = False,
    use_wandb: bool = True,
):
    print(f"\n{'='*60}\n  FOLD {fold_name} | STAGE 1 (Binary)\n{'='*60}")
    train_loader, val_loader = get_fold_dataloaders(fold_name, config, binary=True)
    model = build_model(config, num_classes=NUM_CLASSES_STAGE1).to(device)

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
            print(f"  [WARN] torch.compile stage1 no disponible: {e}")
    elif not use_compile:
        print("  ℹ️ torch.compile desactivado (recomendado en Windows / ahorrar VRAM).")

    criterion = GuidedLoss(
        num_classes=NUM_CLASSES_STAGE1,
        class_weights=config["class_weights_stage1"],
        dice_weight=config["dice_weight_stage1"],
        ce_weight=config["ce_weight_stage1"],
    ).to(device)

    inner = _trainable_model(model)
    encoder_params = [p for p in inner.encoder.parameters() if p.requires_grad]
    decoder_params = list(inner.decoder.parameters())
    enc_ratio = max(1, int(config.get("encoder_lr_ratio", 10)))
    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": config["learning_rate"] / float(enc_ratio)})
    param_groups.append({"params": decoder_params, "lr": config["learning_rate"]})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=int(config.get("lr_plateau_patience", 3)), min_lr=1e-7
    )
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    out_dir = Path(config.get("output_dir", OUTPUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)
    max_epochs = 1 if dry_run else int(config["max_epochs_stage1"])
    patience_max = int(config["patience_stage1"])
    ga = int(config.get("grad_accum_steps", 1))
    thresholds = [float(x) for x in config.get("binary_threshold_candidates", [0.5])]

    best = {"macro_f1": 0.0, "threshold": 0.5, "cm": None, "path": None, "stats": None}
    patience = 0
    print(
        f"  [S1][LR] encoder_lr_ratio={enc_ratio} "
        f"(decoder={config['learning_rate']:.2e})"
    )
    for epoch in range(1, max_epochs + 1):
        print(f"\n  [S1] Epoch {epoch}/{max_epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, grad_accum_steps=ga)
        val_loss, val_bin = validate_binary_with_threshold_search(model, val_loader, criterion, device, thresholds)
        # Sensitivity-aware threshold selection mitigates irreversible misses from the hard stage-1 gate.
        th, sel = select_stage1_threshold(
            val_bin["cms_by_threshold"],
            policy=config.get("stage1_threshold_policy", "macro_f1"),
            min_cancer_recall=float(config.get("stage1_min_cancer_recall", 0.95)),
        )
        macro_f1 = float(sel["macro_f1"])
        scheduler.step(macro_f1)
        ratio = val_loss / (train_loss + 1e-8)
        print(f"  [S1] TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | Val/Train={ratio:.3f}")
        print(
            f"  [S1] Macro-F1(bin)={macro_f1:.4f} @ threshold={th:.2f} | "
            f"policy={config.get('stage1_threshold_policy', 'macro_f1')}"
        )
        print(f"    NC F1: {sel['nc_f1']:.4f}")
        print(f"    Cancer F1: {sel['cancer_f1']:.4f}")
        print(f"    Cancer Precision: {sel['cancer_precision']:.4f}")
        print(f"    Cancer Recall: {sel['cancer_recall']:.4f}")
        if use_wandb and wandb.run is not None:
            enc_lr = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 1 else optimizer.param_groups[-1]["lr"]
            dec_lr = optimizer.param_groups[-1]["lr"]
            wandb.log({
                f"{fold_name}/s1_train_loss": float(train_loss),
                f"{fold_name}/s1_val_loss": float(val_loss),
                f"{fold_name}/s1_val_train_loss_ratio": float(ratio),
                f"{fold_name}/s1_macro_f1": float(macro_f1),
                f"{fold_name}/s1_f1_NC": float(sel["nc_f1"]),
                f"{fold_name}/s1_f1_Cancer": float(sel["cancer_f1"]),
                f"{fold_name}/s1_cancer_precision": float(sel["cancer_precision"]),
                f"{fold_name}/s1_cancer_recall": float(sel["cancer_recall"]),
                f"{fold_name}/s1_best_threshold_epoch": float(th),
                f"{fold_name}/s1_lr_encoder": float(enc_lr),
                f"{fold_name}/s1_lr_decoder": float(dec_lr),
                "epoch": int(epoch),
            })
        if macro_f1 > best["macro_f1"]:
            best["macro_f1"] = macro_f1
            best["threshold"] = th
            best["cm"] = sel["confusion_matrix"]
            best["stats"] = sel
            best_path = out_dir / f"best_stage1_{fold_name}_{macro_f1:.4f}_th{th:.2f}.pth"
            torch.save(_trainable_model(model).state_dict(), best_path)
            best["path"] = best_path
            patience = 0
            print("  [S1] ✓ checkpoint guardado")
        else:
            patience += 1
            if patience >= patience_max:
                print(f"  [S1] Early stopping en epoca {epoch}")
                break
        if dry_run:
            break
    return best


def train_fold_stage2(
    fold_name: str,
    config: dict,
    device: torch.device,
    stage1_path: Path,
    stage1_threshold: float,
    dry_run: bool = False,
    use_wandb: bool = True,
):
    print(f"\n{'='*60}\n  FOLD {fold_name} | STAGE 2 (4-class)\n{'='*60}")
    train_loader, val_loader = get_fold_dataloaders(fold_name, config, binary=False)
    model = build_model(config, num_classes=NUM_CLASSES_STAGE2).to(device)
    stage1_model = build_model(config, num_classes=NUM_CLASSES_STAGE1).to(device)
    stage1_model.load_state_dict(torch.load(stage1_path, map_location=device))
    stage1_model.eval()
    for p in stage1_model.parameters():
        p.requires_grad = False

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
            print(f"  [WARN] torch.compile stage2 no disponible: {e}")
    elif not use_compile:
        print("  ℹ️ torch.compile desactivado (recomendado en Windows / ahorrar VRAM).")

    criterion = GuidedLoss(
        num_classes=NUM_CLASSES_STAGE2,
        class_weights=config["class_weights_stage2"],
        dice_weight=config["dice_weight_stage2"],
        ce_weight=config["ce_weight_stage2"],
        ignore_index=int(config.get("stage2_ignore_index", 255)),
    ).to(device)

    inner = _trainable_model(model)
    encoder_params = [p for p in inner.encoder.parameters() if p.requires_grad]
    decoder_params = list(inner.decoder.parameters())
    enc_ratio = max(1, int(config.get("encoder_lr_ratio", 10)))
    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": config["learning_rate"] / float(enc_ratio)})
    param_groups.append({"params": decoder_params, "lr": config["learning_rate"]})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=int(config.get("lr_plateau_patience", 3)), min_lr=1e-7
    )
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    out_dir = Path(config.get("output_dir", OUTPUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)
    max_epochs = 1 if dry_run else int(config["max_epochs_stage2"])
    patience_max = int(config["patience_stage2"])
    ga = int(config.get("grad_accum_steps", 1))

    best = {"macro_f1": 0.0, "cm": None, "path": None}
    patience = 0
    train_gate_threshold = config.get("stage2_train_threshold")
    if train_gate_threshold is None:
        train_gate_threshold = max(0.05, float(stage1_threshold) - 0.15)
    train_gate_threshold = float(train_gate_threshold)
    train_gate_dilation = int(config.get("stage2_gate_dilation", 0))
    keep_tumor_outside = bool(config.get("stage2_keep_tumor_outside_gate", False))

    print(
        f"  [S2][LR] encoder_lr_ratio={enc_ratio} "
        f"(decoder={config['learning_rate']:.2e}) | gate_threshold(train)={train_gate_threshold:.2f}"
    )
    for epoch in range(1, max_epochs + 1):
        print(f"\n  [S2] Epoch {epoch}/{max_epochs}")
        train_loss = train_one_epoch_stage2_gated(
            model=model,
            stage1_model=stage1_model,
            stage1_threshold=stage1_threshold,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            ignore_index=int(config.get("stage2_ignore_index", 255)),
            gate_threshold=train_gate_threshold,
            gate_dilation=train_gate_dilation,
            keep_tumor_outside_gate=keep_tumor_outside,
            grad_accum_steps=ga,
        )
        val_loss, val_mc = validate_multiclass_stage2_gated(
            model=model,
            stage1_model=stage1_model,
            stage1_threshold=stage1_threshold,
            gate_threshold=train_gate_threshold,
            gate_dilation=train_gate_dilation,
            keep_tumor_outside_gate=keep_tumor_outside,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=NUM_CLASSES_STAGE2,
        )
        macro_f1 = float(val_mc["macro_f1"])
        scheduler.step(macro_f1)
        ratio = val_loss / (train_loss + 1e-8)
        print(f"  [S2] TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | Val/Train={ratio:.3f}")
        print(f"  [S2] Macro-F1(4c)={macro_f1:.4f}")
        for i, name in enumerate(CLASS_NAMES_4):
            print(f"    {name} F1: {val_mc['f1_per_class'][i]:.4f}")
        if use_wandb and wandb.run is not None:
            enc_lr = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 1 else optimizer.param_groups[-1]["lr"]
            dec_lr = optimizer.param_groups[-1]["lr"]
            metrics_dict = {
                f"{fold_name}/s2_train_loss": float(train_loss),
                f"{fold_name}/s2_val_loss": float(val_loss),
                f"{fold_name}/s2_val_train_loss_ratio": float(ratio),
                f"{fold_name}/s2_macro_f1": float(macro_f1),
                f"{fold_name}/s2_lr_encoder": float(enc_lr),
                f"{fold_name}/s2_lr_decoder": float(dec_lr),
                f"{fold_name}/s2_gate_threshold_train": float(train_gate_threshold),
                f"{fold_name}/s2_gate_ratio": float(val_mc.get("gate_ratio", 0.0)),
                "epoch": int(epoch),
            }
            for i, name in enumerate(CLASS_NAMES_4):
                metrics_dict[f"{fold_name}/s2_f1_{name}"] = float(val_mc["f1_per_class"][i])
            wandb.log(metrics_dict)
        if macro_f1 > best["macro_f1"]:
            best["macro_f1"] = macro_f1
            best["cm"] = val_mc["confusion_matrix"]
            best_path = out_dir / f"best_stage2_{fold_name}_{macro_f1:.4f}.pth"
            torch.save(_trainable_model(model).state_dict(), best_path)
            best["path"] = best_path
            patience = 0
            print("  [S2] ✓ checkpoint guardado")
        else:
            patience += 1
            if patience >= patience_max:
                print(f"  [S2] Early stopping en epoca {epoch}")
                break
        if dry_run:
            break
    return best


def evaluate_fold_hierarchical(fold_name: str, config: dict, device: torch.device, stage1_path: Path, stage2_path: Path, stage1_threshold: float):
    _, val_loader_stage2 = get_fold_dataloaders(fold_name, config, binary=False)
    model1 = build_model(config, num_classes=NUM_CLASSES_STAGE1).to(device)
    model2 = build_model(config, num_classes=NUM_CLASSES_STAGE2).to(device)
    model1.load_state_dict(torch.load(stage1_path, map_location=device))
    model2.load_state_dict(torch.load(stage2_path, map_location=device))
    val_h = validate_cascade(
        model1,
        model2,
        val_loader_stage2,
        device,
        stage1_threshold=stage1_threshold,
        infer_gate_dilation=int(config.get("stage1_infer_gate_dilation", 0)),
        enable_stage2_rescue=bool(config.get("enable_stage2_rescue", False)),
        stage2_rescue_threshold=float(config.get("stage2_rescue_threshold", 0.70)),
    )
    return val_h


def _print_cm_4class(cm: np.ndarray, title: str):
    print(f"\n{title}")
    df = pd.DataFrame(cm, index=[f"T_{c}" for c in CLASS_NAMES_4], columns=[f"P_{c}" for c in CLASS_NAMES_4])
    print(df.to_string())
    print("  Per-class F1:")
    for i, cname in enumerate(CLASS_NAMES_4):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else 0.0
        print(f"    {cname:3s}: {f1:.4f}")


def _macro_f1_from_cm(cm: np.ndarray) -> float:
    if cm.sum() == 0:
        return 0.0
    vals = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        if tp + fp + fn == 0:
            vals.append(np.nan)
        else:
            vals.append((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8))
    return float(np.nanmean(np.array(vals, dtype=np.float64)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="1 epoca por stage y fold.")
    parser.add_argument("--unfreeze-last", type=int, default=0)
    parser.add_argument("--weights", type=str, default=None, metavar="PATH")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None, metavar="K")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--sampler-gg5", type=float, default=None)
    parser.add_argument("--sampler-gg4", type=float, default=None)
    parser.add_argument("--sampler-gg3", type=float, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT_DEFAULT)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--fold", type=str, nargs="+", default=None, choices=["Val1", "Val2", "Val3", "Val4"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-epochs-stage1", type=int, default=None)
    parser.add_argument("--max-epochs-stage2", type=int, default=None)
    parser.add_argument("--patience-stage1", type=int, default=None)
    parser.add_argument("--patience-stage2", type=int, default=None)
    parser.add_argument(
        "--stage1-min-cancer-recall",
        type=float,
        default=None,
        help="Recall minimo de cancer para politica constrained de threshold.",
    )
    parser.add_argument(
        "--stage1-threshold-policy",
        type=str,
        choices=["macro_f1", "cancer_recall_constrained", "cancer_f1"],
        default=None,
        help="Politica para seleccionar threshold de Stage1.",
    )
    parser.add_argument(
        "--stage2-train-threshold",
        type=float,
        default=None,
        help="Threshold usado para gate de entrenamiento Stage2. Default=max(0.05, stage1-0.15).",
    )
    parser.add_argument(
        "--stage2-gate-dilation",
        type=int,
        default=None,
        help="Kernel de dilatacion del gate en Stage2 (train/val). 0/1 desactiva.",
    )
    parser.add_argument(
        "--stage2-keep-tumor-outside-gate",
        action="store_true",
        help="Mantener pixeles tumorales GT aunque queden fuera del gate Stage1 en Stage2.",
    )
    parser.add_argument(
        "--enable-stage2-rescue",
        action="store_true",
        help="Permite rescate de tumor por alta confianza de Stage2 aunque Stage1 diga NC.",
    )
    parser.add_argument(
        "--stage2-rescue-threshold",
        type=float,
        default=None,
        help="Confianza minima de Stage2 para activar rescate.",
    )
    parser.add_argument(
        "--stage1-infer-gate-dilation",
        type=int,
        default=None,
        help="Kernel de dilatacion del gate en inferencia jerarquica.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Seed: {args.seed} (cudnn deterministic=True, benchmark=False)")

    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except Exception:
            pass

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
        config["batch_size"] = int(args.batch_size)
    if args.grad_accum is not None:
        config["grad_accum_steps"] = max(1, int(args.grad_accum))
    if args.no_weighted_sampler:
        config["use_weighted_sampler"] = False
    if args.sampler_gg5 is not None:
        config["sampler_weight_gg5"] = float(args.sampler_gg5)
    if args.sampler_gg4 is not None:
        config["sampler_weight_gg4"] = float(args.sampler_gg4)
    if args.sampler_gg3 is not None:
        config["sampler_weight_gg3"] = float(args.sampler_gg3)
    if args.compile and args.no_compile:
        print("  [WARN] --compile y --no-compile a la vez; aplico --no-compile.")
    if args.no_compile:
        config["use_compile"] = False
    elif args.compile:
        config["use_compile"] = True
    if args.max_epochs_stage1 is not None:
        config["max_epochs_stage1"] = int(args.max_epochs_stage1)
    if args.max_epochs_stage2 is not None:
        config["max_epochs_stage2"] = int(args.max_epochs_stage2)
    if args.patience_stage1 is not None:
        config["patience_stage1"] = int(args.patience_stage1)
    if args.patience_stage2 is not None:
        config["patience_stage2"] = int(args.patience_stage2)
    if args.stage1_min_cancer_recall is not None:
        config["stage1_min_cancer_recall"] = float(args.stage1_min_cancer_recall)
    if args.stage1_threshold_policy is not None:
        config["stage1_threshold_policy"] = str(args.stage1_threshold_policy)
    if args.stage2_train_threshold is not None:
        config["stage2_train_threshold"] = float(args.stage2_train_threshold)
    if args.stage2_gate_dilation is not None:
        config["stage2_gate_dilation"] = max(0, int(args.stage2_gate_dilation))
    if args.stage2_keep_tumor_outside_gate:
        config["stage2_keep_tumor_outside_gate"] = True
    if args.enable_stage2_rescue:
        config["enable_stage2_rescue"] = True
    if args.stage2_rescue_threshold is not None:
        config["stage2_rescue_threshold"] = float(args.stage2_rescue_threshold)
    if args.stage1_infer_gate_dilation is not None:
        config["stage1_infer_gate_dilation"] = max(0, int(args.stage1_infer_gate_dilation))
    config["output_dir"] = Path(args.output_dir).resolve() if args.output_dir else OUTPUT_DIR

    print(f"Checkpoints -> {config['output_dir']}")
    eff = config["batch_size"] * config.get("grad_accum_steps", 1)
    print(
        f"CONCH hierarchical: micro_batch={config['batch_size']} × accum={config.get('grad_accum_steps', 1)} "
        f"≈ {eff} | weighted_sampler={config.get('use_weighted_sampler')} | lr={config['learning_rate']}"
    )
    print(
        "  stage1 class_weights (NC,Cancer): "
        f"{config['class_weights_stage1']} | dice/ce: {config['dice_weight_stage1']}/{config['ce_weight_stage1']}"
    )
    print(
        "  stage2 class_weights (NC,GG3,GG4,GG5): "
        f"{config['class_weights_stage2']} | dice/ce: {config['dice_weight_stage2']}/{config['ce_weight_stage2']} | "
        f"stage2_ignore_index={config.get('stage2_ignore_index', 255)}"
    )
    print(
        f"  epochs(S1/S2)={config['max_epochs_stage1']}/{config['max_epochs_stage2']} | "
        f"patience(S1/S2)={config['patience_stage1']}/{config['patience_stage2']} | "
        f"lr_plateau_patience={config.get('lr_plateau_patience', 3)}"
    )
    print(
        f"  stage1_threshold_policy={config.get('stage1_threshold_policy')} | "
        f"stage1_min_cancer_recall={config.get('stage1_min_cancer_recall')}"
    )
    print(
        f"  stage2_train_threshold={config.get('stage2_train_threshold')} | "
        f"stage2_gate_dilation={config.get('stage2_gate_dilation')} | "
        f"keep_tumor_outside_gate={config.get('stage2_keep_tumor_outside_gate')}"
    )
    print(
        f"  infer_gate_dilation={config.get('stage1_infer_gate_dilation')} | "
        f"enable_stage2_rescue={config.get('enable_stage2_rescue')} | "
        f"stage2_rescue_threshold={config.get('stage2_rescue_threshold')}"
    )

    use_wandb = not args.no_wandb
    if use_wandb and not args.dry_run:
        try:
            run_name = args.wandb_name or f"CONCH_hier_bs{config['batch_size']}_ga{config['grad_accum_steps']}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    "script": "training_conch_hierarchical",
                    "img_size": IMG_SIZE,
                    "seed": config["seed"],
                    "batch_size": config["batch_size"],
                    "grad_accum_steps": config["grad_accum_steps"],
                    "effective_batch": eff,
                    "learning_rate": config["learning_rate"],
                    "weight_decay": config["weight_decay"],
                    "unfreeze_last": config["unfreeze_last"],
                    "class_weights_stage1": config["class_weights_stage1"],
                    "class_weights_stage2": config["class_weights_stage2"],
                    "dice_weight_stage1": config["dice_weight_stage1"],
                    "ce_weight_stage1": config["ce_weight_stage1"],
                    "dice_weight_stage2": config["dice_weight_stage2"],
                    "ce_weight_stage2": config["ce_weight_stage2"],
                    "stage2_ignore_index": config["stage2_ignore_index"],
                    "max_epochs_stage1": config["max_epochs_stage1"],
                    "max_epochs_stage2": config["max_epochs_stage2"],
                    "patience_stage1": config["patience_stage1"],
                    "patience_stage2": config["patience_stage2"],
                    "lr_plateau_patience": config.get("lr_plateau_patience", 3),
                    "stage1_min_cancer_recall": config.get("stage1_min_cancer_recall"),
                    "stage1_threshold_policy": config.get("stage1_threshold_policy"),
                    "stage2_train_threshold": config.get("stage2_train_threshold"),
                    "stage2_gate_dilation": config.get("stage2_gate_dilation"),
                    "stage2_keep_tumor_outside_gate": config.get("stage2_keep_tumor_outside_gate"),
                    "stage1_infer_gate_dilation": config.get("stage1_infer_gate_dilation"),
                    "enable_stage2_rescue": config.get("enable_stage2_rescue"),
                    "stage2_rescue_threshold": config.get("stage2_rescue_threshold"),
                    "fold": args.fold,
                },
                tags=["CONCH", "SICAPv2", "hierarchical"],
            )
        except Exception as e:
            print(f"  [WARN] W&B no disponible: {e}")
            use_wandb = False

    fold_names = args.fold if args.fold else ["Val1", "Val2", "Val3", "Val4"]
    aggregated_stage2_cm = np.zeros((NUM_CLASSES_STAGE2, NUM_CLASSES_STAGE2), dtype=np.int64)
    aggregated_hier_cm = np.zeros((NUM_CLASSES_STAGE2, NUM_CLASSES_STAGE2), dtype=np.int64)

    for fold in fold_names:
        s1 = train_fold_stage1(fold, config, device, dry_run=args.dry_run, use_wandb=use_wandb)
        if s1["path"] is None:
            print(f"  [WARN] Fold {fold}: Stage1 sin checkpoint; no se puede entrenar Stage2 gated.")
            continue
        s2 = train_fold_stage2(
            fold_name=fold,
            config=config,
            device=device,
            stage1_path=s1["path"],
            stage1_threshold=float(s1["threshold"]),
            dry_run=args.dry_run,
            use_wandb=use_wandb,
        )
        if s2["cm"] is not None:
            aggregated_stage2_cm += s2["cm"]
        if s1["path"] is None or s2["path"] is None:
            print(f"  [WARN] Fold {fold}: faltan checkpoints para evaluar cascada.")
            continue
        h = evaluate_fold_hierarchical(fold, config, device, s1["path"], s2["path"], float(s1["threshold"]))
        aggregated_hier_cm += h["confusion_matrix"]
        print(f"\n  [Fold {fold}] S1 best macro-F1={s1['macro_f1']:.4f} @th={s1['threshold']:.2f}")
        if s1.get("stats") is not None:
            print(
                f"  [Fold {fold}] S1 cancer recall={s1['stats']['cancer_recall']:.4f} | "
                f"precision={s1['stats']['cancer_precision']:.4f}"
            )
        print(f"  [Fold {fold}] S2 best macro-F1={s2['macro_f1']:.4f}")
        print(
            f"  [Fold {fold}] Hier macro-F1={h['macro_f1']:.4f} | "
            f"rescue_ratio={100.0 * h.get('rescued_ratio', 0.0):.2f}%"
        )
        if use_wandb and wandb.run is not None:
            wandb.log({
                f"{fold}/stage1_best_macro_f1": float(s1["macro_f1"]),
                f"{fold}/stage1_best_threshold": float(s1["threshold"]),
                f"{fold}/stage1_best_cancer_recall": float(s1["stats"]["cancer_recall"]) if s1.get("stats") else 0.0,
                f"{fold}/stage1_best_cancer_precision": float(s1["stats"]["cancer_precision"]) if s1.get("stats") else 0.0,
                f"{fold}/stage2_best_macro_f1": float(s2["macro_f1"]),
                f"{fold}/hier_macro_f1": float(h["macro_f1"]),
                f"{fold}/hier_rescue_ratio": float(h.get("rescued_ratio", 0.0)),
            })

    _print_cm_4class(aggregated_stage2_cm, "Aggregated Stage2 (flat 4-class) confusion matrix")
    _print_cm_4class(aggregated_hier_cm, "Aggregated Hierarchical (Stage1+Stage2) confusion matrix")

    if use_wandb and wandb.run is not None:
        wandb.log({
            "aggregated/stage2_macro_f1": _macro_f1_from_cm(aggregated_stage2_cm),
            "aggregated/hier_macro_f1": _macro_f1_from_cm(aggregated_hier_cm),
        })
        wandb.finish()

    if args.dry_run:
        print("\nDry run completado.")


if __name__ == "__main__":
    main()
