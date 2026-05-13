"""
SICAPv2 Semantic Segmentation — CONCH v2 Multitask Hybrid
==========================================================

Aggressive branch built on top of training_conchv2.py:
- Keeps direct 4-class training as the main task.
- Replaces the plain FPN with a light Hybrid U-Net decoder.
- Adds two auxiliary decomposition heads:
    1. Cancer vs NC
    2. Tumor-only grade prediction (GG3/GG4/GG5)
- Uses deep supervision plus a fused inference path that blends:
    - main 4-class probabilities
    - auxiliary decomposition probabilities

The goal is to get the upside of a cascade without the hard external gate:
the model can still reject pixels back to NC because the main 4-class head
remains the final authority, while the auxiliary heads regularize the decoder
and can be blended during validation/inference.

Usage:
    python src/training_conchv2_multitask.py --no-wandb --dry-run
    python src/training_conchv2_multitask.py --fold Val1 --unfreeze-last 6
    python src/training_conchv2_multitask.py --fold Val1 Val2 Val3 Val4 --no-compile
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
IGNORE_INDEX = 255


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


WANDB_PROJECT_DEFAULT = "SICAPv2_CONCH_MultitaskHybrid"

from paths import IMAGES_DIR, MASKS_DIR, PARTITION_DIR, default_checkpoint_dir

OUTPUT_DIR = default_checkpoint_dir("checkpoints_conch_multitask_hybrid")

NUM_CLASSES = 4
GRADE_CLASSES = 3
CLASS_NAMES = ["NC", "GG3", "GG4", "GG5"]
GRADE_CLASS_NAMES = ["GG3", "GG4", "GG5"]
IMG_SIZE = 512

_PIXEL_FRAC_PARTITION = np.array([0.7781, 0.0605, 0.1175, 0.0439], dtype=np.float64)
_DEFAULT_CE_CLASS_WEIGHTS = (
    np.sqrt(1.0 / _PIXEL_FRAC_PARTITION) / np.sqrt(1.0 / _PIXEL_FRAC_PARTITION[0])
)
DEFAULT_CLASS_WEIGHTS = [float(round(x, 3)) for x in _DEFAULT_CE_CLASS_WEIGHTS]

_TUMOR_PIXEL_FRAC = _PIXEL_FRAC_PARTITION[1:] / _PIXEL_FRAC_PARTITION[1:].sum()
_DEFAULT_GRADE_CLASS_WEIGHTS = (
    np.sqrt(1.0 / _TUMOR_PIXEL_FRAC) / np.min(np.sqrt(1.0 / _TUMOR_PIXEL_FRAC))
)
DEFAULT_GRADE_CLASS_WEIGHTS = [float(round(x, 3)) for x in _DEFAULT_GRADE_CLASS_WEIGHTS]

_FEATURE_BLOCKS = [2, 5, 8, 11]
CONCH_HF_CHECKPOINT = "hf_hub:MahmoodLab/conch"

DEFAULT_CONFIG = {
    "num_classes": NUM_CLASSES,
    "num_workers": 4,
    "weight_decay": 2e-4,
    "max_epochs": 100,
    "patience": 16,
    "learning_rate": 4e-5,
    "lr_plateau_patience": 3,
    "batch_size": 6,
    "grad_accum_steps": 2,
    "unfreeze_last": 0,
    "decoder_channels": 256,
    "decoder_dropout": 0.10,
    "class_weights": list(DEFAULT_CLASS_WEIGHTS),
    "grade_class_weights": list(DEFAULT_GRADE_CLASS_WEIGHTS),
    "main_ce_weight": 0.45,
    "main_dice_weight": 0.20,
    "main_tversky_weight": 0.35,
    "main_label_smoothing": 0.02,
    "aux_weight_1": 0.25,
    "aux_weight_2": 0.10,
    "tumor_loss_weight": 0.30,
    "tumor_pos_weight": 2.5,
    "grade_loss_weight": 0.30,
    "grade_label_smoothing": 0.02,
    "use_weighted_sampler": True,
    "sampler_weight_gg5": 2.5,
    "sampler_weight_gg4": 1.3,
    "sampler_weight_gg3": 1.8,
    "fusion_alpha": 0.30,
    "calibrate_fusion_alpha": True,
    "fusion_alpha_min": 0.0,
    "fusion_alpha_max": 0.45,
    "fusion_alpha_step": 0.15,
    "use_compile": None,
    "conch_checkpoint": None,
    "conch_hf_token": None,
    "seed": DEFAULT_SEED,
}

_MASK_LUT = np.zeros(256, dtype=np.int64)
_MASK_LUT[25:75] = 1
_MASK_LUT[75:175] = 2
_MASK_LUT[175:] = 3


def grade_target_from_mask4(mask4: torch.Tensor) -> torch.Tensor:
    target = torch.full_like(mask4, IGNORE_INDEX)
    target[mask4 == 1] = 0
    target[mask4 == 2] = 1
    target[mask4 == 3] = 2
    return target


def compute_sample_weights(
    image_names: list[str],
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
    def __init__(self, image_names: list[str], images_dir: Path, masks_dir: Path, transform=None):
        self.image_names = image_names
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

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
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.20, p=0.5),
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
        w5 = float(config["sampler_weight_gg5"])
        w4 = float(config["sampler_weight_gg4"])
        w3 = float(config["sampler_weight_gg3"])
        sw = compute_sample_weights(train_names, MASKS_DIR, w5, w4, w3)
        n5 = sum(1 for x in sw if x == w5)
        n4 = sum(1 for x in sw if x == w4)
        n3 = sum(1 for x in sw if x == w3)
        print(
            f"  [Sampler] train={len(train_names)} | GG5×{w5}={n5} | "
            f"GG4×{w4}={n4} | GG3×{w3}={n3}"
        )
        sampler = WeightedRandomSampler(weights=sw, num_samples=len(train_names), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            sampler=sampler,
            drop_last=True,
            **dl_common,
            **kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=True,
            **dl_common,
            **kwargs,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        **dl_common,
        **kwargs,
    )
    return train_loader, val_loader


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
            raise FileNotFoundError(f"Checkpoint not found: {wp}")
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
            print(f"  [ConcHEncoder] Unfrozen the last {unfreeze_last}/{total} blocks")
        else:
            print("  [ConcHEncoder] Frozen encoder (decoder-only training)")

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
            raise RuntimeError(f"Spatial tokens do not form a square grid: N={num_spatial}")

        features = []
        for i, blk in enumerate(self.trunk.blocks):
            x_tok = blk(x_tok)
            if i in self.feature_blocks:
                spatial = x_tok[:, self.num_prefix:self.num_prefix + h_p * w_p, :]
                spatial = spatial.permute(0, 2, 1).reshape(bsz, self.embed_dim, h_p, w_p)
                features.append(spatial)
        return features


class AttentionGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        inter = max(channels // 4, 16)
        self.g_proj = nn.Conv2d(channels, inter, 1, bias=False)
        self.x_proj = nn.Conv2d(channels, inter, 1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(inter, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, guide: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        alpha = self.psi(F.relu(self.g_proj(guide) + self.x_proj(skip), inplace=True))
        return skip * alpha


class HybridMultiTaskDecoder(nn.Module):
    def __init__(self, in_channels: int = 768, dec_ch: int = 256, dropout: float = 0.10):
        super().__init__()
        self.lat = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, dec_ch, 1, bias=False),
                nn.BatchNorm2d(dec_ch),
                nn.ReLU(inplace=True),
            ) for _ in range(4)
        ])
        self.att = nn.ModuleList([AttentionGate(dec_ch) for _ in range(3)])
        self.merge = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dec_ch * 2, dec_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(dec_ch),
                nn.ReLU(inplace=True),
            ) for _ in range(3)
        ])
        self.shared = nn.Sequential(
            nn.Conv2d(dec_ch, dec_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        mid_ch = dec_ch // 2
        self.main_head = nn.Conv2d(mid_ch, NUM_CLASSES, 1)
        self.tumor_head = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, 1),
        )
        self.grade_head = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(mid_ch, GRADE_CLASSES, 1),
        )
        self.aux1 = nn.Conv2d(dec_ch, NUM_CLASSES, 1)
        self.aux2 = nn.Conv2d(dec_ch, NUM_CLASSES, 1)

    def forward(self, features, target_size):
        lats = [lat(f) for lat, f in zip(self.lat, features)]

        x = lats[3]
        out_lvl2 = None
        out_lvl1 = None
        for i in range(2, -1, -1):
            skip = self.att[i](x, lats[i])
            x = self.merge[i](torch.cat([x, skip], dim=1))
            if i == 2:
                out_lvl2 = x
            if i == 1:
                out_lvl1 = x

        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        shared = self.shared(x)
        main_logits = self.main_head(shared)
        tumor_logits = self.tumor_head(shared)
        grade_logits = self.grade_head(shared)
        aux_logits = [
            F.interpolate(self.aux1(out_lvl2), size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(self.aux2(out_lvl1), size=target_size, mode="bilinear", align_corners=False),
        ]
        return {
            "main_logits": main_logits,
            "aux_logits": aux_logits,
            "tumor_logits": tumor_logits,
            "grade_logits": grade_logits,
        }


class CONCHMultiTaskModel(nn.Module):
    def __init__(
        self,
        decoder_channels=256,
        unfreeze_last=0,
        decoder_dropout=0.10,
        weights_path=None,
        hf_token=None,
    ):
        super().__init__()
        self.encoder = ConcHEncoder(
            unfreeze_last=unfreeze_last,
            weights_path=weights_path,
            hf_token=hf_token,
        )
        self.decoder = HybridMultiTaskDecoder(
            in_channels=self.encoder.embed_dim,
            dec_ch=decoder_channels,
            dropout=decoder_dropout,
        )

    def forward(self, x):
        target_size = x.shape[-2:]
        features = self.encoder(x)
        return self.decoder(features, target_size)


def build_model(config: dict):
    model = CONCHMultiTaskModel(
        decoder_channels=config["decoder_channels"],
        unfreeze_last=config["unfreeze_last"],
        decoder_dropout=config.get("decoder_dropout", 0.10),
        weights_path=config.get("conch_checkpoint"),
        hf_token=config.get("conch_hf_token"),
    )
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [Params] Total: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.1f}%)")
    return model


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        class_weights: list[float],
        grade_class_weights: list[float],
        *,
        main_ce_weight: float,
        main_dice_weight: float,
        main_tversky_weight: float,
        aux_weight_1: float,
        aux_weight_2: float,
        tumor_loss_weight: float,
        tumor_pos_weight: float,
        grade_loss_weight: float,
        main_label_smoothing: float,
        grade_label_smoothing: float,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.main_ce_weight = float(main_ce_weight)
        self.main_dice_weight = float(main_dice_weight)
        self.main_tversky_weight = float(main_tversky_weight)
        self.aux_weight_1 = float(aux_weight_1)
        self.aux_weight_2 = float(aux_weight_2)
        self.tumor_loss_weight = float(tumor_loss_weight)
        self.grade_loss_weight = float(grade_loss_weight)

        self.register_buffer("main_weights_tensor", torch.tensor(class_weights, dtype=torch.float32))
        self.register_buffer("grade_weights_tensor", torch.tensor(grade_class_weights, dtype=torch.float32))
        self.register_buffer("tumor_pos_weight_tensor", torch.tensor([tumor_pos_weight], dtype=torch.float32))

        self.main_ce = nn.CrossEntropyLoss(
            weight=self.main_weights_tensor,
            label_smoothing=float(main_label_smoothing),
        )
        self.main_dice = smp.losses.DiceLoss(
            mode="multiclass",
            classes=[0, 1, 2, 3],
            smooth=smooth,
        )
        self.main_tversky = smp.losses.TverskyLoss(
            mode="multiclass",
            classes=[1, 2, 3],
            smooth=smooth,
            alpha=0.30,
            beta=0.70,
        )

        self.tumor_dice = smp.losses.DiceLoss(mode="binary", from_logits=True, smooth=smooth)
        self.tumor_tversky = smp.losses.TverskyLoss(
            mode="binary",
            from_logits=True,
            smooth=smooth,
            alpha=0.30,
            beta=0.70,
        )

        self.grade_ce = nn.CrossEntropyLoss(
            weight=self.grade_weights_tensor,
            ignore_index=IGNORE_INDEX,
            label_smoothing=float(grade_label_smoothing),
        )
        self.grade_dice = smp.losses.DiceLoss(
            mode="multiclass",
            classes=[0, 1, 2],
            ignore_index=IGNORE_INDEX,
            smooth=smooth,
        )
        self.grade_tversky = smp.losses.TverskyLoss(
            mode="multiclass",
            classes=[0, 1, 2],
            ignore_index=IGNORE_INDEX,
            smooth=smooth,
            alpha=0.25,
            beta=0.75,
        )

    def _main_loss(self, logits: torch.Tensor, targets: torch.Tensor):
        logits = logits.float()
        ce = self.main_ce(logits, targets)
        dice = self.main_dice(logits, targets)
        tversky = self.main_tversky(logits, targets)
        total = (
            self.main_ce_weight * ce
            + self.main_dice_weight * dice
            + self.main_tversky_weight * tversky
        )
        return total, ce.detach(), dice.detach(), tversky.detach()

    def _tumor_loss(self, tumor_logits: torch.Tensor, targets: torch.Tensor):
        tumor_logits = tumor_logits.float()
        tumor_targets = (targets > 0).unsqueeze(1).float()
        bce = F.binary_cross_entropy_with_logits(
            tumor_logits,
            tumor_targets,
            pos_weight=self.tumor_pos_weight_tensor,
        )
        dice = self.tumor_dice(tumor_logits, tumor_targets)
        tversky = self.tumor_tversky(tumor_logits, tumor_targets)
        total = 0.35 * bce + 0.20 * dice + 0.45 * tversky
        return total, bce.detach(), dice.detach(), tversky.detach()

    def _grade_loss(self, grade_logits: torch.Tensor, targets: torch.Tensor):
        grade_logits = grade_logits.float()
        grade_targets = grade_target_from_mask4(targets)
        if not torch.any(grade_targets != IGNORE_INDEX):
            zero = grade_logits.sum() * 0.0
            return zero, zero.detach(), zero.detach(), zero.detach()
        ce = self.grade_ce(grade_logits, grade_targets)
        dice = self.grade_dice(grade_logits, grade_targets)
        tversky = self.grade_tversky(grade_logits, grade_targets)
        total = 0.40 * ce + 0.20 * dice + 0.40 * tversky
        return total, ce.detach(), dice.detach(), tversky.detach()

    def forward(self, outputs: dict[str, torch.Tensor], targets: torch.Tensor):
        parts = {}

        main_total, main_ce, main_dice, main_tversky = self._main_loss(outputs["main_logits"], targets)
        total = main_total
        parts["main_total"] = main_total.detach()
        parts["main_ce"] = main_ce
        parts["main_dice"] = main_dice
        parts["main_tversky"] = main_tversky

        if self.aux_weight_1 > 0:
            aux1_total, _, _, _ = self._main_loss(outputs["aux_logits"][0], targets)
            total = total + self.aux_weight_1 * aux1_total
            parts["aux1_total"] = aux1_total.detach()
        else:
            parts["aux1_total"] = torch.zeros_like(main_total.detach())

        if self.aux_weight_2 > 0:
            aux2_total, _, _, _ = self._main_loss(outputs["aux_logits"][1], targets)
            total = total + self.aux_weight_2 * aux2_total
            parts["aux2_total"] = aux2_total.detach()
        else:
            parts["aux2_total"] = torch.zeros_like(main_total.detach())

        tumor_total, tumor_bce, tumor_dice, tumor_tversky = self._tumor_loss(outputs["tumor_logits"], targets)
        total = total + self.tumor_loss_weight * tumor_total
        parts["tumor_total"] = tumor_total.detach()
        parts["tumor_bce"] = tumor_bce
        parts["tumor_dice"] = tumor_dice
        parts["tumor_tversky"] = tumor_tversky

        grade_total, grade_ce, grade_dice, grade_tversky = self._grade_loss(outputs["grade_logits"], targets)
        total = total + self.grade_loss_weight * grade_total
        parts["grade_total"] = grade_total.detach()
        parts["grade_ce"] = grade_ce
        parts["grade_dice"] = grade_dice
        parts["grade_tversky"] = grade_tversky
        parts["loss_total"] = total.detach()
        return total, parts


class SegmentationMetrics:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update_from_preds(self, preds: np.ndarray, targets: np.ndarray):
        mask = (targets >= 0) & (targets < self.num_classes)
        np.add.at(self.confusion_matrix, (targets[mask], preds[mask]), 1)

    def update_from_logits(self, logits: torch.Tensor, targets: torch.Tensor):
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        tgts = targets.detach().cpu().numpy()
        self.update_from_preds(preds, tgts)

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
        return {
            "macro_f1": float(np.nanmean(dice_per_class)),
            "f1_per_class": np.nan_to_num(dice_per_class, nan=0.0),
            "confusion_matrix": cm.copy(),
        }


class BinaryMetrics:
    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)
        self.confusion_matrix = np.zeros((2, 2), dtype=np.int64)

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits.float()).squeeze(1)
        preds = (probs >= self.threshold).long().detach().cpu().numpy()
        tgts = targets.long().detach().cpu().numpy()
        mask = (tgts >= 0) & (tgts < 2)
        np.add.at(self.confusion_matrix, (tgts[mask], preds[mask]), 1)

    def compute(self):
        cm = self.confusion_matrix
        f1 = np.zeros(2)
        precision = np.zeros(2)
        recall = np.zeros(2)
        for c in range(2):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            precision[c] = tp / (tp + fp + 1e-8)
            recall[c] = tp / (tp + fn + 1e-8)
            f1[c] = (2.0 * precision[c] * recall[c]) / (precision[c] + recall[c] + 1e-8)
        return {
            "macro_f1": float(np.nanmean(f1)),
            "f1_per_class": f1,
            "precision_per_class": precision,
            "recall_per_class": recall,
            "confusion_matrix": cm.copy(),
            "cancer_f1": float(f1[1]),
            "cancer_precision": float(precision[1]),
            "cancer_recall": float(recall[1]),
        }


class TumorGradeMetrics:
    def __init__(self):
        self.confusion_matrix = np.zeros((GRADE_CLASSES, GRADE_CLASSES), dtype=np.int64)

    def update(self, logits: torch.Tensor, targets4: torch.Tensor):
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        tgts = grade_target_from_mask4(targets4).detach().cpu().numpy()
        mask = (tgts >= 0) & (tgts < GRADE_CLASSES)
        if np.any(mask):
            np.add.at(self.confusion_matrix, (tgts[mask], preds[mask]), 1)

    def compute(self):
        cm = self.confusion_matrix
        f1 = np.zeros(GRADE_CLASSES)
        for c in range(GRADE_CLASSES):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            if tp + fp + fn == 0:
                f1[c] = np.nan
            else:
                f1[c] = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
        return {
            "macro_f1": float(np.nanmean(f1)),
            "f1_per_class": np.nan_to_num(f1, nan=0.0),
            "confusion_matrix": cm.copy(),
        }


def build_fused_probs(outputs: dict[str, torch.Tensor], alpha: float) -> torch.Tensor:
    alpha = float(alpha)
    main_probs = torch.softmax(outputs["main_logits"].float(), dim=1)
    tumor_prob = torch.sigmoid(outputs["tumor_logits"].float()).clamp(1e-4, 1.0 - 1e-4)
    grade_probs = torch.softmax(outputs["grade_logits"].float(), dim=1)
    aux_probs = torch.cat([1.0 - tumor_prob, tumor_prob * grade_probs], dim=1)
    return (1.0 - alpha) * main_probs + alpha * aux_probs


def build_fusion_alpha_grid(config: dict) -> list[float]:
    if not config.get("calibrate_fusion_alpha", True):
        return [float(config.get("fusion_alpha", 0.30))]
    amin = float(config.get("fusion_alpha_min", 0.0))
    amax = float(config.get("fusion_alpha_max", 0.45))
    astep = float(config.get("fusion_alpha_step", 0.15))
    if astep <= 0:
        return [float(config.get("fusion_alpha", 0.30))]
    grid = np.arange(amin, amax + 1e-12, astep, dtype=np.float64).tolist()
    return [float(round(x, 4)) for x in grid]


def maybe_compile(model: nn.Module, use_compile):
    if use_compile is None:
        import platform
        use_compile = platform.system() != "Windows"
    if use_compile and int(torch.__version__.split(".")[0]) >= 2:
        try:
            import platform
            backend = "inductor" if platform.system() != "Windows" else "aot_eager"
            print("  ⏳ Compiling model with torch.compile...")
            model = torch.compile(model, backend=backend)
            print("  ✅ torch.compile enabled.")
        except Exception as e:
            print(f"  ⚠️ torch.compile unavailable: {e}")
    elif not use_compile:
        print("  ℹ️ torch.compile disabled (recommended on Windows to save VRAM).")
    return model


def _average_loss_parts(acc: dict[str, float], num_batches: int):
    if num_batches <= 0:
        return {k: 0.0 for k in acc}
    return {k: float(v) / float(num_batches) for k, v in acc.items()}


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_accum_steps: int = 1):
    model.train()
    total_loss, num_batches = 0.0, 0
    accum = 0
    part_sums: dict[str, float] = {}

    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()

        if accum == 0:
            optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            outputs = model(images)
            loss, parts = criterion(outputs, masks)

        scaler.scale(loss / grad_accum_steps).backward()
        total_loss += float(loss.item())
        num_batches += 1
        accum += 1
        for key, value in parts.items():
            part_sums[key] = part_sums.get(key, 0.0) + float(value.item())

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

    return {
        "loss": total_loss / max(num_batches, 1),
        "parts": _average_loss_parts(part_sums, num_batches),
    }


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, fusion_alphas: list[float]):
    model.eval()
    total_loss, num_batches = 0.0, 0
    part_sums: dict[str, float] = {}
    main_metrics = SegmentationMetrics(NUM_CLASSES)
    fused_metrics = {float(alpha): SegmentationMetrics(NUM_CLASSES) for alpha in fusion_alphas}
    tumor_metrics = BinaryMetrics(threshold=0.5)
    grade_metrics = TumorGradeMetrics()

    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()

        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            outputs = model(images)
            loss, parts = criterion(outputs, masks)

        total_loss += float(loss.item())
        num_batches += 1
        for key, value in parts.items():
            part_sums[key] = part_sums.get(key, 0.0) + float(value.item())

        main_metrics.update_from_logits(outputs["main_logits"], masks)
        tumor_metrics.update(outputs["tumor_logits"], (masks > 0).long())
        grade_metrics.update(outputs["grade_logits"], masks)

        masks_np = masks.detach().cpu().numpy()
        for alpha, metrics in fused_metrics.items():
            fused_probs = build_fused_probs(outputs, alpha)
            preds = fused_probs.argmax(dim=1).detach().cpu().numpy()
            metrics.update_from_preds(preds, masks_np)

    best_alpha = float(fusion_alphas[0])
    best_fused = fused_metrics[best_alpha].compute()
    for alpha, metrics in fused_metrics.items():
        stats = metrics.compute()
        if stats["macro_f1"] > best_fused["macro_f1"]:
            best_alpha = float(alpha)
            best_fused = stats

    return {
        "loss": total_loss / max(num_batches, 1),
        "parts": _average_loss_parts(part_sums, num_batches),
        "main_metrics": main_metrics.compute(),
        "fused_metrics": best_fused,
        "best_alpha": best_alpha,
        "tumor_metrics": tumor_metrics.compute(),
        "grade_metrics": grade_metrics.compute(),
    }


def print_aggregated_matrices(agg_cm: np.ndarray):
    print(f"\n{'='*60}\n  AGGREGATED CONFUSION MATRICES (ALL FOLDS)\n{'='*60}")

    print("\n[1] 4-Class Confusion Matrix (Rows: True, Cols: Pred)")
    df_4x4 = pd.DataFrame(agg_cm, index=[f"T_{c}" for c in CLASS_NAMES], columns=[f"P_{c}" for c in CLASS_NAMES])
    print(df_4x4.to_string())

    print("\n--- 4-Class Metrics ---")
    for i in range(NUM_CLASSES):
        tp = agg_cm[i, i]
        fp = agg_cm[:, i].sum() - tp
        fn = agg_cm[i, :].sum() - tp
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        print(f"  {CLASS_NAMES[i]:3s}: F1={f1:.4f}  Prec={precision:.4f}  Rec={recall:.4f}")

    print("\n[2] Binary Confusion Matrix (Cancer vs No Cancer)")
    nc_nc, nc_c = agg_cm[0, 0], agg_cm[0, 1:].sum()
    c_nc, c_c = agg_cm[1:, 0].sum(), agg_cm[1:, 1:].sum()
    df_2x2 = pd.DataFrame(
        np.array([[nc_nc, nc_c], [c_nc, c_c]]),
        index=["T_NoCancer", "T_Cancer"],
        columns=["P_NoCancer", "P_Cancer"],
    )
    print(df_2x2.to_string())

    b_tp, b_fp, b_tn, b_fn = c_c, nc_c, nc_nc, c_nc
    b_prec = b_tp / (b_tp + b_fp + 1e-8)
    b_rec = b_tp / (b_tp + b_fn + 1e-8)
    b_f1 = 2 * b_prec * b_rec / (b_prec + b_rec + 1e-8)
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


def save_summary_csv(out_dir: Path, rows: list[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "best_per_fold.csv", index=False)


def train_fold(fold_name: str, config: dict, device: torch.device, dry_run: bool = False, use_wandb: bool = True):
    print(f"\n{'='*60}\n  FOLD: {fold_name}\n{'='*60}")
    train_loader, val_loader = get_fold_dataloaders(fold_name, config)

    model = build_model(config).to(device)
    model = maybe_compile(model, config.get("use_compile"))

    criterion = MultiTaskLoss(
        config["class_weights"],
        config["grade_class_weights"],
        main_ce_weight=float(config["main_ce_weight"]),
        main_dice_weight=float(config["main_dice_weight"]),
        main_tversky_weight=float(config["main_tversky_weight"]),
        aux_weight_1=float(config["aux_weight_1"]),
        aux_weight_2=float(config["aux_weight_2"]),
        tumor_loss_weight=float(config["tumor_loss_weight"]),
        tumor_pos_weight=float(config["tumor_pos_weight"]),
        grade_loss_weight=float(config["grade_loss_weight"]),
        main_label_smoothing=float(config.get("main_label_smoothing", 0.0)),
        grade_label_smoothing=float(config.get("grade_label_smoothing", 0.0)),
    ).to(device)

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = list(model.decoder.parameters())
    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": config["learning_rate"] / 10, "label": "encoder"})
    param_groups.append({"params": decoder_params, "lr": config["learning_rate"], "label": "decoder"})
    for pg in param_groups:
        n_params = sum(p.numel() for p in pg["params"])
        print(f"  [Optim] {pg['label']}: {n_params:,} params, lr={pg['lr']:.2e}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=int(config.get("lr_plateau_patience", 3)),
        min_lr=1e-7,
    )
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    out_dir = Path(config.get("output_dir", OUTPUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)
    max_epochs = 1 if dry_run else int(config["max_epochs"])
    grad_accum_steps = int(config.get("grad_accum_steps", 1))
    fusion_alphas = build_fusion_alpha_grid(config)
    print(f"  [Fusion] alpha_grid={fusion_alphas}")

    best_macro_f1 = 0.0
    best_main_macro_f1 = 0.0
    best_tumor_cancer_f1 = 0.0
    best_grade_macro_f1 = 0.0
    best_alpha = float(config.get("fusion_alpha", 0.30))
    best_cm = None
    best_ckpt_path = None
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        lr_dec = optimizer.param_groups[-1]["lr"]
        print(f"\n  Epoch {epoch}/{max_epochs}  (decoder lr={lr_dec:.2e}, accum={grad_accum_steps})")

        train_res = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            grad_accum_steps=grad_accum_steps,
        )
        val_res = validate_one_epoch(model, val_loader, criterion, device, fusion_alphas)

        train_loss = float(train_res["loss"])
        val_loss = float(val_res["loss"])
        fused_metrics = val_res["fused_metrics"]
        main_metrics = val_res["main_metrics"]
        tumor_metrics = val_res["tumor_metrics"]
        grade_metrics = val_res["grade_metrics"]
        macro_f1 = float(fused_metrics["macro_f1"])
        ratio = val_loss / (train_loss + 1e-8)
        scheduler.step(macro_f1)

        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}  |  Val/Train: {ratio:.3f}")
        print(f"  Fused Macro F1: {macro_f1:.4f} | best_fusion_alpha={val_res['best_alpha']:.2f}")
        print(f"  Main-head Macro F1: {main_metrics['macro_f1']:.4f}")
        print(f"  Aux Tumor Cancer F1: {tumor_metrics['cancer_f1']:.4f}")
        print(f"  Aux Tumor-only Grade Macro F1: {grade_metrics['macro_f1']:.4f}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"    {name} F1: {fused_metrics['f1_per_class'][i]:.4f}")

        if use_wandb and wandb.run is not None:
            enc_lr = float(optimizer.param_groups[0]["lr"]) if encoder_params else 0.0
            dec_lr = float(optimizer.param_groups[-1]["lr"])
            log_dict = {
                "epoch": epoch,
                f"{fold_name}/train_loss": train_loss,
                f"{fold_name}/val_loss": val_loss,
                f"{fold_name}/val_train_loss_ratio": ratio,
                f"{fold_name}/macro_f1_fused": macro_f1,
                f"{fold_name}/macro_f1_main": float(main_metrics["macro_f1"]),
                f"{fold_name}/fusion_alpha": float(val_res["best_alpha"]),
                f"{fold_name}/aux_cancer_f1": float(tumor_metrics["cancer_f1"]),
                f"{fold_name}/aux_grade_macro_f1": float(grade_metrics["macro_f1"]),
                f"{fold_name}/lr_encoder": enc_lr,
                f"{fold_name}/lr_decoder": dec_lr,
            }
            for key, value in train_res["parts"].items():
                log_dict[f"{fold_name}/train_{key}"] = float(value)
            for key, value in val_res["parts"].items():
                log_dict[f"{fold_name}/val_{key}"] = float(value)
            for i, name in enumerate(CLASS_NAMES):
                log_dict[f"{fold_name}/f1_{name}"] = float(fused_metrics["f1_per_class"][i])
            for i, name in enumerate(GRADE_CLASS_NAMES):
                log_dict[f"{fold_name}/aux_grade_f1_{name}"] = float(grade_metrics["f1_per_class"][i])
            wandb.log(log_dict)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_main_macro_f1 = float(main_metrics["macro_f1"])
            best_tumor_cancer_f1 = float(tumor_metrics["cancer_f1"])
            best_grade_macro_f1 = float(grade_metrics["macro_f1"])
            best_alpha = float(val_res["best_alpha"])
            best_cm = fused_metrics["confusion_matrix"]
            best_ckpt_path = out_dir / f"best_{fold_name}_{macro_f1:.4f}_a{best_alpha:.2f}.pth"
            torch.save(model.state_dict(), best_ckpt_path)
            patience_counter = 0
            print(f"  ✓ Model saved (Fused Macro F1={macro_f1:.4f}, alpha={best_alpha:.2f})")
        else:
            patience_counter += 1
            if patience_counter >= int(config["patience"]):
                print(f"  ⛔ Early stopping triggered at epoch {epoch}")
                break

        if dry_run:
            break

    return {
        "fold": fold_name,
        "best_macro_f1": best_macro_f1,
        "best_main_macro_f1": best_main_macro_f1,
        "best_aux_cancer_f1": best_tumor_cancer_f1,
        "best_aux_grade_macro_f1": best_grade_macro_f1,
        "best_alpha": best_alpha,
        "best_cm": best_cm,
        "checkpoint_path": str(best_ckpt_path) if best_ckpt_path is not None else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke test: 1 batch per fold.")
    parser.add_argument("--unfreeze-last", type=int, default=0, help="Unfreeze the last N CONCH blocks.")
    parser.add_argument("--weights", type=str, default=None, metavar="PATH", help="Local CONCH checkpoint.")
    parser.add_argument("--hf-token", type=str, default=None, help="HF token.")
    parser.add_argument("--batch-size", type=int, default=None, help="Micro-batch.")
    parser.add_argument("--grad-accum", type=int, default=None, metavar="K", help="Gradient accumulation.")
    parser.add_argument("--lr", type=float, default=None, help="Decoder LR.")
    parser.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience.")
    parser.add_argument("--lr-plateau-patience", type=int, default=None, help="ReduceLROnPlateau patience.")
    parser.add_argument("--decoder-channels", type=int, default=None, help="Internal decoder width.")
    parser.add_argument("--decoder-dropout", type=float, default=None, help="Dropout in the shared decoder stem.")
    parser.add_argument("--aux-weight-1", type=float, default=None, help="Deep supervision weight for aux head 1.")
    parser.add_argument("--aux-weight-2", type=float, default=None, help="Deep supervision weight for aux head 2.")
    parser.add_argument("--tumor-loss-weight", type=float, default=None, help="Weight of the binary tumor auxiliary loss.")
    parser.add_argument("--tumor-pos-weight", type=float, default=None, help="Positive weight for the tumor BCE loss.")
    parser.add_argument("--grade-loss-weight", type=float, default=None, help="Weight of the tumor-only grade auxiliary loss.")
    parser.add_argument("--main-label-smoothing", type=float, default=None, help="Label smoothing for the main 4-class CE.")
    parser.add_argument("--grade-label-smoothing", type=float, default=None, help="Label smoothing for the tumor-only grade CE.")
    parser.add_argument("--fusion-alpha", type=float, default=None, help="Default fusion alpha.")
    parser.add_argument("--no-fusion-calibration", action="store_true", help="Disable fusion-alpha search on validation.")
    parser.add_argument("--fusion-alpha-min", type=float, default=None, help="Min fusion alpha for calibration.")
    parser.add_argument("--fusion-alpha-max", type=float, default=None, help="Max fusion alpha for calibration.")
    parser.add_argument("--fusion-alpha-step", type=float, default=None, help="Step for fusion alpha calibration.")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--sampler-gg5", type=float, default=None, metavar="W")
    parser.add_argument("--sampler-gg4", type=float, default=None, metavar="W")
    parser.add_argument("--sampler-gg3", type=float, default=None, metavar="W")
    parser.add_argument("--compile", action="store_true", help="Force torch.compile.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")
    parser.add_argument("--no-wandb", action="store_true", help="Do not log to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT_DEFAULT, help="W&B project.")
    parser.add_argument("--wandb-name", type=str, default=None, help="Run name.")
    parser.add_argument(
        "--fold",
        type=str,
        nargs="+",
        default=None,
        choices=["Val1", "Val2", "Val3", "Val4"],
        help="Run one or multiple specific folds.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for checkpoints.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Seed (default: {DEFAULT_SEED}).")
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Seed: {args.seed} (cudnn deterministic=True, benchmark=False)")

    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
            print("\n[System] Anti-sleep enabled.")
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
    if args.lr is not None:
        config["learning_rate"] = float(args.lr)
    if args.weight_decay is not None:
        config["weight_decay"] = float(args.weight_decay)
    if args.patience is not None:
        config["patience"] = int(args.patience)
    if args.lr_plateau_patience is not None:
        config["lr_plateau_patience"] = int(args.lr_plateau_patience)
    if args.decoder_channels is not None:
        config["decoder_channels"] = int(args.decoder_channels)
    if args.decoder_dropout is not None:
        config["decoder_dropout"] = float(args.decoder_dropout)
    if args.aux_weight_1 is not None:
        config["aux_weight_1"] = float(args.aux_weight_1)
    if args.aux_weight_2 is not None:
        config["aux_weight_2"] = float(args.aux_weight_2)
    if args.tumor_loss_weight is not None:
        config["tumor_loss_weight"] = float(args.tumor_loss_weight)
    if args.tumor_pos_weight is not None:
        config["tumor_pos_weight"] = float(args.tumor_pos_weight)
    if args.grade_loss_weight is not None:
        config["grade_loss_weight"] = float(args.grade_loss_weight)
    if args.main_label_smoothing is not None:
        config["main_label_smoothing"] = float(args.main_label_smoothing)
    if args.grade_label_smoothing is not None:
        config["grade_label_smoothing"] = float(args.grade_label_smoothing)
    if args.fusion_alpha is not None:
        config["fusion_alpha"] = float(args.fusion_alpha)
    if args.no_fusion_calibration:
        config["calibrate_fusion_alpha"] = False
    if args.fusion_alpha_min is not None:
        config["fusion_alpha_min"] = float(args.fusion_alpha_min)
    if args.fusion_alpha_max is not None:
        config["fusion_alpha_max"] = float(args.fusion_alpha_max)
    if args.fusion_alpha_step is not None:
        config["fusion_alpha_step"] = float(args.fusion_alpha_step)
    if args.no_weighted_sampler:
        config["use_weighted_sampler"] = False
    if args.sampler_gg5 is not None:
        config["sampler_weight_gg5"] = float(args.sampler_gg5)
    if args.sampler_gg4 is not None:
        config["sampler_weight_gg4"] = float(args.sampler_gg4)
    if args.sampler_gg3 is not None:
        config["sampler_weight_gg3"] = float(args.sampler_gg3)
    if args.compile and args.no_compile:
        print("  [WARN] --compile and --no-compile together; using --no-compile.")
    if args.no_compile:
        config["use_compile"] = False
    elif args.compile:
        config["use_compile"] = True

    config["output_dir"] = Path(args.output_dir).resolve() if args.output_dir else OUTPUT_DIR
    print(f"Checkpoints -> {config['output_dir']}")

    eff = config["batch_size"] * config.get("grad_accum_steps", 1)
    print(
        f"CONCH multitask: micro_batch={config['batch_size']} × accum={config.get('grad_accum_steps', 1)} "
        f"≈ {eff} | weighted_sampler={config.get('use_weighted_sampler')} | lr={config['learning_rate']}"
    )
    print(
        f"  class_weights={config['class_weights']} | grade_class_weights={config['grade_class_weights']} | "
        f"decoder_ch={config['decoder_channels']} | dropout={config['decoder_dropout']}"
    )
    print(
        f"  main(ce/dice/tversky)={config['main_ce_weight']}/{config['main_dice_weight']}/{config['main_tversky_weight']} | "
        f"aux1={config['aux_weight_1']} aux2={config['aux_weight_2']} | "
        f"tumor_w={config['tumor_loss_weight']} grade_w={config['grade_loss_weight']}"
    )
    print(
        f"  fusion_alpha={config['fusion_alpha']} | calibration={config.get('calibrate_fusion_alpha', True)} "
        f"[{config.get('fusion_alpha_min')}, {config.get('fusion_alpha_max')}] step {config.get('fusion_alpha_step')}"
    )

    use_wandb = not args.no_wandb
    if use_wandb and not args.dry_run:
        try:
            wandb_config = {
                "script": "training_conchv2_multitask",
                "img_size": IMG_SIZE,
                "encoder": "CONCH_ViT-B-16_visual",
                "decoder_channels": config["decoder_channels"],
                "decoder_dropout": config["decoder_dropout"],
                "batch_size": config["batch_size"],
                "grad_accum_steps": config.get("grad_accum_steps", 1),
                "effective_batch": eff,
                "learning_rate": config["learning_rate"],
                "weight_decay": config["weight_decay"],
                "max_epochs": config["max_epochs"],
                "patience": config["patience"],
                "class_weights": list(config["class_weights"]),
                "grade_class_weights": list(config["grade_class_weights"]),
                "main_ce_weight": config["main_ce_weight"],
                "main_dice_weight": config["main_dice_weight"],
                "main_tversky_weight": config["main_tversky_weight"],
                "aux_weight_1": config["aux_weight_1"],
                "aux_weight_2": config["aux_weight_2"],
                "tumor_loss_weight": config["tumor_loss_weight"],
                "tumor_pos_weight": config["tumor_pos_weight"],
                "grade_loss_weight": config["grade_loss_weight"],
                "main_label_smoothing": config["main_label_smoothing"],
                "grade_label_smoothing": config["grade_label_smoothing"],
                "unfreeze_last": config["unfreeze_last"],
                "weighted_sampler": config.get("use_weighted_sampler", False),
                "sampler_weight_gg5": config.get("sampler_weight_gg5"),
                "sampler_weight_gg4": config.get("sampler_weight_gg4"),
                "sampler_weight_gg3": config.get("sampler_weight_gg3"),
                "fusion_alpha": config.get("fusion_alpha"),
                "calibrate_fusion_alpha": config.get("calibrate_fusion_alpha"),
                "fusion_alpha_min": config.get("fusion_alpha_min"),
                "fusion_alpha_max": config.get("fusion_alpha_max"),
                "fusion_alpha_step": config.get("fusion_alpha_step"),
                "conch_checkpoint": config.get("conch_checkpoint") or CONCH_HF_CHECKPOINT,
                "output_dir": str(config["output_dir"]),
                "mask_lut": "25:75->GG3, 75:175->GG4, 175:255->GG5, else->NC",
                "pixel_frac_partition": [float(x) for x in _PIXEL_FRAC_PARTITION],
                "seed": config.get("seed", DEFAULT_SEED),
                "fold": args.fold,
            }
            run_name = args.wandb_name or (
                f"CONCH_multitask_u{config['unfreeze_last']}_bs{config['batch_size']}_ga{config.get('grad_accum_steps', 1)}"
            )
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=wandb_config,
                tags=["mask_lut", "CONCH", "SICAPv2", "multitask", "hybrid"],
            )
        except Exception as e:
            print(f"  [WARN] Weights & Biases unavailable: {e}")
            use_wandb = False

    if args.fold:
        fold_names = args.fold if isinstance(args.fold, list) else [args.fold]
    else:
        fold_names = ["Val1", "Val2", "Val3", "Val4"]

    aggregated_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    best_rows = []

    for fold in fold_names:
        res = train_fold(fold, config, device, args.dry_run, use_wandb=use_wandb)
        best_rows.append({
            "fold": res["fold"],
            "macro_f1_fused": res["best_macro_f1"],
            "macro_f1_main": res["best_main_macro_f1"],
            "aux_cancer_f1": res["best_aux_cancer_f1"],
            "aux_grade_macro_f1": res["best_aux_grade_macro_f1"],
            "fusion_alpha": res["best_alpha"],
            "checkpoint_path": res["checkpoint_path"],
        })
        if res["best_cm"] is not None:
            aggregated_cm += res["best_cm"]

    save_summary_csv(Path(config["output_dir"]), best_rows)
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
