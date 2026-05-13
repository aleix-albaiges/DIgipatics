"""
CONCH v2 variant with label denoising and light decoder regularization.

Why this variant exists:
- The latest `training_conchv2.py` run peaks very early on Val1 and then collapses,
  while Val2/Val3/Val4 remain strong.
- Aggregated confusion shows GG3 precision is the weakest point, which is
  consistent with tiny noisy components and ambiguous JPEG boundaries.

Changes versus `training_conchv2.py`:
- Optional connected-component cleanup on mapped masks.
- Light decoder dropout.
- Small CE label smoothing.
- Exposes weight decay / LR plateau patience / mask cleanup hyperparameters by CLI.
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import wandb

import training_conchv2 as base
from paths import IMAGES_DIR, MASKS_DIR, PARTITION_DIR, default_checkpoint_dir

cv2.setNumThreads(0)

WANDB_PROJECT_DEFAULT = "SICAPv2_CONCH_maskLUT_denoised"
OUTPUT_DIR = default_checkpoint_dir("checkpoints_conch_masklut_denoised")

DEFAULT_CONFIG = dict(base.DEFAULT_CONFIG)
DEFAULT_CONFIG.update(
    {
        "weight_decay": 2e-4,
        "decoder_dropout": 0.10,
        "label_smoothing": 0.03,
        "mask_clean_min_area": 16,
        "mask_clean_classes": (3, 2, 1),
        "output_dir": OUTPUT_DIR,
    }
)


def parse_clean_classes(text: str | None) -> tuple[int, ...]:
    if text is None:
        return tuple(DEFAULT_CONFIG["mask_clean_classes"])
    text = text.strip()
    if not text:
        return ()
    values = []
    for token in text.split(","):
        value = int(token.strip())
        if value < 0 or value >= base.NUM_CLASSES:
            raise argparse.ArgumentTypeError(
                f"Invalid class id {value}; expected values in [0,{base.NUM_CLASSES - 1}]"
            )
        values.append(value)
    # Descending order works well for tumor classes: GG5 -> GG4 -> GG3.
    return tuple(sorted(set(values), reverse=True))


def clean_small_components(
    mask: np.ndarray,
    *,
    min_area: int = 0,
    classes: tuple[int, ...] = (),
) -> np.ndarray:
    """Reassign tiny connected components to the local majority neighboring class."""
    if min_area <= 0 or not classes:
        return mask

    cleaned = mask.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)

    for cls in classes:
        binary = (cleaned == cls).astype(np.uint8)
        if binary.max() == 0:
            continue

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for label_id in range(1, n_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area >= min_area:
                continue

            component = labels == label_id
            border = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool)
            border &= ~component

            neighbors = cleaned[border]
            neighbors = neighbors[neighbors != cls]
            if neighbors.size == 0:
                replacement = 0
            else:
                counts = np.bincount(neighbors.astype(np.int64), minlength=base.NUM_CLASSES)
                counts[cls] = 0
                replacement = int(np.argmax(counts))

            cleaned[component] = replacement

    return cleaned


def map_and_clean_mask(
    raw_mask: np.ndarray,
    *,
    min_area: int = 0,
    classes: tuple[int, ...] = (),
) -> np.ndarray:
    mapped = base._MASK_LUT[raw_mask].astype(np.int64)
    return clean_small_components(mapped, min_area=min_area, classes=classes)


def compute_sample_weights(
    image_names: list[str],
    masks_dir: Path,
    weight_gg5: float,
    weight_gg4: float,
    weight_gg3: float,
    *,
    mask_clean_min_area: int = 0,
    mask_clean_classes: tuple[int, ...] = (),
) -> list[float]:
    weights: list[float] = []
    for name in image_names:
        mask_path = masks_dir / name
        weight = 1.0
        if mask_path.exists():
            buf_m = np.fromfile(str(mask_path), dtype=np.uint8)
            mask = cv2.imdecode(buf_m, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mapped = map_and_clean_mask(
                    mask,
                    min_area=mask_clean_min_area,
                    classes=mask_clean_classes,
                )
                if np.any(mapped == 3):
                    weight = weight_gg5
                elif np.any(mapped == 2):
                    weight = weight_gg4
                elif np.any(mapped == 1):
                    weight = weight_gg3
        weights.append(weight)
    return weights


class SICAPv2DenoisedDataset(Dataset):
    def __init__(
        self,
        image_names: list[str],
        images_dir: Path,
        masks_dir: Path,
        *,
        transform=None,
        mask_clean_min_area: int = 0,
        mask_clean_classes: tuple[int, ...] = (),
    ):
        self.image_names = image_names
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_clean_min_area = int(mask_clean_min_area)
        self.mask_clean_classes = tuple(mask_clean_classes)

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
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
            mask_raw = cv2.imdecode(buf_m, cv2.IMREAD_GRAYSCALE)
        else:
            mask_raw = None
        if mask_raw is None:
            mask_raw = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        mask = map_and_clean_mask(
            mask_raw,
            min_area=self.mask_clean_min_area,
            classes=self.mask_clean_classes,
        )

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask


def get_fold_dataloaders(fold_name: str, config: dict):
    fold_dir = PARTITION_DIR / "Validation" / fold_name
    train_df = pd.read_excel(fold_dir / "Train.xlsx")
    val_df = pd.read_excel(fold_dir / "Test.xlsx")
    train_names = train_df["image_name"].tolist()
    val_names = val_df["image_name"].tolist()

    clean_area = int(config.get("mask_clean_min_area", 0))
    clean_classes = tuple(config.get("mask_clean_classes", ()))
    print(
        f"  [Mask cleanup] min_area={clean_area} | classes={list(clean_classes) if clean_classes else 'off'}"
    )

    train_ds = SICAPv2DenoisedDataset(
        train_names,
        IMAGES_DIR,
        MASKS_DIR,
        transform=base.get_train_transforms(),
        mask_clean_min_area=clean_area,
        mask_clean_classes=clean_classes,
    )
    val_ds = SICAPv2DenoisedDataset(
        val_names,
        IMAGES_DIR,
        MASKS_DIR,
        transform=base.get_val_transforms(),
        mask_clean_min_area=clean_area,
        mask_clean_classes=clean_classes,
    )

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", config["num_workers"]))
    kwargs = {"persistent_workers": True, "prefetch_factor": 4} if workers > 0 else {}

    seed = int(config.get("seed", base.DEFAULT_SEED))
    gen = torch.Generator()
    gen.manual_seed(seed)
    winit = partial(base._worker_init_fn, base_seed=seed) if workers > 0 else None
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
        sample_weights = compute_sample_weights(
            train_names,
            MASKS_DIR,
            w5,
            w4,
            w3,
            mask_clean_min_area=clean_area,
            mask_clean_classes=clean_classes,
        )
        n5 = sum(1 for x in sample_weights if x == w5)
        n4 = sum(1 for x in sample_weights if x == w4)
        n3 = sum(1 for x in sample_weights if x == w3)
        print(
            f"  [Sampler] train={len(train_names)} | GG5×{w5}={n5} | GG4×{w4}={n4} | GG3×{w3}={n3}"
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_names),
            replacement=True,
        )
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


class FPNDecoder(nn.Module):
    def __init__(self, in_channels=768, fpn_channels=256, num_classes=4, decoder_dropout=0.0):
        super().__init__()
        self.lat = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, fpn_channels, 1, bias=False),
                    nn.BatchNorm2d(fpn_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(4)
            ]
        )
        self.merge = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(fpn_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(3)
            ]
        )
        self.decoder_dropout = float(decoder_dropout)
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
        if self.decoder_dropout > 0:
            x = F.dropout2d(x, p=self.decoder_dropout, training=self.training)
        return self.head(x)


class CONCHSegModel(nn.Module):
    def __init__(
        self,
        *,
        fpn_channels=256,
        num_classes=4,
        unfreeze_last=0,
        decoder_dropout=0.0,
        weights_path=None,
        hf_token=None,
    ):
        super().__init__()
        self.encoder = base.ConcHEncoder(
            unfreeze_last=unfreeze_last,
            weights_path=weights_path,
            hf_token=hf_token,
        )
        self.decoder = FPNDecoder(
            in_channels=self.encoder.embed_dim,
            fpn_channels=fpn_channels,
            num_classes=num_classes,
            decoder_dropout=decoder_dropout,
        )

    def forward(self, x):
        target_size = x.shape[-2:]
        features = self.encoder(x)
        return self.decoder(features, target_size)


def build_model(config: dict):
    return CONCHSegModel(
        fpn_channels=config["fpn_channels"],
        num_classes=config["num_classes"],
        unfreeze_last=config["unfreeze_last"],
        decoder_dropout=float(config.get("decoder_dropout", 0.0)),
        weights_path=config.get("conch_checkpoint"),
        hf_token=config.get("conch_hf_token"),
    )


class GuidedLoss(nn.Module):
    def __init__(
        self,
        class_weights: list[float],
        dice_weight=0.5,
        ce_weight=0.5,
        smooth=1e-6,
        label_smoothing=0.0,
    ):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.dice_loss = smp.losses.DiceLoss(
            mode="multiclass",
            classes=[0, 1, 2, 3],
            smooth=smooth,
        )
        self.register_buffer("weights_tensor", torch.tensor(class_weights).float())
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.weights_tensor,
            label_smoothing=float(label_smoothing),
        )

    def forward(self, logits, targets):
        logits = logits.float()
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.dice_weight * dice + self.ce_weight * ce


def train_fold(
    fold_name: str,
    config: dict,
    device: torch.device,
    dry_run: bool = False,
    use_wandb: bool = True,
):
    print(f"\n{'=' * 60}\n  FOLD: {fold_name}\n{'=' * 60}")
    train_loader, val_loader = get_fold_dataloaders(fold_name, config)

    model = build_model(config).to(device)

    use_compile = config.get("use_compile")
    if use_compile is None:
        import platform

        use_compile = platform.system() != "Windows"
    if use_compile and int(torch.__version__.split(".")[0]) >= 2:
        try:
            print("  ⏳ Compiling model with torch.compile...")
            import platform

            backend = "inductor" if platform.system() != "Windows" else "aot_eager"
            model = torch.compile(model, backend=backend)
            print("  ✅ torch.compile enabled.")
        except Exception as exc:
            print(f"  ⚠️ torch.compile unavailable: {exc}")
    elif not use_compile:
        print("  ℹ️ torch.compile disabled (recommended on Windows to save VRAM).")

    criterion = GuidedLoss(
        config["class_weights"],
        config["dice_weight"],
        config["ce_weight"],
        label_smoothing=float(config.get("label_smoothing", 0.0)),
    ).to(device)

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = list(model.decoder.parameters())
    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": config["learning_rate"] / 10, "label": "encoder"})
    param_groups.append({"params": decoder_params, "lr": config["learning_rate"], "label": "decoder"})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config["weight_decay"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=int(config.get("lr_plateau_patience", 3)),
        min_lr=1e-7,
    )
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_macro_f1, patience_counter, best_cm = 0.0, 0, None
    out_dir = Path(config.get("output_dir", OUTPUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)
    max_epochs = 1 if dry_run else int(config["max_epochs"])

    ga = int(config.get("grad_accum_steps", 1))
    for epoch in range(1, max_epochs + 1):
        lr_dec = optimizer.param_groups[-1]["lr"]
        print(f"\n  Epoch {epoch}/{max_epochs}  (decoder lr={lr_dec:.2e}, accum={ga})")
        train_loss = base.train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            grad_accum_steps=ga,
        )
        val_loss, val_metrics = base.validate_one_epoch(model, val_loader, criterion, device)

        macro_f1 = val_metrics["macro_f1"]
        scheduler.step(macro_f1)
        ratio = val_loss / (train_loss + 1e-8)
        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss:   {val_loss:.4f}  |  Val/Train: {ratio:.3f}")
        print(f"  Macro F1:   {macro_f1:.4f}")
        for i, name in enumerate(base.CLASS_NAMES):
            print(f"    {name} F1: {val_metrics['f1_per_class'][i]:.4f}")

        if use_wandb and wandb.run is not None:
            if len(optimizer.param_groups) == 2:
                enc_lr = optimizer.param_groups[0]["lr"]
                dec_lr = optimizer.param_groups[1]["lr"]
            else:
                enc_lr = 0.0
                dec_lr = optimizer.param_groups[0]["lr"]
            metrics_dict = {
                f"{fold_name}/train_loss": train_loss,
                f"{fold_name}/val_loss": val_loss,
                f"{fold_name}/val_train_loss_ratio": ratio,
                f"{fold_name}/macro_f1": macro_f1,
                "epoch": epoch,
                f"{fold_name}/lr_encoder": enc_lr,
                f"{fold_name}/lr_decoder": dec_lr,
            }
            for i, name in enumerate(base.CLASS_NAMES):
                metrics_dict[f"{fold_name}/f1_{name}"] = float(val_metrics["f1_per_class"][i])
            wandb.log(metrics_dict)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_cm = val_metrics["confusion_matrix"]
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / f"best_{fold_name}_{macro_f1:.4f}.pth")
            print(f"  ✓ Model saved (Macro F1={macro_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= int(config["patience"]):
                print(f"  ⛔ Early stopping triggered at epoch {epoch}")
                break

        if dry_run:
            break

    return {"fold": fold_name, "best_macro_f1": best_macro_f1, "best_cm": best_cm}


def main():
    parser = argparse.ArgumentParser(description="CONCH v2 + denoised masks + light regularization")
    parser.add_argument("--dry-run", action="store_true", help="Smoke test: 1 batch per fold")
    parser.add_argument("--unfreeze-last", type=int, default=0, help="Unfreeze the last N CONCH ViT-B blocks")
    parser.add_argument("--weights", type=str, default=None, metavar="PATH")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None, metavar="K")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--sampler-gg5", type=float, default=None, metavar="W")
    parser.add_argument("--sampler-gg4", type=float, default=None, metavar="W")
    parser.add_argument("--sampler-gg3", type=float, default=None, metavar="W")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT_DEFAULT)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument(
        "--fold",
        type=str,
        nargs="+",
        default=None,
        choices=["Val1", "Val2", "Val3", "Val4"],
        help="Run one or multiple specific folds. If not set, all 4 folds run.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=base.DEFAULT_SEED)

    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr-plateau-patience", type=int, default=None)
    parser.add_argument("--decoder-dropout", type=float, default=None, metavar="P")
    parser.add_argument("--label-smoothing", type=float, default=None, metavar="EPS")
    parser.add_argument("--mask-clean-min-area", type=int, default=None, metavar="N")
    parser.add_argument(
        "--mask-clean-classes",
        type=parse_clean_classes,
        default=None,
        metavar="IDS",
        help="Comma-separated class ids to clean, e.g. 3 or 3,2,1. Empty string disables cleanup.",
    )
    args = parser.parse_args()

    base.set_seed(args.seed)
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

    config = dict(DEFAULT_CONFIG)
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
    if args.sampler_gg5 is not None:
        config["sampler_weight_gg5"] = args.sampler_gg5
    if args.sampler_gg4 is not None:
        config["sampler_weight_gg4"] = args.sampler_gg4
    if args.sampler_gg3 is not None:
        config["sampler_weight_gg3"] = args.sampler_gg3
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        config["weight_decay"] = args.weight_decay
    if args.patience is not None:
        config["patience"] = args.patience
    if args.lr_plateau_patience is not None:
        config["lr_plateau_patience"] = args.lr_plateau_patience
    if args.decoder_dropout is not None:
        config["decoder_dropout"] = args.decoder_dropout
    if args.label_smoothing is not None:
        config["label_smoothing"] = args.label_smoothing
    if args.mask_clean_min_area is not None:
        config["mask_clean_min_area"] = args.mask_clean_min_area
    if args.mask_clean_classes is not None:
        config["mask_clean_classes"] = tuple(args.mask_clean_classes)
    if args.compile and args.no_compile:
        print("  [WARN] --compile and --no-compile set together; using --no-compile.")
    if args.no_compile:
        config["use_compile"] = False
    elif args.compile:
        config["use_compile"] = True

    config["output_dir"] = Path(args.output_dir).resolve() if args.output_dir else OUTPUT_DIR
    print(f"Checkpoints -> {config['output_dir']}")

    eff = config["batch_size"] * config.get("grad_accum_steps", 1)
    print(
        f"CONCH denoised: micro_batch={config['batch_size']} × accum={config.get('grad_accum_steps', 1)} "
        f"≈ {eff} | weighted_sampler={config.get('use_weighted_sampler')} | lr={config['learning_rate']}"
    )
    print(
        f"  wd={config['weight_decay']} | decoder_dropout={config.get('decoder_dropout', 0.0)} | "
        f"label_smoothing={config.get('label_smoothing', 0.0)} | "
        f"mask_clean_min_area={config.get('mask_clean_min_area', 0)} | "
        f"mask_clean_classes={list(config.get('mask_clean_classes', ()))}"
    )

    use_wandb = not args.no_wandb
    if use_wandb and not args.dry_run:
        try:
            wandb_config = {
                "script": "training_conchv2_denoised",
                "img_size": base.IMG_SIZE,
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
                "sampler_weight_gg5": config.get("sampler_weight_gg5"),
                "sampler_weight_gg4": config.get("sampler_weight_gg4"),
                "sampler_weight_gg3": config.get("sampler_weight_gg3"),
                "decoder_dropout": config.get("decoder_dropout", 0.0),
                "label_smoothing": config.get("label_smoothing", 0.0),
                "mask_clean_min_area": config.get("mask_clean_min_area", 0),
                "mask_clean_classes": list(config.get("mask_clean_classes", ())),
                "conch_checkpoint": config.get("conch_checkpoint") or base.CONCH_HF_CHECKPOINT,
                "output_dir": str(config["output_dir"]),
                "mask_lut": "25:75->GG3, 75:175->GG4, 175:255->GG5, else->NC",
                "pixel_frac_partition": [float(x) for x in base._PIXEL_FRAC_PARTITION],
                "lr_plateau_patience": config.get("lr_plateau_patience", 3),
                "seed": config.get("seed", base.DEFAULT_SEED),
                "fold": args.fold,
            }
            run_name = args.wandb_name or (
                f"CONCH_denoised_bs{config['batch_size']}_ga{config.get('grad_accum_steps', 1)}"
            )
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=wandb_config,
                tags=["mask_lut", "CONCH", "SICAPv2", "denoised"],
            )
        except Exception as exc:
            print(f"  [WARN] Weights & Biases unavailable: {exc}")
            use_wandb = False

    fold_names = args.fold if args.fold else ["Val1", "Val2", "Val3", "Val4"]
    aggregated_cm = np.zeros((base.NUM_CLASSES, base.NUM_CLASSES), dtype=np.int64)

    for fold in fold_names:
        res = train_fold(fold, config, device, args.dry_run, use_wandb=use_wandb)
        if res["best_cm"] is not None:
            aggregated_cm += res["best_cm"]

    base.print_aggregated_matrices(aggregated_cm)
    if use_wandb and wandb.run is not None:
        agg_log = {"aggregated/macro_f1": base._aggregated_macro_f1_from_cm(aggregated_cm)}
        for c, name in enumerate(base.CLASS_NAMES):
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
