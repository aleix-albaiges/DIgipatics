"""
SICAPv2 CONCH segmentation with explicit GG4C as a fifth pixel class.

This is an experimental 5-class continuation of ``training_conchv2.py``.
It keeps the same CONCH + FPN architecture but changes:

    class order: NC, GG3, GG4, GG4C, GG5
    mask LUT   : 25:75 -> GG3
                 75:125 -> GG4
                 125:175 -> GG4C
                 175:255 -> GG5

The 4-class checkpoint is expanded into 5 classes:
    old NC  -> new NC
    old GG3 -> new GG3
    old GG4 -> new GG4
    old GG4 -> new GG4C initialization
    old GG5 -> new GG5

Default base checkpoint:
    artifacts/checkpoints_conch_masklut/best_Val3_0.8201.pth
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
import segmentation_models_pytorch as smp

from paths import IMAGES_DIR, MASKS_DIR, PARTITION_DIR, default_checkpoint_dir
from training_conchv2 import (
    CONCH_NORM_MEAN,
    CONCH_NORM_STD,
    CONCHSegModel,
    get_train_transforms,
    get_val_transforms,
    build_cosine_warmup_scheduler,
)


DEFAULT_SEED = 42
NUM_CLASSES = 5
CLASS_NAMES = ["NC", "GG3", "GG4", "GG4C", "GG5"]
MERGED_CLASS_NAMES = ["NC", "GG3", "GG4_merged", "GG5"]
IMG_SIZE = 512
DEFAULT_BASE_CHECKPOINT = Path("artifacts/checkpoints_conch_masklut/best_Val3_0.8201.pth")
OUTPUT_DIR = default_checkpoint_dir("checkpoints_conch_masklut_g4c_5class")


def set_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int, base_seed: int) -> None:
    seed = int((base_seed + worker_id) % (2**32))
    random.seed(seed)
    np.random.seed(seed)


def build_mask_lut(g4c_gray_min: int) -> np.ndarray:
    if not 76 <= int(g4c_gray_min) <= 174:
        raise ValueError("--g4c-gray-min must be inside the current GG4 range [76, 174].")
    lut = np.zeros(256, dtype=np.int64)
    lut[25:75] = 1
    lut[75:int(g4c_gray_min)] = 2
    lut[int(g4c_gray_min):175] = 3
    lut[175:] = 4
    return lut


def read_split(split_path: Path) -> list[str]:
    df = pd.read_excel(split_path)
    if "image_name" not in df.columns:
        raise KeyError(f"{split_path} does not contain image_name.")
    return df["image_name"].dropna().astype(str).tolist()


def split_paths(fold: str) -> tuple[Path, Path]:
    if fold == "Test":
        return PARTITION_DIR / "Test" / "Train.xlsx", PARTITION_DIR / "Test" / "Test.xlsx"
    return PARTITION_DIR / "Validation" / fold / "Train.xlsx", PARTITION_DIR / "Validation" / fold / "Test.xlsx"


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def collect_final_names(folds: list[str]) -> tuple[list[str], dict]:
    all_names = []
    sources = {}
    for fold in folds:
        train_path, val_path = split_paths(fold)
        for path in (train_path, val_path):
            names = read_split(path)
            sources[str(path)] = len(names)
            all_names.extend(names)
    unique = dedupe_preserve_order(all_names)
    return unique, {
        "sources": sources,
        "rows_before_dedup": len(all_names),
        "unique_images": len(unique),
        "duplicates_removed": len(all_names) - len(unique),
    }


class SICAPv2G4C5Dataset(Dataset):
    def __init__(self, image_names: list[str], mask_lut: np.ndarray, transform=None):
        self.image_names = list(image_names)
        self.mask_lut = mask_lut
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        name = self.image_names[idx]
        image_path = IMAGES_DIR / name
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = MASKS_DIR / name
        if mask_path.exists():
            mask_raw = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        else:
            mask_raw = None
        if mask_raw is None:
            mask_raw = np.zeros(image.shape[:2], dtype=np.uint8)
        mask = self.mask_lut[mask_raw].astype(np.int64)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask.long()


def mask_presence(mask_path: Path, mask_lut: np.ndarray) -> np.ndarray:
    if not mask_path.exists():
        return np.zeros(NUM_CLASSES, dtype=bool)
    raw = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return np.zeros(NUM_CLASSES, dtype=bool)
    mapped = mask_lut[raw]
    present = np.zeros(NUM_CLASSES, dtype=bool)
    present[np.unique(mapped)] = True
    return present


def compute_sample_weights(
    image_names: list[str],
    mask_lut: np.ndarray,
    weight_gg5: float,
    weight_gg4c: float,
    weight_gg4: float,
    weight_gg3: float,
) -> list[float]:
    weights = []
    counts = {name: 0 for name in CLASS_NAMES}
    for name in image_names:
        present = mask_presence(MASKS_DIR / name, mask_lut)
        for idx, cname in enumerate(CLASS_NAMES):
            if present[idx]:
                counts[cname] += 1
        if present[4]:
            weights.append(weight_gg5)
        elif present[3]:
            weights.append(weight_gg4c)
        elif present[2]:
            weights.append(weight_gg4)
        elif present[1]:
            weights.append(weight_gg3)
        else:
            weights.append(1.0)
    print(
        "  [Sampler presence] "
        + " | ".join(f"{name}={counts[name]}" for name in CLASS_NAMES)
    )
    return weights


def compute_pixel_frequencies(image_names: list[str], mask_lut: np.ndarray) -> np.ndarray:
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for name in tqdm(image_names, desc="  Pixel freq", leave=False):
        mask_path = MASKS_DIR / name
        if mask_path.exists():
            raw = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        else:
            raw = None
        if raw is None:
            mapped = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int64)
        else:
            mapped = mask_lut[raw]
        counts += np.bincount(mapped.reshape(-1), minlength=NUM_CLASSES)[:NUM_CLASSES]
    total = max(int(counts.sum()), 1)
    return counts.astype(np.float64) / float(total)


def sqrt_inv_weights(freqs: np.ndarray) -> list[float]:
    freqs = np.maximum(np.asarray(freqs, dtype=np.float64), 1e-12)
    weights = np.sqrt(1.0 / freqs)
    weights = weights / weights[0]
    return [float(round(x, 3)) for x in weights]


def make_loader(names: list[str], mask_lut: np.ndarray, config: dict, train: bool):
    transform = (
        get_train_transforms(
            norm_mean=config["norm_mean"],
            norm_std=config["norm_std"],
            color_aug_enabled=config["color_aug_enabled"],
        )
        if train
        else get_val_transforms(norm_mean=config["norm_mean"], norm_std=config["norm_std"])
    )
    dataset = SICAPv2G4C5Dataset(names, mask_lut=mask_lut, transform=transform)
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", config["num_workers"]))
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    common = dict(
        batch_size=config["batch_size"],
        num_workers=workers,
        pin_memory=True,
        generator=generator,
        worker_init_fn=partial(_worker_init_fn, base_seed=int(config["seed"])) if workers > 0 else None,
    )
    if workers > 0:
        common.update(persistent_workers=True, prefetch_factor=4)
    if train and config["use_weighted_sampler"]:
        weights = compute_sample_weights(
            names,
            mask_lut,
            weight_gg5=config["sampler_weight_gg5"],
            weight_gg4c=config["sampler_weight_gg4c"],
            weight_gg4=config["sampler_weight_gg4"],
            weight_gg3=config["sampler_weight_gg3"],
        )
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=config["sampler_replacement"])
        return DataLoader(dataset, sampler=sampler, drop_last=True, **common)
    return DataLoader(dataset, shuffle=train, drop_last=train, **common)


def clean_state_dict(raw) -> dict:
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break
    out = {}
    for key, value in raw.items():
        name = key
        for prefix in ("module.", "_orig_mod."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        out[name] = value
    return out


def expand_4class_tensor(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    expanded = target.clone()
    expanded[0].copy_(src[0])
    expanded[1].copy_(src[1])
    expanded[2].copy_(src[2])
    expanded[3].copy_(src[2])
    expanded[4].copy_(src[3])
    return expanded


def load_expanded_4class_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    raw = torch.load(checkpoint_path, map_location="cpu")
    source = clean_state_dict(raw)
    target = model.state_dict()
    adapted = {}
    expanded_keys = []
    skipped = []
    for key, target_tensor in target.items():
        if key not in source:
            skipped.append(key)
            adapted[key] = target_tensor
            continue
        src = source[key]
        if tuple(src.shape) == tuple(target_tensor.shape):
            adapted[key] = src
        elif src.ndim >= 1 and target_tensor.ndim >= 1 and src.shape[0] == 4 and target_tensor.shape[0] == 5:
            adapted[key] = expand_4class_tensor(src, target_tensor)
            expanded_keys.append(key)
        else:
            skipped.append(key)
            adapted[key] = target_tensor
    model.load_state_dict(adapted, strict=True)
    print(f"  [Init] Loaded 4-class checkpoint: {checkpoint_path}")
    print(f"  [Init] Expanded keys: {expanded_keys}")
    if skipped:
        print(f"  [Init] Kept initialized keys: {len(skipped)}; first={skipped[:5]}")


class GuidedLoss5(nn.Module):
    def __init__(self, class_weights: list[float], dice_weight: float, ce_weight: float):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", classes=list(range(NUM_CLASSES)), smooth=1e-6)
        self.register_buffer("weights", torch.tensor(class_weights, dtype=torch.float32))
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weights)

    def forward(self, logits, targets):
        logits = logits.float()
        targets = targets.long()
        return self.dice_weight * self.dice_loss(logits, targets) + self.ce_weight * self.ce_loss(logits, targets)


class SegmentationMetrics5:
    def __init__(self):
        self.confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        tgts = targets.detach().cpu().numpy()
        mask = (tgts >= 0) & (tgts < NUM_CLASSES)
        np.add.at(self.confusion_matrix, (tgts[mask], preds[mask]), 1)

    def compute(self) -> dict:
        cm = self.confusion_matrix
        merged_cm = merge_g4_g4c_confusion(cm)
        f1 = np.zeros(NUM_CLASSES, dtype=np.float64)
        precision = np.zeros(NUM_CLASSES, dtype=np.float64)
        recall = np.zeros(NUM_CLASSES, dtype=np.float64)
        for c in range(NUM_CLASSES):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            precision[c] = tp / max(tp + fp, 1)
            recall[c] = tp / max(tp + fn, 1)
            f1[c] = (2.0 * tp) / max((2.0 * tp + fp + fn), 1)
        merged_stats = per_class_stats(merged_cm)
        return {
            "macro_f1": float(np.mean(f1)),
            "f1_per_class": f1,
            "precision_per_class": precision,
            "recall_per_class": recall,
            "confusion_matrix": cm.copy(),
            "merged_macro_f1": float(np.mean(merged_stats["f1_per_class"])),
            "merged_f1_per_class": merged_stats["f1_per_class"],
            "merged_precision_per_class": merged_stats["precision_per_class"],
            "merged_recall_per_class": merged_stats["recall_per_class"],
            "merged_confusion_matrix": merged_cm,
        }


def per_class_stats(cm: np.ndarray) -> dict:
    num_classes = int(cm.shape[0])
    f1 = np.zeros(num_classes, dtype=np.float64)
    precision = np.zeros(num_classes, dtype=np.float64)
    recall = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision[c] = tp / max(tp + fp, 1)
        recall[c] = tp / max(tp + fn, 1)
        f1[c] = (2.0 * tp) / max((2.0 * tp + fp + fn), 1)
    return {
        "f1_per_class": f1,
        "precision_per_class": precision,
        "recall_per_class": recall,
    }


def merge_g4_g4c_confusion(cm5: np.ndarray) -> np.ndarray:
    merged = np.zeros((4, 4), dtype=np.int64)
    # 5-class indices: 0=NC, 1=GG3, 2=GG4, 3=GG4C, 4=GG5
    # 4-class merged indices: 0=NC, 1=GG3, 2=GG4/GG4C, 3=GG5
    index_map = np.array([0, 1, 2, 2, 3], dtype=np.int64)
    for true_idx in range(NUM_CLASSES):
        for pred_idx in range(NUM_CLASSES):
            merged[index_map[true_idx], index_map[pred_idx]] += int(cm5[true_idx, pred_idx])
    return merged


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for p_ema, p in zip(self.ema.parameters(), model.parameters()):
            p_ema.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
        for b_ema, b in zip(self.ema.buffers(), model.buffers()):
            b_ema.copy_(b)


def trainable_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_accum_steps, scheduler=None, ema=None, ema_source=None):
    model.train()
    total_loss = 0.0
    batches = 0
    accum = 0
    optimizer.zero_grad(set_to_none=True)
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            loss = criterion(model(images), masks)
        scaler.scale(loss / grad_accum_steps).backward()
        total_loss += float(loss.item())
        batches += 1
        accum += 1
        if accum >= grad_accum_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            if ema is not None and ema_source is not None:
                ema.update(ema_source)
            accum = 0
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    if accum > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        if ema is not None and ema_source is not None:
            ema.update(ema_source)
    return total_loss / max(batches, 1)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    batches = 0
    metrics = SegmentationMetrics5()
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits, masks)
        total_loss += float(loss.item())
        batches += 1
        metrics.update(logits, masks)
    return total_loss / max(batches, 1), metrics.compute()


def build_optimizer(model: nn.Module, config: dict):
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = list(model.decoder.parameters())
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": config["learning_rate"] / 10, "label": "encoder"})
    groups.append({"params": decoder_params, "lr": config["learning_rate"], "label": "decoder"})
    return torch.optim.AdamW(groups, weight_decay=config["weight_decay"])


def save_checkpoint(path: Path, model: nn.Module, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["model_state_dict"] = trainable_model(model).state_dict()
    torch.save(data, path)
    meta = {k: v for k, v in payload.items() if k != "confusion_matrix"}
    meta_path = path.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def print_metrics(metrics: dict) -> None:
    print(f"  Macro F1 (5-class): {metrics['macro_f1']:.4f}")
    for idx, name in enumerate(CLASS_NAMES):
        print(
            f"    {name:4s} F1={metrics['f1_per_class'][idx]:.4f} "
            f"P={metrics['precision_per_class'][idx]:.4f} "
            f"R={metrics['recall_per_class'][idx]:.4f}"
        )
    print(f"  Macro F1 (4-class merged GG4+GG4C): {metrics['merged_macro_f1']:.4f}")
    for idx, name in enumerate(MERGED_CLASS_NAMES):
        print(
            f"    {name:10s} F1={metrics['merged_f1_per_class'][idx]:.4f} "
            f"P={metrics['merged_precision_per_class'][idx]:.4f} "
            f"R={metrics['merged_recall_per_class'][idx]:.4f}"
        )


def selection_score(metrics: dict, metric_name: str) -> float:
    if metric_name == "macro_f1":
        return float(metrics["macro_f1"])
    if metric_name == "merged_macro_f1":
        return float(metrics["merged_macro_f1"])
    if metric_name == "g4c_f1":
        return float(metrics["f1_per_class"][3])
    if metric_name == "merged_plus_g4c":
        return 0.5 * float(metrics["merged_macro_f1"]) + 0.5 * float(metrics["f1_per_class"][3])
    raise ValueError(f"Unknown selection metric: {metric_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CONCH 5-class segmentation with explicit GG4C.")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--fold", choices=["Val1", "Val2", "Val3", "Val4", "Test"], default="Val3")
    parser.add_argument(
        "--train-xlsx",
        type=Path,
        default=None,
        help="Training split Excel to use directly. Overrides fold/final-fold collection and uses no validation set.",
    )
    parser.add_argument("--final-train", action="store_true")
    parser.add_argument("--final-folds", nargs="+", choices=["Val1", "Val2", "Val3", "Val4", "Test"], default=None)
    parser.add_argument("--g4c-gray-min", type=int, default=125, help="Split inside old GG4 range: [75, min)->GG4, [min,175)->GG4C.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument(
        "--selection-metric",
        choices=["macro_f1", "merged_macro_f1", "g4c_f1", "merged_plus_g4c"],
        default="merged_plus_g4c",
        help=(
            "Metric used to select best checkpoint. Default balances preserving 4-class "
            "merged performance with learning GG4C."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--fpn-channels", type=int, default=256)
    parser.add_argument("--unfreeze-last", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--class-weights", type=str, default=None, help="Comma-separated weights for NC,GG3,GG4,GG4C,GG5.")
    parser.add_argument("--no-auto-class-weights", action="store_true")
    parser.add_argument("--dice-weight", type=float, default=0.55)
    parser.add_argument("--ce-weight", type=float, default=0.45)
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--sampler-gg5", type=float, default=1.8)
    parser.add_argument("--sampler-gg4c", type=float, default=2.2)
    parser.add_argument("--sampler-gg4", type=float, default=1.3)
    parser.add_argument("--sampler-gg3", type=float, default=1.8)
    parser.add_argument("--sampler-replacement", action="store_true")
    parser.add_argument("--no-color-aug", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--conch-checkpoint", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-summary-only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"5-class LUT: 25:75->GG3, 75:{args.g4c_gray_min}->GG4, {args.g4c_gray_min}:175->GG4C, 175:255->GG5")
    mask_lut = build_mask_lut(args.g4c_gray_min)

    if args.train_xlsx is not None:
        train_names = read_split(args.train_xlsx)
        val_names = []
        data_summary = {
            "training_source": str(args.train_xlsx),
            "train_rows": len(train_names),
            "validation": None,
            "deduplicated": False,
        }
    elif args.final_train:
        folds = args.final_folds or ["Val1", "Val2", "Val3", "Val4"]
        train_names, data_summary = collect_final_names(folds)
        data_summary["training_source"] = "final_folds"
        val_names = []
    else:
        train_path, val_path = split_paths(args.fold)
        train_names = read_split(train_path)
        val_names = read_split(val_path)
        data_summary = {"train": str(train_path), "val": str(val_path), "train_rows": len(train_names), "val_rows": len(val_names)}
    print(f"Data summary: {json.dumps(data_summary, indent=2)}")

    if args.class_weights is not None:
        class_weights = [float(x.strip()) for x in args.class_weights.split(",") if x.strip()]
        if len(class_weights) != NUM_CLASSES:
            parser.error("--class-weights must contain exactly 5 values.")
    elif args.no_auto_class_weights:
        class_weights = [1.0, 3.586, 3.0, 3.0, 4.21]
    else:
        freqs = compute_pixel_frequencies(train_names, mask_lut)
        class_weights = sqrt_inv_weights(freqs)
        print("Pixel frequencies: " + ", ".join(f"{n}={freqs[i]:.6f}" for i, n in enumerate(CLASS_NAMES)))
    print(f"Class weights: {class_weights}")

    if args.data_summary_only:
        return

    config = {
        "batch_size": args.batch_size,
        "grad_accum_steps": max(1, args.grad_accum),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "norm_mean": list(CONCH_NORM_MEAN),
        "norm_std": list(CONCH_NORM_STD),
        "color_aug_enabled": not args.no_color_aug,
        "use_weighted_sampler": not args.no_weighted_sampler,
        "sampler_weight_gg5": args.sampler_gg5,
        "sampler_weight_gg4c": args.sampler_gg4c,
        "sampler_weight_gg4": args.sampler_gg4,
        "sampler_weight_gg3": args.sampler_gg3,
        "sampler_replacement": args.sampler_replacement,
    }

    train_loader = make_loader(train_names, mask_lut, config, train=True)
    val_loader = make_loader(val_names, mask_lut, config, train=False) if val_names else None

    model = CONCHSegModel(
        fpn_channels=args.fpn_channels,
        num_classes=NUM_CLASSES,
        unfreeze_last=args.unfreeze_last,
        weights_path=args.conch_checkpoint,
        hf_token=args.hf_token,
    )
    load_expanded_4class_checkpoint(model, args.base_checkpoint)
    model = model.to(device)

    ema = ModelEMA(model, args.ema_decay) if args.ema else None
    if ema is not None:
        print(f"EMA enabled: decay={args.ema_decay}")

    if args.compile and args.no_compile:
        print("  [WARN] both --compile and --no-compile passed; using --no-compile.")
    elif args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled.")
        except Exception as exc:
            print(f"  [WARN] torch.compile failed: {exc}")

    criterion = GuidedLoss5(class_weights, args.dice_weight, args.ce_weight).to(device)
    optimizer = build_optimizer(trainable_model(model), {"learning_rate": args.learning_rate, "weight_decay": args.weight_decay})
    steps_per_epoch = max(1, len(train_loader) // max(1, args.grad_accum))
    total_steps = steps_per_epoch * (1 if args.dry_run else args.max_epochs)
    scheduler = build_cosine_warmup_scheduler(optimizer, total_steps=total_steps, warmup_pct=0.07, min_lr_ratio=0.01)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_score = -1.0
    best_path = None
    patience = 0
    max_epochs = 1 if args.dry_run else args.max_epochs
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ema_source = trainable_model(model) if ema is not None else None

    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch {epoch}/{max_epochs} | lr={optimizer.param_groups[-1]['lr']:.2e}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            grad_accum_steps=max(1, args.grad_accum),
            scheduler=scheduler,
            ema=ema,
            ema_source=ema_source,
        )
        print(f"  Train loss: {train_loss:.4f}")

        if val_loader is None:
            continue

        val_model = ema.ema if ema is not None else model
        val_loss, metrics = validate_one_epoch(val_model, val_loader, criterion, device)
        print(f"  Val loss: {val_loss:.4f}")
        print_metrics(metrics)

        score = selection_score(metrics, args.selection_metric)
        print(f"  Selection score ({args.selection_metric}): {score:.4f}")

        if score > best_score:
            best_score = score
            patience = 0
            name = args.checkpoint_name or f"best_{args.fold}_5class_g4c_{args.selection_metric}_{best_score:.4f}.pth"
            best_path = out_dir / name
            save_checkpoint(
                best_path,
                val_model,
                {
                    "epoch": epoch,
                    "fold": args.fold,
                    "base_checkpoint": str(args.base_checkpoint),
                    "g4c_gray_min": int(args.g4c_gray_min),
                    "class_names": CLASS_NAMES,
                    "class_weights": class_weights,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "selection_metric": args.selection_metric,
                    "selection_score": float(best_score),
                    "macro_f1": float(metrics["macro_f1"]),
                    "f1_per_class": [float(x) for x in metrics["f1_per_class"]],
                    "precision_per_class": [float(x) for x in metrics["precision_per_class"]],
                    "recall_per_class": [float(x) for x in metrics["recall_per_class"]],
                    "confusion_matrix": metrics["confusion_matrix"].tolist(),
                    "merged_class_names": MERGED_CLASS_NAMES,
                    "merged_macro_f1": float(metrics["merged_macro_f1"]),
                    "merged_f1_per_class": [float(x) for x in metrics["merged_f1_per_class"]],
                    "merged_precision_per_class": [float(x) for x in metrics["merged_precision_per_class"]],
                    "merged_recall_per_class": [float(x) for x in metrics["merged_recall_per_class"]],
                    "merged_confusion_matrix": metrics["merged_confusion_matrix"].tolist(),
                    "data_summary": data_summary,
                },
            )
            print(f"  Saved: {best_path}")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if val_loader is None:
        name = args.checkpoint_name or "final_5class_g4c.pth"
        final_path = out_dir / name
        final_model = ema.ema if ema is not None else model
        save_checkpoint(
            final_path,
            final_model,
            {
                "epoch": max_epochs,
                "base_checkpoint": str(args.base_checkpoint),
                "g4c_gray_min": int(args.g4c_gray_min),
                "class_names": CLASS_NAMES,
                "merged_class_names": MERGED_CLASS_NAMES,
                "class_weights": class_weights,
                "training_source": data_summary.get("training_source"),
                "trained_epochs": int(max_epochs),
                "final_train_loss": float(train_loss),
                "data_summary": data_summary,
            },
        )
        print(f"\nFinal checkpoint: {final_path}")
    else:
        print(f"\nBest selection score ({args.selection_metric}): {best_score:.4f}")
        print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
