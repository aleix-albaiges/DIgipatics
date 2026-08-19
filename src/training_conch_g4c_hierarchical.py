"""
Train a hierarchical GG4C model:

  shared CONCH encoder
    ├─ grade head: NC, GG3, GG4_merged, GG5
    └─ GG4C head: non-cribriform GG4 vs cribriform GG4C

The GG4C loss is applied only on true GG4/GG4C pixels. Final 5-class
predictions are hierarchical: GG4C can only be emitted inside pixels that the
grade head predicts as GG4_merged.
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
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
import segmentation_models_pytorch as smp

from paths import IMAGES_DIR, MASKS_DIR, PARTITION_DIR, default_checkpoint_dir
from training_conchv2 import (
    CONCH_NORM_MEAN,
    CONCH_NORM_STD,
    ConcHEncoder,
    FPNDecoder,
    build_cosine_warmup_scheduler,
    get_train_transforms,
    get_val_transforms,
)
from training_conchv2_g4c_5class import (
    CLASS_NAMES as CLASS_NAMES_5,
    MERGED_CLASS_NAMES,
    NUM_CLASSES as NUM_CLASSES_5,
    SegmentationMetrics5,
    build_mask_lut,
    clean_state_dict,
)


DEFAULT_SEED = 42
IMG_SIZE = 512
NUM_GRADE_CLASSES = 4
GRADE_CLASS_NAMES = ["NC", "GG3", "GG4_merged", "GG5"]
DEFAULT_BASE_CHECKPOINT = Path("artifacts/checkpoints_conch_masklut/final_all_folds.pth")
OUTPUT_DIR = default_checkpoint_dir("checkpoints_conch_masklut_g4c_hierarchical")


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
        "training_source": "final_folds",
        "sources": sources,
        "rows_before_dedup": len(all_names),
        "unique_images": len(unique),
        "duplicates_removed": len(all_names) - len(unique),
    }


def build_grade_lut(g4c_gray_min: int) -> np.ndarray:
    lut = np.zeros(256, dtype=np.int64)
    lut[25:75] = 1
    lut[75:175] = 2
    lut[175:] = 3
    return lut


class SICAPv2HierG4CDataset(Dataset):
    def __init__(self, image_names: list[str], mask_lut_5: np.ndarray, grade_lut: np.ndarray, transform=None):
        self.image_names = list(image_names)
        self.mask_lut_5 = mask_lut_5
        self.grade_lut = grade_lut
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
            raw = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        else:
            raw = None
        if raw is None:
            raw = np.zeros(image.shape[:2], dtype=np.uint8)

        mask_5 = self.mask_lut_5[raw].astype(np.int64)
        grade_mask = self.grade_lut[raw].astype(np.int64)
        g4c_target = (mask_5 == 3).astype(np.int64)
        g4_region = ((mask_5 == 2) | (mask_5 == 3)).astype(np.int64)

        if self.transform is not None:
            transformed = self.transform(
                image=image,
                mask=grade_mask,
                masks=[g4c_target, g4_region, mask_5],
            )
            image = transformed["image"]
            grade_mask = transformed["mask"]
            g4c_target, g4_region, mask_5 = transformed["masks"]

        return image, grade_mask.long(), g4c_target.long(), g4_region.long(), mask_5.long()


def compute_training_stats(image_names: list[str], mask_lut_5: np.ndarray, grade_lut: np.ndarray) -> dict:
    grade_counts = np.zeros(NUM_GRADE_CLASSES, dtype=np.int64)
    g4c_pos = 0
    g4c_neg = 0
    for name in tqdm(image_names, desc="  Pixel freq", leave=False):
        mask_path = MASKS_DIR / name
        if mask_path.exists():
            raw = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        else:
            raw = None
        if raw is None:
            raw = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        grade = grade_lut[raw]
        mask_5 = mask_lut_5[raw]
        grade_counts += np.bincount(grade.reshape(-1), minlength=NUM_GRADE_CLASSES)[:NUM_GRADE_CLASSES]
        g4c_pos += int((mask_5 == 3).sum())
        g4c_neg += int((mask_5 == 2).sum())

    grade_freq = grade_counts.astype(np.float64) / max(int(grade_counts.sum()), 1)
    grade_weights = sqrt_inv_weights(grade_freq)
    g4c_pos_weight = float(g4c_neg / max(g4c_pos, 1))
    return {
        "grade_counts": grade_counts,
        "grade_freq": grade_freq,
        "grade_weights": grade_weights,
        "g4c_pos_pixels": int(g4c_pos),
        "g4c_neg_pixels": int(g4c_neg),
        "g4c_pos_weight": g4c_pos_weight,
    }


def sqrt_inv_weights(freqs: np.ndarray) -> list[float]:
    freqs = np.maximum(np.asarray(freqs, dtype=np.float64), 1e-12)
    weights = np.sqrt(1.0 / freqs)
    weights = weights / weights[0]
    return [float(round(x, 3)) for x in weights]


def compute_sample_weights(
    image_names: list[str],
    mask_lut_5: np.ndarray,
    weight_gg5: float,
    weight_gg4c: float,
    weight_gg4: float,
    weight_gg3: float,
) -> list[float]:
    weights = []
    counts = {name: 0 for name in CLASS_NAMES_5}
    for name in image_names:
        mask_path = MASKS_DIR / name
        if mask_path.exists():
            raw = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        else:
            raw = None
        present = np.zeros(NUM_CLASSES_5, dtype=bool)
        if raw is not None:
            mapped = mask_lut_5[raw]
            present[np.unique(mapped)] = True
        for idx, cname in enumerate(CLASS_NAMES_5):
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
    print("  [Sampler presence] " + " | ".join(f"{name}={counts[name]}" for name in CLASS_NAMES_5))
    return weights


def make_loader(names: list[str], mask_lut_5: np.ndarray, grade_lut: np.ndarray, config: dict, train: bool):
    transform = (
        get_train_transforms(
            norm_mean=config["norm_mean"],
            norm_std=config["norm_std"],
            color_aug_enabled=config["color_aug_enabled"],
        )
        if train
        else get_val_transforms(norm_mean=config["norm_mean"], norm_std=config["norm_std"])
    )
    dataset = SICAPv2HierG4CDataset(names, mask_lut_5=mask_lut_5, grade_lut=grade_lut, transform=transform)
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
            mask_lut_5,
            weight_gg5=config["sampler_weight_gg5"],
            weight_gg4c=config["sampler_weight_gg4c"],
            weight_gg4=config["sampler_weight_gg4"],
            weight_gg3=config["sampler_weight_gg3"],
        )
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=config["sampler_replacement"])
        return DataLoader(dataset, sampler=sampler, drop_last=True, **common)
    return DataLoader(dataset, shuffle=train, drop_last=train, **common)


class CONCHHierG4CModel(nn.Module):
    def __init__(self, fpn_channels=256, unfreeze_last=0, weights_path=None, hf_token=None):
        super().__init__()
        self.encoder = ConcHEncoder(
            unfreeze_last=unfreeze_last,
            weights_path=weights_path,
            hf_token=hf_token,
        )
        self.grade_decoder = FPNDecoder(
            in_channels=self.encoder.embed_dim,
            fpn_channels=fpn_channels,
            num_classes=NUM_GRADE_CLASSES,
        )
        self.g4c_decoder = FPNDecoder(
            in_channels=self.encoder.embed_dim,
            fpn_channels=fpn_channels,
            num_classes=2,
        )

    def forward(self, x):
        target_size = x.shape[-2:]
        features = self.encoder(x)
        return {
            "grade_logits": self.grade_decoder(features, target_size),
            "g4c_logits": self.g4c_decoder(features, target_size),
        }


def _copy_old_gg4_to_binary_head(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    out = target.clone()
    out[0].copy_(src[2])
    out[1].copy_(src[2])
    return out


def load_4class_checkpoint(model: CONCHHierG4CModel, checkpoint_path: Path) -> None:
    raw = torch.load(checkpoint_path, map_location="cpu")
    source = clean_state_dict(raw)
    target = model.state_dict()
    adapted = {k: v.clone() for k, v in target.items()}
    loaded_grade = []
    loaded_g4c = []
    skipped = []

    for key, src in source.items():
        if key.startswith("encoder.") and key in adapted and tuple(src.shape) == tuple(adapted[key].shape):
            adapted[key] = src
            continue
        if not key.startswith("decoder."):
            continue
        suffix = key[len("decoder.") :]
        grade_key = "grade_decoder." + suffix
        if grade_key in adapted and tuple(src.shape) == tuple(adapted[grade_key].shape):
            adapted[grade_key] = src
            loaded_grade.append(grade_key)
        else:
            skipped.append(grade_key)
        g4c_key = "g4c_decoder." + suffix
        if g4c_key not in adapted:
            continue
        if tuple(src.shape) == tuple(adapted[g4c_key].shape):
            adapted[g4c_key] = src
            loaded_g4c.append(g4c_key)
        elif src.ndim >= 1 and src.shape[0] == 4 and adapted[g4c_key].shape[0] == 2:
            adapted[g4c_key] = _copy_old_gg4_to_binary_head(src, adapted[g4c_key])
            loaded_g4c.append(g4c_key)

    model.load_state_dict(adapted, strict=True)
    print(f"  [Init] Loaded 4-class checkpoint: {checkpoint_path}")
    print(f"  [Init] Grade decoder keys loaded: {len(loaded_grade)}")
    print(f"  [Init] GG4C decoder keys loaded: {len(loaded_g4c)}")
    if skipped:
        print(f"  [Init] Skipped grade keys: {len(skipped)}; first={skipped[:5]}")


class HierarchicalG4CLoss(nn.Module):
    def __init__(
        self,
        grade_weights: list[float],
        g4c_pos_weight: float,
        grade_dice_weight: float,
        grade_ce_weight: float,
        g4c_dice_weight: float,
        g4c_ce_weight: float,
        g4c_loss_weight: float,
    ):
        super().__init__()
        self.grade_dice_weight = float(grade_dice_weight)
        self.grade_ce_weight = float(grade_ce_weight)
        self.g4c_dice_weight = float(g4c_dice_weight)
        self.g4c_ce_weight = float(g4c_ce_weight)
        self.g4c_loss_weight = float(g4c_loss_weight)
        self.grade_dice = smp.losses.DiceLoss(mode="multiclass", classes=list(range(NUM_GRADE_CLASSES)), smooth=1e-6)
        self.register_buffer("grade_weights", torch.tensor(grade_weights, dtype=torch.float32))
        self.register_buffer("g4c_ce_weights", torch.tensor([1.0, float(g4c_pos_weight)], dtype=torch.float32))
        self.grade_ce = nn.CrossEntropyLoss(weight=self.grade_weights)

    def forward(self, outputs: dict, grade_targets, g4c_targets, g4_region) -> tuple[torch.Tensor, dict]:
        grade_logits = outputs["grade_logits"].float()
        g4c_logits = outputs["g4c_logits"].float()
        grade_targets = grade_targets.long()
        g4c_targets = g4c_targets.long()
        g4_region = g4_region.bool()

        grade_ce = self.grade_ce(grade_logits, grade_targets)
        grade_dice = self.grade_dice(grade_logits, grade_targets)
        grade_loss = self.grade_dice_weight * grade_dice + self.grade_ce_weight * grade_ce

        if g4_region.any():
            g4c_ce_map = F.cross_entropy(g4c_logits, g4c_targets, weight=self.g4c_ce_weights, reduction="none")
            g4c_ce = g4c_ce_map[g4_region].mean()
            probs = torch.softmax(g4c_logits, dim=1)[:, 1]
            target = g4c_targets.float()
            mask = g4_region.float()
            inter = (probs * target * mask).sum()
            denom = (probs * mask).sum() + (target * mask).sum()
            g4c_dice = 1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)
            g4c_loss = self.g4c_dice_weight * g4c_dice + self.g4c_ce_weight * g4c_ce
        else:
            g4c_ce = g4c_logits.sum() * 0.0
            g4c_dice = g4c_logits.sum() * 0.0
            g4c_loss = g4c_logits.sum() * 0.0

        loss = grade_loss + self.g4c_loss_weight * g4c_loss
        return loss, {
            "grade_loss": float(grade_loss.detach().cpu()),
            "g4c_loss": float(g4c_loss.detach().cpu()),
            "g4c_ce": float(g4c_ce.detach().cpu()),
            "g4c_dice": float(g4c_dice.detach().cpu()),
        }


def hierarchical_logits_to_5class(outputs: dict) -> torch.Tensor:
    grade_logits = outputs["grade_logits"]
    g4c_logits = outputs["g4c_logits"]
    grade_pred = grade_logits.argmax(dim=1)
    g4c_pred = g4c_logits.argmax(dim=1)
    pred5 = grade_pred.clone()
    pred5 = torch.where(grade_pred == 3, torch.full_like(pred5, 4), pred5)
    pred5 = torch.where((grade_pred == 2) & (g4c_pred == 1), torch.full_like(pred5, 3), pred5)
    one_hot = F.one_hot(pred5, num_classes=NUM_CLASSES_5).permute(0, 3, 1, 2).float()
    return one_hot


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


def build_optimizer(model: CONCHHierG4CModel, learning_rate: float, weight_decay: float):
    groups = []
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": learning_rate / 10, "label": "encoder"})
    groups.append({"params": model.grade_decoder.parameters(), "lr": learning_rate, "label": "grade_decoder"})
    groups.append({"params": model.g4c_decoder.parameters(), "lr": learning_rate, "label": "g4c_decoder"})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_accum_steps, scheduler=None, ema=None, ema_source=None):
    model.train()
    total_loss = 0.0
    total_grade_loss = 0.0
    total_g4c_loss = 0.0
    batches = 0
    accum = 0
    optimizer.zero_grad(set_to_none=True)
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, grade_masks, g4c_targets, g4_region, _mask5 in pbar:
        images = images.to(device, non_blocking=True)
        grade_masks = grade_masks.to(device, non_blocking=True).long()
        g4c_targets = g4c_targets.to(device, non_blocking=True).long()
        g4_region = g4_region.to(device, non_blocking=True).long()
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            loss, parts = criterion(model(images), grade_masks, g4c_targets, g4_region)
        scaler.scale(loss / grad_accum_steps).backward()
        total_loss += float(loss.item())
        total_grade_loss += parts["grade_loss"]
        total_g4c_loss += parts["g4c_loss"]
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
        pbar.set_postfix(loss=f"{loss.item():.4f}", grade=f"{parts['grade_loss']:.4f}", g4c=f"{parts['g4c_loss']:.4f}")
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
    return {
        "loss": total_loss / max(batches, 1),
        "grade_loss": total_grade_loss / max(batches, 1),
        "g4c_loss": total_g4c_loss / max(batches, 1),
    }


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    batches = 0
    metrics = SegmentationMetrics5()
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for images, grade_masks, g4c_targets, g4_region, mask5 in pbar:
        images = images.to(device, non_blocking=True)
        grade_masks = grade_masks.to(device, non_blocking=True).long()
        g4c_targets = g4c_targets.to(device, non_blocking=True).long()
        g4_region = g4_region.to(device, non_blocking=True).long()
        mask5 = mask5.to(device, non_blocking=True).long()
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            outputs = model(images)
            loss, _parts = criterion(outputs, grade_masks, g4c_targets, g4_region)
        total_loss += float(loss.item())
        batches += 1
        metrics.update(hierarchical_logits_to_5class(outputs), mask5)
    return total_loss / max(batches, 1), metrics.compute()


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


def print_metrics(metrics: dict) -> None:
    print(f"  Macro F1 (5-class hierarchical): {metrics['macro_f1']:.4f}")
    for idx, name in enumerate(CLASS_NAMES_5):
        print(
            f"    {name:4s} F1={metrics['f1_per_class'][idx]:.4f} "
            f"P={metrics['precision_per_class'][idx]:.4f} "
            f"R={metrics['recall_per_class'][idx]:.4f}"
        )
    print(f"  Macro F1 (4-class merged): {metrics['merged_macro_f1']:.4f}")
    for idx, name in enumerate(MERGED_CLASS_NAMES):
        print(
            f"    {name:10s} F1={metrics['merged_f1_per_class'][idx]:.4f} "
            f"P={metrics['merged_precision_per_class'][idx]:.4f} "
            f"R={metrics['merged_recall_per_class'][idx]:.4f}"
        )


def save_checkpoint(path: Path, model: nn.Module, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["model_state_dict"] = trainable_model(model).state_dict()
    torch.save(data, path)
    meta = {k: v for k, v in payload.items() if k not in {"confusion_matrix", "merged_confusion_matrix"}}
    with path.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hierarchical CONCH GG4C model.")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--fold", choices=["Val1", "Val2", "Val3", "Val4", "Test"], default="Val3")
    parser.add_argument("--train-xlsx", type=Path, default=None, help="Training Excel used directly with no validation set.")
    parser.add_argument("--final-train", action="store_true")
    parser.add_argument("--final-folds", nargs="+", choices=["Val1", "Val2", "Val3", "Val4", "Test"], default=None)
    parser.add_argument("--g4c-gray-min", type=int, default=125)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--selection-metric", choices=["macro_f1", "merged_macro_f1", "g4c_f1", "merged_plus_g4c"], default="merged_plus_g4c")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=4.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--fpn-channels", type=int, default=256)
    parser.add_argument("--unfreeze-last", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grade-class-weights", type=str, default=None)
    parser.add_argument("--g4c-pos-weight", type=float, default=None)
    parser.add_argument("--grade-dice-weight", type=float, default=0.55)
    parser.add_argument("--grade-ce-weight", type=float, default=0.45)
    parser.add_argument("--g4c-dice-weight", type=float, default=0.55)
    parser.add_argument("--g4c-ce-weight", type=float, default=0.45)
    parser.add_argument("--g4c-loss-weight", type=float, default=1.0)
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
    print(f"Hierarchical LUT: 75:175->GG4_merged; {args.g4c_gray_min}:175->GG4C inside binary head")
    mask_lut_5 = build_mask_lut(args.g4c_gray_min)
    grade_lut = build_grade_lut(args.g4c_gray_min)

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
        val_names = []
    else:
        train_path, val_path = split_paths(args.fold)
        train_names = read_split(train_path)
        val_names = read_split(val_path)
        data_summary = {
            "training_source": str(train_path),
            "validation": str(val_path),
            "train_rows": len(train_names),
            "val_rows": len(val_names),
        }
    print(f"Data summary: {json.dumps(data_summary, indent=2)}")

    stats = compute_training_stats(train_names, mask_lut_5, grade_lut)
    if args.grade_class_weights is not None:
        grade_weights = [float(x.strip()) for x in args.grade_class_weights.split(",") if x.strip()]
        if len(grade_weights) != NUM_GRADE_CLASSES:
            parser.error("--grade-class-weights must contain exactly 4 values.")
    else:
        grade_weights = stats["grade_weights"]
    g4c_pos_weight = float(args.g4c_pos_weight) if args.g4c_pos_weight is not None else float(stats["g4c_pos_weight"])
    print("Grade pixel frequencies: " + ", ".join(f"{n}={stats['grade_freq'][i]:.6f}" for i, n in enumerate(GRADE_CLASS_NAMES)))
    print(f"Grade class weights: {grade_weights}")
    print(f"GG4C pixels: pos={stats['g4c_pos_pixels']} neg={stats['g4c_neg_pixels']} pos_weight={g4c_pos_weight:.4f}")

    if args.data_summary_only:
        return

    config = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
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
    train_loader = make_loader(train_names, mask_lut_5, grade_lut, config, train=True)
    val_loader = make_loader(val_names, mask_lut_5, grade_lut, config, train=False) if val_names else None

    model = CONCHHierG4CModel(
        fpn_channels=args.fpn_channels,
        unfreeze_last=args.unfreeze_last,
        weights_path=args.conch_checkpoint,
        hf_token=args.hf_token,
    )
    load_4class_checkpoint(model, args.base_checkpoint)
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

    criterion = HierarchicalG4CLoss(
        grade_weights=grade_weights,
        g4c_pos_weight=g4c_pos_weight,
        grade_dice_weight=args.grade_dice_weight,
        grade_ce_weight=args.grade_ce_weight,
        g4c_dice_weight=args.g4c_dice_weight,
        g4c_ce_weight=args.g4c_ce_weight,
        g4c_loss_weight=args.g4c_loss_weight,
    ).to(device)
    optimizer = build_optimizer(trainable_model(model), args.learning_rate, args.weight_decay)
    steps_per_epoch = max(1, len(train_loader) // max(1, args.grad_accum))
    total_steps = steps_per_epoch * (1 if args.dry_run else args.max_epochs)
    scheduler = build_cosine_warmup_scheduler(optimizer, total_steps=total_steps, warmup_pct=0.07, min_lr_ratio=0.01)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_path = None
    patience = 0
    max_epochs = 1 if args.dry_run else args.max_epochs
    ema_source = trainable_model(model) if ema is not None else None
    train_stats = None

    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch {epoch}/{max_epochs} | lr={optimizer.param_groups[-1]['lr']:.2e}")
        train_stats = train_one_epoch(
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
        print(
            f"  Train loss: {train_stats['loss']:.4f} | "
            f"grade={train_stats['grade_loss']:.4f} | g4c={train_stats['g4c_loss']:.4f}"
        )

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
            name = args.checkpoint_name or f"best_{args.fold}_hier_g4c_{args.selection_metric}_{best_score:.4f}.pth"
            best_path = out_dir / name
            save_checkpoint(
                best_path,
                val_model,
                {
                    "epoch": epoch,
                    "fold": args.fold,
                    "architecture": "hierarchical_4class_plus_g4c_binary",
                    "base_checkpoint": str(args.base_checkpoint),
                    "g4c_gray_min": int(args.g4c_gray_min),
                    "grade_class_names": GRADE_CLASS_NAMES,
                    "class_names": CLASS_NAMES_5,
                    "merged_class_names": MERGED_CLASS_NAMES,
                    "grade_class_weights": grade_weights,
                    "g4c_pos_weight": g4c_pos_weight,
                    "train_loss": train_stats,
                    "val_loss": float(val_loss),
                    "selection_metric": args.selection_metric,
                    "selection_score": float(best_score),
                    "macro_f1": float(metrics["macro_f1"]),
                    "f1_per_class": [float(x) for x in metrics["f1_per_class"]],
                    "precision_per_class": [float(x) for x in metrics["precision_per_class"]],
                    "recall_per_class": [float(x) for x in metrics["recall_per_class"]],
                    "confusion_matrix": metrics["confusion_matrix"].tolist(),
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
        final_model = ema.ema if ema is not None else model
        name = args.checkpoint_name or "final_hier_g4c.pth"
        final_path = out_dir / name
        save_checkpoint(
            final_path,
            final_model,
            {
                "epoch": max_epochs,
                "trained_epochs": int(max_epochs),
                "architecture": "hierarchical_4class_plus_g4c_binary",
                "base_checkpoint": str(args.base_checkpoint),
                "g4c_gray_min": int(args.g4c_gray_min),
                "grade_class_names": GRADE_CLASS_NAMES,
                "class_names": CLASS_NAMES_5,
                "merged_class_names": MERGED_CLASS_NAMES,
                "grade_class_weights": grade_weights,
                "g4c_pos_weight": g4c_pos_weight,
                "training_source": data_summary.get("training_source"),
                "final_train_loss": train_stats,
                "data_summary": data_summary,
            },
        )
        print(f"\nFinal checkpoint: {final_path}")
    else:
        print(f"\nBest selection score ({args.selection_metric}): {best_score:.4f}")
        print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
