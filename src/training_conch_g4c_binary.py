"""
Train a GG4 cribiform binary head on top of a trained CONCH GG4 segmenter.

The base model is the existing 4-class CONCH/FPN segmenter. This script loads
its checkpoint, freezes it by default, and trains a patch-level binary head on
the GG4-only Cribfriform partitions:

    input  : GG4 patch image
    target : G4C in {0, 1}

Default checkpoint:
    artifacts/checkpoints_conch_masklut/best_Val3_0.8201.pth

Typical usage:
    python src/training_conch_g4c_binary.py --no-compile
    python src/training_conch_g4c_binary.py --final-train --no-compile
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Sequence

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from paths import IMAGES_DIR, PARTITION_DIR, default_checkpoint_dir


DEFAULT_SEED = 42
IMG_SIZE = 512
CONCH_NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
CONCH_NORM_STD = [0.26862954, 0.26130258, 0.27577711]
DEFAULT_BASE_CHECKPOINT = Path("artifacts/checkpoints_conch_masklut/best_Val3_0.8201.pth")
OUTPUT_DIR = default_checkpoint_dir("checkpoints_conch_g4c_binary")


@dataclass
class DataSummary:
    rows: int
    positives: int
    negatives: int
    positive_fraction: float
    sources: list[str]


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


def read_g4c_table(paths: Sequence[Path]) -> tuple[pd.DataFrame, DataSummary]:
    frames = []
    sources = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Partition file not found: {path}")
        df = pd.read_excel(path)
        required = {"image_name", "G4C"}
        missing = required.difference(df.columns)
        if missing:
            raise KeyError(f"{path} is missing columns: {sorted(missing)}")
        df = df[["image_name", "G4C"]].copy()
        df["image_name"] = df["image_name"].astype(str)
        df["G4C"] = df["G4C"].astype(int)
        df["source"] = str(path)
        frames.append(df)
        sources.append(str(path))

    table = pd.concat(frames, ignore_index=True)
    before = len(table)
    grouped = table.groupby("image_name", sort=False)["G4C"].nunique()
    conflicts = grouped[grouped > 1]
    if not conflicts.empty:
        examples = ", ".join(conflicts.index[:5].tolist())
        raise ValueError(f"Conflicting G4C labels for duplicated images: {examples}")
    table = table.drop_duplicates(subset=["image_name"], keep="first").reset_index(drop=True)

    positives = int(table["G4C"].sum())
    negatives = int(len(table) - positives)
    summary = DataSummary(
        rows=int(len(table)),
        positives=positives,
        negatives=negatives,
        positive_fraction=float(positives / max(len(table), 1)),
        sources=sources + ([f"deduplicated_removed={before - len(table)}"] if before != len(table) else []),
    )
    return table, summary


class G4CDataset(Dataset):
    def __init__(self, table: pd.DataFrame, images_dir: Path, transform=None):
        self.table = table.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        name = str(row["image_name"])
        image_path = self.images_dir / name
        buf = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        target = torch.tensor(float(row["G4C"]), dtype=torch.float32)
        return image, target, name


def get_train_transforms(no_color_aug: bool = False):
    steps = [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.20, p=0.5),
    ]
    if not no_color_aug:
        steps.extend(
            [
                A.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.35),
            ]
        )
    steps.extend(
        [
            A.Normalize(mean=CONCH_NORM_MEAN, std=CONCH_NORM_STD),
            ToTensorV2(),
        ]
    )
    return A.Compose(steps)


def get_val_transforms():
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=CONCH_NORM_MEAN, std=CONCH_NORM_STD),
            ToTensorV2(),
        ]
    )


def make_loader(
    table: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    seed: int,
    train: bool,
    weighted_sampler: bool = False,
    no_color_aug: bool = False,
):
    dataset = G4CDataset(
        table,
        IMAGES_DIR,
        transform=get_train_transforms(no_color_aug=no_color_aug) if train else get_val_transforms(),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    worker_init = partial(_worker_init_fn, base_seed=seed) if num_workers > 0 else None
    kwargs = {"persistent_workers": True, "prefetch_factor": 4} if num_workers > 0 else {}
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        worker_init_fn=worker_init,
        **kwargs,
    )

    if train and weighted_sampler:
        labels = table["G4C"].astype(int).to_numpy()
        pos = max(int(labels.sum()), 1)
        neg = max(int(len(labels) - labels.sum()), 1)
        weights = np.where(labels == 1, neg / pos, 1.0).astype(np.float64)
        sampler = WeightedRandomSampler(weights=weights.tolist(), num_samples=len(weights), replacement=True)
        return DataLoader(dataset, sampler=sampler, drop_last=True, **common)

    return DataLoader(dataset, shuffle=train, drop_last=train, **common)


def _clean_state_dict(raw):
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break
    if not isinstance(raw, dict):
        raise TypeError("Checkpoint does not contain a state_dict-like object.")
    clean = {}
    for key, value in raw.items():
        name = key
        for prefix in ("module.", "_orig_mod."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        clean[name] = value
    return clean


def load_base_segmenter(args) -> nn.Module:
    from training_conchv2 import CONCHSegModel

    model = CONCHSegModel(
        fpn_channels=args.fpn_channels,
        num_classes=4,
        unfreeze_last=args.unfreeze_last if args.fine_tune_backbone else 0,
        weights_path=args.conch_checkpoint,
        hf_token=args.hf_token,
    )
    ckpt_path = Path(args.base_checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu")
    state = _clean_state_dict(raw)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] Missing keys while loading base checkpoint: {len(missing)}")
        print(f"         first: {missing[:5]}")
    if unexpected:
        print(f"  [WARN] Unexpected keys while loading base checkpoint: {len(unexpected)}")
        print(f"         first: {unexpected[:5]}")
    print(f"  [Base] Loaded 4-class checkpoint: {ckpt_path}")
    return model


class G4CHeadModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        hidden_dim: int = 512,
        dropout: float = 0.25,
        freeze_backbone: bool = True,
        use_seg_context: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.freeze_backbone = bool(freeze_backbone)
        self.use_seg_context = bool(use_seg_context)
        self.num_feature_maps = len(base_model.encoder.feature_blocks)
        self.embed_dim = int(base_model.encoder.embed_dim)
        pooled_dim = self.num_feature_maps * self.embed_dim * 2
        context_dim = 8 if self.use_seg_context else 0

        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim + context_dim),
            nn.Linear(pooled_dim + context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 4, 64)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 4, 64), 1),
        )

        if self.freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.base_model.eval()
        return self

    def _extract_inputs(self, images: torch.Tensor) -> torch.Tensor:
        features = self.base_model.encoder(images)
        pooled = []
        for feat in features:
            pooled.append(F.adaptive_avg_pool2d(feat, 1).flatten(1))
            pooled.append(F.adaptive_max_pool2d(feat, 1).flatten(1))
        vectors = [torch.cat(pooled, dim=1)]

        if self.use_seg_context:
            logits4 = self.base_model.decoder(features, images.shape[-2:])
            probs4 = torch.softmax(logits4.float(), dim=1)
            mean_probs = probs4.flatten(2).mean(dim=2)
            max_probs = probs4.flatten(2).amax(dim=2)
            vectors.append(torch.cat([mean_probs, max_probs], dim=1))

        return torch.cat(vectors, dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                x = self._extract_inputs(images)
        else:
            x = self._extract_inputs(images)
        return self.head(x).squeeze(1)


def binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    labels = labels.astype(np.int64)
    preds = (probs >= float(threshold)).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(labels), 1)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def auc_roc(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(probs).rank(method="average").to_numpy(dtype=np.float64)
    rank_sum_pos = float(ranks[labels == 1].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-probs)
    y = labels[order]
    tp = np.cumsum(y == 1)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def tune_threshold(labels: np.ndarray, probs: np.ndarray, min_thr: float, max_thr: float, step: float) -> tuple[float, dict]:
    thresholds = np.arange(float(min_thr), float(max_thr) + 1e-9, float(step))
    best_thr = float(thresholds[0])
    best = None
    for thr in thresholds:
        stats = binary_metrics(labels, probs, float(thr))
        if best is None or stats["f1"] > best["f1"]:
            best = stats
            best_thr = float(thr)
    return best_thr, best


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_accum_steps: int) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    accum = 0
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, targets, _ in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits.float(), targets.float())
        scaler.scale(loss / grad_accum_steps).backward()
        total_loss += float(loss.item())
        batches += 1
        accum += 1
        if accum >= grad_accum_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accum = 0
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    if accum > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float, threshold_args=None) -> tuple[float, dict, float]:
    model.eval()
    total_loss = 0.0
    batches = 0
    all_probs = []
    all_labels = []
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for images, targets, _ in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            logits = model(images)
            loss = criterion(logits.float(), targets.float())
        probs = torch.sigmoid(logits.float()).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(targets.detach().cpu().numpy())
        total_loss += float(loss.item())
        batches += 1

    labels = np.concatenate(all_labels).astype(np.int64)
    probs = np.concatenate(all_probs).astype(np.float64)
    selected_threshold = float(threshold)
    if threshold_args is not None:
        selected_threshold, stats = tune_threshold(labels, probs, *threshold_args)
    else:
        stats = binary_metrics(labels, probs, selected_threshold)
    stats["auroc"] = auc_roc(labels, probs)
    stats["average_precision"] = average_precision(labels, probs)
    stats["positive_fraction"] = float(labels.mean()) if len(labels) else 0.0
    return total_loss / max(batches, 1), stats, selected_threshold


def build_train_paths(args) -> list[Path]:
    if args.final_train:
        if args.final_source == "test":
            return [
                PARTITION_DIR / "Test" / "TrainCribfriform.xlsx",
                PARTITION_DIR / "Test" / "TestCribfriform.xlsx",
            ]
        paths = []
        for fold in ("Val1", "Val2", "Val3", "Val4"):
            paths.append(PARTITION_DIR / "Validation" / fold / "TrainCribfriform.xlsx")
            paths.append(PARTITION_DIR / "Validation" / fold / "TestCribfriform.xlsx")
        return paths
    return [Path(args.train_xlsx)]


def save_checkpoint(path: Path, model: G4CHeadModel, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["model_state_dict"] = model.state_dict()
    data["head_state_dict"] = model.head.state_dict()
    torch.save(data, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a binary GG4c head from CONCH GG4 features.")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--train-xlsx", type=Path, default=PARTITION_DIR / "Test" / "TrainCribfriform.xlsx")
    parser.add_argument("--val-xlsx", type=Path, default=PARTITION_DIR / "Test" / "TestCribfriform.xlsx")
    parser.add_argument("--final-train", action="store_true", help="Train on deduplicated train+test GG4c data and skip validation.")
    parser.add_argument(
        "--final-source",
        choices=["test", "validation-folds"],
        default="test",
        help="Data source used with --final-train. 'test' uses partition/Test Train+Test; validation-folds dedupes Val1-Val4.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pos-weight", type=float, default=None, help="BCE positive weight. Default: negatives / positives.")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--no-color-aug", action="store_true")
    parser.add_argument("--no-seg-context", action="store_true", help="Do not append frozen 4-class probability summaries to the head input.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-threshold-tuning", action="store_true")
    parser.add_argument("--thr-min", type=float, default=0.10)
    parser.add_argument("--thr-max", type=float, default=0.90)
    parser.add_argument("--thr-step", type=float, default=0.02)
    parser.add_argument("--fpn-channels", type=int, default=256)
    parser.add_argument("--conch-checkpoint", type=str, default=None, help="Optional raw CONCH weights path passed to CONCH loader.")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--fine-tune-backbone", action="store_true", help="Also train decoder and optionally last encoder blocks.")
    parser.add_argument("--unfreeze-last", type=int, default=0, help="Last CONCH blocks to unfreeze with --fine-tune-backbone.")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--data-summary-only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")

    train_paths = build_train_paths(args)
    train_table, train_summary = read_g4c_table(train_paths)
    print(f"Train data: {json.dumps(asdict(train_summary), indent=2)}")

    val_table = None
    val_summary = None
    if not args.final_train:
        val_table, val_summary = read_g4c_table([Path(args.val_xlsx)])
        print(f"Val data: {json.dumps(asdict(val_summary), indent=2)}")

    if args.data_summary_only:
        return

    base_model = load_base_segmenter(args)
    model = G4CHeadModel(
        base_model=base_model,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_backbone=not args.fine_tune_backbone,
        use_seg_context=not args.no_seg_context,
    ).to(device)

    use_compile = args.compile and not args.no_compile
    if use_compile and int(torch.__version__.split(".")[0]) >= 2:
        try:
            model = torch.compile(model)
            print("  [Compile] torch.compile enabled.")
        except Exception as exc:
            print(f"  [WARN] torch.compile failed: {exc}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_count:,} / {total_count:,}")

    pos_weight = args.pos_weight
    if pos_weight is None:
        pos_weight = train_summary.negatives / max(train_summary.positives, 1)
    print(f"BCE pos_weight: {pos_weight:.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device, dtype=torch.float32))

    train_loader = make_loader(
        train_table,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train=True,
        weighted_sampler=args.weighted_sampler,
        no_color_aug=args.no_color_aug,
    )
    val_loader = None
    if val_table is not None:
        val_loader = make_loader(
            val_table,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            train=False,
        )

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.max_epochs), eta_min=args.learning_rate * 0.01)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_f1 = -1.0
    best_stats = None
    best_threshold = float(args.threshold)
    best_path = None
    patience = 0

    for epoch in range(1, args.max_epochs + 1):
        print(f"\nEpoch {epoch}/{args.max_epochs} | lr={optimizer.param_groups[0]['lr']:.2e}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, max(1, args.grad_accum))
        scheduler.step()

        if val_loader is None:
            print(f"  Train loss: {train_loss:.4f}")
            continue

        threshold_args = None
        if not args.no_threshold_tuning:
            threshold_args = (args.thr_min, args.thr_max, args.thr_step)
        val_loss, stats, selected_thr = evaluate(model, val_loader, criterion, device, args.threshold, threshold_args)
        print(
            f"  Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
            f"F1: {stats['f1']:.4f} | P: {stats['precision']:.4f} | "
            f"R: {stats['recall']:.4f} | AUROC: {stats['auroc']:.4f} | "
            f"AP: {stats['average_precision']:.4f} | thr={selected_thr:.2f}"
        )
        print(f"  CM: TP={stats['tp']} FP={stats['fp']} TN={stats['tn']} FN={stats['fn']}")

        if stats["f1"] > best_f1:
            best_f1 = float(stats["f1"])
            best_stats = stats
            best_threshold = float(selected_thr)
            patience = 0
            name = args.checkpoint_name or f"best_g4c_head_Val3base_f1{best_f1:.4f}.pth"
            best_path = args.output_dir / name
            save_checkpoint(
                best_path,
                model if not hasattr(model, "_orig_mod") else model._orig_mod,
                {
                    "epoch": epoch,
                    "base_checkpoint": str(args.base_checkpoint),
                    "train_summary": asdict(train_summary),
                    "val_summary": asdict(val_summary),
                    "threshold": best_threshold,
                    "metrics": best_stats,
                    "args": vars(args),
                },
            )
            print(f"  Saved: {best_path}")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if val_loader is None:
        name = args.checkpoint_name or "final_g4c_head_Val3base_all_gg4.pth"
        final_path = args.output_dir / name
        save_checkpoint(
            final_path,
            model if not hasattr(model, "_orig_mod") else model._orig_mod,
            {
                "epoch": args.max_epochs,
                "base_checkpoint": str(args.base_checkpoint),
                "train_summary": asdict(train_summary),
                "threshold": float(args.threshold),
                "metrics": None,
                "args": vars(args),
            },
        )
        print(f"\nFinal head saved: {final_path}")
    else:
        print(f"\nBest F1: {best_f1:.4f} at threshold {best_threshold:.2f}")
        print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
