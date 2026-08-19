from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paths import IMAGES_DIR, default_checkpoint_dir  # noqa: E402
from training_conchv2_g4c_5class import (  # noqa: E402
    CLASS_NAMES,
    CONCH_NORM_MEAN,
    CONCH_NORM_STD,
    CONCHSegModel,
    NUM_CLASSES,
    clean_state_dict,
    get_val_transforms,
)


DEFAULT_CHECKPOINT = (
    default_checkpoint_dir("checkpoints_conch_masklut_g4c_5class")
    / "best_Val3_5class_g4c_merged_plus_g4c_0.7619.pth"
)
DEFAULT_LABELS = ROOT / "partition" / "Test" / "TestCribfriform.xlsx"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "eval_g4c_5class_tile_level"


class TileDataset(Dataset):
    def __init__(self, table: pd.DataFrame):
        self.table = table.reset_index(drop=True)
        self.transform = get_val_transforms(
            norm_mean=list(CONCH_NORM_MEAN),
            norm_std=list(CONCH_NORM_STD),
        )

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int):
        row = self.table.iloc[idx]
        name = str(row["image_name"])
        image_path = IMAGES_DIR / name
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        label = int(row["G4C"])
        return image, label, name


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = scores >= float(threshold)
    y = labels.astype(bool)
    tp = int(np.sum(preds & y))
    fp = int(np.sum(preds & ~y))
    tn = int(np.sum(~preds & ~y))
    fn = int(np.sum(~preds & y))
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
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def auc_roc(labels: np.ndarray, scores: np.ndarray) -> float:
    y = labels.astype(np.int64)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_rank_sum = float(ranks[y == 1].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    y = labels.astype(np.int64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores)
    y = y[order]
    tp = np.cumsum(y == 1)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> dict:
    thresholds = np.unique(scores.astype(np.float64))
    if len(thresholds) == 0:
        thresholds = np.array([0.0], dtype=np.float64)
    candidates = np.unique(np.concatenate(([0.0, 1.0, 10.0, 25.0, 50.0], thresholds)))
    best = None
    for threshold in candidates:
        stats = binary_metrics(labels, scores, float(threshold))
        if best is None or stats["f1"] > best["f1"] or (
            stats["f1"] == best["f1"] and stats["threshold"] > best["threshold"]
        ):
            best = stats
    return best


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = CONCHSegModel(
        fpn_channels=args.fpn_channels,
        num_classes=NUM_CLASSES,
        unfreeze_last=0,
        weights_path=args.conch_checkpoint,
        hf_token=args.hf_token,
    )
    raw = torch.load(args.checkpoint, map_location="cpu")
    state = clean_state_dict(raw)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch. missing={missing[:5]} unexpected={unexpected[:5]}")
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    rows = []
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    for images, labels, names in tqdm(loader, desc="Infer 5-class"):
        images = images.to(device, non_blocking=True)
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            logits = model(images)
            probs = torch.softmax(logits.float(), dim=1)
        preds = logits.argmax(dim=1)
        g4c_pixels = (preds == 3).sum(dim=(1, 2)).detach().cpu().numpy()
        g4c_prob_mean = probs[:, 3].mean(dim=(1, 2)).detach().cpu().numpy()
        g4c_prob_max = probs[:, 3].amax(dim=(1, 2)).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        for name, label, pixels, pmean, pmax in zip(names, labels_np, g4c_pixels, g4c_prob_mean, g4c_prob_max):
            rows.append(
                {
                    "image_name": str(name),
                    "G4C": int(label),
                    "g4c_pixels": int(pixels),
                    "g4c_prob_mean": float(pmean),
                    "g4c_prob_max": float(pmax),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 5-class GG4C segmentation as tile-level GG4C present.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--labels-xlsx", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--pixel-threshold", type=float, default=50.0)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fpn-channels", type=int, default=256)
    parser.add_argument("--conch-checkpoint", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.labels_xlsx.is_file():
        raise FileNotFoundError(args.labels_xlsx)

    table = pd.read_excel(args.labels_xlsx)
    required = {"image_name", "G4C"}
    missing = required.difference(table.columns)
    if missing:
        raise KeyError(f"{args.labels_xlsx} missing columns: {sorted(missing)}")
    table = table[["image_name", "G4C"]].dropna().copy()
    table["image_name"] = table["image_name"].astype(str)
    table["G4C"] = table["G4C"].astype(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Labels: {args.labels_xlsx} rows={len(table)} positives={int(table['G4C'].sum())}")

    dataset = TileDataset(table)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model = load_model(args, device)
    pred_df = run_inference(model, loader, device)

    labels = pred_df["G4C"].to_numpy(dtype=np.int64)
    pixel_scores = pred_df["g4c_pixels"].to_numpy(dtype=np.float64)
    fixed = binary_metrics(labels, pixel_scores, args.pixel_threshold)
    tuned = best_threshold(labels, pixel_scores)
    summary = {
        "checkpoint": str(args.checkpoint),
        "labels_xlsx": str(args.labels_xlsx),
        "class_names": CLASS_NAMES,
        "n_tiles": int(len(pred_df)),
        "positives": int(labels.sum()),
        "negatives": int(len(labels) - labels.sum()),
        "score": "g4c_pixels",
        "fixed_pixel_threshold": fixed,
        "best_pixel_threshold": tuned,
        "auroc_pixels": auc_roc(labels, pixel_scores),
        "average_precision_pixels": average_precision(labels, pixel_scores),
        "score_ranges": {
            "g4c_pixels_min": int(pred_df["g4c_pixels"].min()),
            "g4c_pixels_median": float(pred_df["g4c_pixels"].median()),
            "g4c_pixels_max": int(pred_df["g4c_pixels"].max()),
        },
    }

    for score_col in ("g4c_prob_mean", "g4c_prob_max"):
        scores = pred_df[score_col].to_numpy(dtype=np.float64)
        summary[f"auroc_{score_col}"] = auc_roc(labels, scores)
        summary[f"average_precision_{score_col}"] = average_precision(labels, scores)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "tile_predictions.csv"
    json_path = args.out_dir / "summary.json"
    pred_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved predictions: {csv_path}")
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
