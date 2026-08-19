from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paths import PARTITION_DIR  # noqa: E402
from training_conchv2 import CONCHSegModel, get_val_transforms, tta_forward  # noqa: E402
from training_conchv2_g4c_5class import (  # noqa: E402
    CLASS_NAMES,
    MERGED_CLASS_NAMES,
    NUM_CLASSES,
    SICAPv2G4C5Dataset,
    SegmentationMetrics5,
    build_mask_lut,
    clean_state_dict,
)


DEFAULT_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "checkpoints_conch_masklut_g4c_5class"
    / "best_Val3_5class_g4c_merged_plus_g4c_0.7619.pth"
)
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "eval_5class_holdout_test"
DEFAULT_TEST_XLSX = PARTITION_DIR / "Test" / "Test.xlsx"
BASELINE_MERGED_MACRO_F1 = 0.7097
BASELINE_BINARY_CANCER_F1 = 0.7384


def read_g4c_gray_min(checkpoint: Path, explicit_value: int | None) -> int:
    if explicit_value is not None:
        return int(explicit_value)
    meta_path = checkpoint.with_suffix(".json")
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        value = meta.get("g4c_gray_min")
        if value is not None:
            return int(value)
    return 125


def read_test_names(path: Path) -> list[str]:
    df = pd.read_excel(path)
    if "image_name" not in df.columns:
        raise KeyError(f"{path} does not contain image_name.")
    return df["image_name"].dropna().astype(str).tolist()


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
        raise RuntimeError(f"Checkpoint mismatch. missing={missing[:8]} unexpected={unexpected[:8]}")
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, use_tta: bool) -> dict:
    metrics = SegmentationMetrics5()
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    for images, masks in tqdm(loader, desc="Holdout Test"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            if use_tta:
                logits = tta_forward(model, images, scales=(1.0,), use_d4=True)
            else:
                logits = model(images)
        metrics.update(logits, masks)
    return metrics.compute()


def confusion_to_frame(cm: np.ndarray, names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(cm, index=[f"true_{n}" for n in names], columns=[f"pred_{n}" for n in names])


def stats_from_cm(cm: np.ndarray) -> dict:
    f1 = np.zeros(cm.shape[0], dtype=np.float64)
    precision = np.zeros(cm.shape[0], dtype=np.float64)
    recall = np.zeros(cm.shape[0], dtype=np.float64)
    for idx in range(cm.shape[0]):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        precision[idx] = tp / max(tp + fp, 1)
        recall[idx] = tp / max(tp + fn, 1)
        f1[idx] = (2.0 * tp) / max((2.0 * tp + fp + fn), 1)
    return {
        "macro_f1": float(np.mean(f1)),
        "f1_per_class": f1,
        "precision_per_class": precision,
        "recall_per_class": recall,
    }


def binary_cancer_from_cm5(cm: np.ndarray) -> dict:
    tn = int(cm[0, 0])
    fp = int(cm[0, 1:].sum())
    fn = int(cm[1:, 0].sum())
    tp = int(cm[1:, 1:].sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def print_class_metrics(title: str, names: list[str], stats: dict) -> None:
    print(title)
    for idx, name in enumerate(names):
        print(
            f"  {name:10s} F1={stats['f1_per_class'][idx]:.4f} "
            f"P={stats['precision_per_class'][idx]:.4f} "
            f"R={stats['recall_per_class'][idx]:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 5-class CONCH checkpoint on held-out Test at pixel level.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tta", dest="tta", action="store_true", default=True)
    parser.add_argument("--no-tta", dest="tta", action="store_false")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--g4c-gray-min", type=int, default=None)
    parser.add_argument("--fpn-channels", type=int, default=256)
    parser.add_argument("--conch-checkpoint", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    g4c_gray_min = read_g4c_gray_min(args.checkpoint, args.g4c_gray_min)
    mask_lut = build_mask_lut(g4c_gray_min)
    test_names = read_test_names(DEFAULT_TEST_XLSX)
    transform = get_val_transforms()
    dataset = SICAPv2G4C5Dataset(test_names, mask_lut=mask_lut, transform=transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test split: {DEFAULT_TEST_XLSX} | tiles={len(test_names)}")
    print(f"g4c_gray_min: {g4c_gray_min}")
    print(f"TTA: D4={args.tta} | scales=(1.0,)")

    model = load_model(args, device)
    metrics = evaluate(model, loader, device=device, use_tta=args.tta)
    cm5 = metrics["confusion_matrix"]
    cm_merged = metrics["merged_confusion_matrix"]
    stats5 = stats_from_cm(cm5)
    stats_merged = stats_from_cm(cm_merged)
    cancer = binary_cancer_from_cm5(cm5)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    confusion_to_frame(cm5, CLASS_NAMES).to_csv(args.output_dir / "confusion_matrix_5class.csv")
    confusion_to_frame(cm_merged, MERGED_CLASS_NAMES).to_csv(args.output_dir / "confusion_matrix_merged.csv")

    summary = {
        "checkpoint": str(args.checkpoint),
        "test_xlsx": str(DEFAULT_TEST_XLSX),
        "n_tiles": len(test_names),
        "tta": {"enabled": bool(args.tta), "d4": bool(args.tta), "scales": [1.0]},
        "g4c_gray_min": int(g4c_gray_min),
        "class_names": CLASS_NAMES,
        "merged_class_names": MERGED_CLASS_NAMES,
        "metrics_5class": {
            **stats5,
            "confusion_matrix": cm5,
        },
        "metrics_merged": {
            **stats_merged,
            "confusion_matrix": cm_merged,
        },
        "binary_cancer": cancer,
        "comparison_with_4class_baseline": {
            "merged_macro_f1_baseline": BASELINE_MERGED_MACRO_F1,
            "merged_macro_f1_delta": float(stats_merged["macro_f1"] - BASELINE_MERGED_MACRO_F1),
            "binary_cancer_f1_baseline": BASELINE_BINARY_CANCER_F1,
            "binary_cancer_f1_delta": float(cancer["f1"] - BASELINE_BINARY_CANCER_F1),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")

    print_class_metrics("\n5-class metrics:", CLASS_NAMES, stats5)
    print_class_metrics("\n4-class merged metrics:", MERGED_CLASS_NAMES, stats_merged)
    print(
        "\nBinary cancer: "
        f"F1={cancer['f1']:.4f} P={cancer['precision']:.4f} R={cancer['recall']:.4f} "
        f"Specificity={cancer['specificity']:.4f}"
    )
    print(
        f"\n5-class Macro F1: {stats5['macro_f1']:.4f} | "
        f"4-class merged Macro F1: {stats_merged['macro_f1']:.4f} "
        f"(vs {BASELINE_MERGED_MACRO_F1:.4f} baseline) | "
        f"Binary Cancer F1: {cancer['f1']:.4f} "
        f"(vs {BASELINE_BINARY_CANCER_F1:.4f} baseline)"
    )
    print(f"Saved summary: {args.output_dir / 'summary.json'}")
    print(f"Saved confusion matrices: {args.output_dir}")


if __name__ == "__main__":
    main()
