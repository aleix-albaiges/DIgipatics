#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
VENV_SITE_PACKAGES = REPO_ROOT / "prostata_env" / "Lib" / "site-packages"


def bootstrap_runtime() -> None:
    """Keep NumPy/SciPy/Pandas from the base interpreter, and load PyTorch/CV deps from the repo venv."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # Pin the scientific stack from the current interpreter before prioritising the repo venv.
    for module_name in ("numpy", "pandas", "scipy", "tifffile", "PIL"):
        try:
            __import__(module_name)
        except Exception:
            pass

    if VENV_SITE_PACKAGES.exists():
        sys.path.insert(0, str(VENV_SITE_PACKAGES))
    if SRC_DIR.exists():
        sys.path.insert(0, str(SRC_DIR))

    # `training_conch_final` imports wandb even though inference does not need it.
    sys.modules.setdefault("wandb", types.ModuleType("wandb"))


bootstrap_runtime()

import cv2
import numpy as np
import pandas as pd
import torch

from inspect_pandas_masks import (  # type: ignore
    SICAP_CLASS_NAMES,
    build_color_palette,
    colorize_mask,
    get_tiff_structure,
    normalize_to_uint8,
    overlay,
    read_tiff_level,
    resize_image,
)
from training_conch_final import CONCHSegModel


cv2.setNumThreads(0)

LOGGER = logging.getLogger("run_pandas_sicap_tissue_inference")
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES = [SICAP_CLASS_NAMES[idx] for idx in range(4)]


@dataclass
class CaseInfo:
    case_id: str
    wsi_path: Path
    wsi_structure: Dict[str, object]
    data_provider: Optional[str]
    isup_grade: Optional[str]
    gleason_score: Optional[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SICAPv2 multiclase inference on PANDAS WSI tiles without GT-guided tile selection."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("PANDAS"))
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pandas_tissue_inference_nogt"),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("artifacts/checkpoints_conch_masklut/best_Val3_0.8201.pth"),
    )
    parser.add_argument("--case-id", type=str, nargs="*", default=None)
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--blend-mode", type=str, default="gaussian_sum", choices=("gaussian_sum", "overwrite"))
    parser.add_argument("--gaussian-sigma-scale", type=float, default=4.0)
    parser.add_argument("--target-magnification", type=float, default=10.0)
    parser.add_argument("--source-magnification", type=float, default=40.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--thumb-size", type=int, default=1024)
    parser.add_argument("--tissue-threshold", type=float, default=0.15)
    parser.add_argument(
        "--min-predicted-tumor-fraction",
        type=float,
        default=0.01,
        help="Threshold only for reporting predicted tumor tiles, not for selecting inference tiles.",
    )
    parser.add_argument("--log-level", type=str, default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s | %(levelname)s | %(message)s")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA unavailable. CPU will be used.")
        return torch.device("cpu")
    return torch.device(device_name)


def load_checkpoint_path(path: Path) -> Path:
    checkpoint_path = path
    if not checkpoint_path.is_absolute():
        checkpoint_path = (REPO_ROOT / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    LOGGER.info("Loading checkpoint %s", checkpoint_path)
    model = CONCHSegModel(num_classes=4, unfreeze_last=0)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_train_metadata(train_csv: Optional[Path], data_dir: Path) -> Dict[str, Dict[str, str]]:
    csv_path = train_csv if train_csv is not None else (data_dir / "train.csv")
    if not csv_path.exists():
        LOGGER.warning("train.csv not found in %s. Metadata fields will be empty.", csv_path)
        return {}

    df = pd.read_csv(csv_path)
    if "image_id" not in df.columns:
        LOGGER.warning("%s does not contain image_id. Metadata fields will be empty.", csv_path)
        return {}

    cols = set(df.columns)
    out: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        image_id = str(row["image_id"])
        out[image_id] = {
            "data_provider": str(row["data_provider"]) if "data_provider" in cols and pd.notna(row["data_provider"]) else "",
            "isup_grade": str(row["isup_grade"]) if "isup_grade" in cols and pd.notna(row["isup_grade"]) else "",
            "gleason_score": str(row["gleason_score"]) if "gleason_score" in cols and pd.notna(row["gleason_score"]) else "",
        }
    return out


def discover_cases(
    data_dir: Path,
    explicit_case_ids: Optional[Sequence[str]],
    max_cases: int,
    train_metadata: Dict[str, Dict[str, str]],
) -> List[CaseInfo]:
    explicit = set(explicit_case_ids or [])
    cases: List[CaseInfo] = []

    for wsi_path in sorted(data_dir.glob("*.tiff")):
        if wsi_path.name.endswith("_mask.tiff"):
            continue
        case_id = wsi_path.stem
        if explicit and case_id not in explicit:
            continue

        metadata = train_metadata.get(case_id, {})
        cases.append(
            CaseInfo(
                case_id=case_id,
                wsi_path=wsi_path,
                wsi_structure=get_tiff_structure(wsi_path),
                data_provider=metadata.get("data_provider") or None,
                isup_grade=metadata.get("isup_grade") or None,
                gleason_score=metadata.get("gleason_score") or None,
            )
        )

    if max_cases > 0:
        cases = cases[:max_cases]
    if not cases:
        raise RuntimeError("No WSI cases found for tissue-only inference.")
    return cases


def level_downsamples(structure: Dict[str, object]) -> List[float]:
    level_shapes = structure["level_shapes"]
    base_h, base_w = level_shapes[0][0], level_shapes[0][1]
    factors: List[float] = []
    for shape in level_shapes:
        h, w = shape[0], shape[1]
        factors.append(float((base_h / h + base_w / w) / 2.0))
    return factors


def choose_level_for_magnification(
    structure: Dict[str, object],
    source_magnification: float,
    target_magnification: float,
) -> Tuple[int, float]:
    desired_downsample = float(source_magnification) / float(target_magnification)
    factors = level_downsamples(structure)
    level_index = min(
        range(len(factors)),
        key=lambda idx: abs(math.log(factors[idx] + 1e-8) - math.log(desired_downsample + 1e-8)),
    )
    effective_magnification = float(source_magnification) / factors[level_index]
    return level_index, effective_magnification


def load_wsi_level(case: CaseInfo, level_index: int) -> np.ndarray:
    wsi = read_tiff_level(case.wsi_path, level_index)
    if wsi.ndim == 2:
        wsi = np.repeat(wsi[..., None], 3, axis=2)
    elif wsi.ndim == 3 and wsi.shape[0] in {3, 4} and wsi.shape[-1] not in {3, 4}:
        wsi = np.moveaxis(wsi, 0, -1)
    if wsi.ndim == 3 and wsi.shape[-1] == 4:
        wsi = wsi[..., :3]
    if wsi.dtype != np.uint8:
        wsi = normalize_to_uint8(wsi)
    return wsi


def compute_tissue_fraction(image: np.ndarray) -> float:
    image_f = image.astype(np.float32)
    gray = 0.299 * image_f[..., 0] + 0.587 * image_f[..., 1] + 0.114 * image_f[..., 2]
    tissue = gray < 235.0
    return float(tissue.mean())


def enumerate_tissue_tiles(image: np.ndarray, tile_size: int, stride: int, min_tissue: float) -> List[Dict[str, object]]:
    h, w = image.shape[:2]
    tiles: List[Dict[str, object]] = []
    for y in range(0, max(h - tile_size + 1, 1), stride):
        for x in range(0, max(w - tile_size + 1, 1), stride):
            y1 = min(y + tile_size, h)
            x1 = min(x + tile_size, w)
            if (y1 - y) != tile_size or (x1 - x) != tile_size:
                continue

            img_tile = image[y:y1, x:x1]
            tissue_fraction = compute_tissue_fraction(img_tile)
            if tissue_fraction < min_tissue:
                continue

            tiles.append(
                {
                    "x": x,
                    "y": y,
                    "image": img_tile,
                    "tissue_fraction": tissue_fraction,
                }
            )
    return tiles


def preprocess_tile(tile_rgb: np.ndarray) -> torch.Tensor:
    image = tile_rgb.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).float()


def create_gaussian_kernel(size: int, sigma_scale: float = 4.0) -> np.ndarray:
    sigma = float(size) / max(float(sigma_scale), 1e-6)
    coords = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    g1d = np.exp(-(coords**2) / (2.0 * sigma**2))
    kernel = np.outer(g1d, g1d).astype(np.float32)
    kernel /= max(float(kernel.max()), 1e-8)
    return kernel


def save_case_artifacts(
    case_dir: Path,
    image: np.ndarray,
    pred_canvas: np.ndarray,
    coverage: np.ndarray,
    thumb_size: int,
) -> None:
    from PIL import Image

    palette = build_color_palette(np.arange(len(CLASS_NAMES), dtype=np.int64))
    image_thumb = resize_image(image, thumb_size, is_mask=False)
    pred_thumb = resize_image(pred_canvas, thumb_size, is_mask=True)
    cov_thumb = resize_image(coverage, thumb_size, is_mask=True)
    pred_rgb = colorize_mask(pred_thumb, palette)

    Image.fromarray(image_thumb).save(case_dir / "case_thumbnail.png")
    Image.fromarray(pred_canvas, mode="L").save(case_dir / "case_pred_mask_mosaic_class.png")
    np.save(case_dir / "case_pred_mask_mosaic_class.npy", pred_canvas)
    Image.fromarray(pred_rgb).save(case_dir / "case_pred_mask_mosaic.png")
    Image.fromarray(overlay(image_thumb, pred_rgb)).save(case_dir / "case_pred_overlay_mosaic.png")
    Image.fromarray(cov_thumb).save(case_dir / "case_tissue_tile_coverage.png")


def run_case(
    case: CaseInfo,
    model: torch.nn.Module,
    output_dir: Path,
    tile_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    target_magnification: float,
    source_magnification: float,
    tissue_threshold: float,
    thumb_size: int,
    blend_mode: str,
    gaussian_sigma_scale: float,
    min_predicted_tumor_fraction: float,
) -> Dict[str, object]:
    case_start = time.perf_counter()
    timings = {
        "read_level_seconds": 0.0,
        "tile_selection_seconds": 0.0,
        "preprocess_seconds": 0.0,
        "forward_seconds": 0.0,
        "stitch_seconds": 0.0,
        "write_seconds": 0.0,
    }

    level_index, effective_mag = choose_level_for_magnification(
        case.wsi_structure,
        source_magnification,
        target_magnification,
    )
    LOGGER.info("Case %s -> level %s (effective %.2fx)", case.case_id, level_index, effective_mag)

    t0 = time.perf_counter()
    wsi = load_wsi_level(case, level_index)
    timings["read_level_seconds"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    tiles = enumerate_tissue_tiles(
        image=wsi,
        tile_size=tile_size,
        stride=stride,
        min_tissue=tissue_threshold,
    )
    timings["tile_selection_seconds"] = time.perf_counter() - t1
    if not tiles:
        raise RuntimeError(f"No tissue tiles found for case {case.case_id} at level {level_index}")

    case_dir = output_dir / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    h, w = wsi.shape[:2]
    pred_canvas = np.zeros((h, w), dtype=np.uint8)
    coverage = np.zeros((h, w), dtype=np.uint8)
    tile_rows: List[Dict[str, object]] = []

    effective_blend_mode = blend_mode
    if stride >= tile_size and blend_mode == "gaussian_sum":
        effective_blend_mode = "overwrite"

    if effective_blend_mode == "gaussian_sum":
        prob_accumulator = np.zeros((len(CLASS_NAMES), h, w), dtype=np.float16)
        weight_accumulator = np.zeros((h, w), dtype=np.float16)
        gaussian_kernel = create_gaussian_kernel(tile_size, sigma_scale=gaussian_sigma_scale)

    with torch.inference_mode():
        for start in range(0, len(tiles), batch_size):
            batch_tiles = tiles[start : start + batch_size]

            t_pre = time.perf_counter()
            batch_tensor = torch.stack([preprocess_tile(tile["image"]) for tile in batch_tiles])
            batch_tensor = batch_tensor.to(device, non_blocking=True)
            timings["preprocess_seconds"] += time.perf_counter() - t_pre

            if device.type == "cuda":
                torch.cuda.synchronize()
            t_fwd = time.perf_counter()
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            preds = probs.argmax(axis=1).astype(np.uint8)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings["forward_seconds"] += time.perf_counter() - t_fwd

            t_stitch = time.perf_counter()
            for b_idx, tile in enumerate(batch_tiles):
                idx = start + b_idx
                pred_mask = preds[b_idx]
                y0 = int(tile["y"])
                x0 = int(tile["x"])
                y1 = min(y0 + tile_size, h)
                x1 = min(x0 + tile_size, w)
                ph, pw = (y1 - y0), (x1 - x0)

                coverage[y0:y1, x0:x1] = 255
                if effective_blend_mode == "gaussian_sum":
                    kernel_slice = gaussian_kernel[:ph, :pw]
                    weighted = probs[b_idx, :, :ph, :pw] * kernel_slice[None, :, :]
                    prob_accumulator[:, y0:y1, x0:x1] += weighted.astype(np.float16)
                    weight_accumulator[y0:y1, x0:x1] += kernel_slice.astype(np.float16)
                else:
                    pred_canvas[y0:y1, x0:x1] = pred_mask[:ph, :pw]

                pred_tumor_fraction = float((pred_mask > 0).mean())
                counts = np.bincount(pred_mask.reshape(-1), minlength=len(CLASS_NAMES))
                majority_class = int(counts.argmax())
                tile_rows.append(
                    {
                        "case_id": case.case_id,
                        "tile_index": idx,
                        "x": x0,
                        "y": y0,
                        "level_index": level_index,
                        "effective_magnification": effective_mag,
                        "tissue_fraction": float(tile["tissue_fraction"]),
                        "pred_tumor_fraction": pred_tumor_fraction,
                        "majority_class_id": majority_class,
                        "majority_class_name": CLASS_NAMES[majority_class],
                        **{f"pixels_{CLASS_NAMES[class_idx]}": int(counts[class_idx]) for class_idx in range(len(CLASS_NAMES))},
                    }
                )
            timings["stitch_seconds"] += time.perf_counter() - t_stitch

    if effective_blend_mode == "gaussian_sum":
        t_merge = time.perf_counter()
        norm = np.clip(weight_accumulator.astype(np.float32), 1e-8, None)
        pred_canvas = (prob_accumulator.astype(np.float32) / norm[None, :, :]).argmax(axis=0).astype(np.uint8)
        pred_canvas[coverage == 0] = 0
        timings["stitch_seconds"] += time.perf_counter() - t_merge

    t_write = time.perf_counter()
    tile_df = pd.DataFrame(tile_rows)
    tile_df.to_csv(case_dir / "tile_stats.csv", index=False)

    tumor_tiles_df = tile_df[tile_df["pred_tumor_fraction"] >= float(min_predicted_tumor_fraction)].copy()
    tumor_tiles_df = tumor_tiles_df.sort_values(["pred_tumor_fraction", "tissue_fraction"], ascending=False)
    tumor_tiles_df.to_json(case_dir / "predicted_tumor_tiles.json", orient="records", indent=2)

    save_case_artifacts(
        case_dir=case_dir,
        image=wsi,
        pred_canvas=pred_canvas,
        coverage=coverage,
        thumb_size=thumb_size,
    )
    timings["write_seconds"] = time.perf_counter() - t_write

    class_pixel_counts = {
        CLASS_NAMES[class_idx]: int((pred_canvas == class_idx).sum())
        for class_idx in range(len(CLASS_NAMES))
    }
    tumor_pixels = int((pred_canvas > 0).sum())
    total_case_seconds = time.perf_counter() - case_start

    summary = {
        "case_id": case.case_id,
        "wsi_path": str(case.wsi_path),
        "data_provider": case.data_provider,
        "isup_grade": case.isup_grade,
        "gleason_score": case.gleason_score,
        "level_index": level_index,
        "effective_magnification": effective_mag,
        "tile_size": tile_size,
        "stride": stride,
        "blend_mode": effective_blend_mode,
        "num_tissue_tiles": int(len(tile_rows)),
        "num_predicted_tumor_tiles": int(len(tumor_tiles_df)),
        "predicted_tumor_tile_fraction_threshold": float(min_predicted_tumor_fraction),
        "predicted_tumor_pixels": tumor_pixels,
        "predicted_tumor_pixel_fraction": float(tumor_pixels / max(int((coverage > 0).sum()), 1)),
        "predicted_class_pixel_counts": class_pixel_counts,
        "timings_seconds": {
            **{key: round(value, 4) for key, value in timings.items()},
            "total_case_seconds": round(total_case_seconds, 4),
        },
        "throughput": {
            "tiles_per_second_total": round(len(tile_rows) / max(total_case_seconds, 1e-8), 4),
            "tiles_per_second_forward_only": round(len(tile_rows) / max(timings["forward_seconds"], 1e-8), 4),
        },
    }
    (case_dir / "case_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_metadata = load_train_metadata(args.train_csv, args.data_dir)
    cases = discover_cases(
        data_dir=args.data_dir,
        explicit_case_ids=args.case_id,
        max_cases=args.max_cases,
        train_metadata=train_metadata,
    )

    device = resolve_device(args.device)
    checkpoint_path = load_checkpoint_path(args.checkpoint_path)

    t_model = time.perf_counter()
    model = load_model(checkpoint_path, device=device)
    model_load_seconds = time.perf_counter() - t_model

    case_summaries: List[Dict[str, object]] = []
    for case in cases:
        case_summary = run_case(
            case=case,
            model=model,
            output_dir=args.output_dir,
            tile_size=args.tile_size,
            stride=args.stride,
            batch_size=args.batch_size,
            device=device,
            target_magnification=args.target_magnification,
            source_magnification=args.source_magnification,
            tissue_threshold=args.tissue_threshold,
            thumb_size=args.thumb_size,
            blend_mode=args.blend_mode,
            gaussian_sigma_scale=args.gaussian_sigma_scale,
            min_predicted_tumor_fraction=args.min_predicted_tumor_fraction,
        )
        case_summaries.append(case_summary)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    run_summary = {
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "target_magnification": float(args.target_magnification),
        "source_magnification": float(args.source_magnification),
        "tile_size": int(args.tile_size),
        "stride": int(args.stride),
        "batch_size": int(args.batch_size),
        "tissue_threshold": float(args.tissue_threshold),
        "model_load_seconds": round(model_load_seconds, 4),
        "cases": case_summaries,
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    pd.DataFrame(case_summaries).to_csv(args.output_dir / "case_summary.csv", index=False)
    LOGGER.info("Inference written to %s", args.output_dir)


if __name__ == "__main__":
    main()
