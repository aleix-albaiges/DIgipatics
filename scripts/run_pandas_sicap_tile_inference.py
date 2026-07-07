#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inspect_pandas_masks import (  # type: ignore
    SICAP_CLASS_NAMES,
    build_color_palette,
    colorize_mask,
    ensure_runtime_dependencies,
    extract_label_ids,
    infer_pandas_mask_schema,
    normalize_to_uint8,
    overlay,
    pair_cases,
    read_tiff_level,
    resize_image,
    summarize_mask,
    get_tiff_structure,
)
from wsi_resolution import choose_resolution_plan, resize_to_scale
try:
    from training_conch_final import CONCHSegModel
except ModuleNotFoundError:
    from training_conchv2 import CONCHSegModel

LOGGER = logging.getLogger("run_pandas_sicap_tile_inference")
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
CLASS_NAMES = [SICAP_CLASS_NAMES[idx] for idx in range(4)]
RADBOUD_TO_SICAP = {0: 0, 1: 0, 2: 0, 3: 1, 4: 2, 5: 3}
KAROLINSKA_TO_SICAP = {0: 0, 1: 0, 2: 0}
BINARY_CLASS_NAMES = ["NC", "Cancer"]


@dataclass
class CaseInfo:
    case_id: str
    wsi_path: Path
    mask_path: Path
    mask_structure: Dict[str, object]
    wsi_structure: Dict[str, object]
    schema_name: str
    pandas_to_sicap: Dict[int, int]
    can_map_gleason: bool
    observed_labels: List[int]
    data_provider: str
    isup_grade: Optional[str]
    gleason_score: Optional[str]


@dataclass
class TileRecord:
    tile_index: int
    x: int
    y: int
    level_index: int
    tumor_fraction: float
    tissue_fraction: float
    metrics: Dict[str, object]
    gt_mask: np.ndarray
    pred_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SICAPv2 checkpoint inference on PANDAS WSI tiles at target magnification."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("PANDAS"))
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="CSV metadata de PANDA (si no se indica: <data-dir>/train.csv).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pandas_tile_inference"))
    parser.add_argument(
        "--checkpoints-csv",
        type=Path,
        default=Path("artifacts/checkpoints_conch_masklut/best_per_fold.csv"),
        help="CSV con fold + checkpoint_path (p.ej. Gleason 4 clases: artifacts/checkpoints_conch_masklut/best_per_fold.csv).",
    )
    parser.add_argument("--fold", type=str, default=None, help="Fold to use from the CSV. Defaults to best macro_f1.")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Direct checkpoint path override.")
    parser.add_argument("--case-id", type=str, nargs="*", default=None, help="Optional list of specific PANDAS case ids.")
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--max-tiles-per-case", type=int, default=12)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument(
        "--blend-mode",
        type=str,
        default="gaussian_sum",
        choices=("gaussian_sum", "overwrite"),
        help="How to merge overlapped tiles in WSI mosaic.",
    )
    parser.add_argument(
        "--gaussian-sigma-scale",
        type=float,
        default=4.0,
        help="Gaussian sigma = tile_size / gaussian_sigma_scale (only gaussian_sum).",
    )
    parser.add_argument("--target-magnification", type=float, default=10.0)
    parser.add_argument(
        "--target-mpp",
        type=float,
        default=None,
        help="Target microns per pixel. If omitted, target_magnification is converted with 10/mag.",
    )
    parser.add_argument(
        "--source-magnification",
        type=float,
        default=None,
        help="Fallback source magnification only when TIFF resolution tags do not provide mpp.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--thumb-size", type=int, default=1024)
    parser.add_argument("--tissue-threshold", type=float, default=0.15)
    parser.add_argument("--min-tumor-fraction", type=float, default=0.01)
    parser.add_argument(
        "--binary-cancer-mode",
        action="store_true",
        help="Binary mode: NC vs Cancer. GT cancer is derived from Gleason labels.",
    )
    parser.add_argument("--allow-karolinska", action="store_true")
    parser.add_argument(
        "--skip-gleason-geojson",
        action="store_true",
        help="(4-class) No escribir GeoJSON Gleason (pred + GT suavizado).",
    )
    parser.add_argument(
        "--gt-geojson-blur-sigma",
        type=float,
        default=4.0,
        help="Sigma (píxeles a nivel de inferencia) para suavizar el GT antes de contornos. 0 = sin blur.",
    )
    parser.add_argument(
        "--gt-geojson-fill-opacity",
        type=float,
        default=0.35,
        help="Opacidad de relleno sugerida en propiedades GeoJSON del GT (QuPath/visores que la lean).",
    )
    parser.add_argument(
        "--geojson-include-nc",
        action="store_true",
        help="Incluir polígonos de clase NC en GeoJSON multiclase (más pesado). Por defecto solo GG3/GG4/GG5.",
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


def load_checkpoint_row(csv_path: Path, fold: Optional[str]) -> Tuple[Dict[str, object], Path]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"No checkpoints found in {csv_path}")
    if fold:
        selected = df[df["fold"].astype(str) == str(fold)]
        if selected.empty:
            raise ValueError(f"Fold {fold} not found in {csv_path}")
        row = selected.sort_values("macro_f1", ascending=False).iloc[0]
    else:
        row = df.sort_values("macro_f1", ascending=False).iloc[0]
    checkpoint_path = Path(str(row["checkpoint_path"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = (REPO_ROOT / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        alt = REPO_ROOT / "artifacts" / Path(str(row["checkpoint_path"]))
        if alt.exists():
            checkpoint_path = alt.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return row.to_dict(), checkpoint_path


def load_model(checkpoint_path: Path, device: torch.device, num_classes: int) -> torch.nn.Module:
    LOGGER.info("Loading checkpoint %s", checkpoint_path)
    model = CONCHSegModel(num_classes=num_classes, unfreeze_last=0)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_train_metadata(train_csv: Optional[Path], data_dir: Path) -> Dict[str, Dict[str, str]]:
    csv_path = train_csv if train_csv is not None else (data_dir / "train.csv")
    if not csv_path.exists():
        LOGGER.warning("train.csv not found en %s. Se usará only inferencia por valores de mask.", csv_path)
        return {}
    df = pd.read_csv(csv_path)
    if "image_id" not in df.columns:
        LOGGER.warning("%s no contiene column image_id. Will be ignored para filtrado/anotación.", csv_path)
        return {}
    out: Dict[str, Dict[str, str]] = {}
    cols = set(df.columns)
    for _, row in df.iterrows():
        image_id = str(row["image_id"])
        out[image_id] = {
            "data_provider": str(row["data_provider"]) if "data_provider" in cols and pd.notna(row["data_provider"]) else "",
            "isup_grade": str(row["isup_grade"]) if "isup_grade" in cols and pd.notna(row["isup_grade"]) else "",
            "gleason_score": str(row["gleason_score"]) if "gleason_score" in cols and pd.notna(row["gleason_score"]) else "",
        }
    LOGGER.info("Metadata cargada: %s rows desde %s", len(out), csv_path)
    return out


def inspect_case(
    mask_path: Path,
    wsi_path: Path,
    source_magnification: Optional[float],
    target_magnification: float,
    target_mpp: Optional[float],
    train_meta: Optional[Dict[str, str]] = None,
) -> CaseInfo:
    mask_structure = get_tiff_structure(mask_path)
    wsi_structure = get_tiff_structure(wsi_path)
    # Para decidir esquema/clases, we avoid the lowest level (it can lose labels).
    preview_plan = choose_resolution_plan(
        mask_structure,
        target_magnification=target_magnification,
        source_magnification=source_magnification,
        target_mpp=target_mpp,
    )
    preview_level = preview_plan.level_index
    num_mask_levels = int(mask_structure["num_levels"])
    if num_mask_levels > 2:
        preview_level = min(max(preview_level, 1), num_mask_levels - 2)
    mask_preview = read_tiff_level(mask_path, preview_level)
    mask_summary = summarize_mask(mask_preview, max_unique_report=32, exact_decode=False, structure=mask_structure)
    observed_labels = [int(v) for v in mask_summary.get("derived_label_ids", mask_summary.get("palette_values", []))]
    label_ids = extract_label_ids(mask_preview)
    schema_from_pixels, pandas_to_sicap, _, can_map_gleason = infer_pandas_mask_schema(label_ids)

    provider = (train_meta or {}).get("data_provider", "").strip().lower()
    # Para clases por pixel, we prioritize lo que actually exists en la mask.
    schema_name = schema_from_pixels
    if schema_from_pixels == "radboud_pattern_mask":
        pandas_to_sicap = dict(RADBOUD_TO_SICAP)
        can_map_gleason = True
    else:
        pandas_to_sicap = dict(KAROLINSKA_TO_SICAP)
        can_map_gleason = False

    if provider and ((provider == "radboud" and schema_from_pixels != "radboud_pattern_mask") or (provider == "karolinska" and schema_from_pixels != "karolinska_binary_mask")):
        LOGGER.warning(
            "Case %s: mismatch CSV(%s) vs mask(%s). Para segmentación por pixel is prioritized mask.",
            wsi_path.stem,
            provider,
            schema_from_pixels,
        )
    return CaseInfo(
        case_id=wsi_path.stem,
        wsi_path=wsi_path,
        mask_path=mask_path,
        mask_structure=mask_structure,
        wsi_structure=wsi_structure,
        schema_name=schema_name,
        pandas_to_sicap=pandas_to_sicap,
        can_map_gleason=can_map_gleason,
        observed_labels=observed_labels,
        data_provider=provider or "unknown",
        isup_grade=(train_meta or {}).get("isup_grade") or None,
        gleason_score=(train_meta or {}).get("gleason_score") or None,
    )


def discover_cases(
    data_dir: Path,
    explicit_case_ids: Optional[Sequence[str]],
    allow_karolinska: bool,
    max_cases: int,
    source_magnification: Optional[float],
    target_magnification: float,
    target_mpp: Optional[float],
    train_metadata: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[CaseInfo]:
    candidates: List[CaseInfo] = []
    explicit = set(explicit_case_ids or [])
    for pair in pair_cases(data_dir):
        if explicit and pair.case_id not in explicit:
            continue
        if not pair.wsi_path:
            LOGGER.warning("Skipping %s: no matching WSI.", pair.case_id)
            continue
        case = inspect_case(
            Path(pair.mask_path),
            Path(pair.wsi_path),
            source_magnification=source_magnification,
            target_magnification=target_magnification,
            target_mpp=target_mpp,
            train_meta=(train_metadata or {}).get(pair.case_id),
        )
        if not case.can_map_gleason and not allow_karolinska:
            LOGGER.info("Skipping %s: mask schema %s cannot map to SICAP Gleason classes.", case.case_id, case.schema_name)
            continue
        candidates.append(case)

    candidates.sort(key=lambda item: (not item.can_map_gleason, item.case_id))
    if max_cases > 0:
        candidates = candidates[:max_cases]
    if not candidates:
        raise RuntimeError("No PANDAS cases available for SICAP-compatible tile inference.")
    return candidates


def remap_pandas_mask_to_sicap(mask: np.ndarray, pandas_to_sicap: Dict[int, int]) -> np.ndarray:
    label_ids = extract_label_ids(mask)
    remapped = np.zeros(label_ids.shape, dtype=np.uint8)
    for src_value, dst_value in pandas_to_sicap.items():
        remapped[label_ids == int(src_value)] = int(dst_value)
    return remapped


def remap_pandas_mask_to_binary(mask: np.ndarray, schema_name: str) -> np.ndarray:
    """
    Binary target from WSI mask:
    - radboud_pattern_mask: cancer if Gleason pattern label is 3/4/5
    - karolinska_binary_mask: no Gleason-by-pixel available -> all NC
    """
    label_ids = extract_label_ids(mask)
    out = np.zeros(label_ids.shape, dtype=np.uint8)
    if schema_name == "radboud_pattern_mask":
        out[(label_ids == 3) | (label_ids == 4) | (label_ids == 5)] = 1
    return out


def compute_tissue_fraction(image: np.ndarray) -> float:
    image_f = image.astype(np.float32)
    gray = 0.299 * image_f[..., 0] + 0.587 * image_f[..., 1] + 0.114 * image_f[..., 2]
    tissue = gray < 235.0
    return float(tissue.mean())


def enumerate_tiles(image: np.ndarray, mask: np.ndarray, tile_size: int, stride: int, min_tissue: float, min_tumor_fraction: float) -> List[Dict[str, object]]:
    h, w = mask.shape
    tiles: List[Dict[str, object]] = []
    for y in range(0, max(h - tile_size + 1, 1), stride):
        for x in range(0, max(w - tile_size + 1, 1), stride):
            y1 = min(y + tile_size, h)
            x1 = min(x + tile_size, w)
            if (y1 - y) != tile_size or (x1 - x) != tile_size:
                continue
            img_tile = image[y:y1, x:x1]
            mask_tile = mask[y:y1, x:x1]
            tissue_fraction = compute_tissue_fraction(img_tile)
            tumor_fraction = float((mask_tile > 0).mean())
            if tissue_fraction < min_tissue:
                continue
            if tumor_fraction < min_tumor_fraction:
                continue
            tiles.append(
                {
                    "x": x,
                    "y": y,
                    "image": img_tile,
                    "mask": mask_tile,
                    "tissue_fraction": tissue_fraction,
                    "tumor_fraction": tumor_fraction,
                }
            )
    tiles.sort(key=lambda item: (item["tumor_fraction"], item["tissue_fraction"]), reverse=True)
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


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int = 4) -> np.ndarray:
    valid = (target >= 0) & (target < num_classes)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (target[valid], pred[valid]), 1)
    return matrix


def metrics_from_confusion(cm: np.ndarray, class_names: Sequence[str]) -> Dict[str, object]:
    total = cm.sum()
    acc = float(np.trace(cm) / total) if total > 0 else 0.0
    dice_per_class: Dict[str, float] = {}
    iou_per_class: Dict[str, float] = {}
    for idx, name in enumerate(class_names):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        denom_dice = 2 * tp + fp + fn
        denom_iou = tp + fp + fn
        dice_per_class[name] = float((2 * tp) / denom_dice) if denom_dice > 0 else float("nan")
        iou_per_class[name] = float(tp / denom_iou) if denom_iou > 0 else float("nan")
    macro_dice = float(np.nanmean(list(dice_per_class.values())))
    macro_iou = float(np.nanmean(list(iou_per_class.values())))
    return {
        "pixel_accuracy": acc,
        "dice_per_class": dice_per_class,
        "iou_per_class": iou_per_class,
        "macro_dice": macro_dice,
        "macro_iou": macro_iou,
        "confusion_matrix": cm.tolist(),
    }


def binary_mask_to_geojson(mask: np.ndarray, level_downsample: float, case_id: str) -> Dict[str, object]:
    """
    Convert a binary mask (0=NC, 1=Cancer) into GeoJSON polygons.
    The output coordinates are scaled to level-0 WSI pixels.
    """
    if mask.ndim != 2:
        raise ValueError("binary_mask_to_geojson expects a 2D mask.")

    positive = (mask == 1).astype(np.uint8)
    h, w = positive.shape
    if h == 0 or w == 0 or int(positive.sum()) == 0:
        return {"type": "FeatureCollection", "features": []}

    # Clean tiny artifacts and thin gaps to improve polygon quality.
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(positive, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Pixel-level contours (no polygon simplification) to avoid boxy annotations.
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return {"type": "FeatureCollection", "features": []}

    min_area_px = 64.0
    scale = float(level_downsample)
    features: List[Dict[str, object]] = []
    feature_id = 0
    for contour_idx, contour in enumerate(contours):
        area = float(cv2.contourArea(contour))
        if area < min_area_px:
            continue

        if contour.shape[0] < 3:
            continue

        shell_ring = [[float(pt[0][0]) * scale, float(pt[0][1]) * scale] for pt in contour]
        if shell_ring[0] != shell_ring[-1]:
            shell_ring.append(shell_ring[0])

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "case_id": case_id,
                    "class_id": 1,
                    "class_name": "Cancer",
                    "feature_id": feature_id,
                },
                "geometry": {"type": "Polygon", "coordinates": [shell_ring]},
            }
        )
        feature_id += 1

    return {"type": "FeatureCollection", "features": features}


def multiclass_mask_to_geojson(
    mask: np.ndarray,
    level_downsample: float,
    case_id: str,
    class_names: Sequence[str],
    *,
    skip_nc: bool,
    blur_sigma: float,
    fill_opacity: Optional[float],
    is_ground_truth: bool,
) -> Dict[str, object]:
    """
    Exporta polígonos por clase (Gleason SICAP: NC, GG3, GG4, GG5).
    Para GT difuminado: blur_sigma > 0 suaviza bordes antes de extraer contornos;
    fill_opacity en propiedades para visores/QuPath que respeten opacidad en GeoJSON.
    """
    if mask.ndim != 2:
        raise ValueError("multiclass_mask_to_geojson expects a 2D mask.")

    num_classes = len(class_names)
    scale = float(level_downsample)
    features: List[Dict[str, object]] = []
    feature_id = 0

    class_indices = list(range(num_classes))
    if skip_nc:
        class_indices = [i for i in class_indices if i != 0]

    ksize = 3
    if blur_sigma > 0:
        ksize = max(3, int(6 * blur_sigma + 1) | 1)

    kernel = np.ones((3, 3), dtype=np.uint8)
    min_area_px = 64.0

    for class_id in class_indices:
        binary = (mask == class_id).astype(np.uint8)
        if int(binary.sum()) == 0:
            continue

        if blur_sigma > 0:
            blurred = cv2.GaussianBlur(binary.astype(np.float32), (ksize, ksize), float(blur_sigma))
            binary = (blurred >= 0.5).astype(np.uint8)
            if int(binary.sum()) == 0:
                continue

        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area_px:
                continue
            if contour.shape[0] < 3:
                continue
            shell_ring = [[float(pt[0][0]) * scale, float(pt[0][1]) * scale] for pt in contour]
            if shell_ring[0] != shell_ring[-1]:
                shell_ring.append(shell_ring[0])

            props: Dict[str, object] = {
                "case_id": case_id,
                "class_id": int(class_id),
                "class_name": class_names[class_id],
                "feature_id": feature_id,
                "is_ground_truth": bool(is_ground_truth),
            }
            if fill_opacity is not None:
                props["fillOpacity"] = float(fill_opacity)
                props["strokeOpacity"] = float(min(1.0, fill_opacity + 0.25))

            features.append(
                {
                    "type": "Feature",
                    "properties": props,
                    "geometry": {"type": "Polygon", "coordinates": [shell_ring]},
                }
            )
            feature_id += 1

    return {"type": "FeatureCollection", "features": features}


def save_gleason_geojson_pair(
    case_dir: Path,
    case_id: str,
    pred_canvas: np.ndarray,
    gt_canvas: np.ndarray,
    level_downsample: float,
    class_names: Sequence[str],
    *,
    skip_nc: bool,
    gt_blur_sigma: float,
    gt_fill_opacity: float,
) -> Tuple[str, str]:
    pred_geojson = multiclass_mask_to_geojson(
        pred_canvas,
        level_downsample,
        case_id,
        class_names,
        skip_nc=skip_nc,
        blur_sigma=0.0,
        fill_opacity=None,
        is_ground_truth=False,
    )
    pred_name = "case_pred_gleason_annotations.geojson"
    (case_dir / pred_name).write_text(json.dumps(pred_geojson, indent=2), encoding="utf-8")

    gt_geojson = multiclass_mask_to_geojson(
        gt_canvas,
        level_downsample,
        case_id,
        class_names,
        skip_nc=skip_nc,
        blur_sigma=gt_blur_sigma,
        fill_opacity=gt_fill_opacity,
        is_ground_truth=True,
    )
    gt_name = "case_gt_gleason_annotations_soft.geojson"
    (case_dir / gt_name).write_text(json.dumps(gt_geojson, indent=2), encoding="utf-8")

    return pred_name, gt_name


def save_binary_prediction_geojson(case_dir: Path, case_id: str, pred_canvas: np.ndarray, level_downsample: float) -> str:
    geojson = binary_mask_to_geojson(pred_canvas, level_downsample=level_downsample, case_id=case_id)
    output_name = "case_pred_binary_annotations.geojson"
    output_path = case_dir / output_name
    output_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
    return output_name


def batched_inference(model: torch.nn.Module, images: Sequence[np.ndarray], device: torch.device, batch_size: int) -> List[np.ndarray]:
    outputs: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            tensor = torch.stack([preprocess_tile(tile) for tile in batch]).to(device, non_blocking=True)
            logits = model(tensor)
            preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            outputs.extend([preds[idx] for idx in range(preds.shape[0])])
    return outputs


def save_tile_visuals(case_dir: Path, image_tile: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, tile_name: str) -> Dict[str, object]:
    from PIL import Image
    import matplotlib.pyplot as plt

    classes_present = np.unique(np.concatenate([gt_mask.reshape(-1), pred_mask.reshape(-1)]))
    palette = build_color_palette(classes_present.astype(np.int64))
    gt_rgb = colorize_mask(gt_mask, palette)
    pred_rgb = colorize_mask(pred_mask, palette)
    err_mask = (gt_mask != pred_mask).astype(np.uint8)
    err_rgb = np.zeros((*err_mask.shape, 3), dtype=np.uint8)
    err_rgb[err_mask == 1] = np.asarray((255, 0, 0), dtype=np.uint8)

    Image.fromarray(image_tile).save(case_dir / f"{tile_name}_image.png")
    Image.fromarray(gt_rgb).save(case_dir / f"{tile_name}_gt_mask.png")
    Image.fromarray(pred_rgb).save(case_dir / f"{tile_name}_pred_mask.png")
    Image.fromarray(overlay(image_tile, gt_rgb)).save(case_dir / f"{tile_name}_gt_overlay.png")
    Image.fromarray(overlay(image_tile, pred_rgb)).save(case_dir / f"{tile_name}_pred_overlay.png")
    Image.fromarray(overlay(image_tile, err_rgb, alpha=0.35)).save(case_dir / f"{tile_name}_error_overlay.png")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image_tile)
    axes[0].set_title("Image")
    axes[1].imshow(overlay(image_tile, gt_rgb))
    axes[1].set_title("GT")
    axes[2].imshow(overlay(image_tile, pred_rgb))
    axes[2].set_title("Prediction")
    axes[3].imshow(overlay(image_tile, err_rgb, alpha=0.35))
    axes[3].set_title("Error")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(case_dir / f"{tile_name}_panel.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "image": f"{tile_name}_image.png",
        "gt_mask": f"{tile_name}_gt_mask.png",
        "pred_mask": f"{tile_name}_pred_mask.png",
        "gt_overlay": f"{tile_name}_gt_overlay.png",
        "pred_overlay": f"{tile_name}_pred_overlay.png",
        "error_overlay": f"{tile_name}_error_overlay.png",
        "panel": f"{tile_name}_panel.png",
    }


def save_case_overview(
    case_dir: Path,
    image: np.ndarray,
    mask: np.ndarray,
    pred_tiles: Sequence[TileRecord],
    thumb_size: int,
    pred_canvas: Optional[np.ndarray] = None,
    gt_canvas: Optional[np.ndarray] = None,
    coverage: Optional[np.ndarray] = None,
) -> None:
    from PIL import Image

    palette = build_color_palette(np.unique(np.concatenate([mask.reshape(-1), pred_canvas.reshape(-1)])).astype(np.int64) if pred_canvas is not None else np.unique(mask).astype(np.int64))
    image_thumb = resize_image(image, thumb_size, is_mask=False)
    mask_thumb = resize_image(mask, thumb_size, is_mask=True)
    mask_rgb = colorize_mask(mask_thumb, palette)
    Image.fromarray(image_thumb).save(case_dir / "case_thumbnail.png")
    Image.fromarray(mask_rgb).save(case_dir / "case_gt_mask.png")
    Image.fromarray(overlay(image_thumb, mask_rgb)).save(case_dir / "case_gt_overlay.png")

    # If not provided, fallback to overwrite stitching from per-tile predictions.
    if pred_canvas is None or gt_canvas is None or coverage is None:
        h, w = mask.shape
        pred_canvas = np.zeros((h, w), dtype=np.uint8)
        gt_canvas = np.zeros((h, w), dtype=np.uint8)
        coverage = np.zeros((h, w), dtype=np.uint8)
        for tile in pred_tiles:
            th, tw = tile.pred_mask.shape
            y0, x0 = tile.y, tile.x
            y1, x1 = min(y0 + th, h), min(x0 + tw, w)
            pred_canvas[y0:y1, x0:x1] = tile.pred_mask[: y1 - y0, : x1 - x0]
            gt_canvas[y0:y1, x0:x1] = tile.gt_mask[: y1 - y0, : x1 - x0]
            coverage[y0:y1, x0:x1] = 255

    np.save(case_dir / "case_pred_mask_mosaic_class.npy", pred_canvas)
    np.save(case_dir / "case_gt_mask_mosaic_class.npy", gt_canvas)
    Image.fromarray(pred_canvas, mode="L").save(case_dir / "case_pred_mask_mosaic_class.png")
    Image.fromarray(gt_canvas, mode="L").save(case_dir / "case_gt_mask_mosaic_class.png")

    pred_thumb = resize_image(pred_canvas, thumb_size, is_mask=True)
    gt_thumb = resize_image(gt_canvas, thumb_size, is_mask=True)
    cov_thumb = resize_image(coverage, thumb_size, is_mask=True)
    pred_rgb = colorize_mask(pred_thumb, palette)
    gt_sel_rgb = colorize_mask(gt_thumb, palette)
    Image.fromarray(pred_rgb).save(case_dir / "case_pred_mask_mosaic.png")
    Image.fromarray(gt_sel_rgb).save(case_dir / "case_gt_mask_mosaic.png")
    Image.fromarray(overlay(image_thumb, pred_rgb)).save(case_dir / "case_pred_overlay_mosaic.png")
    Image.fromarray(overlay(image_thumb, gt_sel_rgb)).save(case_dir / "case_gt_overlay_mosaic.png")
    Image.fromarray(cov_thumb).save(case_dir / "case_tile_coverage_mosaic.png")
    summary = [
        {
            "tile_index": tile.tile_index,
            "x": tile.x,
            "y": tile.y,
            "level_index": tile.level_index,
            "tumor_fraction": tile.tumor_fraction,
            "tissue_fraction": tile.tissue_fraction,
        }
        for tile in pred_tiles
    ]
    (case_dir / "selected_tiles.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_case(
    case: CaseInfo,
    model: torch.nn.Module,
    output_dir: Path,
    tile_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    target_magnification: float,
    source_magnification: Optional[float],
    target_mpp: Optional[float],
    max_tiles_per_case: int,
    tissue_threshold: float,
    min_tumor_fraction: float,
    thumb_size: int,
    blend_mode: str,
    gaussian_sigma_scale: float,
    num_classes: int,
    class_names: Sequence[str],
    binary_cancer_mode: bool,
    export_gleason_geojson: bool,
    gt_geojson_blur_sigma: float,
    gt_geojson_fill_opacity: float,
    geojson_skip_nc: bool,
) -> Dict[str, object]:
    plan = choose_resolution_plan(
        case.wsi_structure,
        target_magnification=target_magnification,
        source_magnification=source_magnification,
        target_mpp=target_mpp,
    )
    LOGGER.info(
        "Case %s -> level %s, read_scale %.3f, output %.2fx (%.3f mpp), source=%s",
        case.case_id,
        plan.level_index,
        plan.read_scale,
        plan.output_magnification or target_magnification,
        plan.target_mpp or 0.0,
        plan.source,
    )
    wsi = read_tiff_level(case.wsi_path, plan.level_index)
    mask = read_tiff_level(case.mask_path, plan.level_index)
    if wsi.ndim == 2:
        wsi = np.repeat(wsi[..., None], 3, axis=2)
    elif wsi.ndim == 3 and wsi.shape[0] in {3, 4} and wsi.shape[-1] not in {3, 4}:
        wsi = np.moveaxis(wsi, 0, -1)
    if wsi.ndim == 3 and wsi.shape[-1] == 4:
        wsi = wsi[..., :3]
    if wsi.dtype != np.uint8:
        wsi = normalize_to_uint8(wsi)
    wsi = resize_to_scale(wsi, plan.read_scale, is_mask=False)
    if mask.ndim == 3 and mask.shape[0] in {1, 3, 4} and mask.shape[-1] not in {1, 3, 4}:
        mask = np.moveaxis(mask, 0, -1)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if binary_cancer_mode:
        mapped_mask = remap_pandas_mask_to_binary(mask, case.schema_name)
    else:
        mapped_mask = remap_pandas_mask_to_sicap(mask, case.pandas_to_sicap)
    mapped_mask = resize_to_scale(mapped_mask, plan.read_scale, is_mask=True)
    if mapped_mask.shape[:2] != wsi.shape[:2]:
        mapped_mask = cv2.resize(mapped_mask, (wsi.shape[1], wsi.shape[0]), interpolation=cv2.INTER_NEAREST)

    tiles = enumerate_tiles(
        image=wsi,
        mask=mapped_mask,
        tile_size=tile_size,
        stride=stride,
        min_tissue=tissue_threshold,
        min_tumor_fraction=min_tumor_fraction,
    )[:max_tiles_per_case]
    if not tiles:
        raise RuntimeError(f"No valid tiles found for case {case.case_id} at level {plan.level_index}")

    case_dir = output_dir / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    tile_rows: List[Dict[str, object]] = []
    aggregate_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    tile_records: List[TileRecord] = []
    h, w = mapped_mask.shape
    pred_canvas = np.zeros((h, w), dtype=np.uint8)
    gt_canvas = np.zeros((h, w), dtype=np.uint8)
    coverage = np.zeros((h, w), dtype=np.uint8)

    effective_blend_mode = blend_mode
    if stride >= tile_size and blend_mode == "gaussian_sum":
        effective_blend_mode = "overwrite"
    if effective_blend_mode == "gaussian_sum":
        prob_accumulator = np.zeros((num_classes, h, w), dtype=np.float32)
        weight_accumulator = np.zeros((h, w), dtype=np.float32)
        gaussian_kernel = create_gaussian_kernel(tile_size, sigma_scale=gaussian_sigma_scale)

    with torch.inference_mode():
        for start in range(0, len(tiles), batch_size):
            batch_tiles = tiles[start : start + batch_size]
            batch_tensor = torch.stack([preprocess_tile(t["image"]) for t in batch_tiles]).to(device, non_blocking=True)
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)  # [B, C, H, W]
            preds = probs.argmax(axis=1).astype(np.uint8)  # [B, H, W]

            for b_idx, tile in enumerate(batch_tiles):
                idx = start + b_idx
                pred_mask = preds[b_idx]
                gt_mask = tile["mask"].astype(np.uint8)
                y0 = int(tile["y"])
                x0 = int(tile["x"])
                y1 = min(y0 + tile_size, h)
                x1 = min(x0 + tile_size, w)
                ph, pw = (y1 - y0), (x1 - x0)
                coverage[y0:y1, x0:x1] = 255
                gt_canvas[y0:y1, x0:x1] = gt_mask[:ph, :pw]

                if effective_blend_mode == "gaussian_sum":
                    kernel_slice = gaussian_kernel[:ph, :pw]
                    prob_accumulator[:, y0:y1, x0:x1] += probs[b_idx, :, :ph, :pw] * kernel_slice[None, :, :]
                    weight_accumulator[y0:y1, x0:x1] += kernel_slice
                else:
                    pred_canvas[y0:y1, x0:x1] = pred_mask[:ph, :pw]

                cm = confusion_matrix(pred_mask, gt_mask, num_classes=num_classes)
                metrics = metrics_from_confusion(cm, class_names=class_names)
                aggregate_cm += cm
                tile_name = f"tile_{idx:03d}_x{tile['x']}_y{tile['y']}"
                paths = save_tile_visuals(case_dir, tile["image"], gt_mask, pred_mask, tile_name)
                row = {
                    "case_id": case.case_id,
                    "tile_index": idx,
                    "x": int(tile["x"]),
                    "y": int(tile["y"]),
                    "level_index": plan.level_index,
                    "level_downsample": plan.level_downsample,
                    "output_downsample": plan.output_downsample,
                    "read_scale": plan.read_scale,
                    "effective_magnification": plan.output_magnification,
                    "tumor_fraction": float(tile["tumor_fraction"]),
                    "tissue_fraction": float(tile["tissue_fraction"]),
                    **{f"dice_{name}": metrics["dice_per_class"][name] for name in class_names},
                    **{f"iou_{name}": metrics["iou_per_class"][name] for name in class_names},
                    "pixel_accuracy": metrics["pixel_accuracy"],
                    "macro_dice": metrics["macro_dice"],
                    "macro_iou": metrics["macro_iou"],
                    **paths,
                }
                tile_rows.append(row)
                tile_records.append(
                    TileRecord(
                        tile_index=idx,
                        x=int(tile["x"]),
                        y=int(tile["y"]),
                        level_index=plan.level_index,
                        tumor_fraction=float(tile["tumor_fraction"]),
                        tissue_fraction=float(tile["tissue_fraction"]),
                        metrics=metrics,
                        gt_mask=gt_mask.copy(),
                        pred_mask=pred_mask.copy(),
                    )
                )

    if effective_blend_mode == "gaussian_sum":
        norm = np.clip(weight_accumulator, 1e-8, None)
        pred_canvas = (prob_accumulator / norm[None, :, :]).argmax(axis=0).astype(np.uint8)
        pred_canvas[coverage == 0] = 0

    pd.DataFrame(tile_rows).to_csv(case_dir / "tile_metrics.csv", index=False)
    aggregate_metrics = metrics_from_confusion(aggregate_cm, class_names=class_names)
    save_case_overview(
        case_dir,
        wsi,
        mapped_mask,
        tile_records,
        thumb_size=thumb_size,
        pred_canvas=pred_canvas,
        gt_canvas=gt_canvas,
        coverage=coverage,
    )
    output_downsample = plan.output_downsample

    binary_geojson_name: Optional[str] = None
    pred_gleason_geojson_name: Optional[str] = None
    gt_gleason_geojson_name: Optional[str] = None

    if binary_cancer_mode:
        binary_geojson_name = save_binary_prediction_geojson(
            case_dir=case_dir,
            case_id=case.case_id,
            pred_canvas=pred_canvas,
            level_downsample=output_downsample,
        )
    elif export_gleason_geojson:
        pred_gleason_geojson_name, gt_gleason_geojson_name = save_gleason_geojson_pair(
            case_dir=case_dir,
            case_id=case.case_id,
            pred_canvas=pred_canvas,
            gt_canvas=gt_canvas,
            level_downsample=output_downsample,
            class_names=class_names,
            skip_nc=geojson_skip_nc,
            gt_blur_sigma=gt_geojson_blur_sigma,
            gt_fill_opacity=gt_geojson_fill_opacity,
        )

    summary = {
        "case_id": case.case_id,
        "data_provider": case.data_provider,
        "isup_grade": case.isup_grade,
        "gleason_score": case.gleason_score,
        "schema_name": case.schema_name,
        "binary_cancer_mode": bool(binary_cancer_mode),
        "observed_labels": case.observed_labels,
        **plan.as_dict(),
        "effective_magnification": plan.output_magnification,
        "num_tiles": len(tile_rows),
        "blend_mode": effective_blend_mode,
        "aggregate_metrics": aggregate_metrics,
        "binary_prediction_geojson": binary_geojson_name,
        "pred_gleason_geojson": pred_gleason_geojson_name,
        "gt_gleason_geojson_soft": gt_gleason_geojson_name,
    }
    (case_dir / "case_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    ensure_runtime_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    if args.checkpoint_path is not None:
        checkpoint_row = {"fold": "manual", "macro_f1": None, "checkpoint_path": str(args.checkpoint_path)}
        checkpoint_path = args.checkpoint_path.resolve()
    else:
        checkpoint_row, checkpoint_path = load_checkpoint_row(args.checkpoints_csv, args.fold)

    train_metadata = load_train_metadata(args.train_csv, args.data_dir)
    cases = discover_cases(
        data_dir=args.data_dir,
        explicit_case_ids=args.case_id,
        allow_karolinska=args.allow_karolinska,
        max_cases=args.max_cases,
        source_magnification=args.source_magnification,
        target_magnification=args.target_magnification,
        target_mpp=args.target_mpp,
        train_metadata=train_metadata,
    )
    num_classes = 2 if args.binary_cancer_mode else 4
    class_names = BINARY_CLASS_NAMES if args.binary_cancer_mode else CLASS_NAMES
    model = load_model(checkpoint_path, device=device, num_classes=num_classes)

    case_summaries: List[Dict[str, object]] = []
    for case in cases:
        try:
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
                target_mpp=args.target_mpp,
                max_tiles_per_case=args.max_tiles_per_case,
                tissue_threshold=args.tissue_threshold,
                min_tumor_fraction=args.min_tumor_fraction,
                thumb_size=args.thumb_size,
                blend_mode=args.blend_mode,
                gaussian_sigma_scale=args.gaussian_sigma_scale,
                num_classes=num_classes,
                class_names=class_names,
                binary_cancer_mode=args.binary_cancer_mode,
                export_gleason_geojson=not args.skip_gleason_geojson,
                gt_geojson_blur_sigma=args.gt_geojson_blur_sigma,
                gt_geojson_fill_opacity=args.gt_geojson_fill_opacity,
                geojson_skip_nc=not args.geojson_include_nc,
            )
            case_summaries.append(case_summary)
        except RuntimeError as exc:
            LOGGER.warning("Skipping case %s: %s", case.case_id, exc)
            continue

    run_summary = {
        "checkpoint": checkpoint_row,
        "resolved_checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "target_magnification": float(args.target_magnification),
        "target_mpp": None if args.target_mpp is None else float(args.target_mpp),
        "source_magnification": None if args.source_magnification is None else float(args.source_magnification),
        "tile_size": int(args.tile_size),
        "stride": int(args.stride),
        "batch_size": int(args.batch_size),
        "cases": case_summaries,
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    pd.DataFrame(case_summaries).to_csv(args.output_dir / "case_summary.csv", index=False)
    LOGGER.info("Inference written to %s", args.output_dir)


if __name__ == "__main__":
    main()
