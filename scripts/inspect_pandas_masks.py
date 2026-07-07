#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("inspect_pandas_masks")
MAX_FULL_DECODE_PIXELS = 64_000_000
FIXED_LABEL_COLORS = {
    0: (0, 0, 0),
    1: (46, 204, 113),
    2: (241, 196, 15),
    3: (230, 126, 34),
    4: (231, 76, 60),
}
SICAP_CLASS_NAMES = {
    0: "NC",
    1: "GG3",
    2: "GG4",
    3: "GG5",
}


@dataclass
class PairRecord:
    case_id: str
    wsi_path: Optional[str]
    mask_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect and visualize PANDAS TIFF masks.")
    parser.add_argument("--data-dir", type=Path, default=Path("PANDAS"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pandas_inspection"))
    parser.add_argument("--thumb-size", type=int, default=1536)
    parser.add_argument("--max-unique-report", type=int, default=32)
    parser.add_argument("--log-level", type=str, default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s | %(levelname)s | %(message)s")


def ensure_runtime_dependencies() -> None:
    missing: List[str] = []
    for module_name in ("tifffile", "PIL", "matplotlib", "imagecodecs"):
        try:
            __import__(module_name)
        except Exception:
            missing.append(module_name)
    if missing:
        raise RuntimeError(
            "Missing Python packages: {}. Install them in the active environment before running this script.".format(
                ", ".join(missing)
            )
        )


def open_tiff(path: Path):
    import tifffile

    try:
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            image = series.asarray(maxworkers=1)
    except ValueError as exc:
        message = str(exc)
        if "requires the 'imagecodecs' package" in message:
            raise RuntimeError(
                f"Cannot read TIFF {path.name}: compression requires the 'imagecodecs' package in the active environment."
            ) from exc
        raise
    return image


def _tag_value(page, tag_name: str):
    if tag_name not in page.tags:
        return None
    value = page.tags[tag_name].value
    return getattr(value, "value", value)


def _rational_to_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2:
        numerator, denominator = value
        denominator = float(denominator)
        if denominator == 0:
            return None
        return float(numerator) / denominator
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolution_unit_um(value) -> Optional[float]:
    if value is None:
        return None
    try:
        unit = int(getattr(value, "value", value))
    except (TypeError, ValueError):
        return None
    if unit == 2:
        return 25400.0
    if unit == 3:
        return 10000.0
    return None


def _page_mpp(page) -> Dict[str, Optional[float]]:
    x_resolution = _rational_to_float(_tag_value(page, "XResolution"))
    y_resolution = _rational_to_float(_tag_value(page, "YResolution"))
    unit_um = _resolution_unit_um(_tag_value(page, "ResolutionUnit"))

    mpp_x = unit_um / x_resolution if unit_um is not None and x_resolution else None
    mpp_y = unit_um / y_resolution if unit_um is not None and y_resolution else None
    mpp_values = [value for value in (mpp_x, mpp_y) if value is not None and value > 0]
    mpp = float(sum(mpp_values) / len(mpp_values)) if mpp_values else None
    return {
        "x_resolution": x_resolution,
        "y_resolution": y_resolution,
        "mpp_x": mpp_x,
        "mpp_y": mpp_y,
        "mpp": mpp,
    }


def _serializable_resolution_unit(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def get_tiff_structure(path: Path) -> Dict[str, object]:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        levels = getattr(series, "levels", [series])
        page0 = series.pages[0]
        resolution_unit = _tag_value(page0, "ResolutionUnit")
        return {
            "series_shape": list(series.shape),
            "series_dtype": str(series.dtype),
            "num_levels": int(len(levels)),
            "level_shapes": [list(level.shape) for level in levels],
            "compression": str(getattr(page0, "compression", "unknown")),
            "is_tiled": bool(getattr(page0, "is_tiled", False)),
            "resolution_unit": _serializable_resolution_unit(resolution_unit),
            **_page_mpp(page0),
        }


def read_tiff_level(path: Path, level_index: int):
    import tifffile

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        levels = getattr(series, "levels", [series])
        level = levels[min(level_index, len(levels) - 1)]
        try:
            return level.asarray(maxworkers=1)
        except ValueError as exc:
            message = str(exc)
            if "requires the 'imagecodecs' package" in message:
                raise RuntimeError(
                    f"Cannot read TIFF {path.name}: compression requires the 'imagecodecs' package in the active environment."
                ) from exc
            raise


def load_tiff_preview(path: Path, thumb_size: int):
    structure = get_tiff_structure(path)
    level_shapes = structure["level_shapes"]
    if structure["num_levels"] > 1:
        preview_level = int(structure["num_levels"]) - 1
        LOGGER.debug("Using lowest pyramid level %s for %s", preview_level, path.name)
        image = read_tiff_level(path, preview_level)
        return image, structure, False

    base_shape = level_shapes[0]
    if len(base_shape) >= 2 and int(base_shape[0]) * int(base_shape[1]) > MAX_FULL_DECODE_PIXELS:
        raise RuntimeError(
            f"{path.name} has no pyramid levels and is too large to decode fully for preview "
            f"({base_shape[0]}x{base_shape[1]} pixels)."
        )
    image = open_tiff(path)
    return image, structure, True


def load_rgb_image(path: Path, thumb_size: int) -> Tuple[np.ndarray, Dict[str, object], bool]:
    image, structure, exact_decode = load_tiff_preview(path, thumb_size)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[0] in {3, 4} and image.shape[-1] not in {3, 4}:
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = normalize_to_uint8(image)
    return image, structure, exact_decode


def load_mask(path: Path, thumb_size: int) -> Tuple[np.ndarray, Dict[str, object], bool]:
    mask, structure, exact_decode = load_tiff_preview(path, thumb_size)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    return mask, structure, exact_decode


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    arr = array.astype(np.float32)
    arr -= arr.min()
    max_value = arr.max()
    if max_value > 0:
        arr = arr / max_value
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def resize_image(image: np.ndarray, thumb_size: int, is_mask: bool = False) -> np.ndarray:
    import cv2

    h, w = image.shape[:2]
    scale = min(thumb_size / max(h, w), 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def build_color_palette(values: np.ndarray) -> Dict[int, Tuple[int, int, int]]:
    palette: Dict[int, Tuple[int, int, int]] = {}
    for idx, value in enumerate(values.tolist()):
        label = int(value)
        if label in FIXED_LABEL_COLORS:
            palette[label] = FIXED_LABEL_COLORS[label]
        else:
            fallback = [
                (52, 152, 219),
                (155, 89, 182),
                (26, 188, 156),
                (149, 165, 166),
            ]
            palette[label] = fallback[idx % len(fallback)]
    return palette


def extract_label_ids(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        return mask.astype(np.int64)
    if mask.ndim == 3 and mask.shape[-1] >= 1:
        if np.all(mask[..., 1:] == 0):
            return mask[..., 0].astype(np.int64)
        # Fallback for genuinely color-coded masks
        flat = mask.reshape(-1, mask.shape[-1])
        unique_colors = np.unique(flat, axis=0)
        color_to_id = {tuple(color.tolist()): idx for idx, color in enumerate(unique_colors)}
        out = np.zeros(mask.shape[:2], dtype=np.int64)
        for color, idx in color_to_id.items():
            color_arr = np.asarray(color, dtype=mask.dtype)
            out[np.all(mask == color_arr, axis=-1)] = idx
        return out
    raise ValueError(f"Unsupported mask shape: {mask.shape}")


def infer_pandas_mask_schema(label_ids: np.ndarray) -> Tuple[str, Dict[int, int], Dict[int, str], bool]:
    observed = {int(v) for v in np.unique(label_ids).tolist()}
    if any(v > 2 for v in observed):
        # Radboud pattern-level pixel mask
        pandas_to_sicap = {
            0: 0,  # background / unknown -> NC for training convenience
            1: 0,  # stroma -> NC
            2: 0,  # benign epithelium -> NC
            3: 1,  # Gleason 3 -> GG3
            4: 2,  # Gleason 4 -> GG4
            5: 3,  # Gleason 5 -> GG5
        }
        pandas_meaning = {
            0: "background_or_unknown",
            1: "stroma",
            2: "benign_epithelium",
            3: "gleason_3",
            4: "gleason_4",
            5: "gleason_5",
        }
        return "radboud_pattern_mask", pandas_to_sicap, pandas_meaning, True

    # Karolinska-style masks cannot recover per-pattern Gleason classes
    pandas_to_sicap = {
        0: 0,  # background / unknown -> NC
        1: 0,  # benign tissue -> NC
        2: 0,  # cancer tissue but no per-pattern Gleason
    }
    pandas_meaning = {
        0: "background_or_unknown",
        1: "benign_tissue",
        2: "cancer_tissue_no_pattern",
    }
    return "karolinska_binary_mask", pandas_to_sicap, pandas_meaning, False


def colorize_mask(mask: np.ndarray, palette: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for value, color in palette.items():
        out[mask == value] = np.asarray(color, dtype=np.uint8)
    return out


def overlay(image: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return np.clip(image.astype(np.float32) * (1.0 - alpha) + mask_rgb.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def summarize_mask(mask: np.ndarray, max_unique_report: int, exact_decode: bool, structure: Dict[str, object]) -> Dict[str, object]:
    label_ids = extract_label_ids(mask)
    schema_name, pandas_to_sicap, pandas_meaning, can_map_gleason = infer_pandas_mask_schema(label_ids)
    observed_labels = [int(v) for v in np.unique(label_ids).tolist()]
    observed_sicap_ids = sorted({int(pandas_to_sicap[v]) for v in observed_labels if v in pandas_to_sicap})
    observed_sicap_names = [SICAP_CLASS_NAMES[idx] for idx in observed_sicap_ids]
    if mask.ndim == 2:
        unique, counts = np.unique(mask, return_counts=True)
        values = unique.tolist()
        report = [{"value": int(v), "count": int(c)} for v, c in zip(unique[:max_unique_report], counts[:max_unique_report])]
        return {
            "mask_type": "single_channel_pixel_mask",
            "shape": list(mask.shape),
            "dtype": str(mask.dtype),
            "exact_decode": bool(exact_decode),
            "num_unique_values": int(unique.size),
            "unique_values_report": report,
            "all_unique_values_fit_in_report": bool(unique.size <= max_unique_report),
            "palette_values": values,
            "pandas_mask_schema": schema_name,
            "pandas_label_meaning": {str(k): v for k, v in pandas_meaning.items()},
            "pandas_to_sicap_mapping": {str(k): int(v) for k, v in pandas_to_sicap.items()},
            "observed_sicap_ids": observed_sicap_ids,
            "observed_sicap_names": observed_sicap_names,
            "can_map_to_sicap_gleason_patterns": bool(can_map_gleason),
            "tiff_structure": structure,
        }
    flat = mask.reshape(-1, mask.shape[-1])
    unique_colors, counts = np.unique(flat, axis=0, return_counts=True)
    report = [
        {"rgb": [int(x) for x in color.tolist()], "count": int(count)}
        for color, count in zip(unique_colors[:max_unique_report], counts[:max_unique_report])
    ]
    return {
        "mask_type": "multi_channel_pixel_mask",
        "shape": list(mask.shape),
        "dtype": str(mask.dtype),
        "exact_decode": bool(exact_decode),
        "num_unique_colors": int(unique_colors.shape[0]),
        "unique_colors_report": report,
        "all_unique_colors_fit_in_report": bool(unique_colors.shape[0] <= max_unique_report),
        "derived_label_ids": [int(v) for v in np.unique(label_ids).tolist()],
        "pandas_mask_schema": schema_name,
        "pandas_label_meaning": {str(k): v for k, v in pandas_meaning.items()},
        "pandas_to_sicap_mapping": {str(k): int(v) for k, v in pandas_to_sicap.items()},
        "observed_sicap_ids": observed_sicap_ids,
        "observed_sicap_names": observed_sicap_names,
        "can_map_to_sicap_gleason_patterns": bool(can_map_gleason),
        "tiff_structure": structure,
    }


def pair_cases(data_dir: Path) -> List[PairRecord]:
    masks = sorted(data_dir.glob("*_mask.tiff"))
    pairs: List[PairRecord] = []
    for mask_path in masks:
        case_id = mask_path.stem.replace("_mask", "")
        wsi_path = data_dir / f"{case_id}.tiff"
        pairs.append(PairRecord(case_id=case_id, wsi_path=str(wsi_path) if wsi_path.exists() else None, mask_path=str(mask_path)))
    return pairs


def save_visuals(case_dir: Path, image_thumb: Optional[np.ndarray], mask_thumb: np.ndarray, mask_summary: Dict[str, object]) -> None:
    from PIL import Image

    case_dir.mkdir(parents=True, exist_ok=True)
    label_ids = extract_label_ids(mask_thumb)
    palette = build_color_palette(np.unique(label_ids))
    mask_rgb = colorize_mask(label_ids, palette)
    Image.fromarray(mask_rgb).save(case_dir / "mask_colored.png")
    if image_thumb is not None:
        Image.fromarray(image_thumb).save(case_dir / "wsi_thumbnail.png")
        Image.fromarray(overlay(image_thumb, mask_rgb)).save(case_dir / "overlay.png")
    (case_dir / "mask_summary.json").write_text(json.dumps(mask_summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    ensure_runtime_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = pair_cases(args.data_dir)
    if not pairs:
        raise FileNotFoundError(f"No *_mask.tiff files found in {args.data_dir}")

    summaries: List[Dict[str, object]] = []
    for pair in pairs:
        LOGGER.info("Inspecting %s", pair.case_id)
        mask, mask_structure, mask_exact_decode = load_mask(Path(pair.mask_path), args.thumb_size)
        mask_summary = summarize_mask(mask, args.max_unique_report, mask_exact_decode, mask_structure)
        image_thumb = None
        image_structure = None
        image_exact_decode = None
        if pair.wsi_path is not None:
            image, image_structure, image_exact_decode = load_rgb_image(Path(pair.wsi_path), args.thumb_size)
            image_thumb = resize_image(image, args.thumb_size, is_mask=False)
        mask_thumb = resize_image(mask, args.thumb_size, is_mask=(mask.ndim == 2))
        case_dir = args.output_dir / pair.case_id
        save_visuals(case_dir, image_thumb, mask_thumb, mask_summary)
        summaries.append(
            {
                **asdict(pair),
                **mask_summary,
                "has_matching_wsi": pair.wsi_path is not None,
                "wsi_tiff_structure": image_structure,
                "wsi_exact_decode": image_exact_decode,
            }
        )

    pd_rows = summaries
    try:
        import pandas as pd

        pd.DataFrame(pd_rows).to_csv(args.output_dir / "pandas_mask_inventory.csv", index=False)
    except Exception:
        pass
    (args.output_dir / "pandas_mask_inventory.json").write_text(json.dumps(pd_rows, indent=2), encoding="utf-8")
    LOGGER.info("Inspection written to %s", args.output_dir)


if __name__ == "__main__":
    main()
