"""
Visualize GT-only GG5 examples as side-by-side image+mask pairs.

Creates a single figure:
  - rows = N examples
  - cols = 2 (Original image | GT mask (GG5))

Usage:
  python visualize_gt_gg5_pairs.py --n 8 --use-filenames-txt
"""

import argparse
import sys
from pathlib import Path
import math

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import sicap_imports  # noqa: F401

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from training_conch import (
    PARTITION_DIR,
    MASKS_DIR,
    IMAGES_DIR,
    IMG_SIZE,
    CLASS_NAMES,
    _MASK_LUT,
)

LUT = _MASK_LUT

CLASS_COLORS = {
    0: (50, 50, 50),
    1: (46, 204, 113),
    2: (241, 196, 15),
    3: (231, 76, 60),
}


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        rgb[mask == cls_id] = color
    return rgb


def mask_has_gg5(mask_path: Path) -> bool:
    if not mask_path.exists():
        return False
    buf = np.fromfile(str(mask_path), dtype=np.uint8)
    mask = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
    mapped = LUT[mask]
    return bool(np.any(mapped == 3))


def parse_filenames_txt(txt_path: Path) -> list[tuple[int, str]]:
    out = []
    if not txt_path.exists():
        return out
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        fold_str, name = parts
        fold = int(fold_str.replace("Val", "").strip())
        out.append((fold, name.strip()))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--filenames-txt", type=str, default=None)
    parser.add_argument("--use-filenames-txt", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out).parent if args.out else Path(__file__).resolve().parent / "images prediction"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = out_dir / f"gt_gg5_pairs_n{args.n:02d}.png"

    if args.use_filenames_txt:
        txt_path = Path(args.filenames_txt) if args.filenames_txt else (out_dir / "gt_gg5_only_filenames_n12.txt")
        picked = parse_filenames_txt(txt_path)
        if not picked:
            raise RuntimeError(f"No pude leer nombres desde: {txt_path}")
        picked = picked[: args.n]
    else:
        # Fallback: scan quick across folds for first N GG5 masks
        picked = []
        for fold in [1, 2, 3, 4]:
            fold_dir = PARTITION_DIR / "Validation" / f"Val{fold}"
            val_df = pd.read_excel(fold_dir / "Test.xlsx")
            for name in val_df["image_name"].tolist():
                if mask_has_gg5(MASKS_DIR / name):
                    picked.append((fold, name))
                    if len(picked) >= args.n:
                        break
            if len(picked) >= args.n:
                break

    fig, axes = plt.subplots(args.n, 2, figsize=(10, 4 * args.n))
    if args.n == 1:
        axes = np.array([axes])

    for i, (fold, name) in enumerate(picked):
        # Raw image
        buf = np.fromfile(str(IMAGES_DIR / name), dtype=np.uint8)
        raw = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        raw = cv2.resize(raw, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        # GT mask
        buf_m = np.fromfile(str(MASKS_DIR / name), dtype=np.uint8)
        mask = cv2.imdecode(buf_m, cv2.IMREAD_GRAYSCALE)
        mapped = LUT[mask]
        mapped = cv2.resize(mapped.astype(np.int32), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        axes[i, 0].imshow(raw)
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"Val{fold}: image", fontsize=10)

        axes[i, 1].imshow(colorize_mask(mapped))
        axes[i, 1].axis("off")
        axes[i, 1].set_title(f"Val{fold}: GT mask (GG5)", fontsize=10)

        # Add short filename as a y-label for context
        axes[i, 0].set_ylabel(name[:24] + "...", fontsize=8, rotation=0, labelpad=60)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

