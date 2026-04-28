"""
Visualize Ground Truth (GT) masks where GG5 is present (class id=3).

Creates a grid of GT-only examples (no predictions).

Usage:
  python visualize_gt_gg5.py --n 12 --folds all --out-dir "<DIR>"
  python visualize_gt_gg5.py --n 12 --folds 3 --out-dir "<DIR>"
"""

import argparse
import random
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import sicap_imports  # noqa: F401

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from training_conch import PARTITION_DIR, MASKS_DIR, IMG_SIZE, CLASS_NAMES, _MASK_LUT

LUT = _MASK_LUT

CLASS_COLORS = {
    0: (50, 50, 50),       # NC  — dark gray
    1: (46, 204, 113),     # GG3 — green
    2: (241, 196, 15),     # GG4 — yellow
    3: (231, 76, 60),      # GG5 — red
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12, help="Number of examples to pick.")
    parser.add_argument(
        "--folds",
        type=str,
        default="all",
        help="Which folds to scan: 'all' or a comma list like '1,2,3'.",
    )
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.folds.lower() == "all":
        folds = [1, 2, 3, 4]
    else:
        folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]

    # Gather candidate names from partition Test.xlsx
    candidates = []
    for fold in folds:
        fold_dir = PARTITION_DIR / "Validation" / f"Val{fold}"
        val_df = pd.read_excel(fold_dir / "Test.xlsx")
        candidates.extend([(fold, n) for n in val_df["image_name"].tolist()])

    random.shuffle(candidates)

    picked = []
    seen = set()
    for fold, name in candidates:
        if name in seen:
            continue
        seen.add(name)
        if mask_has_gg5(MASKS_DIR / name):
            picked.append((fold, name))
            if len(picked) >= args.n:
                break

    if not picked:
        raise RuntimeError("No GG5 examples found in the scanned folds.")

    # Plot GT-only colored masks
    cols = 3
    rows = int(np.ceil(len(picked) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    for i, (fold, name) in enumerate(picked[: args.n]):
        r, c = divmod(i, cols)
        ax = axes[r, c]

        # Load & map mask; resize for consistent plotting
        buf = np.fromfile(str(MASKS_DIR / name), dtype=np.uint8)
        m = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        mapped = LUT[m]
        mapped = cv2.resize(mapped, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        ax.imshow(colorize_mask(mapped))
        ax.set_title(f"Val{fold}: {name[:18]}...", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    out_path = out_dir / f"gt_gg5_only_grid_n{len(picked):02d}.png"
    plt.savefig(str(out_path), dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Save filenames for reference in the presentation
    txt_path = out_dir / f"gt_gg5_only_filenames_n{len(picked):02d}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for fold, name in picked[: args.n]:
            f.write(f"Val{fold}\t{name}\n")

    print(f"Saved grid: {out_path}")
    print(f"Saved list: {txt_path}")


if __name__ == "__main__":
    main()

