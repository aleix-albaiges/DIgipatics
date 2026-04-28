"""
Visualize images de parches etiquetados como G5 en partition junto con:
  - mask en escala de grises (valores JPEG decodificados)
  - mask con clases tras _MASK_LUT (NC / GG3 / GG4 / GG5)

Usage:
  python visualize_gg5_masks.py
  python visualize_gg5_masks.py --n 12 --seed 0 --out-dir visualizations_gg5
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import sicap_imports  # noqa: F401

from paths import IMAGES_DIR, MASKS_DIR, PARTITION_DIR
from sicap_imports import REPO_ROOT

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import openpyxl

# Do not import training_conch (pull de torch/smp); misma LUT y rutas (paths.py para datos)

CLASS_NAMES = ["NC", "GG3", "GG4", "GG5"]
_MASK_LUT = np.zeros(256, dtype=np.int64)
_MASK_LUT[43:85] = 1
_MASK_LUT[85:160] = 2
_MASK_LUT[160:] = 3

CLASS_COLORS_RGB = {
    0: (50, 50, 50),
    1: (46, 204, 113),
    2: (241, 196, 15),
    3: (231, 76, 60),
}


def partition_xlsx_files() -> list[Path]:
    patterns = [
        "Validation/*/Train.xlsx",
        "Validation/*/Test.xlsx",
        "Test/Train.xlsx",
        "Test/Test.xlsx",
    ]
    out: list[Path] = []
    for pat in patterns:
        out.extend(sorted(PARTITION_DIR.glob(pat)))
    return out


def load_g5_image_names() -> list[str]:
    """Nombres únicos con G5=1 en partition (sin Cribriform-only files)."""
    names: set[str] = set()
    for xp in partition_xlsx_files():
        if "Crib" in xp.name:
            continue
        wb = openpyxl.load_workbook(xp, read_only=True, data_only=True)
        try:
            ws = wb.active
            it = ws.iter_rows(values_only=True)
            header = next(it)
            idx = {str(h).strip(): i for i, h in enumerate(header) if h is not None}
            i_name = idx["image_name"]
            i_g5 = idx["G5"]
            for row in it:
                if row[i_g5] == 1:
                    names.add(str(row[i_name]).strip())
        finally:
            wb.close()
    return sorted(names)


def load_image(path: Path) -> np.ndarray | None:
    buf = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mask_gray(path: Path) -> np.ndarray | None:
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def colorize_mapped(mapped: np.ndarray) -> np.ndarray:
    h, w = mapped.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c, col in CLASS_COLORS_RGB.items():
        rgb[mapped == c] = col
    return rgb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9, help="Número de parches a mostrar")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out-dir",
        type=str,
        default="visualizations_gg5",
        help="Carpeta de salida (bajo el directorio del dataset)",
    )
    ap.add_argument(
        "--require-mapped-gg5",
        action="store_true",
        help="Solo parches donde la mask LUT tenga al menos un pixel clase GG5 (3)",
    )
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_g5 = load_g5_image_names()
    candidates: list[str] = []
    lut = _MASK_LUT
    for name in all_g5:
        mp = MASKS_DIR / name
        if not mp.is_file():
            continue
        m = load_mask_gray(mp)
        if m is None:
            continue
        mapped = lut[m]
        if args.require_mapped_gg5 and not np.any(mapped == 3):
            continue
        candidates.append(name)

    if not candidates:
        raise SystemExit("No hay parches G5 válidos (revisa partition y masks/).")

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    picked = candidates[: args.n]

    ncols = 4
    nrows = len(picked)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.8 * nrows),
        squeeze=False,
    )

    lut_str = "LUT: [0..42]→NC, [43..84]→GG3, [85..159]→GG4, [160..255]→GG5"

    for r, name in enumerate(picked):
        img_path = IMAGES_DIR / name
        mask_path = MASKS_DIR / name
        rgb = load_image(img_path)
        raw = load_mask_gray(mask_path)
        if rgb is None or raw is None:
            for c in range(ncols):
                axes[r, c].set_visible(False)
            continue
        mapped = lut[raw]
        colored = colorize_mapped(mapped)

        # Fracción de pixeles por clase (información en título)
        bc = np.bincount(mapped.ravel().astype(np.int64), minlength=4)
        tot = float(mapped.size)
        frac = [100.0 * bc[i] / tot for i in range(4)]

        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"{name[:50]}…" if len(name) > 50 else name, fontsize=8)
        axes[r, 0].axis("off")

        axes[r, 1].imshow(raw, cmap="gray", vmin=0, vmax=255)
        axes[r, 1].set_title("Máscara cruda (JPEG, 0–255)", fontsize=9)
        axes[r, 1].axis("off")

        axes[r, 2].imshow(colored)
        axes[r, 2].set_title("Clases tras LUT\n" + lut_str[:40] + "…", fontsize=8)
        axes[r, 2].axis("off")

        overlay = (0.55 * rgb.astype(np.float32) + 0.45 * colored.astype(np.float32)).astype(np.uint8)
        axes[r, 3].imshow(overlay)
        axes[r, 3].set_title(
            f"% pix: NC {frac[0]:.1f} | G3 {frac[1]:.1f} | G4 {frac[2]:.1f} | G5 {frac[3]:.1f}",
            fontsize=8,
        )
        axes[r, 3].axis("off")

    patches = [
        mpatches.Patch(color=np.array(CLASS_COLORS_RGB[k]) / 255.0, label=f"{k}: {CLASS_NAMES[k]}")
        for k in range(4)
    ]
    fig.legend(
        handles=patches,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.98),
        fontsize=10,
    )
    fig.suptitle(
        "Parches con etiqueta G5 en partition | " + lut_str,
        fontsize=10,
        y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = out_dir / "gg5_patches_overview.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {out_png}")

    # Segunda figura: mosaico only RGB + mask mapeada (más legible en informe)
    n_side = int(np.ceil(np.sqrt(len(picked))))
    fig2, axes2 = plt.subplots(n_side, n_side * 2, figsize=(3.2 * n_side * 2, 3 * n_side))
    if n_side == 1:
        axes2 = np.array([[axes2[0], axes2[1]]])
    axes2 = np.atleast_2d(axes2)
    idx = 0
    for i in range(n_side):
        for j in range(n_side):
            if idx >= len(picked):
                for k in range(2):
                    axes2[i, j * 2 + k].set_visible(False)
                continue
            name = picked[idx]
            rgb = load_image(IMAGES_DIR / name)
            raw = load_mask_gray(MASKS_DIR / name)
            idx += 1
            if rgb is None or raw is None:
                continue
            mapped = lut[raw]
            colored = colorize_mapped(mapped)
            axes2[i, j * 2].imshow(rgb)
            axes2[i, j * 2].axis("off")
            axes2[i, j * 2 + 1].imshow(colored)
            axes2[i, j * 2 + 1].axis("off")
    plt.suptitle("G5 (partition): imagen | mask LUT", fontsize=11)
    plt.tight_layout()
    out2 = out_dir / "gg5_grid_compact.png"
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {out2}")


if __name__ == "__main__":
    main()
