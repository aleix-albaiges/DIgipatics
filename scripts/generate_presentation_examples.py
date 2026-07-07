"""
Generate diverse SICAPv2 slide examples: H&E with semi-transparent GT overlay.

Usage:
  python scripts/generate_presentation_examples.py
  python scripts/generate_presentation_examples.py --n 12 --seed 0
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import openpyxl

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import sicap_imports  # noqa: F401
from sicap_imports import REPO_ROOT

from paths import IMAGES_DIR, MASKS_DIR, PARTITION_DIR

CLASS_NAMES = ["NC", "GG3", "GG4", "GG5"]
CLASS_COLORS = {
    0: (50, 50, 50),
    1: (46, 204, 113),
    2: (241, 196, 15),
    3: (231, 76, 60),
}

_MASK_LUT = np.zeros(256, dtype=np.int64)
_MASK_LUT[25:75] = 1
_MASK_LUT[75:175] = 2
_MASK_LUT[175:] = 3


def partition_names(limit: int | None = None) -> list[str]:
    patterns = [
        "Validation/Val1/Train.xlsx",
        "Validation/Val1/Test.xlsx",
        "Validation/Val2/Train.xlsx",
    ]
    names: list[str] = []
    seen: set[str] = set()
    for rel in patterns:
        xp = PARTITION_DIR / rel
        if not xp.is_file():
            continue
        wb = openpyxl.load_workbook(xp, read_only=True, data_only=True)
        try:
            ws = wb.active
            rows = ws.iter_rows(values_only=True)
            header = next(rows)
            idx = {str(h).strip(): i for i, h in enumerate(header) if h is not None}
            for row in rows:
                name = str(row[idx["image_name"]]).strip()
                if name in seen:
                    continue
                seen.add(name)
                names.append(name)
                if limit is not None and len(names) >= limit:
                    return names
        finally:
            wb.close()
    return names


def load_rgb(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    buf = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mapped(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    buf = np.fromfile(str(path), dtype=np.uint8)
    raw = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return None
    return _MASK_LUT[raw]


def colorize_gt(mapped: np.ndarray) -> np.ndarray:
    h, w = mapped.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, col in CLASS_COLORS.items():
        rgb[mapped == class_id] = col
    return rgb


def present_classes(mapped: np.ndarray) -> tuple[int, ...]:
    bc = np.bincount(mapped.ravel().astype(np.int64), minlength=4)
    return tuple(i for i in range(4) if bc[i] > 0)


def class_label(mapped: np.ndarray) -> str:
    bc = np.bincount(mapped.ravel().astype(np.int64), minlength=4)
    parts = [f"{CLASS_NAMES[i]} {100.0 * bc[i] / mapped.size:.1f}%" for i in range(4) if bc[i] > 0]
    return ", ".join(parts)


def blend_overlay(rgb: np.ndarray, mapped: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend colored GT on H&E (NC left unchanged)."""
    base = rgb.astype(np.float32) / 255.0
    colored = colorize_gt(mapped).astype(np.float32) / 255.0
    tissue = mapped != 0
    out = base.copy()
    out[tissue] = (1.0 - alpha) * base[tissue] + alpha * colored[tissue]
    return np.clip(out, 0.0, 1.0)


def draw_overlay(ax, rgb: np.ndarray, mapped: np.ndarray, alpha: float = 0.55) -> None:
    ax.imshow(blend_overlay(rgb, mapped, alpha=alpha))


def class_id_from_name(name: str) -> int:
    key = name.strip().upper()
    if key not in CLASS_NAMES:
        raise ValueError(f"Unknown class {name!r}; use one of {CLASS_NAMES}")
    return CLASS_NAMES.index(key)


def pick_with_class(
    names: list[str],
    class_id: int,
    n: int,
    seed: int,
    *,
    min_class_px: int = 1500,
    mixed: bool = False,
    min_frac: float = 0.10,
    max_frac: float = 0.75,
    target_frac: float = 0.35,
) -> list[str]:
    """Pick patches containing a Gleason class; mixed=True prefers NC + class blend."""
    rng = random.Random(seed)
    candidates: list[tuple[float, float, str]] = []
    for name in names:
        mapped = load_mapped(MASKS_DIR / name)
        if mapped is None or load_rgb(IMAGES_DIR / name) is None:
            continue
        bc = np.bincount(mapped.ravel().astype(np.int64), minlength=4)
        if bc[class_id] < min_class_px or bc[0] < min_class_px:
            continue
        frac = bc[class_id] / mapped.size
        if mixed and (frac < min_frac or frac > max_frac):
            continue
        score = abs(frac - target_frac) if mixed else -frac
        candidates.append((score, frac, name))

    if not candidates:
        return []

    candidates.sort(key=lambda item: (item[0], -item[1]))
    if mixed:
        # Spread picks across similar scores for variety.
        band = candidates[: max(n * 4, n)]
        rng.shuffle(band)
        return [name for _, _, name in band[:n]]

    return [name for _, _, name in candidates[:n]]


def pick_diverse(names: list[str], n: int, seed: int) -> list[str]:
    """Pick patches with varied class combinations (quick scan on partition list)."""
    rng = random.Random(seed)
    shuffled = names[:]
    rng.shuffle(shuffled)

    buckets: dict[tuple[int, ...], list[str]] = {}
    for name in shuffled:
        mapped = load_mapped(MASKS_DIR / name)
        if mapped is None or load_rgb(IMAGES_DIR / name) is None:
            continue
        key = present_classes(mapped)
        if len(key) < 2:
            continue
        buckets.setdefault(key, []).append(name)

    picked: list[str] = []
    seen: set[str] = set()

    # Prefer multi-class combinations first (more interesting for slides).
    for key in sorted(buckets, key=lambda k: (-len(k), k)):
        for name in buckets[key]:
            if name in seen:
                continue
            picked.append(name)
            seen.add(name)
            if len(picked) >= n:
                return picked

    for name in shuffled:
        if name in seen:
            continue
        if load_mapped(MASKS_DIR / name) is None:
            continue
        picked.append(name)
        seen.add(name)
        if len(picked) >= n:
            break

    return picked


def save_single(out_dir: Path, index: int, name: str, alpha: float) -> None:
    rgb = load_rgb(IMAGES_DIR / name)
    mapped = load_mapped(MASKS_DIR / name)
    if rgb is None or mapped is None:
        return

    # Save original tile.
    fig_tile, ax_tile = plt.subplots(1, 1, figsize=(5, 5))
    ax_tile.imshow(rgb)
    ax_tile.set_title("H&E tile", fontsize=9)
    ax_tile.axis("off")
    fig_tile.tight_layout()
    fig_tile.savefig(out_dir / f"tile_{index:02d}.png", dpi=160, bbox_inches="tight")
    plt.close(fig_tile)

    # Save overlayed tile.
    fig_ov, ax_ov = plt.subplots(1, 1, figsize=(5, 5))
    draw_overlay(ax_ov, rgb, mapped, alpha=alpha)
    ax_ov.set_title(class_label(mapped), fontsize=9)
    ax_ov.axis("off")
    fig_ov.tight_layout()
    fig_ov.savefig(out_dir / f"overlay_{index:02d}.png", dpi=160, bbox_inches="tight")
    plt.close(fig_ov)


def save_grid(out_dir: Path, names: list[str], alpha: float) -> None:
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_2d(axes)

    for i, ax in enumerate(axes.ravel()):
        if i >= len(names):
            ax.axis("off")
            continue
        name = names[i]
        rgb = load_rgb(IMAGES_DIR / name)
        mapped = load_mapped(MASKS_DIR / name)
        if rgb is None or mapped is None:
            ax.axis("off")
            continue
        draw_overlay(ax, rgb, mapped, alpha=alpha)
        short = name[:42] + "…" if len(name) > 42 else name
        ax.set_title(short, fontsize=7)
        ax.axis("off")

    patches = [
        mpatches.Patch(color=np.array(CLASS_COLORS[i]) / 255, label=CLASS_NAMES[i])
        for i in range(4)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.suptitle("SICAPv2 — H&E amb GT overlay", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_dir / "overlay_grid.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="Number of examples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.55, help="Overlay opacity for Gleason classes")
    ap.add_argument("--scan", type=int, default=400, help="Partition names to scan for diversity")
    ap.add_argument(
        "--require-class",
        action="append",
        default=[],
        help="Only pick patches containing this class (e.g. GG3, GG5). Repeatable.",
    )
    ap.add_argument("--start-index", type=int, default=1, help="Output numbering start (tile_XX)")
    ap.add_argument("--mixed", action="store_true", help="Prefer NC + class blends (~10-75%% target class)")
    ap.add_argument("--no-grid", action="store_true", help="Skip grid image generation")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "presentacio")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = partition_names(limit=args.scan)
    if args.require_class:
        class_ids = [class_id_from_name(name) for name in args.require_class]
        names: list[str] = []
        seen: set[str] = set()
        per_class = max(1, args.n // len(class_ids))
        for cid in class_ids:
            for name in pick_with_class(pool, cid, per_class, args.seed + cid, mixed=args.mixed):
                if name in seen:
                    continue
                names.append(name)
                seen.add(name)
        if len(names) < args.n:
            for cid in class_ids:
                for name in pick_with_class(pool, cid, args.n, args.seed + cid + 10, mixed=args.mixed):
                    if name in seen:
                        continue
                    names.append(name)
                    seen.add(name)
                    if len(names) >= args.n:
                        break
                if len(names) >= args.n:
                    break
        names = names[: args.n]
    else:
        names = pick_diverse(pool, n=args.n, seed=args.seed)
    if not names:
        raise SystemExit("No valid image/mask pairs found.")

    for offset, name in enumerate(names):
        index = args.start_index + offset
        save_single(out_dir, index, name, alpha=args.alpha)
        mapped = load_mapped(MASKS_DIR / name)
        print(f"{index:02d} {name}")
        if mapped is not None:
            print(f"    {class_label(mapped)}")

    if not args.no_grid:
        save_grid(out_dir, names, alpha=args.alpha)
    print(f"\nSaved {len(names)} tile/overlay pairs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
