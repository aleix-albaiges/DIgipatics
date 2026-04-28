"""
Visualize model predictions vs ground truth masks (CONCH).
Usage:
    python visualize_predictions.py                      # uses Val1 test set, shows 8 random images
    python visualize_predictions.py --fold 2 --n 12      # Val2 test set, 12 images
    python visualize_predictions.py --checkpoint path.pth # custom checkpoint (overrides auto "best")
    python visualize_predictions.py --best-overall      # toma el mejor checkpoint global (macro_f1) y usa su fold
    python visualize_predictions.py --repeat 10 --n 1 --best-overall --out-dir "<DIR>"
"""

import argparse
import random
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import sicap_imports  # noqa: F401

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from torch.amp import autocast

# Reuse model/dataset from training (CONCH)
from training_conch import (
    build_model,
    get_val_transforms,
    SICAPv2Dataset,
    IMAGES_DIR,
    MASKS_DIR,
    PARTITION_DIR,
    OUTPUT_DIR,
    DEFAULT_CONFIG,
    NUM_CLASSES,
    CLASS_NAMES,
    IMG_SIZE,
)

# Class colors (RGB)
CLASS_COLORS = {
    0: (50, 50, 50),       # NC  — dark gray
    1: (46, 204, 113),     # GG3 — green
    2: (241, 196, 15),     # GG4 — yellow
    3: (231, 76, 60),      # GG5 — red
}


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert class-index mask (H,W) to RGB image (H,W,3)."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        rgb[mask == cls_id] = color
    return rgb


def overlay_comparison(image: np.ndarray, pred: np.ndarray, gt: np.ndarray, alpha=0.5) -> np.ndarray:
    """Overlay where GT=Red, Pred=Green, and Overlap=Yellow (Cancer only)"""
    overlay = np.zeros_like(image)
    
    # Binary tumor masks (Any cancer grade = True)
    pred_cancer = pred > 0
    gt_cancer = gt > 0
    
    # Red: False Negative (Ground Truth missing an prediction)
    overlay[gt_cancer & ~pred_cancer] = [231, 76, 60] # Red
    # Green: False Positive (Predicted without GT)
    overlay[pred_cancer & ~gt_cancer] = [46, 204, 113]  # Green
    # Yellow: True Positive (Match)
    overlay[pred_cancer & gt_cancer] = [241, 196, 15]   # Yellow
    
    mask = pred_cancer | gt_cancer
    result = image.copy()
    if mask.any():
        result[mask] = cv2.addWeighted(image[mask], 1 - alpha, overlay[mask], alpha, 0)
    return result


def main():
    parser = argparse.ArgumentParser(description="Visualize predictions vs ground truth")
    parser.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--best-overall", action="store_true",
                        help="Selecciona el mejor checkpoint global (máximo macro_f1) en checkpoints_conch e ignora --fold.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: checkpoints/best_ValX.pth)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Override checkpoint directory (default: OUTPUT_DIR from training_conch.py)")
    parser.add_argument("--n", type=int, default=8, help="Number of images to show")
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure (single run). If --repeat>1, ignored (use --out-dir).")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat visualization sampling this many times (recommended with --n 1).")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="If set, saves outputs into this directory (recommended for presentations).")
    parser.add_argument("--out-base", type=str, default="best_overall_conch_n1",
                        help="Base filename prefix for repeated saves.")
    parser.add_argument("--find-gg5-correct", action="store_true",
                        help="Search for one image where both GT and Pred contain GG5 (class id=3), and save it.")
    parser.add_argument("--gg5-correct-fname", type=str, default="gg5_correct.png",
                        help="Filename for the GG5-correct image (saved into --out-dir).")
    parser.add_argument("--gg5-max-tries", type=int, default=200,
                        help="Max random tries to find a GG5-correct example.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DEFAULT_CONFIG.copy()

    def _parse_macro_f1_from_name(p: Path) -> float:
        # training_conch saves: best_{fold_name}_{macro_f1:.4f}.pth
        stem = p.stem  # without .pth
        # Example: best_Val2_0.7123
        parts = stem.split("_")
        for token in reversed(parts):
            try:
                return float(token)
            except ValueError:
                continue
        return float("-inf")

    def _get_best_checkpoint_for_fold(checkpoint_dir: Path, fold_name: str) -> Path:
        candidates = sorted(checkpoint_dir.glob(f"best_{fold_name}_*.pth"), key=_parse_macro_f1_from_name, reverse=True)
        if not candidates:
            raise FileNotFoundError(
                f"No best checkpoints found for fold '{fold_name}' in {checkpoint_dir}\n"
                f"Expected pattern: best_{fold_name}_*.pth"
            )
        return candidates[0]

    def _parse_fold_from_checkpoint_name(p: Path) -> str:
        # Ejemplo stem: best_Val2_0.7123
        stem = p.stem
        parts = stem.split("_")
        for token in parts:
            if token.startswith("Val") and token[3:].isdigit():
                return token
        raise ValueError(f"No pude extraer fold desde: {p.name}")

    # Load checkpoint
    checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else OUTPUT_DIR
    ckpt_path: Path
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint).resolve()
        fold_name = _parse_fold_from_checkpoint_name(ckpt_path) if not args.best_overall else _parse_fold_from_checkpoint_name(ckpt_path)
    elif args.best_overall:
        candidates = list(checkpoint_dir.glob("best_Val*_*.pth"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoints best_Val*_*.pth encontrados en: {checkpoint_dir}")
        ckpt_path = sorted(candidates, key=_parse_macro_f1_from_name, reverse=True)[0]
        fold_name = _parse_fold_from_checkpoint_name(ckpt_path)
    else:
        fold_name = f"Val{args.fold}"
        ckpt_path = _get_best_checkpoint_for_fold(checkpoint_dir, fold_name)

    print(f"Selected fold: {fold_name}")
    print(f"Loading checkpoint: {ckpt_path}")
    model = build_model(config).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    
    # Remove '_orig_mod.' prefix added by torch.compile during training
    clean_ckpt = {k.replace('_orig_mod.', ''): v for k, v in ckpt.items()}
    model.load_state_dict(clean_ckpt)
    
    model.eval()

    # Load val/test images for this fold
    fold_dir = PARTITION_DIR / "Validation" / fold_name
    val_df = pd.read_excel(fold_dir / "Test.xlsx")
    val_names = val_df["image_name"].tolist()
    n = min(args.n, len(val_names))
    repeat = max(1, int(args.repeat))

    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    def _plot_one(selected_name: str, save_path: Path | None):
        dataset = SICAPv2Dataset([selected_name], IMAGES_DIR, MASKS_DIR, transform=get_val_transforms())
        img_tensor, mask_tensor = dataset[0]

        # Raw image for display
        buf = np.fromfile(str(IMAGES_DIR / selected_name), dtype=np.uint8)
        raw_image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

        # Inference
        with torch.no_grad():
            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(img_tensor.unsqueeze(0).to(device))
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        gt = mask_tensor.numpy()

        fig, axes = plt.subplots(1, 4, figsize=(22, 6))
        axes = axes[np.newaxis, :]  # unify indexing
        col_titles = ["Original Image", "Ground Truth", "Prediction", "Overlay (GT=Rojo, Pred=Verde)"]
        for ax, title in zip(axes[0], col_titles):
            ax.set_title(title, fontsize=18, fontweight="bold", pad=15)

        axes[0, 0].imshow(raw_image)
        axes[0, 0].set_ylabel(selected_name[:30] + "...", fontsize=9, rotation=0, labelpad=130, va="center")
        axes[0, 1].imshow(colorize_mask(gt))
        axes[0, 2].imshow(colorize_mask(pred))
        axes[0, 3].imshow(overlay_comparison(raw_image, pred, gt, alpha=0.5))
        for j in range(4):
            axes[0, j].axis("off")

        legend_patches = [mpatches.Patch(color=np.array(c) / 255, label=CLASS_NAMES[k]) for k, c in CLASS_COLORS.items()]
        legend_patches.extend([
            mpatches.Patch(color=np.array([241, 196, 15]) / 255, label="Tumor Coincide (Amarillo)"),
            mpatches.Patch(color=np.array([231, 76, 60]) / 255, label="Faltó GT (Rojo)"),
            mpatches.Patch(color=np.array([46, 204, 113]) / 255, label="Sobró Pred (Verde)"),
        ])
        fig.legend(handles=legend_patches, loc="lower center", ncol=7, fontsize=13, frameon=True, fancybox=True)

        plt.tight_layout(rect=[0, 0.05, 1, 0.98])
        plt.subplots_adjust(bottom=0.10, top=0.95, hspace=0.1)

        pred_has_gg5 = bool((pred == 3).any())
        gt_has_gg5 = bool((gt == 3).any())
        # "Predice GG5 y acierta" -> ambos contienen la clase 3 (no exige pixeles solapados).
        gg5_overlap = bool(pred_has_gg5 and gt_has_gg5)

        if save_path is not None:
            plt.savefig(str(save_path), dpi=180, bbox_inches="tight")
            print(f"Saved to {save_path}")
            plt.close(fig)
        else:
            # No mostrar ventanas durante búsquedas automáticas
            if out_dir is None and not args.save:
                plt.show()
            else:
                plt.close(fig)
        return pred_has_gg5, gt_has_gg5, gg5_overlap

    def _gg5_overlap_only(selected_name: str) -> bool:
        """Infer only para saber si pred y GT comparten al menos un pixel GG5 (clase 3)."""
        dataset = SICAPv2Dataset([selected_name], IMAGES_DIR, MASKS_DIR, transform=get_val_transforms())
        img_tensor, mask_tensor = dataset[0]

        with torch.no_grad():
            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(img_tensor.unsqueeze(0).to(device))
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        gt = mask_tensor.numpy()
        pred_has_gg5 = bool((pred == 3).any())
        gt_has_gg5 = bool((gt == 3).any())
        return bool(pred_has_gg5 and gt_has_gg5)

    def _gg5_pred_only(selected_name: str) -> bool:
        """Infer only para saber si el modelo predice alguna vez GG5 (clase 3)."""
        dataset = SICAPv2Dataset([selected_name], IMAGES_DIR, MASKS_DIR, transform=get_val_transforms())
        img_tensor, _ = dataset[0]
        with torch.no_grad():
            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(img_tensor.unsqueeze(0).to(device))
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        return bool((pred == 3).any())

    # Repeat sampling
    # If repeat>1, we expect --n==1 (n images per figure). We implement n=1 images per saved PNG.
    if repeat > 1 and n != 1:
        print("  [WARN] --repeat>1 funciona mejor con --n 1. Forzando n=1.")
        n = 1

    gg5_saved = False
    gg5_overlap_name = None

    for k in range(repeat):
        name = random.choice(val_names)
        save_path = None
        if out_dir is not None:
            save_path = out_dir / f"{args.out_base}_{k+1:02d}.png"
        _pred_has, _gt_has, overlap = _plot_one(name, save_path=save_path)
        if args.find_gg5_correct and overlap and not gg5_saved:
            gg5_saved = True
            gg5_overlap_name = name
            if out_dir is not None:
                gg5_path = out_dir / args.gg5_correct_fname
                _plot_one(name, save_path=gg5_path)

    # If not found in repeat, do extra tries (only if requested)
    if args.find_gg5_correct and not gg5_saved:
        if out_dir is None:
            raise ValueError("--find-gg5-correct requiere --out-dir para saver la imagen encontrada.")
        for _ in range(args.gg5_max_tries):
            name = random.choice(val_names)
            overlap = _gg5_overlap_only(name)
            if overlap:
                gg5_path = out_dir / args.gg5_correct_fname
                _plot_one(name, save_path=gg5_path)
                gg5_overlap_name = name
                gg5_saved = True
                break

    # Fallback: si no hay ejemplo que cumpla (pred y GT tienen GG5), al menos save uno donde predice GG5.
    if args.find_gg5_correct and not gg5_saved:
        pred_only_path = out_dir / "gg5_pred_only.png" if out_dir is not None else None
        if pred_only_path is not None:
            for _ in range(args.gg5_max_tries):
                name = random.choice(val_names)
                if _gg5_pred_only(name):
                    _plot_one(name, save_path=pred_only_path)
                    gg5_overlap_name = name
                    gg5_saved = True
                    break

    if out_dir is None and (not args.save):
        print("  [INFO] Nothing saved. Use --out-dir or --save.")


if __name__ == "__main__":
    main()
