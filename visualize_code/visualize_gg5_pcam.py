"""
Find and visualize one PCam (PCam->TAH_UNet) example where:
  - prediction contains GG5 (class id=3)
  - GT mask contains GG5 (class id=3)

Output is saved for presentation.

Usage:
  python visualize_gg5_pcam.py --fold 3 --out-dir "<DIR>"
  python visualize_gg5_pcam.py --fold 1 --out-dir "<DIR>" --max-scan 300
"""

import argparse
import sys
from pathlib import Path
import random

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import sicap_imports  # noqa: F401

from paths import IMAGES_DIR, MASKS_DIR, PARTITION_DIR, default_checkpoint_dir

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch.amp import autocast

from training_pcam import (
    build_model,
    get_val_transforms,
    SICAPv2Dataset,
    DEFAULT_CONFIG,
    ENCODER_PRESETS,
    CLASS_NAMES,
    NUM_CLASSES,
)


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
    """Overlay where GT=Red, Pred=Green, Overlap=Yellow (Cancer only)."""
    overlay = np.zeros_like(image)
    pred_cancer = pred > 0
    gt_cancer = gt > 0
    overlay[gt_cancer & ~pred_cancer] = [231, 76, 60]
    overlay[pred_cancer & ~gt_cancer] = [46, 204, 113]
    overlay[pred_cancer & gt_cancer] = [241, 196, 15]

    mask = pred_cancer | gt_cancer
    result = image.copy()
    if mask.any():
        result[mask] = cv2.addWeighted(image[mask], 1 - alpha, overlay[mask], alpha, 0)
    return result


def _strip_compile_prefix(state_dict: dict) -> dict:
    # torch.compile may wrap modules; remove `_orig_mod.` prefix if present.
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def _predict_with_gg5_threshold(logits: torch.Tensor, gg5_threshold: float) -> torch.Tensor:
    """PCam script uses asymmetric thresholding: GG5 if prob(GG5) > threshold."""
    probs = torch.softmax(logits, dim=1)  # [B, 4, H, W]
    base_preds = probs[:, : NUM_CLASSES - 1].argmax(dim=1)  # [B,H,W] over 0..2
    gg5_conf_mask = probs[:, NUM_CLASSES - 1] > gg5_threshold  # [B,H,W]
    preds = base_preds.clone()
    preds[gg5_conf_mask] = NUM_CLASSES - 1
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=3, choices=[1, 2, 3, 4])
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(default_checkpoint_dir("checkpoints_nature_pcam")),
    )
    parser.add_argument("--max-scan", type=int, default=200)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--out-name", type=str, default="gg5_pcam_correct.png")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PCam uses best_Val{fold}.pth (observed in your folder)
    ckpt_path = Path(args.checkpoint_dir) / f"best_Val{args.fold}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Build PCam model with PCam pretrained encoder (patchcamelyon)
    config = dict(DEFAULT_CONFIG)
    config.update(ENCODER_PRESETS["pcam_resnet50"])

    model = build_model(config).to(device)
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(_strip_compile_prefix(state))
    model.eval()

    gg5_threshold = float(config["gg5_inference_threshold"])

    fold_dir = PARTITION_DIR / "Validation" / f"Val{args.fold}"
    val_df = pd.read_excel(fold_dir / "Test.xlsx")
    # Para encontrar un caso rápido para la presentación, barajamos las images.
    # (Val3 puede tener ~1800 samples; en orden pueden tardar en aparecer TN/TP correctos de GG5.)
    val_names = val_df["image_name"].tolist()
    random.shuffle(val_names)
    val_names = val_names[: args.max_scan]

    chosen_name = None
    chosen_pred = None
    chosen_gt = None

    # Iterate; pick first that matches (pred GG5 and GT GG5)
    for name in val_names:
        ds = SICAPv2Dataset([name], IMAGES_DIR, MASKS_DIR, transform=get_val_transforms())
        img_tensor, mask_tensor = ds[0]
        img_batch = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(img_batch)
            pred_tensor = _predict_with_gg5_threshold(logits, gg5_threshold=gg5_threshold)

        pred = pred_tensor.squeeze(0).cpu().numpy().astype(np.int64)
        gt = mask_tensor.cpu().numpy().astype(np.int64)

        if (pred == 3).any() and (gt == 3).any():
            chosen_name = name
            chosen_pred = pred
            chosen_gt = gt
            break

    if chosen_name is None:
        raise RuntimeError(
            f"No PCam GG5-correct sample found in Val{args.fold} (first {len(val_names)} images) "
            f"using gg5_threshold={gg5_threshold}."
        )

    # Load raw image for display
    buf = np.fromfile(str(IMAGES_DIR / chosen_name), dtype=np.uint8)
    raw_image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    # Plot and save
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    axes = axes[np.newaxis, :]
    col_titles = ["Original Image", "Ground Truth", "Prediction", "Overlay (GT=Rojo, Pred=Verde)"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=18, fontweight="bold", pad=15)

    axes[0, 0].imshow(raw_image)
    axes[0, 0].set_ylabel(chosen_name[:30] + "...", fontsize=9, rotation=0, labelpad=130, va="center")
    axes[0, 1].imshow(colorize_mask(chosen_gt))
    axes[0, 2].imshow(colorize_mask(chosen_pred))
    axes[0, 3].imshow(overlay_comparison(raw_image, chosen_pred, chosen_gt, alpha=0.5))

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

    plt.savefig(str(out_path), dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"Chosen image: {chosen_name} | GG5 threshold={gg5_threshold}")


if __name__ == "__main__":
    main()

