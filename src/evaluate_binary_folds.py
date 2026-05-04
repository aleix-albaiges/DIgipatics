import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Fix relative imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training_conch_binary import (
    build_model,
    get_fold_dataloaders,
    validate_one_epoch,
    GuidedLoss,
    print_aggregated_matrices,
    NUM_CLASSES,
    CLASS_NAMES,
    DEFAULT_CONFIG,
    _DEFAULT_CE_CLASS_WEIGHTS,
    _PIXEL_FRAC_PARTITION
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CSV
    csv_path = Path("artifacts/checkpoints_conch_binary/best_per_fold.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} folds for evaluation from {csv_path}")

    # Reconstruct configuration
    config = DEFAULT_CONFIG.copy()
    config["class_weights"] = _DEFAULT_CE_CLASS_WEIGHTS
    config["pos_weight"] = float(_PIXEL_FRAC_PARTITION[0] / (_PIXEL_FRAC_PARTITION[1] + 1e-8))
    config["use_weighted_sampler"] = False # No need for sampler in evaluation

    aggregated_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    best_thresholds = []

    # Threshold grid for tuning during evaluation
    tmin, tmax, tstep = 0.30, 0.70, 0.05
    threshold_grid = np.arange(tmin, tmax + 1e-12, tstep, dtype=np.float64).tolist()

    for idx, row in df.iterrows():
        fold_name = row['fold']
        ckpt_path = row['checkpoint_path']
        
        print(f"\n{'='*60}\n  EVALUATING FOLD {fold_name}\n{'='*60}")
        print(f"Checkpoint: {ckpt_path}")
        
        if not os.path.exists(ckpt_path):
            print(f"  [WARN] Checkpoint not found: {ckpt_path}")
            continue

        # 1. Load Dataloader
        _, val_loader = get_fold_dataloaders(fold_name, config)

        # 2. Build Model and load weights
        config["num_classes"] = NUM_CLASSES
        model = build_model(config)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)

        # 3. Loss (only needed for validation step, though we only care about metrics)
        criterion = GuidedLoss(config["pos_weight"], config["dice_weight"], config["ce_weight"]).to(device)

        # 4. Evaluate
        print("  Running inference on validation set (with threshold tuning)...")
        val_loss, val_metrics, best_thr, best_cancer_f1 = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            eval_threshold=0.5,
            threshold_grid=threshold_grid
        )
        
        print(f"  [Fold {fold_name}] Val Loss: {val_loss:.4f}")
        print(f"  [Fold {fold_name}] Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"  [Fold {fold_name}] Best Threshold: {best_thr:.2f} (Cancer F1: {best_cancer_f1:.4f})")
        
        best_thresholds.append(best_thr)
        aggregated_cm += val_metrics["confusion_matrix"]

    print("\n\n" + "#"*60)
    print("  FINAL AGGREGATED EVALUATION")
    print("#"*60)
    print_aggregated_matrices(aggregated_cm)
    if best_thresholds:
        print(f"Best thresholds by fold: {[round(x, 3) for x in best_thresholds]}")
        print(f"Mean best threshold: {float(np.mean(best_thresholds)):.3f}")

if __name__ == "__main__":
    main()
