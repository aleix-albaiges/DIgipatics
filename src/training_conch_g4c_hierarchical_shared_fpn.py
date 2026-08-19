"""
Train a hierarchical GG4C model with a shared FPN:

  shared CONCH encoder
    -> shared FPN trunk
       -> grade head: NC, GG3, GG4_merged, GG5
       -> GG4C head: non-cribriform GG4 vs cribriform GG4C

This keeps the hierarchical constraint from training_conch_g4c_hierarchical.py,
but avoids running two full FPN decoders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler

from paths import default_checkpoint_dir
from training_conchv2 import (
    CONCH_NORM_MEAN,
    CONCH_NORM_STD,
    ConcHEncoder,
    build_cosine_warmup_scheduler,
)
from training_conchv2_g4c_5class import (
    CLASS_NAMES as CLASS_NAMES_5,
    MERGED_CLASS_NAMES,
    NUM_CLASSES as NUM_CLASSES_5,
    build_mask_lut,
    clean_state_dict,
)
from training_conch_g4c_hierarchical import (
    DEFAULT_BASE_CHECKPOINT,
    DEFAULT_SEED,
    GRADE_CLASS_NAMES,
    NUM_GRADE_CLASSES,
    OUTPUT_DIR as HIER_OUTPUT_DIR,
    HierarchicalG4CLoss,
    ModelEMA,
    build_grade_lut,
    collect_final_names,
    compute_training_stats,
    hierarchical_logits_to_5class,
    make_loader,
    print_metrics,
    read_split,
    save_checkpoint,
    selection_score,
    set_seed,
    split_paths,
    train_one_epoch,
    trainable_model,
    validate_one_epoch,
)


OUTPUT_DIR = default_checkpoint_dir("checkpoints_conch_masklut_g4c_hierarchical_shared_fpn")
CONFIG_LINEAGE = {
    "source": "Run I training_conchv2 4-class config",
    "heldout_test_macro_f1_4class": 0.7097,
    "heldout_test_binary_cancer_f1": 0.7384,
    "changes": [
        "shared FPN decoder trunk",
        "4-class grade head for NC/GG3/GG4_merged/GG5",
        "binary GG4C head supervised only on true GG4/GG4C pixels",
    ],
}


class SharedFPNHierDecoder(nn.Module):
    def __init__(self, in_channels=768, fpn_channels=256):
        super().__init__()
        self.lat = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, fpn_channels, 1, bias=False),
                    nn.BatchNorm2d(fpn_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(4)
            ]
        )
        self.merge = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(fpn_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(3)
            ]
        )
        self.head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.grade_head = nn.Conv2d(fpn_channels // 2, NUM_GRADE_CLASSES, 1)
        self.g4c_head = nn.Conv2d(fpn_channels // 2, 2, 1)

    def forward(self, features, target_size):
        lats = [lat(f) for lat, f in zip(self.lat, features)]
        x = lats[3]
        for i in range(2, -1, -1):
            x = F.interpolate(x, size=lats[i].shape[-2:], mode="bilinear", align_corners=False)
            x = self.merge[i](x + lats[i])
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = self.head(x)
        return {
            "grade_logits": self.grade_head(x),
            "g4c_logits": self.g4c_head(x),
        }


class CONCHHierG4CSharedFPNModel(nn.Module):
    def __init__(self, fpn_channels=256, unfreeze_last=0, weights_path=None, hf_token=None):
        super().__init__()
        self.encoder = ConcHEncoder(
            unfreeze_last=unfreeze_last,
            weights_path=weights_path,
            hf_token=hf_token,
        )
        self.decoder = SharedFPNHierDecoder(
            in_channels=self.encoder.embed_dim,
            fpn_channels=fpn_channels,
        )

    def forward(self, x):
        target_size = x.shape[-2:]
        features = self.encoder(x)
        return self.decoder(features, target_size)


def _copy_old_gg4_to_binary_head(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    out = target.clone()
    out[0].copy_(src[2])
    out[1].copy_(src[2])
    return out


def load_4class_checkpoint(model: CONCHHierG4CSharedFPNModel, checkpoint_path: Path) -> None:
    raw = torch.load(checkpoint_path, map_location="cpu")
    source = clean_state_dict(raw)
    target = model.state_dict()
    adapted = {k: v.clone() for k, v in target.items()}
    loaded_shared = []
    loaded_grade = []
    loaded_g4c = []
    skipped = []

    for key, src in source.items():
        if key.startswith("encoder.") and key in adapted and tuple(src.shape) == tuple(adapted[key].shape):
            adapted[key] = src
            continue
        if not key.startswith("decoder."):
            continue

        suffix = key[len("decoder.") :]
        shared_key = "decoder." + suffix
        if shared_key in adapted and tuple(src.shape) == tuple(adapted[shared_key].shape):
            adapted[shared_key] = src
            loaded_shared.append(shared_key)
            continue

        if suffix.startswith("head.3."):
            param_name = suffix[len("head.3.") :]
            grade_key = "decoder.grade_head." + param_name
            g4c_key = "decoder.g4c_head." + param_name
            if grade_key in adapted and tuple(src.shape) == tuple(adapted[grade_key].shape):
                adapted[grade_key] = src
                loaded_grade.append(grade_key)
            else:
                skipped.append(grade_key)
            if g4c_key in adapted and src.ndim >= 1 and src.shape[0] == 4 and adapted[g4c_key].shape[0] == 2:
                adapted[g4c_key] = _copy_old_gg4_to_binary_head(src, adapted[g4c_key])
                loaded_g4c.append(g4c_key)

    model.load_state_dict(adapted, strict=True)
    print(f"  [Init] Loaded 4-class checkpoint: {checkpoint_path}")
    print(f"  [Init] Shared FPN keys loaded: {len(loaded_shared)}")
    print(f"  [Init] Grade head keys loaded: {len(loaded_grade)}")
    print(f"  [Init] GG4C head keys loaded: {len(loaded_g4c)}")
    if skipped:
        print(f"  [Init] Skipped keys: {len(skipped)}; first={skipped[:5]}")


def build_optimizer(model: CONCHHierG4CSharedFPNModel, learning_rate: float, weight_decay: float):
    groups = []
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": learning_rate / 10, "label": "encoder"})
    groups.append({"params": model.decoder.parameters(), "lr": learning_rate, "label": "shared_fpn_decoder"})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hierarchical CONCH GG4C model with shared FPN.")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--fold", choices=["Val1", "Val2", "Val3", "Val4", "Test"], default="Val3")
    parser.add_argument("--train-xlsx", type=Path, default=None, help="Training Excel used directly with no validation set.")
    parser.add_argument("--final-train", action="store_true")
    parser.add_argument("--final-folds", nargs="+", choices=["Val1", "Val2", "Val3", "Val4", "Test"], default=None)
    parser.add_argument("--g4c-gray-min", type=int, default=125)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint-name", type=str, default=None)
    parser.add_argument("--selection-metric", choices=["macro_f1", "merged_macro_f1", "g4c_f1", "merged_plus_g4c"], default="merged_plus_g4c")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=4.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--fpn-channels", type=int, default=256)
    parser.add_argument("--unfreeze-last", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--grade-class-weights", type=str, default=None)
    parser.add_argument("--g4c-pos-weight", type=float, default=None)
    parser.add_argument("--grade-dice-weight", type=float, default=0.55)
    parser.add_argument("--grade-ce-weight", type=float, default=0.45)
    parser.add_argument("--g4c-dice-weight", type=float, default=0.55)
    parser.add_argument("--g4c-ce-weight", type=float, default=0.45)
    parser.add_argument("--g4c-loss-weight", type=float, default=1.0)
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--sampler-gg5", type=float, default=1.8)
    parser.add_argument("--sampler-gg4c", type=float, default=2.2)
    parser.add_argument("--sampler-gg4", type=float, default=1.3)
    parser.add_argument("--sampler-gg3", type=float, default=1.8)
    parser.add_argument("--sampler-replacement", action="store_true")
    parser.add_argument("--no-color-aug", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--conch-checkpoint", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-summary-only", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Architecture: shared FPN hierarchical GG4C")
    print(f"Hierarchical LUT: 75:175->GG4_merged; {args.g4c_gray_min}:175->GG4C inside binary head")
    mask_lut_5 = build_mask_lut(args.g4c_gray_min)
    grade_lut = build_grade_lut(args.g4c_gray_min)

    if args.train_xlsx is not None:
        train_names = read_split(args.train_xlsx)
        val_names = []
        data_summary = {
            "training_source": str(args.train_xlsx),
            "train_rows": len(train_names),
            "validation": None,
            "deduplicated": False,
        }
    elif args.final_train:
        folds = args.final_folds or ["Val1", "Val2", "Val3", "Val4"]
        train_names, data_summary = collect_final_names(folds)
        val_names = []
    else:
        train_path, val_path = split_paths(args.fold)
        train_names = read_split(train_path)
        val_names = read_split(val_path)
        data_summary = {
            "training_source": str(train_path),
            "validation": str(val_path),
            "train_rows": len(train_names),
            "val_rows": len(val_names),
        }
    print(f"Data summary: {json.dumps(data_summary, indent=2)}")

    stats = compute_training_stats(train_names, mask_lut_5, grade_lut)
    if args.grade_class_weights is not None:
        grade_weights = [float(x.strip()) for x in args.grade_class_weights.split(",") if x.strip()]
        if len(grade_weights) != NUM_GRADE_CLASSES:
            parser.error("--grade-class-weights must contain exactly 4 values.")
    else:
        grade_weights = stats["grade_weights"]
    g4c_pos_weight = float(args.g4c_pos_weight) if args.g4c_pos_weight is not None else float(stats["g4c_pos_weight"])
    print("Grade pixel frequencies: " + ", ".join(f"{n}={stats['grade_freq'][i]:.6f}" for i, n in enumerate(GRADE_CLASS_NAMES)))
    print(f"Grade class weights: {grade_weights}")
    print(f"GG4C pixels: pos={stats['g4c_pos_pixels']} neg={stats['g4c_neg_pixels']} pos_weight={g4c_pos_weight:.4f}")

    if args.data_summary_only:
        return

    config = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "norm_mean": list(CONCH_NORM_MEAN),
        "norm_std": list(CONCH_NORM_STD),
        "color_aug_enabled": not args.no_color_aug,
        "use_weighted_sampler": not args.no_weighted_sampler,
        "sampler_weight_gg5": args.sampler_gg5,
        "sampler_weight_gg4c": args.sampler_gg4c,
        "sampler_weight_gg4": args.sampler_gg4,
        "sampler_weight_gg3": args.sampler_gg3,
        "sampler_replacement": args.sampler_replacement,
    }
    train_loader = make_loader(train_names, mask_lut_5, grade_lut, config, train=True)
    val_loader = make_loader(val_names, mask_lut_5, grade_lut, config, train=False) if val_names else None

    model = CONCHHierG4CSharedFPNModel(
        fpn_channels=args.fpn_channels,
        unfreeze_last=args.unfreeze_last,
        weights_path=args.conch_checkpoint,
        hf_token=args.hf_token,
    )
    load_4class_checkpoint(model, args.base_checkpoint)
    model = model.to(device)

    ema = ModelEMA(model, args.ema_decay) if args.ema else None
    if ema is not None:
        print(f"EMA enabled: decay={args.ema_decay}")

    if args.compile and args.no_compile:
        print("  [WARN] both --compile and --no-compile passed; using --no-compile.")
    elif args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled.")
        except Exception as exc:
            print(f"  [WARN] torch.compile failed: {exc}")

    criterion = HierarchicalG4CLoss(
        grade_weights=grade_weights,
        g4c_pos_weight=g4c_pos_weight,
        grade_dice_weight=args.grade_dice_weight,
        grade_ce_weight=args.grade_ce_weight,
        g4c_dice_weight=args.g4c_dice_weight,
        g4c_ce_weight=args.g4c_ce_weight,
        g4c_loss_weight=args.g4c_loss_weight,
    ).to(device)
    optimizer = build_optimizer(trainable_model(model), args.learning_rate, args.weight_decay)
    steps_per_epoch = max(1, len(train_loader) // max(1, args.grad_accum))
    total_steps = steps_per_epoch * (1 if args.dry_run else args.max_epochs)
    scheduler = build_cosine_warmup_scheduler(optimizer, total_steps=total_steps, warmup_pct=0.07, min_lr_ratio=0.01)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_path = None
    patience = 0
    max_epochs = 1 if args.dry_run else args.max_epochs
    ema_source = trainable_model(model) if ema is not None else None
    train_stats = None

    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch {epoch}/{max_epochs} | lr={optimizer.param_groups[-1]['lr']:.2e}")
        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            grad_accum_steps=max(1, args.grad_accum),
            scheduler=scheduler,
            ema=ema,
            ema_source=ema_source,
        )
        print(
            f"  Train loss: {train_stats['loss']:.4f} | "
            f"grade={train_stats['grade_loss']:.4f} | g4c={train_stats['g4c_loss']:.4f}"
        )

        if val_loader is None:
            continue

        val_model = ema.ema if ema is not None else model
        val_loss, metrics = validate_one_epoch(val_model, val_loader, criterion, device)
        print(f"  Val loss: {val_loss:.4f}")
        print_metrics(metrics)
        score = selection_score(metrics, args.selection_metric)
        print(f"  Selection score ({args.selection_metric}): {score:.4f}")

        if score > best_score:
            best_score = score
            patience = 0
            name = args.checkpoint_name or f"best_{args.fold}_shared_fpn_hier_g4c_{args.selection_metric}_{best_score:.4f}.pth"
            best_path = out_dir / name
            save_checkpoint(
                best_path,
                val_model,
                {
                    "epoch": epoch,
                    "fold": args.fold,
                    "architecture": "hierarchical_shared_fpn_4class_plus_g4c_binary",
                    "config_lineage": CONFIG_LINEAGE,
                    "base_checkpoint": str(args.base_checkpoint),
                    "g4c_gray_min": int(args.g4c_gray_min),
                    "grade_class_names": GRADE_CLASS_NAMES,
                    "class_names": CLASS_NAMES_5,
                    "merged_class_names": MERGED_CLASS_NAMES,
                    "num_classes": int(NUM_CLASSES_5),
                    "grade_class_weights": grade_weights,
                    "g4c_pos_weight": g4c_pos_weight,
                    "train_loss": train_stats,
                    "val_loss": float(val_loss),
                    "selection_metric": args.selection_metric,
                    "selection_score": float(best_score),
                    "macro_f1": float(metrics["macro_f1"]),
                    "f1_per_class": [float(x) for x in metrics["f1_per_class"]],
                    "precision_per_class": [float(x) for x in metrics["precision_per_class"]],
                    "recall_per_class": [float(x) for x in metrics["recall_per_class"]],
                    "confusion_matrix": metrics["confusion_matrix"].tolist(),
                    "merged_macro_f1": float(metrics["merged_macro_f1"]),
                    "merged_f1_per_class": [float(x) for x in metrics["merged_f1_per_class"]],
                    "merged_precision_per_class": [float(x) for x in metrics["merged_precision_per_class"]],
                    "merged_recall_per_class": [float(x) for x in metrics["merged_recall_per_class"]],
                    "merged_confusion_matrix": metrics["merged_confusion_matrix"].tolist(),
                    "data_summary": data_summary,
                },
            )
            print(f"  Saved: {best_path}")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if val_loader is None:
        final_model = ema.ema if ema is not None else model
        name = args.checkpoint_name or "final_shared_fpn_hier_g4c.pth"
        final_path = out_dir / name
        save_checkpoint(
            final_path,
            final_model,
            {
                "epoch": max_epochs,
                "trained_epochs": int(max_epochs),
                "architecture": "hierarchical_shared_fpn_4class_plus_g4c_binary",
                "config_lineage": CONFIG_LINEAGE,
                "base_checkpoint": str(args.base_checkpoint),
                "g4c_gray_min": int(args.g4c_gray_min),
                "grade_class_names": GRADE_CLASS_NAMES,
                "class_names": CLASS_NAMES_5,
                "merged_class_names": MERGED_CLASS_NAMES,
                "num_classes": int(NUM_CLASSES_5),
                "grade_class_weights": grade_weights,
                "g4c_pos_weight": g4c_pos_weight,
                "training_source": data_summary.get("training_source"),
                "final_train_loss": train_stats,
                "data_summary": data_summary,
            },
        )
        print(f"\nFinal checkpoint: {final_path}")
    else:
        print(f"\nBest selection score ({args.selection_metric}): {best_score:.4f}")
        print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
