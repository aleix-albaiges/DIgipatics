"""
CONCH v2 variant focused on improving GG3 without collapsing GG5.

Why this variant exists:
- The current v2 run over-weights any tile with at least one GG5 pixel.
- Aggregated confusion shows GG3 precision is the weakest point, with many
  false positives coming from NC and GG4.
- The hierarchical sampler hides mixed GG3+GG4 tiles under the GG4/GG5 bucket,
  so the model sees too few ambiguous GG3 examples.

Changes versus training_conchv2.py:
- Sampler uses per-class minimum pixel thresholds instead of np.any(...).
- Optional additive multi-label sampler instead of hierarchical priority.
- Extra boost for tiles containing both GG3 and GG4.

Everything else stays aligned with training_conchv2.py so comparisons remain
clean.
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import wandb

import training_conchv2 as base
from paths import IMAGES_DIR, MASKS_DIR, PARTITION_DIR, default_checkpoint_dir

cv2.setNumThreads(0)

WANDB_PROJECT_DEFAULT = "SICAPv2_CONCH_GG3Focus"
OUTPUT_DIR = default_checkpoint_dir("checkpoints_conch_gg3_focus")

DEFAULT_CONFIG = dict(base.DEFAULT_CONFIG)
DEFAULT_CONFIG.update(
    {
        "sampler_mode": "additive",
        "sampler_boost_gg3": 1.00,
        "sampler_boost_gg4": 0.35,
        "sampler_boost_gg5": 0.85,
        "sampler_mix_boost_gg3_gg4": 1.15,
        "sampler_min_pixels_gg3": 1024,
        "sampler_min_pixels_gg4": 1024,
        "sampler_min_pixels_gg5": 512,
        "output_dir": OUTPUT_DIR,
    }
)


def _present_above_threshold(counts: np.ndarray, cls: int, min_pixels: int) -> bool:
    return int(counts[cls]) >= int(max(0, min_pixels))


def compute_sample_weights(
    image_names: list[str],
    masks_dir: Path,
    *,
    sampler_mode: str,
    boost_gg3: float,
    boost_gg4: float,
    boost_gg5: float,
    mix_boost_gg3_gg4: float,
    min_pixels_gg3: int,
    min_pixels_gg4: int,
    min_pixels_gg5: int,
) -> tuple[list[float], dict[str, int | float]]:
    weights: list[float] = []
    stats = {
        "gg3_tiles": 0,
        "gg4_tiles": 0,
        "gg5_tiles": 0,
        "gg3_gg4_tiles": 0,
        "tumor_tiles": 0,
    }

    for name in image_names:
        mask_path = masks_dir / name
        weight = 1.0
        has_gg3 = False
        has_gg4 = False
        has_gg5 = False

        if mask_path.exists():
            buf_m = np.fromfile(str(mask_path), dtype=np.uint8)
            mask = cv2.imdecode(buf_m, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mapped = base._MASK_LUT[mask]
                counts = np.bincount(mapped.ravel(), minlength=base.NUM_CLASSES)
                has_gg3 = _present_above_threshold(counts, 1, min_pixels_gg3)
                has_gg4 = _present_above_threshold(counts, 2, min_pixels_gg4)
                has_gg5 = _present_above_threshold(counts, 3, min_pixels_gg5)

                if sampler_mode == "hierarchical":
                    if has_gg5:
                        weight += boost_gg5
                    elif has_gg4:
                        weight += boost_gg4
                    elif has_gg3:
                        weight += boost_gg3
                else:
                    if has_gg3:
                        weight += boost_gg3
                    if has_gg4:
                        weight += boost_gg4
                    if has_gg5:
                        weight += boost_gg5
                    if has_gg3 and has_gg4:
                        weight *= mix_boost_gg3_gg4

        if has_gg3:
            stats["gg3_tiles"] += 1
        if has_gg4:
            stats["gg4_tiles"] += 1
        if has_gg5:
            stats["gg5_tiles"] += 1
        if has_gg3 and has_gg4:
            stats["gg3_gg4_tiles"] += 1
        if has_gg3 or has_gg4 or has_gg5:
            stats["tumor_tiles"] += 1

        weights.append(float(weight))

    if weights:
        stats["weight_mean"] = float(np.mean(weights))
        stats["weight_max"] = float(np.max(weights))
    else:
        stats["weight_mean"] = 0.0
        stats["weight_max"] = 0.0
    return weights, stats


def get_fold_dataloaders(fold_name: str, config: dict):
    fold_dir = PARTITION_DIR / "Validation" / fold_name
    train_df = pd.read_excel(fold_dir / "Train.xlsx")
    val_df = pd.read_excel(fold_dir / "Test.xlsx")
    train_names = train_df["image_name"].tolist()
    val_names = val_df["image_name"].tolist()

    train_ds = base.SICAPv2Dataset(
        train_names,
        IMAGES_DIR,
        MASKS_DIR,
        transform=base.get_train_transforms(),
    )
    val_ds = base.SICAPv2Dataset(
        val_names,
        IMAGES_DIR,
        MASKS_DIR,
        transform=base.get_val_transforms(),
    )

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", config["num_workers"]))
    kwargs = {"persistent_workers": True, "prefetch_factor": 4} if workers > 0 else {}

    seed = int(config.get("seed", base.DEFAULT_SEED))
    gen = torch.Generator()
    gen.manual_seed(seed)
    winit = partial(base._worker_init_fn, base_seed=seed) if workers > 0 else None
    dl_common = dict(
        num_workers=workers,
        pin_memory=True,
        generator=gen,
        worker_init_fn=winit,
    )

    if config.get("use_weighted_sampler", False):
        weights, stats = compute_sample_weights(
            train_names,
            MASKS_DIR,
            sampler_mode=str(config.get("sampler_mode", "additive")),
            boost_gg3=float(config.get("sampler_boost_gg3", 1.0)),
            boost_gg4=float(config.get("sampler_boost_gg4", 0.35)),
            boost_gg5=float(config.get("sampler_boost_gg5", 0.85)),
            mix_boost_gg3_gg4=float(config.get("sampler_mix_boost_gg3_gg4", 1.15)),
            min_pixels_gg3=int(config.get("sampler_min_pixels_gg3", 1024)),
            min_pixels_gg4=int(config.get("sampler_min_pixels_gg4", 1024)),
            min_pixels_gg5=int(config.get("sampler_min_pixels_gg5", 512)),
        )
        print(
            "  [Sampler] "
            f"mode={config.get('sampler_mode')} | train={len(train_names)} | "
            f"GG3={stats['gg3_tiles']} | GG4={stats['gg4_tiles']} | GG5={stats['gg5_tiles']} | "
            f"GG3+GG4={stats['gg3_gg4_tiles']} | mean_w={stats['weight_mean']:.3f} | "
            f"max_w={stats['weight_max']:.3f}"
        )
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_names), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            sampler=sampler,
            drop_last=True,
            **dl_common,
            **kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=True,
            **dl_common,
            **kwargs,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        **dl_common,
        **kwargs,
    )
    return train_loader, val_loader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CONCH v2 variant with GG3-focused thresholded/additive sampler"
    )
    parser.add_argument("--dry-run", action="store_true", help="Run exactly 1 epoch to validate the pipeline.")
    parser.add_argument("--weights", type=str, default=None, help="Local CONCH checkpoint (.bin/.pt).")
    parser.add_argument("--hf-token", type=str, default=None, help="Optional HF token for gated CONCH download.")
    parser.add_argument("--unfreeze-last", type=int, default=6, help="Last ViT blocks to unfreeze.")
    parser.add_argument("--batch-size", type=int, default=None, help="Micro-batch size.")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps.")
    parser.add_argument("--no-weighted-sampler", action="store_true", help="Disable sampler and use shuffle=True.")
    parser.add_argument(
        "--sampler-mode",
        type=str,
        choices=["additive", "hierarchical"],
        default=None,
        help="Additive uses all present classes; hierarchical keeps one priority bucket.",
    )
    parser.add_argument("--sampler-boost-gg3", type=float, default=None, metavar="B", help="Additive/hierarchical GG3 boost.")
    parser.add_argument("--sampler-boost-gg4", type=float, default=None, metavar="B", help="Additive/hierarchical GG4 boost.")
    parser.add_argument("--sampler-boost-gg5", type=float, default=None, metavar="B", help="Additive/hierarchical GG5 boost.")
    parser.add_argument(
        "--sampler-mix-boost-gg3-gg4",
        type=float,
        default=None,
        metavar="M",
        help="Extra multiplicative boost for tiles containing both GG3 and GG4.",
    )
    parser.add_argument("--sampler-min-pixels-gg3", type=int, default=None, metavar="N", help="Minimum GG3 pixels to count for the sampler.")
    parser.add_argument("--sampler-min-pixels-gg4", type=int, default=None, metavar="N", help="Minimum GG4 pixels to count for the sampler.")
    parser.add_argument("--sampler-min-pixels-gg5", type=int, default=None, metavar="N", help="Minimum GG5 pixels to count for the sampler.")
    parser.add_argument("--compile", action="store_true", help="Force torch.compile.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile.")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT_DEFAULT, help="wandb project.")
    parser.add_argument("--wandb-name", type=str, default=None, help="Optional wandb run name.")
    parser.add_argument(
        "--fold",
        type=str,
        nargs="+",
        default=None,
        choices=["Val1", "Val2", "Val3", "Val4"],
        help="Run one or multiple folds. If omitted, all 4 folds run.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for .pth checkpoints.")
    parser.add_argument("--seed", type=int, default=base.DEFAULT_SEED, help=f"Random seed (default: {base.DEFAULT_SEED}).")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    base.set_seed(args.seed)
    print(f"Seed: {args.seed} (cudnn deterministic=True, benchmark=False)")

    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
            print("\n[System] Anti-sleep enabled.")
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = DEFAULT_CONFIG.copy()
    config["seed"] = args.seed
    config["unfreeze_last"] = args.unfreeze_last
    if args.weights:
        config["conch_checkpoint"] = args.weights
    if args.hf_token:
        config["conch_hf_token"] = args.hf_token
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        config["grad_accum_steps"] = max(1, args.grad_accum)
    if args.no_weighted_sampler:
        config["use_weighted_sampler"] = False
    if args.sampler_mode is not None:
        config["sampler_mode"] = args.sampler_mode
    if args.sampler_boost_gg3 is not None:
        config["sampler_boost_gg3"] = float(args.sampler_boost_gg3)
    if args.sampler_boost_gg4 is not None:
        config["sampler_boost_gg4"] = float(args.sampler_boost_gg4)
    if args.sampler_boost_gg5 is not None:
        config["sampler_boost_gg5"] = float(args.sampler_boost_gg5)
    if args.sampler_mix_boost_gg3_gg4 is not None:
        config["sampler_mix_boost_gg3_gg4"] = float(args.sampler_mix_boost_gg3_gg4)
    if args.sampler_min_pixels_gg3 is not None:
        config["sampler_min_pixels_gg3"] = int(args.sampler_min_pixels_gg3)
    if args.sampler_min_pixels_gg4 is not None:
        config["sampler_min_pixels_gg4"] = int(args.sampler_min_pixels_gg4)
    if args.sampler_min_pixels_gg5 is not None:
        config["sampler_min_pixels_gg5"] = int(args.sampler_min_pixels_gg5)

    if args.compile and args.no_compile:
        print("  [WARN] --compile and --no-compile provided together; using --no-compile.")
    if args.no_compile:
        config["use_compile"] = False
    elif args.compile:
        config["use_compile"] = True

    config["output_dir"] = Path(args.output_dir).resolve() if args.output_dir else OUTPUT_DIR
    print(f"Checkpoints -> {config['output_dir']}")

    eff = config["batch_size"] * config.get("grad_accum_steps", 1)
    print(
        f"CONCH GG3-focus: micro_batch={config['batch_size']} x accum={config.get('grad_accum_steps', 1)} "
        f"~= {eff} | weighted_sampler={config.get('use_weighted_sampler')} | lr={config['learning_rate']}"
    )
    print(
        "  class_weights (CE, sqrt-inv freq partition): "
        f"{config['class_weights']} | dice/ce: {config['dice_weight']}/{config['ce_weight']} | "
        f"lr_plateau_patience={config.get('lr_plateau_patience', 3)}"
    )
    print(
        "  sampler cfg: "
        f"mode={config.get('sampler_mode')} | boosts="
        f"({config.get('sampler_boost_gg3')}, {config.get('sampler_boost_gg4')}, {config.get('sampler_boost_gg5')}) | "
        f"mix_gg3_gg4={config.get('sampler_mix_boost_gg3_gg4')} | "
        f"min_pixels=({config.get('sampler_min_pixels_gg3')}, {config.get('sampler_min_pixels_gg4')}, {config.get('sampler_min_pixels_gg5')})"
    )

    use_wandb = not args.no_wandb
    if use_wandb and not args.dry_run:
        try:
            wandb_config = {
                "script": "training_conchv2_gg3_focus",
                "img_size": base.IMG_SIZE,
                "encoder": "CONCH_ViT-B-16_visual",
                "fpn_channels": config["fpn_channels"],
                "batch_size": config["batch_size"],
                "grad_accum_steps": config.get("grad_accum_steps", 1),
                "effective_batch": eff,
                "learning_rate": config["learning_rate"],
                "weight_decay": config["weight_decay"],
                "max_epochs": config["max_epochs"],
                "patience": config["patience"],
                "dice_weight": config["dice_weight"],
                "ce_weight": config["ce_weight"],
                "class_weights": list(config["class_weights"]),
                "unfreeze_last": config["unfreeze_last"],
                "weighted_sampler": config.get("use_weighted_sampler", False),
                "sampler_mode": config.get("sampler_mode"),
                "sampler_boost_gg3": config.get("sampler_boost_gg3"),
                "sampler_boost_gg4": config.get("sampler_boost_gg4"),
                "sampler_boost_gg5": config.get("sampler_boost_gg5"),
                "sampler_mix_boost_gg3_gg4": config.get("sampler_mix_boost_gg3_gg4"),
                "sampler_min_pixels_gg3": config.get("sampler_min_pixels_gg3"),
                "sampler_min_pixels_gg4": config.get("sampler_min_pixels_gg4"),
                "sampler_min_pixels_gg5": config.get("sampler_min_pixels_gg5"),
                "conch_checkpoint": config.get("conch_checkpoint") or base.CONCH_HF_CHECKPOINT,
                "output_dir": str(config["output_dir"]),
                "mask_lut": "43:85->GG3, 85:160->GG4, 160:255->GG5, else->NC",
                "pixel_frac_partition": [float(x) for x in base._PIXEL_FRAC_PARTITION],
                "lr_plateau_patience": config.get("lr_plateau_patience", 3),
                "seed": config.get("seed", base.DEFAULT_SEED),
                "fold": args.fold,
            }
            run_name = args.wandb_name or f"CONCH_GG3Focus_bs{config['batch_size']}_ga{config.get('grad_accum_steps', 1)}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=wandb_config,
                tags=["mask_lut", "CONCH", "SICAPv2", "gg3_focus"],
            )
        except Exception as exc:
            print(f"  [WARN] Weights & Biases unavailable: {exc}")
            use_wandb = False

    base.get_fold_dataloaders = get_fold_dataloaders

    if args.fold:
        fold_names = args.fold if isinstance(args.fold, list) else [args.fold]
    else:
        fold_names = ["Val1", "Val2", "Val3", "Val4"]

    aggregated_cm = np.zeros((base.NUM_CLASSES, base.NUM_CLASSES), dtype=np.int64)

    for fold in fold_names:
        res = base.train_fold(fold, config, device, args.dry_run, use_wandb=use_wandb)
        if res["best_cm"] is not None:
            aggregated_cm += res["best_cm"]

    base.print_aggregated_matrices(aggregated_cm)
    if use_wandb and wandb.run is not None:
        agg_log = {"aggregated/macro_f1": base._aggregated_macro_f1_from_cm(aggregated_cm)}
        for c, name in enumerate(base.CLASS_NAMES):
            tp = aggregated_cm[c, c]
            fp = aggregated_cm[:, c].sum() - tp
            fn = aggregated_cm[c, :].sum() - tp
            f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else 0.0
            agg_log[f"aggregated/f1_{name}"] = float(f1)
        wandb.log(agg_log)
    if args.dry_run:
        print("\nDry run completed successfully.")
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
