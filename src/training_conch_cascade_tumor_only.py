"""
Binary-to-grade cascade for SICAPv2 with CONCH.

Motivation:
- Direct 4-class training spends most of its capacity separating NC vs Cancer.
- The existing hierarchical branch still trained Stage 2 as a 4-class problem, so
  it never fully focused on GG3/GG4/GG5 discrimination.

This variant changes the factorization:
1. Stage 1: reuse a strong binary NC-vs-Cancer checkpoint.
2. Stage 2: train a 3-class grade segmenter only on tumor grades {GG3, GG4, GG5}.
3. Final prediction: Stage 1 gates cancer area, Stage 2 assigns the grade there.

That is a much more aggressive change than tuning v2 and has materially higher upside.
"""

from __future__ import annotations

import argparse
import os
import platform
import warnings
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import wandb

import training_conch_hierarchical as base
from paths import IMAGES_DIR, MASKS_DIR, PARTITION_DIR, default_checkpoint_dir

warnings.filterwarnings("ignore", category=UserWarning)
cv2.setNumThreads(0)

WANDB_PROJECT_DEFAULT = "SICAPv2_CONCH_cascade_tumor_only"
OUTPUT_DIR = default_checkpoint_dir("checkpoints_conch_cascade_tumor_only")
STAGE1_BINARY_CKPT_DIR_DEFAULT = default_checkpoint_dir("checkpoints_conch_binary")

CLASS_NAMES_4 = ["NC", "GG3", "GG4", "GG5"]
CLASS_NAMES_3 = ["GG3", "GG4", "GG5"]
NUM_CLASSES_STAGE1 = 2
NUM_CLASSES_STAGE2 = 3
IGNORE_INDEX = 255

_TUMOR_PIXEL_FRAC = np.array([0.0605, 0.1175, 0.0439], dtype=np.float64)


def _sqrt_inv_weights(freqs: np.ndarray) -> list[float]:
    w = np.sqrt(1.0 / np.asarray(freqs, dtype=np.float64))
    w = w / w.min()
    return [float(round(x, 3)) for x in w]


DEFAULT_CLASS_WEIGHTS_STAGE2 = _sqrt_inv_weights(_TUMOR_PIXEL_FRAC)

DEFAULT_CONFIG = dict(base.DEFAULT_CONFIG)
DEFAULT_CONFIG.update(
    {
        "weight_decay": 2e-4,
        "decoder_dropout": 0.1,
        "label_smoothing": 0.0,
        "class_weights_stage2": list(DEFAULT_CLASS_WEIGHTS_STAGE2),
        "max_epochs_stage2": 100,
        "patience_stage2": 18,
        "sampler_weight_gg3": 1.0,
        "sampler_weight_gg4": 1.5,
        "sampler_weight_gg5": 3.0,
        "train_only_tumor_images": True,
        "stage1_external_ckpt_dir": str(STAGE1_BINARY_CKPT_DIR_DEFAULT.resolve()),
        "stage1_external_top_k": 3,
        "stage1_threshold_policy": "macro_f1",
        "stage1_min_cancer_recall": 0.95,
        "stage2_gate_dilation": 1,
        "stage1_infer_gate_dilation": 1,
        "calibrate_stage1_threshold": True,
        "stage1_calibration_min": 0.10,
        "stage1_calibration_max": 0.60,
        "stage1_calibration_step": 0.05,
        "keep_tumor_outside_gate": True,
        "enable_stage2_rescue": False,
        "stage2_rescue_threshold": 0.90,
        "output_dir": OUTPUT_DIR,
    }
)


def read_mask4(name: str, image_shape: tuple[int, int] | None = None) -> np.ndarray:
    mask_path = MASKS_DIR / name
    if mask_path.exists():
        buf = np.fromfile(str(mask_path), dtype=np.uint8)
        raw = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    else:
        raw = None
    if raw is None:
        if image_shape is None:
            image_shape = (512, 512)
        return np.zeros(image_shape, dtype=np.uint8)
    return base._MASK_LUT[raw].astype(np.uint8)


def grade_target_from_mask4(mask4: torch.Tensor) -> torch.Tensor:
    target = torch.full_like(mask4, IGNORE_INDEX)
    target[mask4 == 1] = 0
    target[mask4 == 2] = 1
    target[mask4 == 3] = 2
    return target


def scan_mask_properties(name: str) -> tuple[bool, int]:
    mask4 = read_mask4(name)
    has_tumor = bool(np.any(mask4 > 0))
    highest_grade = int(mask4.max())
    return has_tumor, highest_grade


def compute_stage2_sample_weights(
    image_names: list[str],
    weight_gg5: float,
    weight_gg4: float,
    weight_gg3: float,
) -> list[float]:
    weights: list[float] = []
    for name in image_names:
        _, highest_grade = scan_mask_properties(name)
        if highest_grade >= 3:
            weights.append(float(weight_gg5))
        elif highest_grade == 2:
            weights.append(float(weight_gg4))
        else:
            weights.append(float(weight_gg3))
    return weights


def checkpoint_score_from_name(path: Path) -> float | None:
    stem = path.stem
    try:
        if stem.startswith("best_stage1_"):
            return float(stem.split("_")[3])
        return float(stem.split("_")[-1])
    except (IndexError, ValueError):
        return None


def find_external_stage1_candidates(ckpt_dir: Path, fold_name: str, top_k: int = 0) -> list[Path]:
    candidates: list[tuple[float, Path]] = []
    for pat in (f"best_{fold_name}_*.pth", f"best_stage1_{fold_name}_*.pth"):
        for path in ckpt_dir.glob(pat):
            score = checkpoint_score_from_name(path)
            if score is None:
                continue
            candidates.append((float(score), path))
    candidates.sort(key=lambda x: (x[0], x[1].name), reverse=True)
    paths = [path for _, path in candidates]
    if top_k and top_k > 0:
        paths = paths[:top_k]
    return paths


def stage1_candidate_key(stats: dict, policy: str, min_cancer_recall: float) -> tuple:
    pol = str(policy).strip().lower()
    macro_f1 = float(stats["macro_f1"])
    cancer_f1 = float(stats["cancer_f1"])
    cancer_precision = float(stats["cancer_precision"])
    cancer_recall = float(stats["cancer_recall"])
    if pol == "cancer_recall_constrained":
        feasible = 1 if cancer_recall >= float(min_cancer_recall) else 0
        return (feasible, macro_f1, cancer_f1, cancer_precision, cancer_recall)
    if pol == "cancer_f1":
        return (cancer_f1, macro_f1, cancer_precision, cancer_recall)
    return (macro_f1, cancer_f1, cancer_precision, cancer_recall)


class GradeOnlyDataset(Dataset):
    def __init__(self, image_names: list[str], *, transform=None):
        self.image_names = image_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        name = self.image_names[idx]
        img_path = IMAGES_DIR / name
        buf = np.fromfile(str(img_path), dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask4 = read_mask4(name, image_shape=image.shape[:2])
        if self.transform:
            transformed = self.transform(image=image, mask=mask4)
            image = transformed["image"]
            mask4 = transformed["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask4 = torch.from_numpy(mask4)

        if not torch.is_tensor(mask4):
            mask4 = torch.from_numpy(np.asarray(mask4))
        mask4 = mask4.long()
        grade_target = grade_target_from_mask4(mask4)
        return image, grade_target, mask4


def get_stage2_dataloaders(fold_name: str, config: dict):
    fold_dir = PARTITION_DIR / "Validation" / fold_name
    train_df = pd.read_excel(fold_dir / "Train.xlsx")
    val_df = pd.read_excel(fold_dir / "Test.xlsx")
    train_names = train_df["image_name"].tolist()
    val_names = val_df["image_name"].tolist()

    if config.get("train_only_tumor_images", True):
        train_names = [name for name in train_names if scan_mask_properties(name)[0]]

    train_ds = GradeOnlyDataset(train_names, transform=base.get_train_transforms())
    val_ds = GradeOnlyDataset(val_names, transform=base.get_val_transforms())

    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", config["num_workers"]))
    kwargs = {"persistent_workers": True, "prefetch_factor": 4} if workers > 0 else {}
    seed = int(config.get("seed", base.DEFAULT_SEED))
    gen = torch.Generator()
    gen.manual_seed(seed)
    winit = partial(base._worker_init_fn, base_seed=seed) if workers > 0 else None
    dl_common = dict(num_workers=workers, pin_memory=True, generator=gen, worker_init_fn=winit)

    if config.get("use_weighted_sampler", False):
        w5 = float(config["sampler_weight_gg5"])
        w4 = float(config["sampler_weight_gg4"])
        w3 = float(config["sampler_weight_gg3"])
        sw = compute_stage2_sample_weights(train_names, w5, w4, w3)
        n5 = sum(1 for x in sw if x == w5)
        n4 = sum(1 for x in sw if x == w4)
        n3 = sum(1 for x in sw if x == w3)
        print(
            f"  [Stage2 Sampler] train={len(train_names)} | "
            f"GG5×{w5}={n5} | GG4×{w4}={n4} | GG3×{w3}={n3}"
        )
        sampler = WeightedRandomSampler(weights=sw, num_samples=len(train_names), replacement=True)
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
    print(
        f"  [Stage2 Data] train_images={len(train_names)} | val_images={len(val_names)} | "
        f"tumor_only_train={bool(config.get('train_only_tumor_images', True))}"
    )
    return train_loader, val_loader


class TumorGradeLoss(nn.Module):
    def __init__(
        self,
        class_weights: list[float],
        *,
        dice_weight: float = 0.55,
        ce_weight: float = 0.45,
        smooth: float = 1e-6,
        label_smoothing: float = 0.0,
        ignore_index: int = IGNORE_INDEX,
    ):
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.ignore_index = int(ignore_index)
        self.dice_loss = smp.losses.DiceLoss(
            mode="multiclass",
            classes=list(range(NUM_CLASSES_STAGE2)),
            smooth=smooth,
            ignore_index=self.ignore_index,
        )
        self.register_buffer("weights_tensor", torch.tensor(class_weights).float())
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.weights_tensor,
            ignore_index=self.ignore_index,
            label_smoothing=float(label_smoothing),
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.dice_weight * dice + self.ce_weight * ce


def maybe_compile(model: nn.Module, use_compile) -> nn.Module:
    if use_compile is None:
        use_compile = platform.system() != "Windows"
    if use_compile and int(torch.__version__.split(".")[0]) >= 2:
        try:
            backend = "inductor" if platform.system() != "Windows" else "aot_eager"
            print("  ⏳ Compiling model with torch.compile...")
            model = torch.compile(model, backend=backend)
            print("  ✅ torch.compile enabled.")
        except Exception as exc:
            print(f"  ⚠️ torch.compile unavailable: {exc}")
    elif not use_compile:
        print("  ℹ️ torch.compile disabled (recommended on Windows to save VRAM).")
    return model


def load_stage1_from_external_binary(
    fold_name: str,
    config: dict,
    device: torch.device,
    use_wandb: bool,
) -> dict:
    ckpt_dir = Path(config["stage1_external_ckpt_dir"]).resolve()
    top_k = int(config.get("stage1_external_top_k", 0) or 0)
    candidates = find_external_stage1_candidates(ckpt_dir, fold_name, top_k=top_k)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint matching best_{fold_name}_*.pth or best_stage1_{fold_name}_*.pth in {ckpt_dir}"
        )

    print(
        f"  [Stage1 External] Evaluating {len(candidates)} binary checkpoint candidate(s) "
        f"from {ckpt_dir}"
    )
    _, val_loader = base.get_fold_dataloaders(fold_name, config, binary=True)
    criterion = base.GuidedLoss(
        num_classes=NUM_CLASSES_STAGE1,
        class_weights=config["class_weights_stage1"],
        dice_weight=config["dice_weight_stage1"],
        ce_weight=config["ce_weight_stage1"],
    ).to(device)
    thresholds = [float(x) for x in config.get("binary_threshold_candidates", [0.5])]
    ext_th = config.get("stage1_external_threshold")
    policy = str(config.get("stage1_threshold_policy", "macro_f1"))
    min_rec = float(config.get("stage1_min_cancer_recall", 0.95))
    drop_s1 = float(config.get("stage1_decoder_dropout_for_external", 0.0))

    best: dict | None = None
    for ckpt_path in candidates:
        model = base.build_model({**config, "decoder_dropout": drop_s1}, num_classes=NUM_CLASSES_STAGE1).to(device)
        state = torch.load(ckpt_path, map_location=device)
        res = model.load_state_dict(state, strict=False)
        if res and (res.missing_keys or res.unexpected_keys):
            print(
                f"  [WARN] Stage1 load strict=False missing={len(res.missing_keys)} "
                f"unexpected={len(res.unexpected_keys)}"
            )

        val_loss, val_bin = base.validate_binary_with_threshold_search(model, val_loader, criterion, device, thresholds)
        if ext_th is not None:
            th = float(ext_th)
            cms = val_bin["cms_by_threshold"]
            cm = None
            for key, value in cms.items():
                if abs(float(key) - th) < 1e-6:
                    cm = value
                    break
            if cm is None:
                cm = base.accumulate_binary_cm_at_threshold(model, val_loader, device, th)
            stats = base.compute_binary_stats_from_cm(cm)
            stats["confusion_matrix"] = cm
        else:
            th, stats = base.select_stage1_threshold(
                val_bin["cms_by_threshold"],
                policy=policy,
                min_cancer_recall=min_rec,
            )
        score = stage1_candidate_key(stats, policy=policy, min_cancer_recall=min_rec)
        print(
            f"  [Stage1 External] {ckpt_path.name} | th={float(th):.2f} | "
            f"macro-F1={float(stats['macro_f1']):.4f} | "
            f"cancer_recall={float(stats['cancer_recall']):.4f} | "
            f"cancer_precision={float(stats['cancer_precision']):.4f}"
        )
        candidate = {
            "fold": fold_name,
            "macro_f1": float(stats["macro_f1"]),
            "threshold": float(th),
            "cm": stats["confusion_matrix"].copy(),
            "path": ckpt_path,
            "stats": {k: v for k, v in stats.items() if k != "confusion_matrix"},
            "decoder_dropout_override": drop_s1,
            "from_external_binary": True,
            "val_loss_external": float(val_loss),
            "score": score,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    assert best is not None
    print(
        f"  [Stage1 External] Selected {best['path'].name} | threshold={best['threshold']:.4f} | "
        f"macro-F1={best['macro_f1']:.4f} | "
        f"cancer_recall={best['stats']['cancer_recall']:.4f} | "
        f"cancer_precision={best['stats']['cancer_precision']:.4f}"
    )
    if use_wandb and wandb.run is not None:
        wandb.log(
            {
                f"{fold_name}/s1_external_path": str(best["path"]),
                f"{fold_name}/s1_external_macro_f1": float(best["macro_f1"]),
                f"{fold_name}/s1_external_threshold": float(best["threshold"]),
                f"{fold_name}/s1_external_cancer_recall": float(best["stats"]["cancer_recall"]),
                f"{fold_name}/s1_external_cancer_precision": float(best["stats"]["cancer_precision"]),
            }
        )
    best.pop("score", None)
    return best


def load_stage1_model(
    config: dict,
    device: torch.device,
    stage1_path: Path,
    *,
    decoder_dropout_override: float | None = None,
):
    s1_cfg = dict(config)
    if decoder_dropout_override is not None:
        s1_cfg["decoder_dropout"] = float(decoder_dropout_override)
    model = base.build_model(s1_cfg, num_classes=NUM_CLASSES_STAGE1).to(device)
    state = torch.load(stage1_path, map_location=device)
    res = model.load_state_dict(state, strict=False)
    if res and (res.missing_keys or res.unexpected_keys):
        print(
            f"  [WARN] Stage1 load strict=False missing={len(res.missing_keys)} "
            f"unexpected={len(res.unexpected_keys)}"
        )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def build_stage2_model(config: dict) -> nn.Module:
    model = base.build_model(config, num_classes=NUM_CLASSES_STAGE2)
    return model


def train_one_epoch_stage2_cascade(
    model,
    stage1_model,
    stage1_threshold: float,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    *,
    gate_threshold: float,
    gate_dilation: int,
    keep_tumor_outside_gate: bool,
    grad_accum_steps: int = 1,
):
    model.train()
    stage1_model.eval()
    total_loss, num_batches = 0.0, 0
    accum = 0
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pbar = tqdm(loader, desc="  TrainS2", leave=False)
    for images, grade_targets, mask4 in pbar:
        images = images.to(device, non_blocking=True)
        grade_targets = grade_targets.to(device, non_blocking=True).long()
        mask4 = mask4.to(device, non_blocking=True).long()

        if accum == 0:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
                logits1 = stage1_model(images)
            p_cancer = torch.softmax(logits1.float(), dim=1)[:, 1]
            gate_pred = p_cancer >= float(gate_threshold)
            gate_pred = base.dilate_gate_tensor(gate_pred, gate_dilation)

        valid_mask = gate_pred
        if keep_tumor_outside_gate:
            valid_mask = valid_mask | (mask4 > 0)

        valid_pixels = int(valid_mask.sum().item())
        if valid_pixels == 0:
            pbar.set_postfix(loss="skip(empty-gate)")
            continue

        targets = grade_targets.clone()
        targets[~valid_mask] = IGNORE_INDEX

        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            logits3 = model(images)
            loss = criterion(logits3, targets)

        scaler.scale(loss / grad_accum_steps).backward()
        total_loss += float(loss.item())
        num_batches += 1
        accum += 1
        if accum >= grad_accum_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            accum = 0
        gate_ratio = float(valid_mask.float().mean().item())
        pbar.set_postfix(loss=f"{loss.item():.4f}", gate_pct=f"{gate_ratio * 100:.1f}")

    if accum > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate_cascade(
    stage1_model,
    stage2_model,
    loader,
    device,
    *,
    stage1_threshold: float,
    infer_gate_dilation: int,
    enable_stage2_rescue: bool,
    stage2_rescue_threshold: float,
    threshold_grid: list[float] | None = None,
):
    stage1_model.eval()
    stage2_model.eval()
    metrics3 = base.SegmentationMetrics(3)
    if not threshold_grid:
        threshold_grid = [float(stage1_threshold)]
    threshold_grid = [float(x) for x in threshold_grid]
    metrics4 = {th: base.SegmentationMetrics(4) for th in threshold_grid}
    gate_ratio_sum = {th: 0.0 for th in threshold_grid}
    gate_batches = {th: 0 for th in threshold_grid}
    rescued_pixels = {th: 0 for th in threshold_grid}
    total_pixels = {th: 0 for th in threshold_grid}
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    pbar = tqdm(loader, desc="  ValC ", leave=False)

    for images, _grade_targets, mask4 in pbar:
        images = images.to(device, non_blocking=True)
        mask4 = mask4.to(device, non_blocking=True).long()

        with autocast("cuda", enabled=(device.type == "cuda"), dtype=amp_dtype):
            logits1 = stage1_model(images)
            logits3 = stage2_model(images)

        p_cancer = torch.softmax(logits1.float(), dim=1)[:, 1]
        prob3 = torch.softmax(logits3.float(), dim=1)
        pred3 = prob3.argmax(dim=1)
        conf3 = prob3.max(dim=1).values

        tumor_mask = mask4 > 0
        if int(tumor_mask.sum().item()) > 0:
            metrics3.update_batch(
                pred3[tumor_mask].cpu().numpy(),
                (mask4[tumor_mask] - 1).cpu().numpy(),
            )

        mask4_np = mask4.cpu().numpy()
        for th in threshold_grid:
            gate = p_cancer >= float(th)
            gate = base.dilate_gate_tensor(gate, infer_gate_dilation)
            pred4 = torch.zeros_like(mask4)
            pred4[gate] = pred3[gate] + 1

            if enable_stage2_rescue:
                rescue = (~gate) & (conf3 >= float(stage2_rescue_threshold))
                pred4[rescue] = pred3[rescue] + 1
                rescued_pixels[th] += int(rescue.sum().item())
                total_pixels[th] += int(rescue.numel())
            else:
                total_pixels[th] += int(gate.numel())

            gate_ratio_sum[th] += float(gate.float().mean().item())
            gate_batches[th] += 1
            metrics4[th].update_batch(pred4.cpu().numpy(), mask4_np)

    out3 = metrics3.compute()
    best_out = None
    for th in threshold_grid:
        out4 = metrics4[th].compute()
        out4["gate_ratio"] = gate_ratio_sum[th] / max(gate_batches[th], 1)
        out4["rescued_ratio"] = float(rescued_pixels[th] / max(total_pixels[th], 1))
        out4["grade_macro_f1_tumor_only"] = out3["macro_f1"]
        out4["grade_f1_per_class_tumor_only"] = out3["f1_per_class"]
        out4["best_stage1_threshold"] = float(th)
        if best_out is None:
            best_out = out4
            continue
        key_cur = (
            float(out4["macro_f1"]),
            float(out4["f1_per_class"][0]),
            float(out4["f1_per_class"][3]),
            float(th),
        )
        key_best = (
            float(best_out["macro_f1"]),
            float(best_out["f1_per_class"][0]),
            float(best_out["f1_per_class"][3]),
            float(best_out["best_stage1_threshold"]),
        )
        if key_cur > key_best:
            best_out = out4
    assert best_out is not None
    return best_out


def train_fold_stage2(
    fold_name: str,
    config: dict,
    device: torch.device,
    stage1_path: Path,
    stage1_threshold: float,
    stage1_decoder_dropout_override: float | None,
    *,
    dry_run: bool,
    use_wandb: bool,
):
    print(f"\n{'=' * 60}\n  FOLD {fold_name} | STAGE 2 (Tumor-only GG3/GG4/GG5)\n{'=' * 60}")
    train_loader, val_loader = get_stage2_dataloaders(fold_name, config)
    model = build_stage2_model(config).to(device)
    model = maybe_compile(model, config.get("use_compile"))
    stage1_model = load_stage1_model(
        config,
        device,
        stage1_path,
        decoder_dropout_override=stage1_decoder_dropout_override,
    )

    criterion = TumorGradeLoss(
        config["class_weights_stage2"],
        dice_weight=float(config.get("dice_weight_stage2", 0.55)),
        ce_weight=float(config.get("ce_weight_stage2", 0.45)),
        label_smoothing=float(config.get("label_smoothing", 0.0)),
        ignore_index=IGNORE_INDEX,
    ).to(device)

    inner = base._trainable_model(model)
    encoder_params = [p for p in inner.encoder.parameters() if p.requires_grad]
    decoder_params = list(inner.decoder.parameters())
    enc_ratio = max(1, int(config.get("encoder_lr_ratio", 10)))
    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": config["learning_rate"] / float(enc_ratio)})
    param_groups.append({"params": decoder_params, "lr": config["learning_rate"]})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=float(config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=int(config.get("lr_plateau_patience", 3)),
        min_lr=1e-7,
    )
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    out_dir = Path(config.get("output_dir", OUTPUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)
    max_epochs = 1 if dry_run else int(config.get("max_epochs_stage2", 100))
    patience_max = int(config.get("patience_stage2", 18))
    ga = int(config.get("grad_accum_steps", 1))

    best = {"macro_f1": 0.0, "cm": None, "path": None}
    patience = 0
    train_gate_threshold = config.get("stage2_train_threshold")
    if train_gate_threshold is None:
        train_gate_threshold = float(stage1_threshold)
    train_gate_threshold = float(train_gate_threshold)
    train_gate_dilation = int(config.get("stage2_gate_dilation", 1))
    keep_tumor_outside = bool(config.get("keep_tumor_outside_gate", True))
    infer_gate_dilation = int(config.get("stage1_infer_gate_dilation", 1))
    threshold_grid = None
    if config.get("calibrate_stage1_threshold", True):
        tmin = float(config.get("stage1_calibration_min", 0.10))
        tmax = float(config.get("stage1_calibration_max", 0.60))
        tstep = float(config.get("stage1_calibration_step", 0.05))
        threshold_grid = np.arange(tmin, tmax + 1e-12, tstep, dtype=np.float64).tolist()
        if float(stage1_threshold) not in threshold_grid:
            threshold_grid.append(float(stage1_threshold))
            threshold_grid = sorted(set(float(x) for x in threshold_grid))

    print(
        f"  [S2][LR] encoder_lr_ratio={enc_ratio} (decoder={config['learning_rate']:.2e}) | "
        f"train_gate_threshold={train_gate_threshold:.2f} | infer_gate_threshold(seed)={float(stage1_threshold):.2f}"
    )

    for epoch in range(1, max_epochs + 1):
        lr_dec = optimizer.param_groups[-1]["lr"]
        print(f"\n  [S2] Epoch {epoch}/{max_epochs}  (decoder lr={lr_dec:.2e}, accum={ga})")
        train_loss = train_one_epoch_stage2_cascade(
            model=model,
            stage1_model=stage1_model,
            stage1_threshold=stage1_threshold,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            gate_threshold=train_gate_threshold,
            gate_dilation=train_gate_dilation,
            keep_tumor_outside_gate=keep_tumor_outside,
            grad_accum_steps=ga,
        )
        val_cascade = validate_cascade(
            stage1_model,
            model,
            val_loader,
            device,
            stage1_threshold=float(stage1_threshold),
            infer_gate_dilation=infer_gate_dilation,
            enable_stage2_rescue=bool(config.get("enable_stage2_rescue", False)),
            stage2_rescue_threshold=float(config.get("stage2_rescue_threshold", 0.90)),
            threshold_grid=threshold_grid,
        )

        macro_f1 = float(val_cascade["macro_f1"])
        scheduler.step(macro_f1)
        print(f"  [S2] TrainLoss={train_loss:.4f}")
        print(
            f"  [S2] Cascade Macro-F1(4c)={macro_f1:.4f} | "
            f"best_stage1_threshold={val_cascade['best_stage1_threshold']:.2f}"
        )
        for i, name in enumerate(CLASS_NAMES_4):
            print(f"    {name} F1: {val_cascade['f1_per_class'][i]:.4f}")
        print(
            f"  [S2] Tumor-only grade Macro-F1(3c)={val_cascade['grade_macro_f1_tumor_only']:.4f} | "
            f"gate_ratio={val_cascade['gate_ratio']:.3f} | rescued_ratio={val_cascade['rescued_ratio']:.4f}"
        )

        if use_wandb and wandb.run is not None:
            enc_lr = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 1 else optimizer.param_groups[-1]["lr"]
            dec_lr = optimizer.param_groups[-1]["lr"]
            log_dict = {
                "epoch": int(epoch),
                f"{fold_name}/s2_train_loss": float(train_loss),
                f"{fold_name}/s2_macro_f1_4class": macro_f1,
                f"{fold_name}/s2_macro_f1_tumor_only": float(val_cascade["grade_macro_f1_tumor_only"]),
                f"{fold_name}/s2_gate_ratio": float(val_cascade["gate_ratio"]),
                f"{fold_name}/s2_rescued_ratio": float(val_cascade["rescued_ratio"]),
                f"{fold_name}/s2_lr_encoder": float(enc_lr),
                f"{fold_name}/s2_lr_decoder": float(dec_lr),
                f"{fold_name}/s2_train_gate_threshold": float(train_gate_threshold),
                f"{fold_name}/s1_infer_threshold_seed": float(stage1_threshold),
                f"{fold_name}/s1_infer_threshold": float(val_cascade["best_stage1_threshold"]),
            }
            for i, name in enumerate(CLASS_NAMES_4):
                log_dict[f"{fold_name}/f1_{name}"] = float(val_cascade["f1_per_class"][i])
            for i, name in enumerate(CLASS_NAMES_3):
                log_dict[f"{fold_name}/tumor_f1_{name}"] = float(val_cascade["grade_f1_per_class_tumor_only"][i])
            wandb.log(log_dict)

        if macro_f1 > best["macro_f1"]:
            best["macro_f1"] = macro_f1
            best["cm"] = val_cascade["confusion_matrix"]
            best["stage1_threshold"] = float(val_cascade["best_stage1_threshold"])
            best_path = out_dir / (
                f"best_stage2_tumoronly_{fold_name}_{macro_f1:.4f}_th{best['stage1_threshold']:.2f}.pth"
            )
            torch.save(base._trainable_model(model).state_dict(), best_path)
            best["path"] = best_path
            patience = 0
            print("  [S2] ✓ checkpoint saved")
        else:
            patience += 1
            if patience >= patience_max:
                print(f"  [S2] Early stopping at epoch {epoch}")
                break
        if dry_run:
            break

    return best


def evaluate_fold(
    fold_name: str,
    config: dict,
    device: torch.device,
    stage1_path: Path,
    stage2_path: Path,
    stage1_threshold: float,
    stage1_decoder_dropout_override: float | None,
):
    _, val_loader = get_stage2_dataloaders(fold_name, config)
    stage1_model = load_stage1_model(
        config,
        device,
        stage1_path,
        decoder_dropout_override=stage1_decoder_dropout_override,
    )
    stage2_model = build_stage2_model(config).to(device)
    stage2_model.load_state_dict(torch.load(stage2_path, map_location=device))
    return validate_cascade(
        stage1_model,
        stage2_model,
        val_loader,
        device,
        stage1_threshold=float(stage1_threshold),
        infer_gate_dilation=int(config.get("stage1_infer_gate_dilation", 1)),
        enable_stage2_rescue=bool(config.get("enable_stage2_rescue", False)),
        stage2_rescue_threshold=float(config.get("stage2_rescue_threshold", 0.90)),
    )


def print_aggregated_results(cm: np.ndarray):
    print(f"\n{'=' * 60}\n  AGGREGATED CONFUSION MATRIX (4-CLASS)\n{'=' * 60}")
    df = pd.DataFrame(
        cm,
        index=[f"T_{name}" for name in CLASS_NAMES_4],
        columns=[f"P_{name}" for name in CLASS_NAMES_4],
    )
    print(df.to_string())
    print("\n--- 4-Class Metrics ---")
    for i, cname in enumerate(CLASS_NAMES_4):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = (2.0 * precision * recall) / (precision + recall + 1e-8)
        print(f"  {cname:3s}: F1={f1:.4f}  Prec={precision:.4f}  Rec={recall:.4f}")

    tp_c = cm[1:, 1:].sum()
    fp_c = cm[0, 1:].sum()
    fn_c = cm[1:, 0].sum()
    precision_c = tp_c / (tp_c + fp_c + 1e-8)
    recall_c = tp_c / (tp_c + fn_c + 1e-8)
    f1_c = (2.0 * precision_c * recall_c) / (precision_c + recall_c + 1e-8)
    acc = cm.trace() / (cm.sum() + 1e-8)
    print("\n--- Binary Derived Metrics ---")
    print(f"  Cancer F1 (derived) : {f1_c:.4f}")
    print(f"  Overall Accuracy    : {acc:.4f}")


def save_summary_csv(out_dir: Path, rows: list[dict]):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "best_per_fold.csv", index=False)
    df.sort_values("macro_f1", ascending=False).to_csv(out_dir / "checkpoint_ranking.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="CONCH cascade: binary gate + tumor-only grade segmenter")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--unfreeze-last", type=int, default=0)
    parser.add_argument("--weights", type=str, default=None, metavar="PATH")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None, metavar="K")
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--sampler-gg5", type=float, default=None)
    parser.add_argument("--sampler-gg4", type=float, default=None)
    parser.add_argument("--sampler-gg3", type=float, default=None)
    parser.add_argument("--decoder-dropout", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT_DEFAULT)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--fold", type=str, nargs="+", default=None, choices=["Val1", "Val2", "Val3", "Val4"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    parser.add_argument("--max-epochs-stage2", type=int, default=None)
    parser.add_argument("--patience-stage2", type=int, default=None)
    parser.add_argument("--stage1-ckpt-dir", type=str, default=None)
    parser.add_argument("--stage1-external-top-k", type=int, default=None)
    parser.add_argument("--stage1-external-threshold", type=float, default=None)
    parser.add_argument(
        "--stage1-threshold-policy",
        type=str,
        choices=["macro_f1", "cancer_recall_constrained", "cancer_f1"],
        default=None,
    )
    parser.add_argument("--stage1-min-cancer-recall", type=float, default=None)
    parser.add_argument("--no-threshold-calibration", action="store_true")
    parser.add_argument("--calibration-th-min", type=float, default=None)
    parser.add_argument("--calibration-th-max", type=float, default=None)
    parser.add_argument("--calibration-th-step", type=float, default=None)
    parser.add_argument("--stage2-train-threshold", type=float, default=None)
    parser.add_argument("--stage2-gate-dilation", type=int, default=None)
    parser.add_argument("--stage1-infer-gate-dilation", type=int, default=None)
    parser.add_argument("--enable-stage2-rescue", action="store_true")
    parser.add_argument("--stage2-rescue-threshold", type=float, default=None)
    parser.add_argument("--train-all-images", action="store_true")
    parser.add_argument("--no-keep-tumor-outside-gate", action="store_true")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    if args.unfreeze_last is not None:
        config["unfreeze_last"] = int(args.unfreeze_last)
    if args.weights is not None:
        config["conch_checkpoint"] = args.weights
    if args.hf_token is not None:
        config["conch_hf_token"] = args.hf_token
    if args.batch_size is not None:
        config["batch_size"] = int(args.batch_size)
    if args.grad_accum is not None:
        config["grad_accum_steps"] = int(args.grad_accum)
    if args.no_weighted_sampler:
        config["use_weighted_sampler"] = False
    if args.sampler_gg5 is not None:
        config["sampler_weight_gg5"] = float(args.sampler_gg5)
    if args.sampler_gg4 is not None:
        config["sampler_weight_gg4"] = float(args.sampler_gg4)
    if args.sampler_gg3 is not None:
        config["sampler_weight_gg3"] = float(args.sampler_gg3)
    if args.decoder_dropout is not None:
        config["decoder_dropout"] = float(args.decoder_dropout)
    if args.weight_decay is not None:
        config["weight_decay"] = float(args.weight_decay)
    if args.label_smoothing is not None:
        config["label_smoothing"] = float(args.label_smoothing)
    if args.compile:
        config["use_compile"] = True
    if args.no_compile:
        config["use_compile"] = False
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.max_epochs_stage2 is not None:
        config["max_epochs_stage2"] = int(args.max_epochs_stage2)
    if args.patience_stage2 is not None:
        config["patience_stage2"] = int(args.patience_stage2)
    if args.stage1_ckpt_dir is not None:
        config["stage1_external_ckpt_dir"] = str(Path(args.stage1_ckpt_dir).resolve())
    if args.stage1_external_top_k is not None:
        config["stage1_external_top_k"] = int(args.stage1_external_top_k)
    if args.stage1_external_threshold is not None:
        config["stage1_external_threshold"] = float(args.stage1_external_threshold)
    if args.stage1_threshold_policy is not None:
        config["stage1_threshold_policy"] = args.stage1_threshold_policy
    if args.stage1_min_cancer_recall is not None:
        config["stage1_min_cancer_recall"] = float(args.stage1_min_cancer_recall)
    if args.no_threshold_calibration:
        config["calibrate_stage1_threshold"] = False
    if args.calibration_th_min is not None:
        config["stage1_calibration_min"] = float(args.calibration_th_min)
    if args.calibration_th_max is not None:
        config["stage1_calibration_max"] = float(args.calibration_th_max)
    if args.calibration_th_step is not None:
        config["stage1_calibration_step"] = float(args.calibration_th_step)
    if args.stage2_train_threshold is not None:
        config["stage2_train_threshold"] = float(args.stage2_train_threshold)
    if args.stage2_gate_dilation is not None:
        config["stage2_gate_dilation"] = int(args.stage2_gate_dilation)
    if args.stage1_infer_gate_dilation is not None:
        config["stage1_infer_gate_dilation"] = int(args.stage1_infer_gate_dilation)
    if args.enable_stage2_rescue:
        config["enable_stage2_rescue"] = True
    if args.stage2_rescue_threshold is not None:
        config["stage2_rescue_threshold"] = float(args.stage2_rescue_threshold)
    if args.train_all_images:
        config["train_only_tumor_images"] = False
    if args.no_keep_tumor_outside_gate:
        config["keep_tumor_outside_gate"] = False

    folds = args.fold or ["Val1", "Val2", "Val3", "Val4"]
    config["seed"] = int(args.seed)
    base.set_seed(config["seed"])

    use_wandb = not args.no_wandb
    if use_wandb:
        rn = args.wandb_name or f"cascade_tumoronly_u{config['unfreeze_last']}_bs{config['batch_size']}"
        wandb.init(
            project=args.wandb_project,
            name=rn,
            config={
                "script": "training_conch_cascade_tumor_only",
                "fold": folds,
                "batch_size": config["batch_size"],
                "grad_accum_steps": config["grad_accum_steps"],
                "learning_rate": config["learning_rate"],
                "weight_decay": config["weight_decay"],
                "unfreeze_last": config["unfreeze_last"],
                "decoder_dropout": config["decoder_dropout"],
                "label_smoothing": config["label_smoothing"],
                "stage1_external_ckpt_dir": config["stage1_external_ckpt_dir"],
                "stage1_external_top_k": config.get("stage1_external_top_k", 0),
                "stage1_threshold_policy": config["stage1_threshold_policy"],
                "stage1_min_cancer_recall": config["stage1_min_cancer_recall"],
                "calibrate_stage1_threshold": config.get("calibrate_stage1_threshold", True),
                "stage1_calibration_min": config.get("stage1_calibration_min"),
                "stage1_calibration_max": config.get("stage1_calibration_max"),
                "stage1_calibration_step": config.get("stage1_calibration_step"),
                "stage2_gate_dilation": config["stage2_gate_dilation"],
                "stage1_infer_gate_dilation": config["stage1_infer_gate_dilation"],
                "enable_stage2_rescue": config["enable_stage2_rescue"],
                "stage2_rescue_threshold": config["stage2_rescue_threshold"],
                "train_only_tumor_images": config["train_only_tumor_images"],
                "keep_tumor_outside_gate": config["keep_tumor_outside_gate"],
                "class_weights_stage2": config["class_weights_stage2"],
            },
            tags=["CONCH", "SICAPv2", "cascade", "tumor-only", "grades"],
        )

    eff = int(config["batch_size"]) * int(config.get("grad_accum_steps", 1))
    print(
        f"CONCH cascade tumor-only: micro_batch={config['batch_size']} × accum={config.get('grad_accum_steps', 1)} "
        f"≈ {eff} | weighted_sampler={config.get('use_weighted_sampler')} | lr={config['learning_rate']}"
    )
    print(
        f"  stage1 ckpt dir={config['stage1_external_ckpt_dir']} | "
        f"top_k={config.get('stage1_external_top_k', 0)} | "
        f"threshold_policy={config['stage1_threshold_policy']} | "
        f"min_cancer_recall={config['stage1_min_cancer_recall']}"
    )
    print(
        f"  stage2 class_weights={config['class_weights_stage2']} | "
        f"dice/ce={config['dice_weight_stage2']}/{config['ce_weight_stage2']} | "
        f"weight_decay={config['weight_decay']} | decoder_dropout={config['decoder_dropout']} | "
        f"label_smoothing={config.get('label_smoothing', 0.0)}"
    )
    print(
        f"  train_only_tumor_images={config['train_only_tumor_images']} | "
        f"keep_tumor_outside_gate={config['keep_tumor_outside_gate']} | "
        f"train_gate_dilation={config['stage2_gate_dilation']} | infer_gate_dilation={config['stage1_infer_gate_dilation']}"
    )
    print(
        f"  threshold_calibration={config.get('calibrate_stage1_threshold', True)} | "
        f"calibration_grid=[{config.get('stage1_calibration_min')}, {config.get('stage1_calibration_max')}] "
        f"step {config.get('stage1_calibration_step')}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aggregated_cm = np.zeros((4, 4), dtype=np.int64)
    best_rows: list[dict] = []

    for fold in folds:
        s1 = load_stage1_from_external_binary(fold, config, device, use_wandb)
        s2 = train_fold_stage2(
            fold,
            config,
            device,
            stage1_path=s1["path"],
            stage1_threshold=float(s1["threshold"]),
            stage1_decoder_dropout_override=s1.get("decoder_dropout_override"),
            dry_run=args.dry_run,
            use_wandb=use_wandb,
        )
        if s2["path"] is None:
            raise RuntimeError(f"Stage2 did not produce a checkpoint for {fold}")
        final_stage1_threshold = float(s2.get("stage1_threshold", s1["threshold"]))
        val_h = evaluate_fold(
            fold,
            config,
            device,
            stage1_path=s1["path"],
            stage2_path=s2["path"],
            stage1_threshold=final_stage1_threshold,
            stage1_decoder_dropout_override=s1.get("decoder_dropout_override"),
        )
        aggregated_cm += val_h["confusion_matrix"]
        best_rows.append(
            {
                "fold": fold,
                "macro_f1": float(val_h["macro_f1"]),
                "tumor_macro_f1": float(val_h["grade_macro_f1_tumor_only"]),
                "stage1_threshold_seed": float(s1["threshold"]),
                "stage1_threshold_final": final_stage1_threshold,
                "stage1_path": str(s1["path"]),
                "stage2_path": str(s2["path"]),
                "f1_NC": float(val_h["f1_per_class"][0]),
                "f1_GG3": float(val_h["f1_per_class"][1]),
                "f1_GG4": float(val_h["f1_per_class"][2]),
                "f1_GG5": float(val_h["f1_per_class"][3]),
            }
        )
        if use_wandb and wandb.run is not None:
            wandb.log(
                {
                    f"{fold}/best_macro_f1_4class": float(val_h["macro_f1"]),
                    f"{fold}/best_macro_f1_tumor_only": float(val_h["grade_macro_f1_tumor_only"]),
                    f"{fold}/best_f1_NC": float(val_h["f1_per_class"][0]),
                    f"{fold}/best_f1_GG3": float(val_h["f1_per_class"][1]),
                    f"{fold}/best_f1_GG4": float(val_h["f1_per_class"][2]),
                    f"{fold}/best_f1_GG5": float(val_h["f1_per_class"][3]),
                    f"{fold}/best_stage1_threshold": final_stage1_threshold,
                    f"{fold}/best_stage1_threshold_seed": float(s1["threshold"]),
                }
            )

    macro_f1 = base._macro_f1_from_cm(aggregated_cm)
    print_aggregated_results(aggregated_cm)
    out_dir = Path(config.get("output_dir", OUTPUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_summary_csv(out_dir, best_rows)

    if use_wandb and wandb.run is not None:
        per_class = []
        for i in range(4):
            tp = aggregated_cm[i, i]
            fp = aggregated_cm[:, i].sum() - tp
            fn = aggregated_cm[i, :].sum() - tp
            per_class.append(float((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)))
        wandb.log(
            {
                "aggregated/macro_f1": float(macro_f1),
                "aggregated/f1_NC": per_class[0],
                "aggregated/f1_GG3": per_class[1],
                "aggregated/f1_GG4": per_class[2],
                "aggregated/f1_GG5": per_class[3],
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
