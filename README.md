# SICAPv2 — Prostate histopathology tiles & pixel-level cancer segmentation

This repository supports **research on digitized prostate tissue**: extracting **tiles from whole-slide images (WSI)**, training **semantic segmentation** models, and performing **pixel-level inference for cancer detection and grading patterns** (non-cancerous tissue vs Gleason-grade regions encoded in dataset masks).

## Focus

- **WSI tiling**: fixed-size patches (e.g. 512×512) from prostate H&E slides, with patient-aware splits.
- **Pixel-level inference**: multi-class segmentation (e.g. NC, GG3, GG4, GG5) for computer-aided analysis of tumor extent and grade-related patterns.
- **Training & evaluation**: PyTorch pipelines using strong histopathology encoders (CONCH, UNI/UNI2, Nature/PCam variants) with FPN/U-Net style decoders, cross-validation over `partition/`, and optional Weights & Biases logging.

Large assets (images, masks, checkpoints, W&B runs) stay **out of Git**; this repository is code-first and keeps only lightweight metadata and documentation.

## Repository layout

| Path | Description |
|------|-------------|
| [`src/paths.py`](src/paths.py) | Single source of truth for **data root** and **artifacts** (`DIGIPATICS_ROOT`, `ARTIFACTS_ROOT`). |
| [`src/training_*.py`](src/) | Training entrypoints (CONCH hierarchical, UNI2, PCam, etc.). |
| [`scripts/`](scripts/) | Utilities: mask/partition audits, feature precompute, W&B helpers, quick verification scripts. Use [`scripts/sicap_imports.py`](scripts/sicap_imports.py) so imports resolve. |
| [`visualize_code/`](visualize_code/) | Figures and qualitative comparisons (GT vs predictions). |
| [`notebooks/`](notebooks/) | Exploratory analysis (partition, scores, masks). |
| [`data/README.md`](data/README.md) | How to place or link `images/`, `masks/`, `partition/`. |
| [`slurm/`](slurm/) | Cluster job templates (GPU training). |
| [`docs/`](docs/) | Environment and configuration details. |

Legacy layout is supported: if `images/`, `masks/`, and `partition/` sit next to this `README.md`, nothing else is required. Optionally consolidate under `data/` or set `DIGIPATICS_ROOT` to another folder (see [`docs/configuration.md`](docs/configuration.md)).

## Quick start (local)

From the repository root (activate your virtual environment first):

```bash
pip install -r requirements.txt
# CONCH (gated): huggingface-cli login  → https://huggingface.co/MahmoodLab/conch

python src/training_conch_hierarchical.py --dry-run --no-wandb
```

Full training example:

```bash
python src/training_conch_hierarchical.py --fold Val1 --no-wandb
```

Checkpoints and logs default to **`artifacts/<checkpoint_folder_name>/`** (see `src/paths.py`). If you already have runs under old `checkpoints_*` at repo root, copy or symlink them into `artifacts/` **with the same subfolder names**, or set `ARTIFACTS_ROOT` to point at a directory that contains those folders.

## Cluster (Slurm)

See [`slurm/README.md`](slurm/README.md). Create `mkdir -p slurm/logs` on Linux. Set `DIGIPATICS_ROOT` (and optionally `HF_TOKEN`, `WANDB_API_KEY`) before launching Python so paths resolve at import time.

## Dataset

The **SICAPv2** patch database uses paired JPEG tiles and label masks; patient-based folds live under `partition/`. A short description of folders and labels is in the original [`readme.txt`](readme.txt) (dataset provenance).

Some utilities expect a **`sicap_mapping.py`** helper at the repository root (same convention as older training scripts): if your clone does not include it, keep your local copy next to `README.md` or adjust imports.

## Requirements

See [`requirements.txt`](requirements.txt) and [`docs/environment.md`](docs/environment.md). A CUDA-enabled PyTorch build must match your installed NVIDIA driver/runtime.

## License & attribution

Respect the **SICAPv2** data terms and any **foundation model** licenses (CONCH, UNI/UNI2, etc.). Cite the corresponding papers and datasets in academic work.
