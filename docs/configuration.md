# Configuration

## Path variables

| Variable | Purpose |
|----------|---------|
| `DIGIPATICS_ROOT` | Absolute path to the folder that contains **`images/`**, **`masks/`**, and **`partition/`**. Set **before** `python` when data is not under the default locations (see [`src/paths.py`](../src/paths.py)). |
| `ARTIFACTS_ROOT` | Root for checkpoints and caches. Default: **`artifacts/`** under the repo. Each training script uses `default_checkpoint_dir("checkpoints_…")` under this root. |
| `HF_TOKEN` | Hugging Face token for gated models (CONCH, UNI, etc.) if not using CLI login. |
| `UNI2_FEATURES_DIR` | Override directory for precomputed UNI2 `.pt` features (used by `training_uni2_fast.py` before falling back to AppData / `artifacts/uni2_features`). |

## CLI overrides

Most training scripts accept `--output-dir` to override the default artifact directory for a run.

## Data layout options

1. **Legacy**: `images`, `masks`, `partition` at repository root (same level as `README.md`).
2. **Centralized**: `data/images`, `data/masks`, `data/partition` — detected automatically when present.
3. **External**: export `DIGIPATICS_ROOT=/path/to/dataset`.

## Migrating old checkpoints

Previous layouts wrote under `checkpoints_*` at repo root. New defaults use `artifacts/checkpoints_*`. Either:

- Move folders into `artifacts/` preserving names, or  
- Symlink `artifacts/checkpoints_<name>` → old location, or  
- Pass `--output-dir` / set `ARTIFACTS_ROOT` explicitly.
