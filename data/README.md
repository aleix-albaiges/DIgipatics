# Dataset layout (`DIGIPATICS_ROOT`)

Training scripts resolve data paths via [`src/paths.py`](../src/paths.py).

Expected layout under your data root (either **repository root** with legacy folders, or **`data/`** subfolder):

```text
<DIGIPATICS_ROOT>/
  images/       # JPEG tiles from WSI (512 px patches, etc.)
  masks/        # Grayscale masks aligned with images (label encoding per project)
  partition/    # Train/validation splits (e.g. Validation/Val1 …)
```

## Options

1. **Legacy (current workspace)** — Leave `images`, `masks`, and `partition` next to `README.md` at the repo root. No `DIGIPATICS_ROOT` needed.

2. **Centralized `data/`** — Create `data/images`, `data/masks`, `data/partition` (copy or symlink/junction). Scripts pick `PROJECT_ROOT/data` automatically when those folders exist.

3. **External disk / cluster scratch** — Set once per shell or job:

   ```bash
   export DIGIPATICS_ROOT=/path/to/SICAPv2_dataset
   ```

Do **not** commit large image or mask blobs; only this README (and optional small metadata) belong in Git.
