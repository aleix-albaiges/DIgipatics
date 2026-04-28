# Slurm job templates

These scripts target a typical Linux GPU cluster using **bash** and **Slurm**. Adapt **`#SBATCH`** lines for your site (partition name, QoS, account, GPU type, memory, wall time).

## Prerequisites

From the repo root on the cluster:

- Python venv or conda env with dependencies from **`requirements.txt`** installed.
- Dataset reachable at **`DIGIPATICS_ROOT`** (exported in the `.slurm` file).
- **`HF_TOKEN`** if using gated Hugging Face models without cached weights.

## Logs

Jobs write stdout/stderr under **`logs/`** (create once: `mkdir -p logs` from repo root).

## Usage

Submit from the repository root:

```bash
mkdir -p logs
export DIGIPATICS_ROOT=/path/to/dataset_root   # contains images/, masks/, partition/
export HF_TOKEN=...                           # optional
sbatch slurm/train_conch_hierarchical.slurm
```

Append training CLI arguments after editing the **`python`** line inside the script, or duplicate the script per experiment.

## Variables set in templates

| Variable | Role |
|----------|------|
| `REPO_ROOT` | Repo root (`cd` target). |
| `PYTHONPATH` | Includes repo root so `python src/...` runs cleanly. |
| `DIGIPATICS_ROOT` | Defaults to `$REPO_ROOT` if unset (legacy layout with data at repo root). |
