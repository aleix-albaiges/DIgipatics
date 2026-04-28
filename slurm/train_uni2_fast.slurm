#!/usr/bin/env bash
#SBATCH --job-name=sicap_uni2_fast
#SBATCH --partition=CHANGE_ME_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export DIGIPATICS_ROOT="${DIGIPATICS_ROOT:-$REPO_ROOT}"

# Precompute UNI2 features once (heavy I/O); point cache to scratch if needed:
# export UNI2_FEATURES_DIR=/scratch/$USER/uni2_features

python "$REPO_ROOT/src/training_uni2_fast.py" --no-wandb "$@"
