#!/usr/bin/env bash
#SBATCH --job-name=sicap_smoke
#SBATCH --partition=CHANGE_ME_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export DIGIPATICS_ROOT="${DIGIPATICS_ROOT:-$REPO_ROOT}"

python "$REPO_ROOT/src/training_conch_hierarchical.py" --dry-run --no-wandb "$@"
