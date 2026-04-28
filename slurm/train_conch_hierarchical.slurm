#!/usr/bin/env bash
#SBATCH --job-name=sicap_conch_hier
#SBATCH --partition=CHANGE_ME_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export DIGIPATICS_ROOT="${DIGIPATICS_ROOT:-$REPO_ROOT}"

# Optional Hugging Face token for gated CONCH weights:
# export HF_TOKEN="..."

# Uncomment one:
# source .venv/bin/activate
# module load cuda/12.x
# conda activate sicap

python "$REPO_ROOT/src/training_conch_hierarchical.py" --no-wandb "$@"
