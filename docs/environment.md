# Environment

## Python

Use **Python 3.10+** (3.11 recommended). Create a virtual environment at the repo root, or reuse your existing environment (e.g., `prostata_env`, kept **gitignored**).

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

## PyTorch / CUDA

Install a **Torch build** that matches your **NVIDIA driver** (see [pytorch.org](https://pytorch.org/get-started/locally/)). For CPU-only smoke tests, CPU wheels are enough; training expects a CUDA GPU.

## CONCH (Mahmood Lab)

The CONCH vision encoder is **gated on Hugging Face**. Accept the model card, then either:

- `huggingface-cli login`, or  
- pass `--hf-token` / set `HF_TOKEN` for non-interactive jobs.

## Optional services

- **Weights & Biases**: `wandb login` or `WANDB_API_KEY` in the environment. Use `--no-wandb` for offline runs.

## Slurm / cluster

On shared systems, load your site's **CUDA module** or activate **conda** before `sbatch`. Do not assume partition names from templates; replace `CHANGE_ME_gpu` (and QoS/account fields if your cluster requires them).
