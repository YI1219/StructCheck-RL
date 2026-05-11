# Slurm example scripts

These templates target clusters that run jobs inside an **Apptainer/Singularity** (or compatible) image via `srun --container-image`. They are **not** one-size-fits-all: edit `#SBATCH` directives and exports for your site.

## Common environment variables

| Variable | Role |
|----------|------|
| `REPO_ROOT` | Checkout root (default: `SLURM_SUBMIT_DIR` at submit time). |
| `CONTAINER_IMAGE` | Path or URI to `.sqsh` / image (**required** in the templates). |
| `CONTAINER_MOUNTS` | Bind mounts for `srun` (default: `${HOME}:${HOME}`). |
| `WORKDIR` | Working directory inside the job (default: `REPO_ROOT`). |
| `BASE_MODEL` | Base VLM directory (**required** for train + eval examples). |
| `TRAIN_JSONL` / `EVAL_JSONL` | Data paths (**required** for training template). |
| `OUT_DIR` | Training output (default: `REPO_ROOT/outputs/structcheck_exp_sft_<timestamp>`). |
| `LOG_ROOT` | Host path for rank logs (default: `$HOME/logs`). |
| `PY` / `PIP` / `TORCHRUN` | Override if not using `/opt/conda/envs/llm/...` inside the image. |
| `ADAPTER_DIR` | SFT output dir (**required** for `eval_adapter_1gpu_test_s.sbatch`). |

Training also honors `GPUS_PER_NODE`, `MAX_STEPS`, `LR`, `MAX_LENGTH`, etc. (see the `.sbatch` file).

## Outputs

Slurm stdout/stderr default to `slurm-<jobname>-<jobid>_*.out` / `.err` in the submit directory unless your `#SBATCH` lines override them.

## Site policy example

`SITE_POLICY.txt` documents one cluster’s rule about **not** using `#SBATCH --gres`; your center may differ.
