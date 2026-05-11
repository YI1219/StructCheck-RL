#!/usr/bin/env bash
# Cluster policy: do NOT add #SBATCH --gres=... or srun --gres=... (jobs may stall PD; GPUs come from partition/node).
# See scripts/slurm/SITE_POLICY.txt
# Default name test_s_1; override at submit: sbatch --job-name=test_s_7 ...  or: ./scripts/slurm/submit_test_s_sft.sh 7
#
# Portable usage: submit from repo root (or set REPO_ROOT) and export CONTAINER_IMAGE, BASE_MODEL, TRAIN_JSONL, EVAL_JSONL.
# See scripts/slurm/README.md and the root README.md.
#
# --- Edit for your Slurm site ---
#SBATCH --job-name=test_s_1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --partition=b3-poc-polyu
#SBATCH --account=b3-poc-polyu
#SBATCH --qos=qos_b3-poc-polyu
#SBATCH --output=slurm-%x-%j_%N.out
#SBATCH --error=slurm-%x-%j_%N.err

set -euo pipefail
set -x

: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to your Apptainer/Singularity image (.sqsh or URI)}"
: "${BASE_MODEL:?Set BASE_MODEL to the base VLM directory}"
: "${TRAIN_JSONL:?Set TRAIN_JSONL to training JSONL}"
: "${EVAL_JSONL:?Set EVAL_JSONL to eval JSONL}"

: "${GPUS_PER_NODE:=8}"
: "${MAX_STEPS:=800}"
: "${LR:=1e-5}"
: "${MAX_LENGTH:=2048}"
: "${PER_DEVICE_BATCH_SIZE:=1}"
: "${GRAD_ACCUM:=1}"
: "${SFT_EXTRA_ARGS:=--balance-max-factor 8}"
: "${LORA_R:=32}"
: "${LORA_ALPHA:=0}"
: "${COMPLETION_EXTRA_COPIES:=Completed:2,In Progress:1}"
: "${COMPLETION_VOTES:=1}"
: "${EXTRA_TRAIN_JSONL:=}"
: "${RUN_GROUP:=test_s}"
: "${LOG_ROOT:=$HOME/logs}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
WORKDIR="${WORKDIR:-$REPO_ROOT}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${HOME}:${HOME}}"
PY="${PY:-/opt/conda/envs/llm/bin/python3.10}"
PIP="${PIP:-/opt/conda/envs/llm/bin/pip}"
TORCHRUN="${TORCHRUN:-/opt/conda/envs/llm/bin/torchrun}"

mkdir -p "${LOG_ROOT}/${RUN_GROUP}/slurm" "${LOG_ROOT}/${RUN_GROUP}/ranks"

ts=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/structcheck_exp_sft_${ts}}"
mkdir -p "${OUT_DIR}"

export PYTHONUNBUFFERED=1
export EXTRA_TRAIN_JSONL
export COMPLETION_EXTRA_COPIES
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TOKENIZERS_PARALLELISM=false

rdzv_endpoint="127.0.0.1:29500"

srun \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --mem=256G \
  --container-image "${CONTAINER_IMAGE}" \
  --container-mounts "${CONTAINER_MOUNTS}" \
  --container-workdir "${WORKDIR}" \
  --output="${LOG_ROOT}/${RUN_GROUP}/ranks/%j_%t_%N.log" \
  --error="${LOG_ROOT}/${RUN_GROUP}/ranks/%j_%t_%N.err" \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH=\"\$(pwd)/src:\${PYTHONPATH:-}\"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
    PY=${PY}
    PIP=${PIP}
    TORCHRUN=${TORCHRUN}
    \$PIP freeze | sort > \"${OUT_DIR}/pip_freeze_llm_start.txt\"

    \$PY - <<'PY'
import torch
print('cuda_available', torch.cuda.is_available())
print('cuda_device_count', torch.cuda.device_count())
if torch.cuda.is_available():
  print('device0', torch.cuda.get_device_name(0))
PY

    echo \"OUT_DIR=${OUT_DIR}\"
    EXTRA_TRAIN_ARG=
    if [ -n \"\${EXTRA_TRAIN_JSONL:-}\" ]; then
      EXTRA_TRAIN_ARG=\"--extra-train-jsonl \${EXTRA_TRAIN_JSONL}\"
    fi
    TORCH_COMP_COPY=()
    if [ -n \"\${COMPLETION_EXTRA_COPIES:-}\" ]; then
      TORCH_COMP_COPY=(--completion-extra-copies \"\${COMPLETION_EXTRA_COPIES}\")
    fi
    \$TORCHRUN \
      --nproc_per_node ${GPUS_PER_NODE} \
      --nnodes 1 \
      --rdzv_backend c10d \
      --rdzv_endpoint ${rdzv_endpoint} \
      -m structcheck.train.sft_train \
        --model-path \"${BASE_MODEL}\" \
        --train-jsonl \"${TRAIN_JSONL}\" \
        --eval-jsonl \"${EVAL_JSONL}\" \
        --output-dir \"${OUT_DIR}\" \
        --epochs 1 \
        --max-steps ${MAX_STEPS} \
        --per-device-batch-size ${PER_DEVICE_BATCH_SIZE} \
        --gradient-accumulation-steps ${GRAD_ACCUM} \
        --learning-rate ${LR} \
        --max-length ${MAX_LENGTH} \
        --lora-r ${LORA_R} \
        --lora-alpha ${LORA_ALPHA} \
        ${SFT_EXTRA_ARGS} \${EXTRA_TRAIN_ARG} \${TORCH_COMP_COPY[@]}

    export CUDA_VISIBLE_DEVICES=0
    \$PY -m structcheck.eval.run_inference \
      --base-model-path \"${BASE_MODEL}\" \
      --adapter-path \"${OUT_DIR}\" \
      --input-jsonl \"${EVAL_JSONL}\" \
      --output-jsonl \"${OUT_DIR}/predictions_eval_2stage.jsonl\" \
      --require-image \
      --compact-prompt \
      --max-length ${MAX_LENGTH} \
      --max-new-tokens 128 \
      --completion-votes ${COMPLETION_VOTES}

    \$PY -m structcheck.eval.evaluate_predictions \
      --predictions-jsonl \"${OUT_DIR}/predictions_eval_2stage.jsonl\" \
      --output-json \"${OUT_DIR}/metrics_eval_2stage.json\"

    cat \"${OUT_DIR}/metrics_eval_2stage.json\"

    \$PIP freeze | sort > \"${OUT_DIR}/pip_freeze_llm_end.txt\"
    diff -u \"${OUT_DIR}/pip_freeze_llm_start.txt\" \"${OUT_DIR}/pip_freeze_llm_end.txt\" > \"${OUT_DIR}/pip_freeze_llm_diff.txt\" || true
  "
