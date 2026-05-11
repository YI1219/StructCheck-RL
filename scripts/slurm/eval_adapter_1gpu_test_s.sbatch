#!/usr/bin/env bash
# Re-run inference + metrics on an existing adapter (validates run_inference / evaluate_predictions).
# Usage:
#   ADAPTER_DIR=/path/to/exp_dir sbatch scripts/slurm/eval_adapter_1gpu_test_s.sbatch
# Optional: PRED_SUFFIX=realign_metrics MAX_LENGTH=2048
# Older adapters (trained before SFT/infer tail alignment): LEGACY_SFT_PROMPT=1
# No --gres (site policy); GPU comes from the allocated node / container defaults.
# See scripts/slurm/SITE_POLICY.txt
#
# Also set CONTAINER_IMAGE, BASE_MODEL, EVAL_JSONL (and optionally REPO_ROOT / CONTAINER_MOUNTS).
# See scripts/slurm/README.md
#
# --- Edit for your Slurm site ---
#SBATCH --job-name=test_s_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=b3-poc-polyu
#SBATCH --account=b3-poc-polyu
#SBATCH --qos=qos_b3-poc-polyu
#SBATCH --output=slurm-%x-%j_%N.out
#SBATCH --error=slurm-%x-%j_%N.err

set -euo pipefail
set -x

: "${ADAPTER_DIR:?Export ADAPTER_DIR=/path/to/sft/output (directory with adapter)}"
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to your Apptainer/Singularity image}"
: "${BASE_MODEL:?Set BASE_MODEL to the base VLM directory}"
: "${EVAL_JSONL:?Set EVAL_JSONL to eval JSONL}"

: "${MAX_LENGTH:=2048}"
: "${MAX_NEW_TOKENS:=128}"
: "${COMPLETION_VOTES:=1}"
: "${LEGACY_SFT_PROMPT:=0}"
: "${PRED_SUFFIX:=metrics_realign}"
: "${RUN_GROUP:=test_s}"
: "${LOG_ROOT:=$HOME/logs}"

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
WORKDIR="${WORKDIR:-$REPO_ROOT}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${HOME}:${HOME}}"
PY="${PY:-/opt/conda/envs/llm/bin/python3.10}"
PIP="${PIP:-/opt/conda/envs/llm/bin/pip}"

mkdir -p "${LOG_ROOT}/${RUN_GROUP}/slurm"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export LEGACY_SFT_PROMPT MAX_LENGTH MAX_NEW_TOKENS COMPLETION_VOTES PRED_SUFFIX

srun \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --mem=64G \
  --container-image "${CONTAINER_IMAGE}" \
  --container-mounts "${CONTAINER_MOUNTS}" \
  --container-workdir "${WORKDIR}" \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH=\"\$(pwd)/src:\${PYTHONPATH:-}\"
    export CUDA_VISIBLE_DEVICES=0
    PY=${PY}
    PIP=${PIP}
    AD=\"${ADAPTER_DIR}\"
    EVAL_FREEZE_TS=\$(date +%Y%m%d_%H%M%S)
    \$PIP freeze | sort > \"\${AD}/pip_freeze_llm_eval_start_\${EVAL_FREEZE_TS}.txt\"
    OUT_PRED=\"\${AD}/predictions_eval_2stage_${PRED_SUFFIX}.jsonl\"
    OUT_MET=\"\${AD}/metrics_eval_2stage_${PRED_SUFFIX}.json\"
    echo \"ADAPTER_DIR=\${AD}\"
    echo \"OUT_PRED=\${OUT_PRED}\"
    EXTRA_LEGACY=()
    if [ \"\${LEGACY_SFT_PROMPT:-0}\" != \"0\" ]; then
      EXTRA_LEGACY=(--legacy-sft-prompt)
    fi
    \$PY -m structcheck.eval.run_inference \
      --base-model-path \"${BASE_MODEL}\" \
      --adapter-path \"\${AD}\" \
      --input-jsonl \"${EVAL_JSONL}\" \
      --output-jsonl \"\${OUT_PRED}\" \
      --require-image \
      --compact-prompt \
      --max-length ${MAX_LENGTH} \
      --max-new-tokens ${MAX_NEW_TOKENS} \
      --completion-votes ${COMPLETION_VOTES} \
      \"\${EXTRA_LEGACY[@]}\"
    \$PY -m structcheck.eval.evaluate_predictions \
      --predictions-jsonl \"\${OUT_PRED}\" \
      --output-json \"\${OUT_MET}\"
    cat \"\${OUT_MET}\"

    \$PIP freeze | sort > \"\${AD}/pip_freeze_llm_eval_end_\${EVAL_FREEZE_TS}.txt\"
    diff -u \"\${AD}/pip_freeze_llm_eval_start_\${EVAL_FREEZE_TS}.txt\" \"\${AD}/pip_freeze_llm_eval_end_\${EVAL_FREEZE_TS}.txt\" > \"\${AD}/pip_freeze_llm_eval_diff_\${EVAL_FREEZE_TS}.txt\" || true
  "
