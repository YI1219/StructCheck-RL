#!/usr/bin/env bash
# Submit 8-GPU SFT+eval with Slurm job name test_s_<NUM>.
# No --gres in sbatch (see SITE_POLICY.txt in this directory).
# Usage: ./scripts/slurm/submit_test_s_sft.sh 12
set -euo pipefail
N="${1:?usage: $0 NUM (e.g. 12 for job name test_s_12)}"
SCRIPT="$(cd "$(dirname "$0")" && pwd)/train_structcheck_sft_1node_8gpu_nogres.sbatch"
exec sbatch --job-name="test_s_${N}" "$SCRIPT"
