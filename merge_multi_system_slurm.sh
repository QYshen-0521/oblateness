#!/bin/bash
# =============================================================================
# 合并各 chunk 输出（依赖：全部 chunk 作业 afterok 成功后再提交本作业）
# 写出：results/$OBLATE_MULTI_RESULTS_SUBDIR/multi_system_summary_a1only.csv、run_timing_a1only.txt
# =============================================================================

#SBATCH --job-name=oblat-ms-merge
#SBATCH --partition=64c512g
#SBATCH --account=acct-tdlffb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/merge-%j.out

set -euo pipefail

WORKDIR="${WORKDIR:-/dssg/home/acct-tdlffb/tdlffb-user1/workspace/RV_astrometry_detect/shen/oblateness}"

_oblateness_parent="$(cd "$(dirname "$WORKDIR")" && pwd)"
if [[ -n "${CONDA_ROOT:-}" && -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
elif [[ -f "${_oblateness_parent}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${_oblateness_parent}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found. Set CONDA_ROOT." >&2
  exit 1
fi
conda activate "${CONDA_ENV:-oblateness}"

cd "$WORKDIR" || { echo "cd failed: $WORKDIR" >&2; exit 1; }
export PYTHONPATH="${WORKDIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p logs

echo "=== merge_multi_system_chunks START $(date -Is) ==="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-n/a}"
export OBLATE_MULTI_RESULTS_SUBDIR="${OBLATE_MULTI_RESULTS_SUBDIR:-multi_system_a1_only}"
echo "OBLATE_MULTI_RESULTS_SUBDIR=$OBLATE_MULTI_RESULTS_SUBDIR"

exec python -m oblateness.merge_multi_system_chunks
