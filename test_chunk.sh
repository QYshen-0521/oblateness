#!/bin/bash
# =============================================================================
# 单 chunk 多系统批量（见 worklog「多系统 Slurm 并行与汇总」）
# 由 submit_multi_system_chunks.sh 提交；勿多作业写同一 multi_system_summary.csv。
#
# 必设环境变量：CHUNK_ID（0..N_GROUPS-1）、ROWS（逗号分隔行号）
# 输出：$WORKDIR/results/${OBLATE_MULTI_RESULTS_SUBDIR:-multi_system_a1_only}/_chunks_tierb/chunk_XX/
# 历史：results/multi_system/_chunks/ …
# =============================================================================

#SBATCH --job-name=oblat-ms-chunk
#SBATCH --partition=64c512g
#SBATCH --account=acct-tdlffb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/chunk-%j.out
# #SBATCH --time=48:00:00

set -euo pipefail

WORKDIR="${WORKDIR:-/dssg/home/acct-tdlffb/tdlffb-user1/workspace/RV_astrometry_detect/shen/oblateness}"

# Conda：Slurm 批作业须先 source conda.sh，再 conda activate（勿用「source activate」）
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
  echo "ERROR: conda.sh not found. Set CONDA_ROOT to your Miniconda/Anaconda root." >&2
  exit 1
fi
conda activate "${CONDA_ENV:-oblateness}"

cd "$WORKDIR" || { echo "cd failed: $WORKDIR" >&2; exit 1; }

export PYTHONPATH="${WORKDIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

# NASA 凌星 CSV：未设时 Python 默认 Tier B（见 noiseless_grid_chi2_inversion）。重跑原 578：
# export OBLATE_CSV_PATH="data/nasa_archive/ps_tran_oblate_inputs_valid_20260416.csv"
[[ -n "${OBLATE_CSV_PATH:-}" ]] && echo "OBLATE_CSV_PATH=${OBLATE_CSV_PATH}"

if [[ -z "${CHUNK_ID+x}" ]] || [[ -z "${ROWS:-}" ]]; then
  echo "ERROR: set CHUNK_ID and ROWS (e.g. submit_multi_system_chunks.sh)" >&2
  exit 1
fi

printf -v CHUNK_TAG "%02d" "${CHUNK_ID}"
# A1 两态新结果根（勿覆盖历史 results/multi_system/）
OBLATE_MULTI_RESULTS_SUBDIR="${OBLATE_MULTI_RESULTS_SUBDIR:-multi_system_a1_only}"
# 原路径（注释保留）：RESULTS_DIR="${WORKDIR}/results/multi_system/${OBLATE_CHUNKS_BASE}/chunk_${CHUNK_TAG}"
OBLATE_CHUNKS_BASE="${OBLATE_CHUNKS_BASE:-_chunks_tierb}"
RESULTS_DIR="${WORKDIR}/results/${OBLATE_MULTI_RESULTS_SUBDIR}/${OBLATE_CHUNKS_BASE}/chunk_${CHUNK_TAG}"
export OBLATE_CHUNK_ID="${CHUNK_TAG}"
export OBLATE_MULTI_OUTPUT_DIR="${RESULTS_DIR}"

ROWS_LOG="chunk_${CHUNK_TAG}"
mkdir -p logs "${RESULTS_DIR}"

echo "=== chunk ${CHUNK_TAG} START ==="
echo "timestamp_local: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-n/a}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "ROWS=${ROWS:0:120}..."
SECONDS=0

set +e
python -m oblateness.multi_system_batch_noiseless \
  --rows "${ROWS}" \
  --output-dir "${RESULTS_DIR}" \
  --summary-overwrite
PY_EXIT=$?
set -e

echo "=== chunk ${CHUNK_TAG} END ==="
echo "elapsed_wall_seconds=${SECONDS}"
echo "python_exit_code=${PY_EXIT}"
TIMING_LINE="$(date '+%Y-%m-%dT%H:%M:%S%z') job=${SLURM_JOB_ID:-local} chunk=${CHUNK_TAG} elapsed_s=${SECONDS} exit=${PY_EXIT} results=${RESULTS_DIR}"
echo "${TIMING_LINE}" >> "${WORKDIR}/logs/batch_timing.log"

exit "${PY_EXIT}"
