#!/bin/bash
# =============================================================================
# 超算 Slurm：多系统无噪声批量（见 worklog「多系统批量无噪声测试」）
# 日常修改：WORKDIR 下 RESULTS_DIR / ROWS；环境与分区按集群策略调整。
# 提交：cd 到本仓库根目录后执行  sbatch test.sh
# =============================================================================

#SBATCH --job-name=oblateness-multi
#SBATCH --partition=64c512g
#SBATCH --account=acct-tdlffb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%j.out
# 若集群要求最长运行时间，按分区加上，例如：
# #SBATCH --time=48:00:00

set -euo pipefail

# --- Conda 环境（与你在 Startrack 任务中一致；若用 module load / 其他 env 请改此处）---
source activate oblateness

# --- 本仓库在服务器上的绝对路径 ---
WORKDIR="/dssg/home/acct-tdlffb/tdlffb-user1/workspace/RV_astrometry_detect/shen/oblateness"
cd "$WORKDIR" || { echo "cd failed: $WORKDIR" >&2; exit 1; }

# --- 结果保存路径（二选一，见下）---
# 方式 A（默认）：写入仓库内 results/<子目录>，适合与代码同盘、小体量。
RESULTS_SUBDIR="multi_system"
# 方式 B：写入任意绝对路径（大存储 / NFS）；非空则优先于 RESULTS_SUBDIR，并导出给 Python。
# 例：RESULTS_DIR="/dssg/home/acct-tdlffb/scratch/oblateness_multi_system_20260418"
RESULTS_DIR=""

# --- 要跑的 CSV 行号（0 = 首条数据行），逗号分隔 ---
ROWS="0,1,2,3,4"

mkdir -p logs

# --- 耗时：写入 Slurm 标准输出日志（#SBATCH --output=logs/%j.out）---
echo "=== oblateness-multi START ==="
echo "timestamp_local: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "hostname: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-n/a}"
SECONDS=0

set +e
if [[ -n "${RESULTS_DIR}" ]]; then
  mkdir -p "${RESULTS_DIR}"
  export OBLATE_MULTI_OUTPUT_DIR="${RESULTS_DIR}"
  python -m oblateness.multi_system_batch_noiseless --rows "${ROWS}" --output-dir "${RESULTS_DIR}"
else
  unset OBLATE_MULTI_OUTPUT_DIR || true
  mkdir -p "results/${RESULTS_SUBDIR}"
  python -m oblateness.multi_system_batch_noiseless --rows "${ROWS}" --output-subdir "${RESULTS_SUBDIR}"
fi
PY_EXIT=$?
set -e

RESULT_PATH="${RESULTS_DIR:-$WORKDIR/results/$RESULTS_SUBDIR}"
echo "=== oblateness-multi END ==="
echo "timestamp_local: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "elapsed_wall_seconds=${SECONDS}"
echo "python_exit_code=${PY_EXIT}"
echo "Done. Results under: ${RESULT_PATH}"

# 追加一行到 logs/，便于多作业对比（与 %j.out 内容一致的信息摘要）
TIMING_LINE="$(date '+%Y-%m-%dT%H:%M:%S%z') job=${SLURM_JOB_ID:-local} elapsed_s=${SECONDS} exit=${PY_EXIT} rows=${ROWS} results=${RESULT_PATH}"
echo "${TIMING_LINE}" >> "${WORKDIR}/logs/batch_timing.log"
echo "timing_line_appended_to: ${WORKDIR}/logs/batch_timing.log"

exit "${PY_EXIT}"
