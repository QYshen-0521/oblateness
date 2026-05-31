#!/bin/bash
# =============================================================================
# 提交 N_GROUPS 个 chunk 作业 + 1 个 merge 作业（--dependency=afterok:...）
# 用法：在仓库根目录  bash submit_multi_system_chunks.sh
# 可调：N_GROUPS（默认 20）、WORKDIR、conda 环境名
# =============================================================================

set -euo pipefail

N_GROUPS="${N_GROUPS:-20}"
WORKDIR="${WORKDIR:-/dssg/home/acct-tdlffb/tdlffb-user1/workspace/RV_astrometry_detect/shen/oblateness}"

cd "$WORKDIR" || { echo "cd failed: $WORKDIR" >&2; exit 1; }
export PYTHONPATH="${WORKDIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

# 多系统按 ``planet_config_for_row`` 读入 NASA 表。代码默认已指向 Tier B 派生表；若要在服务器上重跑原 578 行，在下一行前取消注释：
# export OBLATE_CSV_PATH="data/nasa_archive/ps_tran_oblate_inputs_valid_20260416.csv"
# Tier B 显式写（与代码默认二选一，可不设）：
# export OBLATE_CSV_PATH="data/nasa_archive/ps_tran_oblate_inputs_tier_b_derived_20260416.csv"
# **合并表 ~742颗星**（Tier A+B 不覆盖旧多系统目录）示例：另设结果子目录 + 合并 CSV
#   export OBLATE_CSV_PATH="data/nasa_archive/ps_tran_oblate_inputs_combined_tierab_20260416.csv"
#   export OBLATE_MULTI_RESULTS_SUBDIR="multi_system_combined742_run1"
if [[ -n "${OBLATE_CSV_PATH:-}" ]]; then
  export OBLATE_CSV_PATH
  echo "OBLATE_CSV_PATH=$OBLATE_CSV_PATH"
fi

export OBLATE_MULTI_RESULTS_SUBDIR="${OBLATE_MULTI_RESULTS_SUBDIR:-multi_system_a1_only}"
echo "OBLATE_MULTI_RESULTS_SUBDIR=$OBLATE_MULTI_RESULTS_SUBDIR"

# 非交互 bash 同样需要 conda.sh（与 test_chunk.sh 一致）
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

mkdir -p logs "${WORKDIR}/results/${OBLATE_MULTI_RESULTS_SUBDIR}/_chunks_tierb"
# 旧结果目录（勿删、默认勿写）：results/multi_system/、multi_system_a1_only/ 等

echo "WORKDIR=$WORKDIR N_GROUPS=$N_GROUPS"
python -m oblateness.multi_system_chunks --list --n-groups "${N_GROUPS}"

JOB_IDS=()
for ((cid = 0; cid < N_GROUPS; cid++)); do
  ROWS=$(python -c "from oblateness.multi_system_chunks import rows_csv_for_chunk, csv_data_row_count; print(rows_csv_for_chunk(${cid}, ${N_GROUPS}, csv_data_row_count()))")
  echo "--- submitting chunk ${cid} rows (first 80 chars): ${ROWS:0:80}..."
  export CHUNK_ID="${cid}"
  export ROWS
  jid=$(sbatch --parsable --export=ALL "${WORKDIR}/test_chunk.sh")
  JOB_IDS+=("${jid}")
  echo "  -> job_id=${jid}"
done

# afterok: id1:id2:...
dep=$(IFS=:; echo "${JOB_IDS[*]}")
echo "Dependency string: afterok:${dep}"

MERGE_JID=$(sbatch --parsable --export=ALL --dependency="afterok:${dep}" "${WORKDIR}/merge_multi_system_slurm.sh")
echo "merge job_id=${MERGE_JID}"
echo "Done. Monitor: squeue -u \$USER"
