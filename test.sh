#!/usr/bin/env bash
# =============================================================================
# 超算 Slurm：多系统无噪声批量（见 worklog「多系统批量无噪声测试」）
# 提交：在仓库根目录执行  sbatch test.sh
# =============================================================================

#SBATCH -J oblateness-multi
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# --- 本仓库在服务器上的绝对路径（与本地 git clone 路径无关）---
WORKDIR="/dssg/home/acct-tdlffb/tdlffb-user1/workspace/RV_astrometry_detect/shen/oblateness"
cd "$WORKDIR" || { echo "cd failed: $WORKDIR"; exit 1; }

# --- 要跑的 CSV 行号（0 = 首条数据行），逗号分隔；在此修改即可 ---
ROWS="0,1,2,3,4"

# --- 可选：Python 环境（按服务器实际路径修改）---
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source ".venv/bin/activate"
fi

mkdir -p logs results/multi_system

# 需要 squishy 依赖：pip install -e ".[squishy]"
exec python -m oblateness.multi_system_batch_noiseless --rows "${ROWS}"
