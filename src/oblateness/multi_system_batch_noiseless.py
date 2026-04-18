"""
多系统批量无噪声测试（见 ``worklog.md``「多系统批量无噪声测试」）。

对 ``ps_tran_oblate_inputs_valid_*.csv`` 中**多行**（每行一颗行星）重复与
``batch_noiseless_recovery`` 相同的注入网格、粗/细搜索、A1/C1（含去重）。

**运行**::

    python -m oblateness.multi_system_batch_noiseless --rows 0,1,2

或设置环境变量 ``OBLATE_ROW_INDICES=0,1,2``（命令行优先）。

每系统输出至 ``results/<output_subdir>/``：状态图 ``status_row{idx:04d}_<name>.png``、
``batch_row{idx:04d}_<name>.npz``；汇总表 ``multi_system_summary.csv``。

超算：在仓库根目录用 ``sbatch hpc_batch_noiseless.sh``（见该脚本内工作目录与行号）。
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import traceback
from typing import Any

from oblateness.batch_noiseless_recovery import (
    BATCH_RUN_CONFIG,
    _save_npz,
    _save_status_figure,
    run_batch,
    planet_config_for_row,
)
from oblateness.noiseless_grid_chi2_inversion import _repo_root

# =============================================================================
# 默认要跑的系统（CSV 行号，0 = 首条数据行）；可用 CLI / 环境变量覆盖
# =============================================================================
MULTI_SYSTEM_CONFIG: dict[str, Any] = {
    "row_indices": [0, 1, 2],
    "output_subdir": "multi_system",
    "summary_csv": "multi_system_summary.csv",
}


def _safe_name_fragment(name: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", name.strip(), flags=re.ASCII)
    s = s.strip("_")[:80]
    return s or "planet"


def run_multi_system(
    row_indices: list[int],
    output_subdir: str | None = None,
    write_summary: bool = True,
) -> list[dict[str, Any]]:
    """
    依次对 ``row_indices`` 中每个索引跑 ``run_batch``，写每系统图/npz；可选写汇总 CSV。

    返回每行一条记录的列表（含 ``ok``、路径或 ``error``）。
    """
    sub = output_subdir or str(MULTI_SYSTEM_CONFIG["output_subdir"])
    root = _repo_root()
    out_dir = root / "results" / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_name = str(MULTI_SYSTEM_CONFIG.get("summary_csv", "multi_system_summary.csv"))
    summary_rows: list[dict[str, Any]] = []

    for ri in row_indices:
        rec: dict[str, Any] = {
            "row_index": ri,
            "pl_name": "",
            "n_clean": "",
            "n_degenerate": "",
            "n_failed": "",
            "figure_path": "",
            "npz_path": "",
            "error": "",
        }
        try:
            pc = planet_config_for_row(ri)
            results = run_batch(pc)
            if not results:
                raise RuntimeError("empty results")
            pl_name = str(results[0]["pl_name"])
            rec["pl_name"] = pl_name
            n_clean = sum(1 for r in results if r["status"] == "clean")
            n_deg = sum(1 for r in results if r["status"] == "degenerate")
            n_fail = sum(1 for r in results if r["status"] == "failed")
            rec["n_clean"] = n_clean
            rec["n_degenerate"] = n_deg
            rec["n_failed"] = n_fail

            frag = _safe_name_fragment(pl_name)
            fig_name = f"status_row{ri:04d}_{frag}.png"
            npz_name = f"batch_row{ri:04d}_{frag}.npz"
            fig_path = out_dir / fig_name
            npz_path = out_dir / npz_name

            _save_status_figure(results, fig_path)
            rec["figure_path"] = str(fig_path.relative_to(root))
            if BATCH_RUN_CONFIG.get("save_results_npz", True):
                _save_npz(results, npz_path)
                rec["npz_path"] = str(npz_path.relative_to(root))
        except Exception as exc:  # noqa: BLE001 — 记录 skip，不静默
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[SKIP] row_index={ri}: {rec['error']}", file=sys.stderr)
            traceback.print_exc()
        summary_rows.append(rec)

    if write_summary and summary_rows:
        summary_path = out_dir / summary_name
        fieldnames = [
            "row_index",
            "pl_name",
            "n_clean",
            "n_degenerate",
            "n_failed",
            "figure_path",
            "npz_path",
            "error",
        ]
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in summary_rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})
        print("summary:", summary_path)

    return summary_rows


def _parse_row_indices_from_env_and_args(args: argparse.Namespace) -> list[int]:
    if getattr(args, "rows", None):
        return [int(x.strip()) for x in args.rows.split(",") if x.strip()]
    env = os.environ.get("OBLATE_ROW_INDICES", "").strip()
    if env:
        return [int(x.strip()) for x in env.split(",") if x.strip()]
    return [int(x) for x in MULTI_SYSTEM_CONFIG["row_indices"]]


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-system noiseless batch (same pipeline as batch_noiseless_recovery).")
    p.add_argument(
        "--rows",
        type=str,
        default=None,
        help="Comma-separated CSV row indices, e.g. 0,1,5. Overrides env OBLATE_ROW_INDICES and defaults.",
    )
    p.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help="Under results/, default from MULTI_SYSTEM_CONFIG.",
    )
    args = p.parse_args()
    rows = _parse_row_indices_from_env_and_args(args)
    if not rows:
        print("No row indices; set --rows or OBLATE_ROW_INDICES or MULTI_SYSTEM_CONFIG.")
        return
    print("row_indices:", rows)
    run_multi_system(rows, output_subdir=args.output_subdir)


if __name__ == "__main__":
    main()
