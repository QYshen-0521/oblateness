"""
多系统批量无噪声测试（见 ``worklog.md``「多系统批量无噪声测试」）。

对 NASA 可建模输入表（``noiseless_grid_chi2_inversion.PLANET_CONFIG['csv_path']``，默认
``ps_tran_oblate_inputs_tier_b_derived_20260416.csv``；可 ``export OBLATE_CSV_PATH=...`` 或改回
原 ``ps_tran_oblate_inputs_valid_*.csv``）中**多行**（每行一颗行星）重复与
``batch_noiseless_recovery`` 相同的注入网格、粗/细搜索、**A1 两态**（C1 仍算入每条结果作诊断、不参与分类）。

**运行**::

    python -m oblateness.multi_system_batch_noiseless --rows 0,1,2

或设置环境变量 ``OBLATE_ROW_INDICES=0,1,2``（命令行优先）。

每系统输出至 ``results/<output_subdir>/``（默认 ``multi_system_a1_only/``；或 ``--output-dir``）：状态图 ``status_row{idx:04d}_<name>.png``、
``batch_row{idx:04d}_<name>.npz``；汇总表 ``multi_system_summary.csv``（默认**追加**新行，不覆盖历史；
``--summary-overwrite`` 可整表重写）；任务墙钟时间 ``run_timing.txt``。

超算：单作业 ``test.sh``；**578 系统分 ≈20 组并行**见 ``submit_multi_system_chunks.sh``、``test_chunk.sh``、
``merge_multi_system_slurm.sh``（``worklog``「多系统 Slurm 并行与汇总」）。

若设置 ``--output-dir`` 或环境变量 ``OBLATE_MULTI_OUTPUT_DIR``，则**直接**写入该目录（可与仓库 ``results/`` 分离，便于 NFS/大存储）。chunk 作业可设 ``OBLATE_CHUNK_ID``（写入 ``run_timing.txt``）。
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import traceback
from pathlib import Path
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
    "output_subdir": "multi_system_a1_only",
    "summary_csv": "multi_system_summary.csv",
    "append_summary": True,
}


def _safe_name_fragment(name: str) -> str:
    s = re.sub(r"[^\w.\-]+", "_", name.strip(), flags=re.ASCII)
    s = s.strip("_")[:80]
    return s or "planet"


def _path_for_summary(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_existing_summary_rows(summary_path: Path, fieldnames: list[str]) -> list[dict[str, Any]]:
    if not summary_path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with summary_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        for raw in reader:
            rows.append({k: (raw.get(k) or "").strip() for k in fieldnames})
    return rows


def run_multi_system(
    row_indices: list[int],
    output_subdir: str | None = None,
    output_dir: Path | str | None = None,
    write_summary: bool = True,
    append_summary: bool | None = None,
) -> list[dict[str, Any]]:
    """
    依次对 ``row_indices`` 中每个索引跑 ``run_batch``，写每系统图/npz；可选写汇总 CSV。

    ``output_dir``：若给定，为**绝对输出根目录**（覆盖 ``output_subdir`` / 默认 ``results/<subdir>``）。

    返回每行一条记录的列表（含 ``ok``、路径或 ``error``）。
    """
    root = _repo_root()
    if output_dir is not None:
        out_dir = Path(output_dir).expanduser().resolve()
    else:
        sub = output_subdir or str(MULTI_SYSTEM_CONFIG["output_subdir"])
        out_dir = (root / "results" / sub).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_name = str(MULTI_SYSTEM_CONFIG.get("summary_csv", "multi_system_summary.csv"))
    summary_rows: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    for ri in row_indices:
        rec: dict[str, Any] = {
            "row_index": ri,
            "pl_name": "",
            "n_success": "",
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
            n_ok = sum(1 for r in results if r["status"] == "success")
            n_fail = sum(1 for r in results if r["status"] == "failed")
            rec["n_success"] = n_ok
            rec["n_failed"] = n_fail

            frag = _safe_name_fragment(pl_name)
            fig_name = f"status_row{ri:04d}_{frag}.png"
            npz_name = f"batch_row{ri:04d}_{frag}.npz"
            fig_path = out_dir / fig_name
            npz_path = out_dir / npz_name

            _save_status_figure(results, fig_path)
            rec["figure_path"] = _path_for_summary(fig_path, root)
            if BATCH_RUN_CONFIG.get("save_results_npz", True):
                _save_npz(results, npz_path)
                rec["npz_path"] = _path_for_summary(npz_path, root)
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
            "n_success",
            "n_failed",
            "figure_path",
            "npz_path",
            "error",
        ]
        do_append = (
            bool(MULTI_SYSTEM_CONFIG.get("append_summary", True))
            if append_summary is None
            else append_summary
        )
        if do_append:
            prior = _load_existing_summary_rows(summary_path, fieldnames)
            combined = prior + [
                {k: row.get(k, "") for k in fieldnames} for row in summary_rows
            ]
            print(f"summary: append mode — prior_rows={len(prior)} + new_rows={len(summary_rows)}")
        else:
            combined = [{k: row.get(k, "") for k in fieldnames} for row in summary_rows]
            print("summary: overwrite mode")
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in combined:
                w.writerow(row)
        print("summary:", summary_path)

    elapsed = time.perf_counter() - t0
    timing_path = out_dir / "run_timing.txt"
    timing_lines = [
        f"wall_seconds={elapsed:.6f}",
        f"n_row_indices={len(row_indices)}",
    ]
    if row_indices:
        timing_lines.append(f"row_index_min={min(row_indices)}")
        timing_lines.append(f"row_index_max={max(row_indices)}")
    chunk_e = os.environ.get("OBLATE_CHUNK_ID", "").strip()
    if chunk_e:
        timing_lines.append(f"chunk_id={chunk_e}")
    jid = os.environ.get("SLURM_JOB_ID", "").strip()
    if jid:
        timing_lines.append(f"slurm_job_id={jid}")
    timing_path.write_text("\n".join(timing_lines) + "\n", encoding="utf-8")
    print(f"timing: wall_seconds={elapsed:.6f} -> {timing_path}")

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
        help="Under repo results/, default from MULTI_SYSTEM_CONFIG. Ignored if --output-dir is set.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Absolute directory for png/npz/summary (overrides --output-subdir). Env: OBLATE_MULTI_OUTPUT_DIR.",
    )
    p.add_argument(
        "--summary-overwrite",
        action="store_true",
        help="Rewrite multi_system_summary.csv from this run only (default: append new rows to existing file).",
    )
    args = p.parse_args()
    rows = _parse_row_indices_from_env_and_args(args)
    if not rows:
        print("No row indices; set --rows or OBLATE_ROW_INDICES or MULTI_SYSTEM_CONFIG.")
        return
    od_env = os.environ.get("OBLATE_MULTI_OUTPUT_DIR", "").strip()
    output_dir = args.output_dir or od_env or None
    print("row_indices:", rows)
    if output_dir:
        print("output_dir:", Path(output_dir).expanduser().resolve())
    run_multi_system(
        rows,
        output_subdir=args.output_subdir,
        output_dir=output_dir,
        append_summary=not args.summary_overwrite,
    )


if __name__ == "__main__":
    main()
