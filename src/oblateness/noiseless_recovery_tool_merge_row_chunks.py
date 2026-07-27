"""
合并各 chunk 独占目录下的 per-chunk ``multi_system_summary.csv`` 与 ``run_timing.txt``（见 ``worklog.md``）。

**默认（A1 两态管线）**：读 ``results/multi_system_a1_only/_chunks_tierb/chunk_*/``，写出
``results/multi_system_a1_only/multi_system_summary_a1only.csv`` 与 ``run_timing_a1only.txt``，
**不覆盖**旧 ``results/multi_system/*``。旧 578 chunk 在 ``multi_system/_chunks/``。

默认**追加**最终汇总（保留已有 ``*_a1only*`` 文件内容；同一次合并中仍按 chunk 内 ``row_index`` 去重）。
仅当当前 ``OBLATE_CSV_PATH`` / 默认表的数据行数 **小于 578**（典型 Tier-B-only）且 chunk 路径含
``_chunks_tierb`` 时，写出前才把 ``row_index`` 加上 **578**；**合并表 ~742 行** 自动不加。
可 ``export OBLATE_MERGE_NO_TIERB_ROW_OFFSET=1`` 强制不加、``OBLATE_MERGE_FORCE_TIERB_ROW_OFFSET=1`` 强制加。
默认结果根由 ``OBLATE_MULTI_RESULTS_SUBDIR``（默认 ``multi_system_a1_only``）决定。需整文件重跑时用 ``--no-append``。

Per-chunk 汇总可为新列 ``n_success`` / ``n_failed``；若仍为旧三列 ``n_clean`` / ``n_degenerate`` / ``n_failed``，
合并时会规范化为 ``n_success = n_clean + n_degenerate``。

运行::

    python -m oblateness.merge_multi_system_chunks
    python -m oblateness.merge_multi_system_chunks --chunk-root results/multi_system_a1_only/_chunks_tierb
    python -m oblateness.merge_multi_system_chunks --chunk-root results/multi_system/_chunks --no-append
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oblateness.noiseless_grid_recovery_single_injection import _repo_root

# 各 chunk 目录内文件名（与 multi_system_batch_noiseless 一致）
CHUNK_SUMMARY_NAME = "multi_system_summary.csv"
CHUNK_TIMING_NAME = "run_timing.txt"
# 合并到项目结果根下的最终文件名（勿与旧 ``results/multi_system/multi_system_summary.csv`` 同名）
FINAL_SUMMARY_NAME = "multi_system_summary_a1only.csv"
FINAL_TIMING_NAME = "run_timing_a1only.txt"

# 与 ``ps_tran_oblate_inputs_valid_*.csv`` **数据行数**一致（仅 Tier A 时 n=578）。仅当 CSV **少于**
# 此行数**且** chunk 路径为 ``_chunks_tierb`` 时，合并才把 ``row_index`` 加上本底
#（Tier B-only 跑法：chunk 内 0..n-1 → 总表 578…）。合并表（742）或整表重泡 Tier A（578）均不加。
TIERB_GLOBAL_ROW_INDEX_BASE = 578
TIER_A_VALID_N_DATA_ROWS = 578


def _chunk_root_is_tierb(chunk_root: Path) -> bool:
    return "_chunks_tierb" in chunk_root.resolve().parts


def should_apply_tierb_row_index_offset(chunk_root: Path) -> bool:
    """是否对本次合并结果做 +578（仅 Tier-B-only 且路径含 ``_chunks_tierb``）。"""
    if not _chunk_root_is_tierb(chunk_root):
        return False
    env_off = os.environ.get("OBLATE_MERGE_NO_TIERB_ROW_OFFSET", "").strip().lower()
    if env_off in ("1", "true", "yes"):
        return False
    env_on = os.environ.get("OBLATE_MERGE_FORCE_TIERB_ROW_OFFSET", "").strip().lower()
    if env_on in ("1", "true", "yes"):
        return True
    try:
        from oblateness.noiseless_recovery_tool_split_row_chunks import csv_data_row_count

        n = csv_data_row_count()
    except Exception:
        return True
    if n >= TIER_A_VALID_N_DATA_ROWS:
        return False
    return True


def apply_tierb_global_row_index_offset(
    rows: list[dict[str, str]], *, base: int = TIERB_GLOBAL_ROW_INDEX_BASE
) -> None:
    """将 Tier B 的 chunk 内行号变为总表行号（578, 579, …）。就地修改 ``row_index`` 字段。"""
    for r in rows:
        try:
            local = int((r.get("row_index") or "").strip())
        except ValueError:
            continue
        r["row_index"] = str(local + int(base))


FIELDNAMES = [
    "row_index",
    "pl_name",
    "n_success",
    "n_failed",
    "figure_path",
    "npz_path",
    "error",
]


def _normalize_summary_row(raw: dict[str, str]) -> dict[str, str]:
    """统一为 ``FIELDNAMES``；若存在旧列 ``n_clean`` / ``n_degenerate``，则规范化为 A1 两态计数。"""
    row = {k: (raw.get(k) or "").strip() for k in FIELDNAMES}
    nc = (raw.get("n_clean") or "").strip()
    nd = (raw.get("n_degenerate") or "").strip()
    if nc != "" or nd != "":
        nf = (raw.get("n_failed") or "").strip()
        try:
            c = int(nc) if nc else 0
            d = int(nd) if nd else 0
            f = int(nf) if nf else 0
            row["n_success"] = str(c + d)
            row["n_failed"] = str(f)
        except ValueError:
            pass
    return row


def _load_summary(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            return []
        for raw in r:
            rows.append(_normalize_summary_row(dict(raw)))
    return rows


def _read_timing_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def merge_summaries(chunk_root: Path) -> tuple[list[dict[str, str]], list[str]]:
    """返回合并后的行列表与警告信息。"""
    warnings: list[str] = []
    chunk_dirs = sorted(chunk_root.glob("chunk_*"), key=lambda p: p.name)
    if not chunk_dirs:
        warnings.append(f"no chunk_* under {chunk_root}")
    all_rows: list[dict[str, str]] = []
    for d in chunk_dirs:
        p = d / CHUNK_SUMMARY_NAME
        if not p.is_file():
            warnings.append(f"missing {p}")
            continue
        all_rows.extend(_load_summary(p))

    by_ri: dict[int, dict[str, str]] = {}
    for r in all_rows:
        try:
            ri = int(r["row_index"])
        except (KeyError, ValueError):
            warnings.append(f"bad row_index in row: {r!r}")
            continue
        if ri in by_ri:
            warnings.append(f"duplicate row_index {ri}; keeping later row")
        by_ri[ri] = r

    merged = [by_ri[k] for k in sorted(by_ri.keys())]
    return merged, warnings


def build_merged_timing(
    chunk_root: Path,
    *,
    merge_wall_s: float,
    n_expected_rows: int | None,
) -> str:
    lines: list[str] = []
    lines.append("# merge_multi_system_chunks — combined chunk timings + merge metadata")
    lines.append(f"merge_utc={datetime.now(timezone.utc).isoformat()}")
    lines.append(f"merge_wall_seconds={merge_wall_s:.6f}")
    if n_expected_rows is not None:
        lines.append(f"n_summary_rows_expected={n_expected_rows}")
    lines.append("")

    chunk_dirs = sorted(chunk_root.glob("chunk_*"), key=lambda p: p.name)
    for d in chunk_dirs:
        name = d.name
        tt = _read_timing_text(d / CHUNK_TIMING_NAME)
        lines.append(f"### {name}")
        if tt.strip():
            for ln in tt.strip().splitlines():
                lines.append(ln)
        else:
            lines.append("(no run_timing.txt)")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Merge per-chunk multi_system outputs (A1-only default paths).")
    p.add_argument(
        "--chunk-root",
        type=str,
        default=None,
        help=(
            "Directory containing chunk_00, chunk_01, ... "
            "(default: <repo>/results/$OBLATE_MULTI_RESULTS_SUBDIR/_chunks_tierb, "
            "subdir default multi_system_a1_only)."
        ),
    )
    p.add_argument(
        "--final-dir",
        type=str,
        default=None,
        help="Merged CSV/timing directory (default: <repo>/results/$OBLATE_MULTI_RESULTS_SUBDIR).",
    )
    p.add_argument(
        "--no-append",
        action="store_true",
        help="Replace final *_a1only* summary and timing instead of appending.",
    )
    args = p.parse_args()
    root = _repo_root()
    ms_sub = (os.environ.get("OBLATE_MULTI_RESULTS_SUBDIR") or "multi_system_a1_only").strip() or "multi_system_a1_only"
    chunk_root = (
        Path(args.chunk_root)
        if args.chunk_root
        else root / "results" / ms_sub / "_chunks_tierb"
    )
    chunk_root = chunk_root.resolve()
    final_dir = Path(args.final_dir) if args.final_dir else root / "results" / ms_sub
    final_dir = final_dir.resolve()
    final_dir.mkdir(parents=True, exist_ok=True)
    append = not args.no_append
    if args.no_append:
        print(
            "NOTE: --no-append replaces merged *_a1only* files in final-dir entirely.",
            file=sys.stderr,
        )

    if not any(chunk_root.glob("chunk_*")):
        print(
            f"ERROR: no chunk_* directories under {chunk_root}; refusing to write final summary.",
            file=sys.stderr,
        )
        sys.exit(1)

    t_merge0 = time.perf_counter()
    merged, warns = merge_summaries(chunk_root)
    for w in warns:
        print(f"WARN: {w}", file=sys.stderr)

    if should_apply_tierb_row_index_offset(chunk_root):
        apply_tierb_global_row_index_offset(merged)
        print(
            f"row_index: Tier B chunk-local -> global (+{TIERB_GLOBAL_ROW_INDEX_BASE}; root={chunk_root.name})",
            file=sys.stderr,
        )

    out_csv = final_dir / FINAL_SUMMARY_NAME
    out_timing = final_dir / FINAL_TIMING_NAME
    csv_existed_nonempty = out_csv.is_file() and out_csv.stat().st_size > 0
    mode = "a" if append and csv_existed_nonempty else "w"
    with out_csv.open(mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if mode == "w":
            w.writeheader()
        for row in merged:
            w.writerow({k: row.get(k, "") for k in FIELDNAMES})

    n_exp: int | None = None
    try:
        from oblateness.noiseless_recovery_tool_split_row_chunks import csv_data_row_count

        n_exp = csv_data_row_count()
    except Exception:
        pass
    timing_body = build_merged_timing(
        chunk_root,
        merge_wall_s=time.perf_counter() - t_merge0,
        n_expected_rows=n_exp,
    )
    if append and out_timing.is_file() and out_timing.stat().st_size > 0:
        sep = (
            f"\n\n### APPEND {datetime.now(timezone.utc).isoformat()} "
            f"chunk_root={chunk_root}\n\n"
        )
        with out_timing.open("a", encoding="utf-8") as f:
            f.write(sep)
            f.write(timing_body)
    else:
        out_timing.write_text(timing_body, encoding="utf-8")

    act = "appended" if append and csv_existed_nonempty else "wrote"
    print(f"{act} {len(merged)} data rows -> {out_csv} (append={append})")
    print(f"merged timing -> {out_timing} (append={append})")
    if n_exp is not None and len(merged) != n_exp:
        print(f"WARN: row count {len(merged)} != CSV n_total {n_exp}", file=sys.stderr)


if __name__ == "__main__":
    main()
