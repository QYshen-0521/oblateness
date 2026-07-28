"""
将 ``row_index`` 连续区间 ``0..n_total-1`` 划分为 ``n_groups`` 个无重叠、无遗漏的 chunk（见 ``worklog.md``
「多系统 Slurm 并行与汇总」）。

**分组规则**：令 ``base = n_total // n_groups``，``rem = n_total % n_groups``；前 ``rem`` 个 chunk 各含
``base+1`` 行，其余各含 ``base`` 行（例如 578 行、20 组 → 前 18 组 29 行、后 2 组 28 行）。

Slurm 当前约定（A1 两态）：新结果 ``results/multi_system_a1_only/_chunks_tierb/chunk_{id:02d}/``；旧物在 ``multi_system/``。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from oblateness.noiseless_grid_recovery_single_planet_batch import planet_config_for_row
from oblateness.noiseless_grid_recovery_single_injection import _repo_root

DEFAULT_N_GROUPS = 20


def csv_data_row_count() -> int:
    """与 ``planet_config_for_row`` 使用同一 CSV 的数据行数（无表头）。"""
    root = _repo_root()
    pc = planet_config_for_row(0)
    path = root / pc["csv_path"]
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def partition_indices(n_total: int, n_groups: int) -> list[list[int]]:
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")
    if n_total < 0:
        raise ValueError("n_total must be non-negative")
    base = n_total // n_groups
    rem = n_total % n_groups
    chunks: list[list[int]] = []
    start = 0
    for g in range(n_groups):
        size = base + (1 if g < rem else 0)
        chunks.append(list(range(start, start + size)))
        start += size
    if start != n_total:
        raise RuntimeError("internal partition error")
    return chunks


def chunk_output_dir(
    chunk_id: int, *, base_relative: str = "multi_system_a1_only/_chunks_tierb"
) -> Path:
    """仓库内相对 ``results/`` 的 chunk 目录：``results/<base_relative>/chunk_XX/``。旧名 ``_chunks`` 勿删。"""
    return _repo_root() / "results" / base_relative / f"chunk_{int(chunk_id):02d}"


def rows_csv_for_chunk(chunk_id: int, n_groups: int, n_total: int) -> str:
    """逗号分隔、供 ``--rows`` 使用。"""
    chunks = partition_indices(n_total, n_groups)
    if chunk_id < 0 or chunk_id >= len(chunks):
        raise IndexError(f"chunk_id={chunk_id} out of range for n_groups={n_groups}")
    return ",".join(str(i) for i in chunks[chunk_id])


def main() -> None:
    p = argparse.ArgumentParser(description="Partition CSV row indices into Slurm chunks.")
    p.add_argument("--n-total", type=int, default=None, help="Default: count from NASA valid CSV.")
    p.add_argument("--n-groups", type=int, default=DEFAULT_N_GROUPS)
    p.add_argument(
        "--print-rows",
        type=int,
        metavar="CHUNK_ID",
        help="Print comma-separated row indices for this chunk to stdout (for bash ROWS=).",
    )
    p.add_argument("--list", action="store_true", help="Print each chunk id, size, and row range.")
    args = p.parse_args()
    n_total = args.n_total if args.n_total is not None else csv_data_row_count()
    n_groups = args.n_groups
    if args.list:
        chunks = partition_indices(n_total, n_groups)
        for i, ch in enumerate(chunks):
            print(f"chunk_{i:02d} n={len(ch)} rows {ch[0]}..{ch[-1]}")
        return
    if args.print_rows is not None:
        sys.stdout.write(rows_csv_for_chunk(args.print_rows, n_groups, n_total))
        return
    p.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
