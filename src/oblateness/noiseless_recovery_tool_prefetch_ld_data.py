"""
在**可联网**机器上预下载 ExoTiC-LD 所需的恒星网格与仪器通带文件到 ``data/exotic_ld_data/``，
便于拷贝到无外网的 HPC 后复用（与 ``noiseless_grid_chi2_inversion._ld_coeffs_from_exotic`` 路径一致）。

ExoTiC-LD 会在 ``ld_data_path`` 下按需创建::

    <ld_data_path>/mps1/MH<met>/teff<teff>/logg<logg>/mps1_spectra.dat
    <ld_data_path>/Sensitivity_files/<mode>_throughput.csv

运行（需 ``pip install -e ".[squishy]"``、网络畅通）::

    python -m oblateness.prefetch_exotic_ld_data --rows 0,1,2,5-20
    python -m oblateness.prefetch_exotic_ld_data --all-csv

``--all-csv``：对 ``planet_config_for_row(0)`` 所指 CSV 的**全部数据行**逐行预取（578 颗量级；
已存在的 ``mps1_spectra.dat`` 不会重复下载）。

与多系统批量使用相同的 ``planet_config_for_row`` 与 CSV 恒星参数列。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from oblateness.noiseless_grid_recovery_single_planet_batch import planet_config_for_row
from oblateness.noiseless_grid_recovery_single_injection import (
    _col,
    _ld_coeffs_from_exotic,
    _load_csv_row,
    _repo_root,
)


def _ld_coeffs_from_exotic_with_retry(
    pc: dict,
    teff: int,
    logg: float,
    met: float,
    *,
    max_attempts: int = 5,
) -> None:
    """对瞬时网络错误做指数退避重试（全表预取时远端偶发断连）。"""
    delay = 1.0
    for attempt in range(max_attempts):
        try:
            _ld_coeffs_from_exotic(pc, teff, logg, met)
            return
        except Exception as exc:
            err = str(exc).lower()
            retriable = any(
                s in err for s in ("connection", "timeout", "http", "temporar", "reset")
            )
            if attempt < max_attempts - 1 and retriable:
                time.sleep(delay)
                delay = min(delay * 2.0, 45.0)
                continue
            raise


def prefetch_rows(row_indices: list[int]) -> int:
    """对每个 ``row_index`` 调用与管线相同的 LD 计算以触发下载。返回失败行数。"""
    n_fail = 0
    root = _repo_root()
    n_total = len(row_indices)
    log_every = 1 if n_total <= 50 else 50

    for j, ri in enumerate(row_indices):
        pc = planet_config_for_row(ri)
        if pc.get("ld_source") != "exotic":
            print(f"[skip] row {ri}: ld_source={pc.get('ld_source')} (not exotic)", flush=True)
            continue
        csv_path = root / pc["csv_path"]
        row = _load_csv_row(csv_path, ri)
        name = row["pl_name"].strip()
        teff = int(round(_col("st_teff", row)))
        logg = _col("st_logg", row)
        met = _col("st_met", row)
        log_line = (
            f"[{j + 1}/{n_total}] row {ri} {name}: Teff={teff} logg={logg} [M/H]={met}"
        )
        try:
            _ld_coeffs_from_exotic_with_retry(pc, teff, logg, met)
            if j % log_every == 0 or j == n_total - 1:
                print(f"{log_line} -> ok", flush=True)
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"{log_line} -> FAILED: {exc}", file=sys.stderr, flush=True)
    return n_fail


def _parse_rows(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a.strip()), int(b.strip())
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    # 保序去重
    seen: set[int] = set()
    unique: list[int] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            unique.append(x)
    return unique


def _csv_data_row_count(csv_path: Path) -> int:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser(description="Prefetch ExoTiC-LD data files into data/exotic_ld_data/.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--rows",
        type=str,
        default=None,
        help="Comma-separated indices and/or ranges, e.g. 0,1,5-20",
    )
    g.add_argument(
        "--all-csv",
        action="store_true",
        help="Prefetch for every data row in ps_tran_oblate_inputs_valid_*.csv (same as batch pipeline).",
    )
    args = p.parse_args()
    root = _repo_root()
    if args.all_csv:
        pc0 = planet_config_for_row(0)
        csv_path = root / pc0["csv_path"]
        n = _csv_data_row_count(csv_path)
        rows = list(range(n))
        print(f"--all-csv: {csv_path} -> {n} rows (indices 0..{n - 1})", flush=True)
    else:
        rows = _parse_rows(args.rows or "")
    if not rows:
        print("No rows parsed.", file=sys.stderr)
        sys.exit(1)
    out_root = root / "data" / "exotic_ld_data"
    print(f"ld_data_path (root-relative): {out_root}", flush=True)
    print(f"row_indices ({len(rows)}): {rows[:20]}{'...' if len(rows) > 20 else ''}", flush=True)
    n_fail = prefetch_rows(rows)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
