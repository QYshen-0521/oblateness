"""
自 ``worklog.md``「NASA 可建模输入：Tier B + 合并表」：从 keyparams+default 筛 Tier B 派生几何，
**不修改** 既有 ``ps_tran_oblate_inputs_valid_20260416.csv``；写出 Tier B 专表与 A+B 合并表。

- ``data/nasa_archive/ps_tran_oblate_inputs_tier_b_derived_20260416.csv``：仅 Tier B（派生量）。
- ``data/nasa_archive/ps_tran_oblate_inputs_combined_tierab_20260416.csv``：578 行 A + 新 B 行，含元数据列。

单位（见 NASA PS 文档）：`pl_orbsmax` [AU]，`st_rad` [R_sun]；派生
:math:`(a/R_\\star) = a_\\mathrm{AU}\\cdot(1~\\mathrm{AU}) / (R_\\star^{R_\\odot} \\cdot R_\\odot)`，与
``oblateness.constants`` 的 ``au``、``R_sun`` 一致。`pl_rade` 为地球半径倍数时，
:math:`R_p/R_\\star = (\\texttt{pl_rade} \\cdot R_\\oplus) / (R_\\star^{R_\\odot} R_\\odot)`，
地半径取 **6.3781e6 m**（与常见 IAU 型数值一致，见脚本内常量）。

运行::

    python -m oblateness.build_tier_b_oblate_inputs
"""

from __future__ import annotations

import csv
from pathlib import Path

# 地球赤道半径 [m]（用于 pl_rade×R_⊕；NASA 表 pl_rade 为地球半径单位）
R_EARTH_SI: float = 6.3781e6

KEYPARAMS = "data/nasa_archive/ps_tran_flag_keyparams_20260416.csv"
DEFAULT_TABLE = "data/nasa_archive/ps_tran_flag_default_20260416.csv"
TIER_A_FILE = "data/nasa_archive/ps_tran_oblate_inputs_valid_20260416.csv"
SUMMARY_578 = "results/multi_system/multi_system_summary.csv"
OUT_TIER_B = "data/nasa_archive/ps_tran_oblate_inputs_tier_b_derived_20260416.csv"
OUT_COMBINED = "data/nasa_archive/ps_tran_oblate_inputs_combined_tierab_20260416.csv"

# Tier A 九列（直接有效）
TIER_A_COLS = [
    "pl_orbper",
    "pl_ratdor",
    "pl_ratror",
    "pl_orbincl",
    "pl_tranmid",
    "pl_orbeccen",
    "st_teff",
    "st_logg",
    "st_met",
]
# Tier B 仍须直接的七列
CORE7 = [
    "pl_orbper",
    "pl_orbincl",
    "pl_tranmid",
    "pl_orbeccen",
    "st_teff",
    "st_logg",
    "st_met",
]

EXTRA_META = [
    "oblateness_input_tier",
    "pl_ratdor_source",
    "pl_ratror_source",
    "keyparams_row_index",
    "tier_a_legacy_row_index",
    "injection_batch_578_completed",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _lim_ok(d: dict[str, str], col: str) -> bool:
    lim_key = f"{col}lim"
    raw = d.get(lim_key, "")
    s = (raw or "").strip()
    if s == "":
        return True
    try:
        return float(s) == 0.0
    except ValueError:
        return False


def _val_non_empty(d: dict[str, str], col: str) -> bool:
    return bool((d.get(col) or "").strip())


def _direct_ok(d: dict[str, str], col: str) -> bool:
    return _val_non_empty(d, col) and _lim_ok(d, col)


def _tier_a_nine_ok(row: dict[str, str]) -> bool:
    return all(_direct_ok(row, c) for c in TIER_A_COLS)


def _core7_ok(row: dict[str, str]) -> bool:
    return all(_direct_ok(row, c) for c in CORE7)


def _sane_ratdor(x: float) -> bool:
    return 0.0 < x < 500.0  # 宽松：凌星主带


def _sane_ratror(x: float) -> bool:
    return 0.0 < x <= 1.0


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_has_multi_system_batch_counts(r: dict[str, str]) -> bool:
    if (r.get("n_success") or "").strip() != "":
        return True
    if (r.get("n_failed") or "").strip() != "":
        return True
    if (r.get("n_clean") or "").strip() != "":
        return True
    if (r.get("n_degenerate") or "").strip() != "":
        return True
    return False


def _load_summary_pl_injected(root: Path) -> set[str]:
    p = root / SUMMARY_578
    if not p.is_file():
        return set()
    out: set[str] = set()
    with p.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            err = (r.get("error") or "").strip()
            name = (r.get("pl_name") or "").strip()
            if not name or err:
                continue
            if not _row_has_multi_system_batch_counts(r):
                continue
            out.add(name)
    return out


def _derive_ratdor_au(orb_max_au: float, st_rad_rsun: float) -> float:
    from oblateness.constants import R_sun, au

    a_m = float(orb_max_au) * au
    rstar_m = float(st_rad_rsun) * R_sun
    return a_m / rstar_m


def _derive_ratror_rade(
    pl_rade: float,
    st_rad_rsun: float,
) -> float:
    from oblateness.constants import R_sun

    rp_m = float(pl_rade) * R_EARTH_SI
    rstar_m = float(st_rad_rsun) * R_sun
    return rp_m / rstar_m


def _copy_row_add_meta(
    base: dict[str, str],
    meta: dict[str, str],
) -> dict[str, str]:
    o = {**base, **meta}
    return o


def build() -> None:
    root = _repo_root()
    keyparams_path = root / KEYPARAMS
    default_path = root / DEFAULT_TABLE
    tier_a_path = root / TIER_A_FILE

    kp_rows = _load_csv(keyparams_path)
    def_rows = _load_csv(default_path)
    default_by_name: dict[str, dict[str, str]] = {}
    for d in def_rows:
        n = d.get("pl_name", "").strip()
        if n:
            default_by_name[n] = d

    tier_a_rows = _load_csv(tier_a_path)
    tier_a_names = {r["pl_name"].strip() for r in tier_a_rows}
    plname_to_keyparams_idx: dict[str, int] = {}
    for i, r in enumerate(kp_rows):
        n = r.get("pl_name", "").strip()
        if n and n not in plname_to_keyparams_idx:
            plname_to_keyparams_idx[n] = i

    completed_578 = _load_summary_pl_injected(root)

    tier_b_rows: list[dict[str, str]] = []

    for kpi, row in enumerate(kp_rows):
        name = row.get("pl_name", "").strip()
        if not name or name in tier_a_names:
            continue
        ddef = default_by_name.get(name)
        if ddef is None:
            continue
        try:
            tr = int(float((ddef.get("tran_flag") or "0").strip() or 0))
            df = int(float((ddef.get("default_flag") or "0").strip() or 0))
        except ValueError:
            continue
        if tr != 1 or df != 1:
            continue
        if _tier_a_nine_ok(row):
            # 理论上也应在 Tier A；若与 578 不同步，跳过 B
            continue
        if not _core7_ok(row):
            continue
        # B 专指「档案无可用 pl_ratdor、需由 orbsmax+st_rad 派生」
        if _direct_ok(row, "pl_ratdor"):
            continue

        if not (_direct_ok(row, "pl_orbsmax") and _direct_ok(row, "st_rad")):
            continue
        if not ( _val_non_empty(row, "pl_orbsmax") and _val_non_empty(row, "st_rad") ):
            continue
        try:
            orb_au = float(row["pl_orbsmax"].strip())
            rsun = float(row["st_rad"].strip())
        except ValueError:
            continue
        try:
            ratdor = _derive_ratdor_au(orb_au, rsun)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue
        if not _sane_ratdor(ratdor):
            continue

        ratror_source = ""
        ratror: float | None = None
        if _direct_ok(row, "pl_ratror"):
            try:
                ratror = float(row["pl_ratror"].strip())
                ratror_source = "archive"
            except ValueError:
                ratror = None
        if ratror is None:
            prm = f"{(ddef.get('pl_rade') or '').strip()}"
            lmk = f"{(ddef.get('pl_radelim') or '').strip()}"
            l_ok = lmk == "" or (lmk and float(lmk) == 0.0)
            if prm and l_ok:
                try:
                    pr = float(prm)
                    ratror = _derive_ratror_rade(pr, rsun)
                    ratror_source = "derived_rade_st_rad"
                except (ValueError, ZeroDivisionError, FloatingPointError):
                    ratror = None
        if ratror is None or not _sane_ratror(ratror):
            continue

        out = dict(row)
        out["pl_ratdor"] = f"{ratdor:.16g}"
        out["pl_ratdorerr1"] = "0"
        out["pl_ratdorerr2"] = "0"
        out["pl_ratdorlim"] = "0"
        if ratror_source == "derived_rade_st_rad":
            out["pl_ratror"] = f"{ratror:.16g}"
            out["pl_ratrorerr1"] = "0"
            out["pl_ratrorerr2"] = "0"
            out["pl_ratrorlim"] = "0"
        # archive 的 pl_ratror 已在 dict(row) 中保留

        meta = {
            "oblateness_input_tier": "B_derived_geometry",
            "pl_ratdor_source": "derived_orbsmax_st_rad",
            "pl_ratror_source": ratror_source,
            "keyparams_row_index": str(plname_to_keyparams_idx.get(name, kpi)),
            "tier_a_legacy_row_index": "-1",
            "injection_batch_578_completed": "0",
        }
        for k, v in meta.items():
            out[k] = v
        tier_b_rows.append(out)

    # 若 header 以 tier_a 为准
    if tier_a_rows:
        out_keys = list(tier_a_rows[0].keys()) + EXTRA_META
    else:
        out_keys = list(kp_rows[0].keys()) + EXTRA_META

    def _row_write(r: dict[str, str]) -> dict[str, str]:
        return {k: r.get(k, "") for k in out_keys}

    for r in tier_b_rows:
        for k in out_keys:
            r.setdefault(k, r.get(k, ""))

    with (root / OUT_TIER_B).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_keys, extrasaction="ignore")
        w.writeheader()
        for r in tier_b_rows:
            w.writerow(_row_write(r))

    # Combined: A + B
    combined: list[dict[str, str]] = []
    for i, r in enumerate(tier_a_rows):
        name = r["pl_name"].strip()
        kidx = plname_to_keyparams_idx.get(name, "")
        inj = "1" if name in completed_578 else "0"
        meta = {
            "oblateness_input_tier": "A_archive_direct",
            "pl_ratdor_source": "archive",
            "pl_ratror_source": "archive",
            "keyparams_row_index": str(kidx) if kidx != "" else "",
            "tier_a_legacy_row_index": str(i),
            "injection_batch_578_completed": inj,
        }
        combined.append(_copy_row_add_meta(r, meta))

    for r in tier_b_rows:
        combined.append(_row_write(r))

    with (root / OUT_COMBINED).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_keys, extrasaction="ignore")
        w.writeheader()
        for r in combined:
            w.writerow({k: r.get(k, "") for k in out_keys})

    print("Tier A n=", len(tier_a_rows), "file unchanged:", TIER_A_FILE)
    print("Tier B n=", len(tier_b_rows), "->", root / OUT_TIER_B)
    print("Combined n=", len(combined), "->", root / OUT_COMBINED, "(expected", len(tier_a_rows) + len(tier_b_rows), ")")

    toi = [r for r in tier_b_rows if "TOI-2537" in r.get("pl_name", "")]
    if toi:
        print("TOI-2537* in Tier B sample:", toi[0].get("pl_name"), toi[0].get("pl_ratdor_source"))


def main() -> None:
    build()


if __name__ == "__main__":
    main()
