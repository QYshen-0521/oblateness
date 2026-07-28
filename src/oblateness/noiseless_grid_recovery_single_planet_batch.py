"""
批量无噪声反演：注入网格、粗网格 + 局部细化、A1/C1 判据与 :math:`f`–:math:`\\theta` 状态图。

数学定义与 ``worklog.md`` 中「批量无噪声反演」节一致；:math:`\\chi^2` 与单点脚本相同，
用 :math:`\\sum_k(F_{\\mathrm{obl},k}-F_{\\mathrm{inj},k})^2`（不显式算 :math:`F_0`）。

**仅改下方配置字典**即可调整行星/LD、注入扫参、粗/细网格（粗：\\(f\\) 步长与 \\(\\theta\\) 度步长；
细：`f_step`、`theta_deg_step`）、判据与输出文件名。

运行（需 ``pip install -e ".[squishy]"``）::

    python -m oblateness.batch_noiseless_recovery

默认会遍历全部注入格点（计算量较大）；调试可将 ``BATCH_RUN_CONFIG['max_injections']`` 设为小整数。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from oblateness.noiseless_grid_recovery_single_injection import (
    PLANET_CONFIG,
    _col,
    _compute_ld_u,
    _load_csv_row,
    _repo_root,
)

# =============================================================================
# 行星 + LD + 时间轴（与单点反演一致；不含注入 —— 注入由 INJECTION_GRID_CONFIG 扫参）
# =============================================================================
BASE_PLANET_CONFIG: dict[str, Any] = {
    k: v
    for k, v in PLANET_CONFIG.items()
    if k not in ("injected_projected_f", "injected_projected_theta_deg")
}

# =============================================================================
# 注入扫参网格（Planner 定案：f 步长 0.01；θ 步长 20°）
# =============================================================================
INJECTION_GRID_CONFIG: dict[str, Any] = {
    "f_min": 0.01,
    "f_max": 0.15,
    "f_step": 0.01,
    "theta_inj_deg_min": 0.0,
    "theta_inj_deg_max": 160.0,
    "theta_inj_deg_step": 20.0,
}

# =============================================================================
# 粗网格 :math:`(f,\\theta)`（试探参数；\\(\\theta\\) 用度定义步长，内部转弧度；\\([0^\\circ,180^\\circ)\\)）
# =============================================================================
COARSE_GRID_CONFIG: dict[str, Any] = {
    "f_min": 0.0,
    "f_max": 0.2,
    "f_step": 0.02,
    "theta_deg_min": 0.0,
    "theta_deg_max": 180.0,
    "theta_deg_step": 5.0,
}

# =============================================================================
# 局部细化：在粗 :math:`\\arg\\min` 邻域内按**固定步长**密化（邻域半宽可配）
# =============================================================================
REFINE_GRID_CONFIG: dict[str, Any] = {
    "f_step": 0.005,
    "theta_deg_step": 1.0,
    "f_abs_half_width": 0.03,
    "theta_deg_half_width": 15.0,
    "f_min_clip": 0.0,
    "f_max_clip": 0.35,
}

# =============================================================================
# C1：次小与最小 :math:`\\chi^2` 间隙阈值（无噪声建议与浮点残差量级匹配）
# =============================================================================
DELTA_CHI2_CONFIG: dict[str, Any] = {
    "floor": 1e-12,
    "rtol": 1e-9,
}

# =============================================================================
# C1：粗+细网格合并后按 \((f,\\theta)\) 去重（同键保留 \(\min\\chi^2\)），再取 \(\chi^2_{(1)},\chi^2_{(2)}\)
# =============================================================================
C1_DEDUP_CONFIG: dict[str, Any] = {
    "ndigits_f": 8,
    "ndigits_theta_rad": 10,
}

# =============================================================================
# 运行与输出
# =============================================================================
BATCH_RUN_CONFIG: dict[str, Any] = {
    # None = 全部注入格点；设为例如 3 便于调试
    "max_injections": None,
    "status_figure_filename": "batch_noiseless_f_theta_status.png",
    "save_results_npz": True,
    "results_npz_filename": "batch_noiseless_recovery.npz",
}


Status = Literal["clean", "degenerate", "failed"]


def _theta_distance_deg(a_deg: float, b_deg: float) -> float:
    """:math:`[0^\\circ,180^\\circ)` 上最短角距（度）。"""
    a = float(a_deg) % 180.0
    b = float(b_deg) % 180.0
    d = abs(a - b)
    return float(min(d, 180.0 - d))


def _injection_grid() -> tuple[np.ndarray, np.ndarray]:
    ig = INJECTION_GRID_CONFIG
    n_f = int(round((ig["f_max"] - ig["f_min"]) / ig["f_step"])) + 1
    fs = np.linspace(ig["f_min"], ig["f_max"], n_f)
    n_th = int(round((ig["theta_inj_deg_max"] - ig["theta_inj_deg_min"]) / ig["theta_inj_deg_step"])) + 1
    ths = np.linspace(ig["theta_inj_deg_min"], ig["theta_inj_deg_max"], n_th)
    return fs.astype(np.float64), ths.astype(np.float64)


def _coarse_axes() -> tuple[np.ndarray, np.ndarray]:
    cg = COARSE_GRID_CONFIG
    n_f = int(round((cg["f_max"] - cg["f_min"]) / cg["f_step"])) + 1
    fs = np.linspace(cg["f_min"], cg["f_max"], n_f)
    # θ：度域 [theta_deg_min, theta_deg_max)，步长 theta_deg_step（与 0–180° 一致）
    ths_deg = np.arange(cg["theta_deg_min"], cg["theta_deg_max"], cg["theta_deg_step"])
    ths = np.deg2rad(ths_deg.astype(np.float64))
    return fs, ths


def _chi2_grid(
    system: Any,
    f_inj_arr: np.ndarray,
    fs: np.ndarray,
    ths_rad: np.ndarray,
) -> np.ndarray:
    chi2 = np.empty((len(fs), len(ths_rad)), dtype=np.float64)
    for i, f in enumerate(fs):
        for j, th in enumerate(ths_rad):
            f_obl = np.array(
                system.lightcurve(params={"projected_f": float(f), "projected_theta": float(th)})
            )
            chi2[i, j] = float(np.sum((f_obl - f_inj_arr) ** 2))
    return chi2


def _ravel_mesh(fs: np.ndarray, ths_rad: np.ndarray, chi2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ff, tt = np.meshgrid(fs, ths_rad, indexing="ij")
    return ff.ravel(), tt.ravel(), chi2.ravel()


def _refine_axes(
    f_center: float,
    th_center_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    rg = REFINE_GRID_CONFIG
    f_step = float(rg["f_step"])
    th_step_deg = float(rg["theta_deg_step"])

    f_lo = max(float(rg["f_min_clip"]), f_center - float(rg["f_abs_half_width"]))
    f_hi = min(float(rg["f_max_clip"]), f_center + float(rg["f_abs_half_width"]))
    if f_hi <= f_lo:
        f_hi = f_lo + f_step

    fs = np.arange(f_lo, f_hi + 0.5 * f_step, f_step, dtype=np.float64)
    if fs.size < 2:
        fs = np.array([f_lo, min(f_hi, f_lo + f_step)], dtype=np.float64)

    half_th_deg = float(rg["theta_deg_half_width"])
    th_center_deg = float(np.rad2deg(th_center_rad))
    th_lo_deg = max(0.0, th_center_deg - half_th_deg)
    th_hi_deg = min(180.0 - 1e-9, th_center_deg + half_th_deg)
    if th_hi_deg <= th_lo_deg:
        th_hi_deg = th_lo_deg + th_step_deg

    ths_deg = np.arange(th_lo_deg, th_hi_deg + 0.5 * th_step_deg, th_step_deg, dtype=np.float64)
    if ths_deg.size < 2:
        ths_deg = np.array([th_lo_deg, min(th_hi_deg, th_lo_deg + th_step_deg)], dtype=np.float64)
    ths = np.deg2rad(ths_deg)
    return fs, ths


def _delta_chi2_threshold(chi2_min: float, cfg: dict[str, Any]) -> float:
    return float(max(cfg["floor"], cfg["rtol"] * max(chi2_min, 1.0)))


def _dedupe_chi2_by_f_theta(
    f_flat: np.ndarray,
    th_rad_flat: np.ndarray,
    chi_flat: np.ndarray,
    ndigits_f: int,
    ndigits_theta_rad: int,
) -> np.ndarray:
    """对物理格点 (f,θ) 去重，同一键只保留最小 χ²（见 worklog「C1 实现勘误」）。"""
    best: dict[tuple[float, float], float] = {}
    for i in range(int(f_flat.size)):
        k = (
            round(float(f_flat[i]), int(ndigits_f)),
            round(float(th_rad_flat[i]), int(ndigits_theta_rad)),
        )
        c = float(chi_flat[i])
        if k not in best or c < best[k]:
            best[k] = c
    return np.array(list(best.values()), dtype=np.float64)


def _c1_gap(chi2_values: np.ndarray) -> tuple[float, float, float]:
    """返回 (:math:`\\chi^2_{(1)}`, :math:`\\chi^2_{(2)}`, gap)。至少两点；否则 gap=0。"""
    flat = np.asarray(chi2_values, dtype=np.float64).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return float("nan"), float("nan"), 0.0
    s = np.sort(flat)
    c1 = float(s[0])
    c2 = float(s[1]) if s.size > 1 else c1
    return c1, c2, float(c2 - c1)


def _build_system(pc: dict[str, Any]) -> tuple[Any, str, np.ndarray]:
    import jax.numpy as jnp
    from squishyplanet import OblateSystem

    root = _repo_root()
    csv_path = root / pc["csv_path"]
    row = _load_csv_row(csv_path, pc["row_index"])
    name = row["pl_name"].strip()

    period = _col("pl_orbper", row)
    a = _col("pl_ratdor", row)
    rp = _col("pl_ratror", row)
    inc_deg = _col("pl_orbincl", row)
    t0 = _col("pl_tranmid", row)
    ecc = _col("pl_orbeccen", row)
    teff = int(round(_col("st_teff", row)))
    logg = _col("st_logg", row)
    met = _col("st_met", row)

    inc = float(np.deg2rad(inc_deg))
    ld_u = _compute_ld_u(pc, teff, logg, met)
    ld_j = jnp.array(ld_u, dtype=jnp.float64)

    half = float(pc["time_half_width_days"])
    times = jnp.linspace(t0 - half, t0 + half, int(pc["n_time"]))

    common = dict(
        times=times,
        t0=float(t0),
        period=float(period),
        a=float(a),
        tidally_locked=bool(pc["tidally_locked"]),
        e=float(ecc),
        i=inc,
        omega=0.0,
        Omega=float(pc["Omega_rad"]),
        obliq=0.0,
        prec=0.0,
        ld_u_coeffs=ld_j,
        parameterize_with_projected_ellipse=True,
        projected_effective_r=float(rp),
    )
    system = OblateSystem(**common, projected_f=0.0, projected_theta=0.0)
    return system, name, np.array(times)


def run_single_injection(
    system: Any,
    f_inj: float,
    theta_inj_deg: float,
) -> dict[str, Any]:
    """一次注入：粗网格 :math:`\\arg\\min` → 邻域细化 → 全局最优 + A1/C1。"""
    th_inj_rad = float(np.deg2rad(theta_inj_deg))
    f_inj_arr = np.array(
        system.lightcurve(params={"projected_f": float(f_inj), "projected_theta": th_inj_rad})
    )

    fs_c, ths_c = _coarse_axes()
    chi2_c = _chi2_grid(system, f_inj_arr, fs_c, ths_c)
    ii, jj = np.unravel_index(int(np.argmin(chi2_c)), chi2_c.shape)
    f0_hat = float(fs_c[ii])
    th0_hat = float(ths_c[jj])

    fs_r, ths_r = _refine_axes(f0_hat, th0_hat)
    chi2_r = _chi2_grid(system, f_inj_arr, fs_r, ths_r)

    fc, tc, vc = _ravel_mesh(fs_c, ths_c, chi2_c)
    fr, tr, vr = _ravel_mesh(fs_r, ths_r, chi2_r)
    f_all = np.concatenate([fc, fr])
    th_all = np.concatenate([tc, tr])
    chi_all = np.concatenate([vc, vr])

    k = int(np.argmin(chi_all))
    f_hat = float(f_all[k])
    theta_hat_rad = float(th_all[k])
    chi2_min = float(chi_all[k])

    # A1：ε 取「细网格」步长之半（显式定义于 REFINE_GRID_CONFIG）
    rg = REFINE_GRID_CONFIG
    epsilon_f = float(rg["f_step"] / 2.0)
    epsilon_theta_deg = float(rg["theta_deg_step"] / 2.0)

    theta_hat_deg = float(np.rad2deg(theta_hat_rad))
    a1 = bool(abs(f_hat - f_inj) < epsilon_f and _theta_distance_deg(theta_hat_deg, theta_inj_deg) < epsilon_theta_deg)

    dcfg = C1_DEDUP_CONFIG
    chi_flat_c1 = _dedupe_chi2_by_f_theta(
        f_all,
        th_all,
        chi_all,
        ndigits_f=int(dcfg["ndigits_f"]),
        ndigits_theta_rad=int(dcfg["ndigits_theta_rad"]),
    )
    chi2_1, chi2_2, gap = _c1_gap(chi_flat_c1)
    dcfg = DELTA_CHI2_CONFIG
    gap_thr = _delta_chi2_threshold(chi2_1, dcfg)
    c1 = bool((gap) > gap_thr)

    if a1 and c1:
        status: Status = "clean"
    elif a1 and not c1:
        status = "degenerate"
    else:
        status = "failed"

    return {
        "f_inj": f_inj,
        "theta_inj_deg": float(theta_inj_deg),
        "f_hat": f_hat,
        "theta_hat_rad": theta_hat_rad,
        "theta_hat_deg": theta_hat_deg,
        "chi2_min": chi2_min,
        "chi2_1": chi2_1,
        "chi2_2": chi2_2,
        "chi2_gap": gap,
        "delta_chi2_threshold": gap_thr,
        "delta_chi2_floor": dcfg["floor"],
        "delta_chi2_rtol": dcfg["rtol"],
        "epsilon_f": epsilon_f,
        "epsilon_theta_deg": epsilon_theta_deg,
        "A1": a1,
        "C1": c1,
        "status": status,
        "f_hat_coarse": f0_hat,
        "theta_hat_coarse_rad": th0_hat,
        "n_chi2_evaluated": int(f_all.size),
        "n_chi2_unique_c1": int(chi_flat_c1.size),
    }


def planet_config_for_row(row_index: int) -> dict[str, Any]:
    """返回指定 CSV 行的行星配置（与 ``BASE_PLANET_CONFIG`` 同构，仅 ``row_index`` 不同）。"""
    return {**BASE_PLANET_CONFIG, "row_index": int(row_index)}


def run_batch(planet_cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """对 **单个** 行星跑完整注入网格；``planet_cfg`` 默认 ``BASE_PLANET_CONFIG``（见 ``PLANET_CONFIG['row_index']``）。"""
    pc = planet_cfg if planet_cfg is not None else BASE_PLANET_CONFIG
    root = _repo_root()
    csv_path = root / pc["csv_path"]
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    system, pl_name, _ = _build_system(pc)

    f_grid, theta_inj_grid = _injection_grid()
    pairs: list[tuple[float, float]] = [(float(f), float(th)) for f in f_grid for th in theta_inj_grid]

    br = BATCH_RUN_CONFIG
    max_n = br.get("max_injections")
    if max_n is not None:
        pairs = pairs[: int(max_n)]

    results: list[dict[str, Any]] = []
    for f_inj, theta_inj_deg in pairs:
        r = run_single_injection(system, f_inj, theta_inj_deg)
        r["pl_name"] = pl_name
        r["row_index"] = int(pc["row_index"])
        results.append(r)
    return results


def _save_status_figure(
    results: list[dict[str, Any]],
    figure_path: Path | str | None = None,
) -> Path:
    matplotlib = __import__("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = {"clean": "#2ca02c", "degenerate": "#ff7f0e", "failed": "#d62728"}
    fig, ax = plt.subplots(figsize=(9.0, 6.0), dpi=120)
    for s in ("clean", "degenerate", "failed"):
        pts = [r for r in results if r["status"] == s]
        if not pts:
            continue
        ax.scatter(
            [p["f_inj"] for p in pts],
            [p["theta_inj_deg"] for p in pts],
            c=color[s],
            s=55,
            marker="o",
            edgecolors="k",
            linewidths=0.3,
            label=s,
            zorder=3,
        )
    ax.set_xlabel(r"injected $f_{\mathrm{inj}}$")
    ax.set_ylabel(r"injected $\theta_{\mathrm{inj}}$ (deg)")
    title = results[0]["pl_name"] if results else ""
    ax.set_title(f"{title} — noiseless batch recovery (A1 ∧ C1 = clean)")
    ax.set_xlim(INJECTION_GRID_CONFIG["f_min"] - 0.005, INJECTION_GRID_CONFIG["f_max"] + 0.005)
    ax.set_ylim(
        INJECTION_GRID_CONFIG["theta_inj_deg_min"] - 5.0,
        INJECTION_GRID_CONFIG["theta_inj_deg_max"] + 5.0,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    root = _repo_root()
    out = Path(figure_path) if figure_path is not None else root / "results" / str(BATCH_RUN_CONFIG["status_figure_filename"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def _save_npz(
    results: list[dict[str, Any]],
    npz_path: Path | str | None = None,
) -> Path:
    """将列表结果压成列向量存入 npz。"""
    keys = [
        "f_inj",
        "theta_inj_deg",
        "f_hat",
        "theta_hat_deg",
        "chi2_min",
        "chi2_1",
        "chi2_2",
        "chi2_gap",
        "delta_chi2_threshold",
        "epsilon_f",
        "epsilon_theta_deg",
        "A1",
        "C1",
        "n_chi2_evaluated",
        "n_chi2_unique_c1",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        out[k] = np.array([float(r[k]) if k not in ("A1", "C1") else float(bool(r[k])) for r in results])
    out["status"] = np.array([r["status"] for r in results], dtype=object)
    out["pl_name"] = results[0]["pl_name"] if results else ""
    if results and "row_index" in results[0]:
        out["row_index"] = int(results[0]["row_index"])
    root = _repo_root()
    path = Path(npz_path) if npz_path is not None else root / "results" / str(BATCH_RUN_CONFIG["results_npz_filename"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return path


def main() -> None:
    results = run_batch(None)
    print("n_results:", len(results))
    if not results:
        return
    n_clean = sum(1 for r in results if r["status"] == "clean")
    n_deg = sum(1 for r in results if r["status"] == "degenerate")
    n_fail = sum(1 for r in results if r["status"] == "failed")
    print("status counts — clean:", n_clean, "degenerate:", n_deg, "failed:", n_fail)

    fig_path = _save_status_figure(results)
    print("saved figure:", fig_path)
    if BATCH_RUN_CONFIG.get("save_results_npz"):
        p = _save_npz(results)
        print("saved npz:", p)


if __name__ == "__main__":
    main()
