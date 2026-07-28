"""20次公共噪声实现的批量含噪声反演与连续恢复能力地图。

每个 cadence 注入标准差为 137 ppm 的独立白色高斯噪声，对应论文中针对 TOI-2537 b 的
NIRISS/SOSS 经验噪声估计。噪声缓存中的 ``t_exp_s`` 仍用于设置模型时间采样；缓存噪声值和
``nint`` 仅保留为输出元数据，不再用于放大注入噪声。

每个注入格点使用同一组20个随机种子（公共随机数），避免相邻格点因使用完全不同的噪声样本而
形成单次随机红黄绿图。固定细网格上的模型光变只预计算一次；之后用矩阵运算批量计算每组噪声
的 χ²。输出每个格点20次恢复的中位误差、分位数和覆盖率，并以连续色标展示恢复能力。

运行（需 ``pip install -e ".[squishy]"``）::

    python -m oblateness.noised_grid_recovery_single_planet_batch_v03

默认遍历全部注入格点；调试可将 ``BATCH_RUN_CONFIG['max_injections']`` 设为小整数。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is only for terminal progress display.
    tqdm = None

from oblateness.noiseless_recovery_tool_ingress_egress_sampling_v02 import (
    build_lightcurve_time_array_days,
)
from oblateness.noised_grid_recovery_single_injection import (
    NOISE_CONFIG,
    PLANET_CONFIG,
    _col,
    _compute_ld_u,
    _load_success_noise_for_target,
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
    "f_max": 0.35,
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
# Profile-χ² 置信区间
# =============================================================================
CONFIDENCE_CONFIG: dict[str, float] = {
    # 一维 profile likelihood：一个感兴趣参数。
    "profile_delta_chi2_68": 1.0,
    "profile_delta_chi2_95": 4.0,
    # 二维联合置信区域：f 与 theta 两个感兴趣参数。
    "joint_delta_chi2_68": 2.30,
    "joint_delta_chi2_95": 6.18,
}

# =============================================================================
# 粗、细网格合并后的物理格点去重精度
# =============================================================================
C1_DEDUP_CONFIG: dict[str, Any] = {
    "ndigits_f": 8,
    "ndigits_theta_rad": 10,
}

# =============================================================================
# 运行与输出
# =============================================================================
NOISE_INJECTION_CONFIG: dict[str, float] = {
    "sigma_ppm_per_cadence": 137.0,
}

BATCH_RUN_CONFIG: dict[str, Any] = {
    # None = 全部注入格点；设为例如 3 便于调试
    "max_injections": None,
    "n_noise_realizations": 20,
    "summary_figure_filename": "batch_noised_mc20_ie60/batch_noised_recovery_metrics.png",
    "save_results_npz": True,
    "results_npz_filename": "batch_noised_mc20_ie60/batch_noised_recovery_realizations.npz",
    "save_realizations_csv": True,
    "realizations_csv_filename": "batch_noised_mc20_ie60/batch_noised_recovery_realizations.csv",
    "save_summary_csv": True,
    "summary_csv_filename": "batch_noised_mc20_ie60/batch_noised_recovery_summary.csv",
}

MODEL_GRID_CONFIG: dict[str, float] = {
    "f_min": 0.0,
    "f_max": 0.35,
    "f_step": 0.005,
    "theta_deg_min": 0.0,
    "theta_deg_max": 180.0,
    "theta_deg_step": 1.0,
}


Coverage = Literal["inside_68", "inside_95", "outside_95"]


def _theta_distance_deg(a_deg: float, b_deg: float) -> float:
    """Return the shortest angular distance on the 180-degree periodic domain."""
    a = float(a_deg) % 180.0
    b = float(b_deg) % 180.0
    distance = abs(a - b)
    return float(min(distance, 180.0 - distance))


def _injection_grid() -> tuple[np.ndarray, np.ndarray]:
    ig = INJECTION_GRID_CONFIG
    n_f = int(round((ig["f_max"] - ig["f_min"]) / ig["f_step"])) + 1
    fs = np.linspace(ig["f_min"], ig["f_max"], n_f)
    n_th = int(round((ig["theta_inj_deg_max"] - ig["theta_inj_deg_min"]) / ig["theta_inj_deg_step"])) + 1
    ths = np.linspace(ig["theta_inj_deg_min"], ig["theta_inj_deg_max"], n_th)
    return fs.astype(np.float64), ths.astype(np.float64)


def _coarse_axes() -> tuple[np.ndarray, np.ndarray]:
    cg = COARSE_GRID_CONFIG
    fs = np.arange(
        cg["f_min"],
        cg["f_max"] + 0.5 * cg["f_step"],
        cg["f_step"],
        dtype=np.float64,
    )
    # θ：度域 [theta_deg_min, theta_deg_max)，步长 theta_deg_step（与 0–180° 一致）
    ths_deg = np.arange(cg["theta_deg_min"], cg["theta_deg_max"], cg["theta_deg_step"])
    ths = np.deg2rad(ths_deg.astype(np.float64))
    return fs, ths


def _chi2_grid(
    system: Any,
    f_obs_arr: np.ndarray,
    sigma_flux: float,
    fs: np.ndarray,
    ths_rad: np.ndarray,
) -> np.ndarray:
    chi2 = np.empty((len(fs), len(ths_rad)), dtype=np.float64)
    sigma_flux = float(sigma_flux)
    if not np.isfinite(sigma_flux) or sigma_flux <= 0.0:
        raise ValueError(f"invalid sigma_flux: {sigma_flux}")
    for i, f in enumerate(fs):
        for j, th in enumerate(ths_rad):
            f_obl = np.array(
                system.lightcurve(params={"projected_f": float(f), "projected_theta": float(th)})
            )
            chi2[i, j] = float(np.sum(((f_obl - f_obs_arr) / sigma_flux) ** 2))
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


def _dedupe_grid_points(
    f_flat: np.ndarray,
    th_rad_flat: np.ndarray,
    chi_flat: np.ndarray,
    ndigits_f: int,
    ndigits_theta_rad: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对物理格点去重，并为每个 ``(f, theta)`` 保留最小 χ²。"""
    best: dict[tuple[float, float], tuple[float, float, float]] = {}
    for i in range(int(f_flat.size)):
        f = float(f_flat[i])
        theta = float(th_rad_flat[i]) % np.pi
        k = (
            round(f, int(ndigits_f)),
            round(theta, int(ndigits_theta_rad)),
        )
        c = float(chi_flat[i])
        if k not in best or c < best[k][2]:
            best[k] = (f, theta, c)
    values = list(best.values())
    return (
        np.array([v[0] for v in values], dtype=np.float64),
        np.array([v[1] for v in values], dtype=np.float64),
        np.array([v[2] for v in values], dtype=np.float64),
    )


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


def _profile_minimum(
    parameter_values: np.ndarray,
    chi2_values: np.ndarray,
    ndigits: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按单个参数分组并对另一个参数取最小 χ²。"""
    best: dict[float, tuple[float, float]] = {}
    for value, chi2 in zip(parameter_values, chi2_values, strict=True):
        v = float(value)
        c = float(chi2)
        key = round(v, int(ndigits))
        if key not in best or c < best[key][1]:
            best[key] = (v, c)
    ordered = sorted(best.values(), key=lambda item: item[0])
    return (
        np.array([item[0] for item in ordered], dtype=np.float64),
        np.array([item[1] for item in ordered], dtype=np.float64),
    )


def _linear_profile_interval(
    values: np.ndarray,
    profile_chi2: np.ndarray,
    chi2_min: float,
    delta_chi2: float,
) -> tuple[float, float]:
    delta = profile_chi2 - float(chi2_min)
    threshold = float(delta_chi2)
    mask = (delta <= threshold) | np.isclose(delta, threshold, rtol=1e-10, atol=1e-12)
    allowed = values[mask]
    if allowed.size == 0:
        best = float(values[int(np.argmin(profile_chi2))])
        return best, best
    return float(np.min(allowed)), float(np.max(allowed))


def _theta_profile_interval(
    theta_deg: np.ndarray,
    profile_chi2: np.ndarray,
    chi2_min: float,
    delta_chi2: float,
    theta_hat_deg: float,
) -> tuple[float, float, float, float]:
    """返回周期 180° 下相对于最佳值的下/上边界及非对称误差。"""
    delta = profile_chi2 - float(chi2_min)
    threshold = float(delta_chi2)
    mask = (delta <= threshold) | np.isclose(delta, threshold, rtol=1e-10, atol=1e-12)
    allowed = theta_deg[mask]
    if allowed.size == 0:
        center = float(theta_hat_deg) % 180.0
        return center, center, 0.0, 0.0

    center = float(theta_hat_deg) % 180.0
    offsets = (allowed - center + 90.0) % 180.0 - 90.0
    offset_low = min(float(np.min(offsets)), 0.0)
    offset_high = max(float(np.max(offsets)), 0.0)
    low = (center + offset_low) % 180.0
    high = (center + offset_high) % 180.0
    return low, high, -offset_low, offset_high


def _confidence_summary(
    f_values: np.ndarray,
    theta_rad_values: np.ndarray,
    chi2_values: np.ndarray,
    f_hat: float,
    theta_hat_deg: float,
) -> dict[str, float | bool]:
    """由去重后的搜索格点生成一维 profile-χ² 置信区间。"""
    chi2_min = float(np.min(chi2_values))
    f_axis, f_profile = _profile_minimum(f_values, chi2_values, ndigits=8)
    theta_axis, theta_profile = _profile_minimum(
        np.rad2deg(theta_rad_values) % 180.0,
        chi2_values,
        ndigits=8,
    )

    out: dict[str, float | bool] = {}
    for label in ("68", "95"):
        delta = CONFIDENCE_CONFIG[f"profile_delta_chi2_{label}"]
        f_low, f_high = _linear_profile_interval(f_axis, f_profile, chi2_min, delta)
        theta_low, theta_high, theta_minus, theta_plus = _theta_profile_interval(
            theta_axis,
            theta_profile,
            chi2_min,
            delta,
            theta_hat_deg,
        )
        out[f"f_ci{label}_low"] = f_low
        out[f"f_ci{label}_high"] = f_high
        out[f"f_ci{label}_err_minus"] = max(0.0, float(f_hat) - f_low)
        out[f"f_ci{label}_err_plus"] = max(0.0, f_high - float(f_hat))
        out[f"theta_ci{label}_low_deg"] = theta_low
        out[f"theta_ci{label}_high_deg"] = theta_high
        out[f"theta_ci{label}_err_minus_deg"] = theta_minus
        out[f"theta_ci{label}_err_plus_deg"] = theta_plus

    f_min = float(np.min(f_axis))
    f_max = float(np.max(f_axis))
    out["f_ci95_touches_search_boundary"] = bool(
        np.isclose(out["f_ci95_low"], f_min) or np.isclose(out["f_ci95_high"], f_max)
    )
    return out


def _build_system(
    pc: dict[str, Any],
    *,
    cadence_seconds: float | None = None,
) -> tuple[Any, str, np.ndarray]:
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
    transit_duration_hours = _col("pl_trandur", row)
    teff = int(round(_col("st_teff", row)))
    logg = _col("st_logg", row)
    met = _col("st_met", row)

    omega_text = row.get("pl_orblper", "").strip()
    if omega_text:
        omega_deg = float(omega_text)
    elif np.isclose(ecc, 0.0):
        omega_deg = 0.0
    else:
        raise ValueError(
            f"{name} has eccentricity e={ecc}, but pl_orblper is missing; "
            "cannot construct an eccentric transit consistently"
        )

    inc = float(np.deg2rad(inc_deg))
    omega = float(np.deg2rad(omega_deg))
    ld_u = _compute_ld_u(pc, teff, logg, met)
    ld_j = jnp.array(ld_u, dtype=jnp.float64)

    raw_ts = pc.get("time_sampling")
    ts = dict(raw_ts) if isinstance(raw_ts, dict) else {}
    if cadence_seconds is not None:
        ts["cadence_seconds"] = float(cadence_seconds)
    times_np = build_lightcurve_time_array_days(
        t0_days=float(t0),
        period_days=float(period),
        pl_ratdor=float(a),
        k_rp_over_rs=float(rp),
        inc_deg=float(inc_deg),
        ecc=float(ecc),
        time_half_width_days=float(pc["time_half_width_days"]),
        n_time_uniform=int(pc["n_time"]),
        time_sampling=ts,
        omega_deg=omega_deg,
        transit_duration_hours=transit_duration_hours,
    )
    times = jnp.array(times_np, dtype=jnp.float64)

    common = dict(
        times=times,
        t0=float(t0),
        period=float(period),
        a=float(a),
        tidally_locked=bool(pc["tidally_locked"]),
        e=float(ecc),
        i=inc,
        omega=omega,
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
    sigma_flux_per_cadence: float,
    noise_seed: int,
) -> dict[str, Any]:
    """一次注入：加噪、粗网格定位、局部细化，并计算 profile-χ² 区间。"""
    th_inj_rad = float(np.deg2rad(theta_inj_deg))
    f_inj_arr = np.array(
        system.lightcurve(params={"projected_f": float(f_inj), "projected_theta": th_inj_rad})
    )
    rng = np.random.default_rng(int(noise_seed))
    gaussian_noise = rng.normal(0.0, float(sigma_flux_per_cadence), size=f_inj_arr.shape)
    f_obs_arr = f_inj_arr + gaussian_noise

    fs_c, ths_c = _coarse_axes()
    chi2_c = _chi2_grid(system, f_obs_arr, sigma_flux_per_cadence, fs_c, ths_c)
    ii, jj = np.unravel_index(int(np.argmin(chi2_c)), chi2_c.shape)
    f0_hat = float(fs_c[ii])
    th0_hat = float(ths_c[jj])

    fs_r, ths_r = _refine_axes(f0_hat, th0_hat)
    chi2_r = _chi2_grid(system, f_obs_arr, sigma_flux_per_cadence, fs_r, ths_r)

    fc, tc, vc = _ravel_mesh(fs_c, ths_c, chi2_c)
    fr, tr, vr = _ravel_mesh(fs_r, ths_r, chi2_r)
    f_all = np.concatenate([fc, fr])
    th_all = np.concatenate([tc, tr])
    chi_all = np.concatenate([vc, vr])

    dcfg = C1_DEDUP_CONFIG
    f_unique, theta_unique, chi2_unique = _dedupe_grid_points(
        f_all,
        th_all,
        chi_all,
        ndigits_f=int(dcfg["ndigits_f"]),
        ndigits_theta_rad=int(dcfg["ndigits_theta_rad"]),
    )
    k = int(np.argmin(chi2_unique))
    f_hat = float(f_unique[k])
    theta_hat_rad = float(theta_unique[k])
    theta_hat_deg = float(np.rad2deg(theta_hat_rad) % 180.0)
    chi2_min = float(chi2_unique[k])
    confidence = _confidence_summary(
        f_unique,
        theta_unique,
        chi2_unique,
        f_hat,
        theta_hat_deg,
    )

    chi2_true = float(np.sum((gaussian_noise / float(sigma_flux_per_cadence)) ** 2))
    delta_chi2_true = max(0.0, chi2_true - chi2_min)
    truth_better_than_grid = bool(chi2_true < chi2_min)
    if delta_chi2_true <= CONFIDENCE_CONFIG["joint_delta_chi2_68"]:
        coverage: Coverage = "inside_68"
    elif delta_chi2_true <= CONFIDENCE_CONFIG["joint_delta_chi2_95"]:
        coverage = "inside_95"
    else:
        coverage = "outside_95"

    chi2_1, chi2_2, gap = _c1_gap(chi2_unique)
    n_data = int(f_obs_arr.size)
    degrees_of_freedom = max(1, n_data - 2)
    f_abs_error = abs(f_hat - float(f_inj))
    theta_abs_error_deg = _theta_distance_deg(theta_hat_deg, theta_inj_deg)
    recovery_distance_grid_steps = float(
        np.hypot(
            f_abs_error / float(REFINE_GRID_CONFIG["f_step"]),
            theta_abs_error_deg / float(REFINE_GRID_CONFIG["theta_deg_step"]),
        )
    )

    return {
        "f_inj": f_inj,
        "theta_inj_deg": float(theta_inj_deg),
        "f_hat": f_hat,
        "theta_hat_rad": theta_hat_rad,
        "theta_hat_deg": theta_hat_deg,
        "f_abs_error": f_abs_error,
        "theta_abs_error_deg": theta_abs_error_deg,
        "recovery_distance_grid_steps": recovery_distance_grid_steps,
        "chi2_min": chi2_min,
        "reduced_chi2": chi2_min / degrees_of_freedom,
        "chi2_true": chi2_true,
        "delta_chi2_true": delta_chi2_true,
        "truth_better_than_grid": truth_better_than_grid,
        "coverage": coverage,
        "profile_delta_chi2_68": CONFIDENCE_CONFIG["profile_delta_chi2_68"],
        "profile_delta_chi2_95": CONFIDENCE_CONFIG["profile_delta_chi2_95"],
        "joint_delta_chi2_68": CONFIDENCE_CONFIG["joint_delta_chi2_68"],
        "joint_delta_chi2_95": CONFIDENCE_CONFIG["joint_delta_chi2_95"],
        "chi2_1": chi2_1,
        "chi2_2": chi2_2,
        "chi2_gap": gap,
        "sigma_flux_per_cadence": float(sigma_flux_per_cadence),
        "noise_random_seed": int(noise_seed),
        "f_hat_coarse": f0_hat,
        "theta_hat_coarse_rad": th0_hat,
        "n_data": n_data,
        "degrees_of_freedom": degrees_of_freedom,
        "n_chi2_evaluated": int(chi_all.size),
        "n_chi2_unique": int(chi2_unique.size),
        **confidence,
    }


def planet_config_for_row(row_index: int) -> dict[str, Any]:
    """返回指定 CSV 行的行星配置（与 ``BASE_PLANET_CONFIG`` 同构，仅 ``row_index`` 不同）。"""
    return {**BASE_PLANET_CONFIG, "row_index": int(row_index)}


def _model_grid_axes() -> tuple[np.ndarray, np.ndarray]:
    cfg = MODEL_GRID_CONFIG
    fs = np.arange(
        cfg["f_min"],
        cfg["f_max"] + 0.5 * cfg["f_step"],
        cfg["f_step"],
        dtype=np.float64,
    )
    theta_deg = np.arange(
        cfg["theta_deg_min"],
        cfg["theta_deg_max"],
        cfg["theta_deg_step"],
        dtype=np.float64,
    )
    return fs, theta_deg


def _precompute_model_library(
    system: Any,
    fs: np.ndarray,
    theta_deg: np.ndarray,
) -> np.ndarray:
    """Compute every fixed-grid model once; return ``(n_f*n_theta, n_time)``."""
    n_models = int(fs.size * theta_deg.size)
    first = np.asarray(
        system.lightcurve(params={"projected_f": float(fs[0]), "projected_theta": np.deg2rad(theta_deg[0])}),
        dtype=np.float64,
    )
    models = np.empty((n_models, first.size), dtype=np.float64)
    models[0] = first

    iterator = range(1, n_models)
    if tqdm is not None:
        iterator = tqdm(iterator, total=n_models - 1, desc="precompute model grid", unit="model")
    n_theta = int(theta_deg.size)
    for flat_index in iterator:
        i_f, i_theta = divmod(flat_index, n_theta)
        models[flat_index] = np.asarray(
            system.lightcurve(
                params={
                    "projected_f": float(fs[i_f]),
                    "projected_theta": float(np.deg2rad(theta_deg[i_theta])),
                }
            ),
            dtype=np.float64,
        )
    return models


def _chi2_from_model_library(
    model_fluxes: np.ndarray,
    observed_fluxes: np.ndarray,
    sigma_flux_per_cadence: float,
) -> np.ndarray:
    """Return χ² with shape ``(n_models, n_noise_realizations)`` using one matrix product."""
    sigma = float(sigma_flux_per_cadence)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"invalid sigma_flux_per_cadence: {sigma}")
    models = np.asarray(model_fluxes, dtype=np.float64) - 1.0
    observations = np.asarray(observed_fluxes, dtype=np.float64) - 1.0
    model_norm = np.sum(models * models, axis=1)[:, None]
    observation_norm = np.sum(observations * observations, axis=1)[None, :]
    chi2 = (model_norm - 2.0 * (models @ observations.T) + observation_norm) / sigma**2
    return np.maximum(chi2, 0.0)


def _confidence_summary_rectangular(
    fs: np.ndarray,
    theta_deg: np.ndarray,
    chi2_grid: np.ndarray,
    f_hat: float,
    theta_hat_deg: float,
) -> dict[str, float | bool]:
    """Profile-χ² intervals on the fixed rectangular model grid."""
    chi2_min = float(np.min(chi2_grid))
    f_profile = np.min(chi2_grid, axis=1)
    theta_profile = np.min(chi2_grid, axis=0)
    out: dict[str, float | bool] = {}
    for label in ("68", "95"):
        delta = CONFIDENCE_CONFIG[f"profile_delta_chi2_{label}"]
        f_low, f_high = _linear_profile_interval(fs, f_profile, chi2_min, delta)
        theta_low, theta_high, theta_minus, theta_plus = _theta_profile_interval(
            theta_deg,
            theta_profile,
            chi2_min,
            delta,
            theta_hat_deg,
        )
        out[f"f_ci{label}_low"] = f_low
        out[f"f_ci{label}_high"] = f_high
        out[f"f_ci{label}_err_minus"] = max(0.0, float(f_hat) - f_low)
        out[f"f_ci{label}_err_plus"] = max(0.0, f_high - float(f_hat))
        out[f"theta_ci{label}_low_deg"] = theta_low
        out[f"theta_ci{label}_high_deg"] = theta_high
        out[f"theta_ci{label}_err_minus_deg"] = theta_minus
        out[f"theta_ci{label}_err_plus_deg"] = theta_plus
    out["f_ci95_touches_search_boundary"] = bool(
        np.isclose(out["f_ci95_low"], fs[0]) or np.isclose(out["f_ci95_high"], fs[-1])
    )
    return out


def _nearest_axis_index(axis: np.ndarray, value: float, name: str) -> int:
    index = int(np.argmin(np.abs(axis - float(value))))
    if not np.isclose(axis[index], float(value), rtol=0.0, atol=1e-10):
        raise ValueError(f"{name}={value} is not represented on the fixed model grid")
    return index


def _run_injection_realizations(
    model_fluxes: np.ndarray,
    fs: np.ndarray,
    theta_deg: np.ndarray,
    standard_noise: np.ndarray,
    f_inj: float,
    theta_inj_deg: float,
    sigma_flux_per_cadence: float,
    base_seed: int,
) -> list[dict[str, Any]]:
    """Fit all common-noise realizations for one injected ``(f, theta)``."""
    i_f_true = _nearest_axis_index(fs, f_inj, "f_inj")
    i_theta_true = _nearest_axis_index(theta_deg, theta_inj_deg, "theta_inj_deg")
    n_theta = int(theta_deg.size)
    true_flat_index = i_f_true * n_theta + i_theta_true
    clean_flux = model_fluxes[true_flat_index]
    observed = clean_flux[None, :] + float(sigma_flux_per_cadence) * standard_noise
    chi2_flat = _chi2_from_model_library(model_fluxes, observed, sigma_flux_per_cadence)
    chi2_cube = chi2_flat.reshape(fs.size, theta_deg.size, standard_noise.shape[0])

    n_data = int(clean_flux.size)
    degrees_of_freedom = max(1, n_data - 2)
    results: list[dict[str, Any]] = []
    for realization_index in range(int(standard_noise.shape[0])):
        chi2_values = chi2_flat[:, realization_index]
        best_flat_index = int(np.argmin(chi2_values))
        i_f_hat, i_theta_hat = divmod(best_flat_index, n_theta)
        f_hat = float(fs[i_f_hat])
        theta_hat_deg = float(theta_deg[i_theta_hat])
        chi2_min = float(chi2_values[best_flat_index])
        chi2_true = float(chi2_values[true_flat_index])
        delta_chi2_true = max(0.0, chi2_true - chi2_min)
        if delta_chi2_true <= CONFIDENCE_CONFIG["joint_delta_chi2_68"]:
            coverage: Coverage = "inside_68"
        elif delta_chi2_true <= CONFIDENCE_CONFIG["joint_delta_chi2_95"]:
            coverage = "inside_95"
        else:
            coverage = "outside_95"

        confidence = _confidence_summary_rectangular(
            fs,
            theta_deg,
            chi2_cube[:, :, realization_index],
            f_hat,
            theta_hat_deg,
        )
        f_abs_error = abs(f_hat - float(f_inj))
        theta_abs_error_deg = _theta_distance_deg(theta_hat_deg, theta_inj_deg)
        recovery_distance = float(
            np.hypot(
                f_abs_error / float(MODEL_GRID_CONFIG["f_step"]),
                theta_abs_error_deg / float(MODEL_GRID_CONFIG["theta_deg_step"]),
            )
        )
        two_smallest = np.partition(chi2_values, 1)[:2]
        chi2_1 = float(np.min(two_smallest))
        chi2_2 = float(np.max(two_smallest))
        results.append(
            {
                "f_inj": float(f_inj),
                "theta_inj_deg": float(theta_inj_deg),
                "f_hat": f_hat,
                "theta_hat_deg": theta_hat_deg,
                "f_abs_error": f_abs_error,
                "theta_abs_error_deg": theta_abs_error_deg,
                "recovery_distance_grid_steps": recovery_distance,
                "chi2_min": chi2_min,
                "reduced_chi2": chi2_min / degrees_of_freedom,
                "chi2_true": chi2_true,
                "delta_chi2_true": delta_chi2_true,
                "truth_better_than_grid": bool(chi2_true < chi2_min - 1e-9),
                "coverage": coverage,
                "profile_delta_chi2_68": CONFIDENCE_CONFIG["profile_delta_chi2_68"],
                "profile_delta_chi2_95": CONFIDENCE_CONFIG["profile_delta_chi2_95"],
                "joint_delta_chi2_68": CONFIDENCE_CONFIG["joint_delta_chi2_68"],
                "joint_delta_chi2_95": CONFIDENCE_CONFIG["joint_delta_chi2_95"],
                "chi2_1": chi2_1,
                "chi2_2": chi2_2,
                "chi2_gap": chi2_2 - chi2_1,
                "sigma_flux_per_cadence": float(sigma_flux_per_cadence),
                "noise_random_seed": int(base_seed + realization_index),
                "noise_realization_index": int(realization_index),
                "n_data": n_data,
                "degrees_of_freedom": degrees_of_freedom,
                "n_chi2_evaluated": int(chi2_values.size),
                "n_chi2_unique": int(chi2_values.size),
                "is_closest_recovery": False,
                **confidence,
            }
        )
    return results


def _circular_median_deg(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64) % 180.0
    vector = np.mean(np.exp(2j * np.deg2rad(values)))
    reference = 0.0 if np.isclose(abs(vector), 0.0) else float(np.rad2deg(np.angle(vector)) / 2.0) % 180.0
    offsets = (values - reference + 90.0) % 180.0 - 90.0
    return float((reference + np.median(offsets)) % 180.0)


def _aggregate_realizations(realizations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for result in realizations:
        grouped.setdefault(int(result["injection_index"]), []).append(result)

    summaries: list[dict[str, Any]] = []
    for injection_index in sorted(grouped):
        rows = grouped[injection_index]
        values = lambda key: np.array([float(row[key]) for row in rows], dtype=np.float64)
        coverage = [row["coverage"] for row in rows]
        f_hat = values("f_hat")
        theta_hat = values("theta_hat_deg")
        distance = values("recovery_distance_grid_steps")
        summary = {
            "pl_name": rows[0]["pl_name"],
            "row_index": int(rows[0]["row_index"]),
            "noise_cache_target": rows[0]["noise_cache_target"],
            "noise_cache_row_index": int(rows[0]["noise_cache_row_index"]),
            "model_cadence_seconds": float(rows[0]["model_cadence_seconds"]),
            "injection_index": injection_index,
            "f_inj": float(rows[0]["f_inj"]),
            "theta_inj_deg": float(rows[0]["theta_inj_deg"]),
            "n_noise_realizations": len(rows),
            "f_hat_median": float(np.median(f_hat)),
            "f_hat_p16": float(np.percentile(f_hat, 16.0)),
            "f_hat_p84": float(np.percentile(f_hat, 84.0)),
            "theta_hat_circular_median_deg": _circular_median_deg(theta_hat),
            "f_abs_error_median": float(np.median(values("f_abs_error"))),
            "f_abs_error_p16": float(np.percentile(values("f_abs_error"), 16.0)),
            "f_abs_error_p84": float(np.percentile(values("f_abs_error"), 84.0)),
            "theta_abs_error_deg_median": float(np.median(values("theta_abs_error_deg"))),
            "theta_abs_error_deg_p16": float(np.percentile(values("theta_abs_error_deg"), 16.0)),
            "theta_abs_error_deg_p84": float(np.percentile(values("theta_abs_error_deg"), 84.0)),
            "recovery_distance_grid_steps_median": float(np.median(distance)),
            "recovery_distance_grid_steps_p16": float(np.percentile(distance, 16.0)),
            "recovery_distance_grid_steps_p84": float(np.percentile(distance, 84.0)),
            "coverage_68_fraction": float(np.mean([item == "inside_68" for item in coverage])),
            "coverage_95_fraction": float(np.mean([item != "outside_95" for item in coverage])),
            "outside_95_fraction": float(np.mean([item == "outside_95" for item in coverage])),
            "f_ci68_width_median": float(np.median(values("f_ci68_high") - values("f_ci68_low"))),
            "theta_ci68_width_deg_median": float(
                np.median(values("theta_ci68_err_minus_deg") + values("theta_ci68_err_plus_deg"))
            ),
            "reduced_chi2_median": float(np.median(values("reduced_chi2"))),
            "f_ci95_boundary_fraction": float(
                np.mean([bool(row["f_ci95_touches_search_boundary"]) for row in rows])
            ),
            "is_best_median_recovery": False,
        }
        summaries.append(summary)

    if summaries:
        best_index = int(np.argmin([row["recovery_distance_grid_steps_median"] for row in summaries]))
        summaries[best_index]["is_best_median_recovery"] = True
    return summaries


def run_batch(
    planet_cfg: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all injections with the same set of 20 Gaussian-noise realizations."""
    pc = planet_cfg if planet_cfg is not None else BASE_PLANET_CONFIG
    root = _repo_root()
    csv_path = root / pc["csv_path"]
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    input_row = _load_csv_row(csv_path, int(pc["row_index"]))
    pl_name = input_row["pl_name"].strip()
    noise_row = _load_success_noise_for_target(root / str(NOISE_CONFIG["cache_csv_path"]), pl_name)
    sigma_flux_full_transit = float(noise_row["sigma_flux"])
    sigma_frac_ppm_full_transit = float(noise_row["sigma_frac_ppm"])
    try:
        nint = int(round(float(noise_row["nint"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"noise cache has invalid nint for {pl_name}: {noise_row.get('nint')!r}") from exc
    if nint <= 0:
        raise ValueError(f"noise cache nint must be positive for {pl_name}: {nint}")
    try:
        cadence_seconds = float(noise_row["t_exp_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"noise cache has invalid t_exp_s for {pl_name}: {noise_row.get('t_exp_s')!r}"
        ) from exc
    if not np.isfinite(cadence_seconds) or cadence_seconds <= 0.0:
        raise ValueError(
            f"noise cache t_exp_s must be a positive finite value for {pl_name}: "
            f"{cadence_seconds}"
        )

    system, system_name, _ = _build_system(pc, cadence_seconds=cadence_seconds)
    if system_name != pl_name:
        raise ValueError(f"target name changed while building system: {pl_name!r} != {system_name!r}")

    sigma_frac_ppm_per_cadence = float(
        NOISE_INJECTION_CONFIG["sigma_ppm_per_cadence"]
    )
    sigma_flux_per_cadence = sigma_frac_ppm_per_cadence * 1e-6
    base_seed = int(NOISE_CONFIG["random_seed"])

    fs_model, theta_model_deg = _model_grid_axes()
    model_fluxes = _precompute_model_library(system, fs_model, theta_model_deg)
    n_realizations = int(BATCH_RUN_CONFIG["n_noise_realizations"])
    if n_realizations <= 0:
        raise ValueError(f"n_noise_realizations must be positive: {n_realizations}")
    standard_noise = np.stack(
        [
            np.random.default_rng(base_seed + realization_index).normal(size=model_fluxes.shape[1])
            for realization_index in range(n_realizations)
        ],
        axis=0,
    )

    f_grid, theta_inj_grid = _injection_grid()
    pairs: list[tuple[float, float]] = [(float(f), float(th)) for f in f_grid for th in theta_inj_grid]

    br = BATCH_RUN_CONFIG
    max_n = br.get("max_injections")
    if max_n is not None:
        pairs = pairs[: int(max_n)]

    realizations: list[dict[str, Any]] = []
    iterator = enumerate(pairs)
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            total=len(pairs),
            desc=f"{pl_name} injections x {n_realizations} noises",
            unit="injection",
        )

    for injection_index, (f_inj, theta_inj_deg) in iterator:
        injection_results = _run_injection_realizations(
            model_fluxes,
            fs_model,
            theta_model_deg,
            standard_noise,
            f_inj,
            theta_inj_deg,
            sigma_flux_per_cadence,
            base_seed,
        )
        for result in injection_results:
            result["pl_name"] = pl_name
            result["row_index"] = int(pc["row_index"])
            result["noise_cache_target"] = noise_row["target"]
            result["noise_cache_row_index"] = int(noise_row["row_index"])
            result["noise_cache_nint"] = nint
            result["model_cadence_seconds"] = cadence_seconds
            result["sigma_flux_full_transit"] = sigma_flux_full_transit
            result["sigma_frac_ppm_full_transit"] = sigma_frac_ppm_full_transit
            result["sigma_frac_ppm_per_cadence"] = sigma_frac_ppm_per_cadence
            result["injection_index"] = int(injection_index)
        realizations.extend(injection_results)

    return realizations, _aggregate_realizations(realizations)


def _save_metric_figure(
    summaries: list[dict[str, Any]],
    figure_path: Path | str | None = None,
) -> Path:
    matplotlib = __import__("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not summaries:
        raise ValueError("cannot plot empty Monte Carlo summary")

    metrics = [
        ("f_abs_error_median", r"median $|\hat f-f_{\rm inj}|$", "median absolute f error"),
        (
            "theta_abs_error_deg_median",
            r"median angular error (deg)",
            "median absolute theta error",
        ),
        (
            "recovery_distance_grid_steps_median",
            "median distance (fine-grid steps)",
            "combined median recovery distance",
        ),
    ]
    x = np.array([row["f_inj"] for row in summaries], dtype=np.float64)
    y = np.array([row["theta_inj_deg"] for row in summaries], dtype=np.float64)
    best = next(row for row in summaries if row["is_best_median_recovery"])
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2), dpi=120, sharex=True, sharey=True)
    for ax, (key, colorbar_label, panel_title) in zip(axes, metrics, strict=True):
        values = np.array([row[key] for row in summaries], dtype=np.float64)
        points = ax.scatter(
            x,
            y,
            c=values,
            cmap="viridis",
            s=115,
            marker="s",
            edgecolors="black",
            linewidths=0.25,
        )
        ax.scatter(
            [best["f_inj"]],
            [best["theta_inj_deg"]],
            c="#ffd43b",
            s=250,
            marker="*",
            edgecolors="black",
            linewidths=0.9,
            zorder=5,
        )
        ax.set_title(panel_title)
        ax.set_xlabel(r"injected $f_{\mathrm{inj}}$")
        ax.set_xlim(INJECTION_GRID_CONFIG["f_min"] - 0.005, INJECTION_GRID_CONFIG["f_max"] + 0.005)
        ax.set_ylim(
            INJECTION_GRID_CONFIG["theta_inj_deg_min"] - 5.0,
            INJECTION_GRID_CONFIG["theta_inj_deg_max"] + 5.0,
        )
        ax.grid(True, alpha=0.2)
        fig.colorbar(points, ax=ax, label=colorbar_label, pad=0.02)
    axes[0].set_ylabel(r"injected $\theta_{\mathrm{inj}}$ (deg)")
    title = summaries[0]["pl_name"]
    fig.suptitle(
        f"{title} — {summaries[0]['n_noise_realizations']} common-noise recovery realizations; "
        "star = minimum median distance"
    )
    fig.tight_layout()
    root = _repo_root()
    out = (
        Path(figure_path)
        if figure_path is not None
        else root / "results" / str(BATCH_RUN_CONFIG["summary_figure_filename"])
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


RESULT_NUMERIC_KEYS = [
    "f_inj",
    "theta_inj_deg",
    "f_hat",
    "f_abs_error",
    "theta_abs_error_deg",
    "recovery_distance_grid_steps",
    "f_ci68_low",
    "f_ci68_high",
    "f_ci68_err_minus",
    "f_ci68_err_plus",
    "f_ci95_low",
    "f_ci95_high",
    "f_ci95_err_minus",
    "f_ci95_err_plus",
    "theta_hat_deg",
    "theta_ci68_low_deg",
    "theta_ci68_high_deg",
    "theta_ci68_err_minus_deg",
    "theta_ci68_err_plus_deg",
    "theta_ci95_low_deg",
    "theta_ci95_high_deg",
    "theta_ci95_err_minus_deg",
    "theta_ci95_err_plus_deg",
    "chi2_min",
    "reduced_chi2",
    "chi2_true",
    "delta_chi2_true",
    "profile_delta_chi2_68",
    "profile_delta_chi2_95",
    "joint_delta_chi2_68",
    "joint_delta_chi2_95",
    "chi2_1",
    "chi2_2",
    "chi2_gap",
    "sigma_flux_full_transit",
    "sigma_frac_ppm_full_transit",
    "sigma_flux_per_cadence",
    "sigma_frac_ppm_per_cadence",
    "noise_cache_nint",
    "model_cadence_seconds",
    "noise_random_seed",
    "noise_realization_index",
    "injection_index",
    "n_data",
    "degrees_of_freedom",
    "n_chi2_evaluated",
    "n_chi2_unique",
]

RESULT_BOOLEAN_KEYS = [
    "truth_better_than_grid",
    "f_ci95_touches_search_boundary",
    "is_closest_recovery",
]


def _save_npz(
    results: list[dict[str, Any]],
    npz_path: Path | str | None = None,
) -> Path:
    """将列表结果压成列向量存入 npz。"""
    out: dict[str, Any] = {}
    for key in RESULT_NUMERIC_KEYS:
        out[key] = np.array([float(r[key]) for r in results], dtype=np.float64)
    for key in RESULT_BOOLEAN_KEYS:
        out[key] = np.array([bool(r[key]) for r in results], dtype=bool)
    out["coverage"] = np.array([r["coverage"] for r in results], dtype=object)
    out["pl_name"] = results[0]["pl_name"] if results else ""
    if results and "row_index" in results[0]:
        out["row_index"] = int(results[0]["row_index"])
        out["noise_cache_target"] = results[0]["noise_cache_target"]
        out["noise_cache_row_index"] = int(results[0]["noise_cache_row_index"])
    root = _repo_root()
    path = Path(npz_path) if npz_path is not None else root / "results" / str(BATCH_RUN_CONFIG["results_npz_filename"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return path


def _save_realizations_csv(
    results: list[dict[str, Any]],
    csv_path: Path | str | None = None,
) -> Path:
    """Save all injection-by-noise realization results."""
    root = _repo_root()
    path = (
        Path(csv_path)
        if csv_path is not None
        else root / "results" / str(BATCH_RUN_CONFIG["realizations_csv_filename"])
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_keys = [
        "pl_name",
        "row_index",
        "noise_cache_target",
        "noise_cache_row_index",
        "coverage",
    ]
    fieldnames = metadata_keys + RESULT_NUMERIC_KEYS + RESULT_BOOLEAN_KEYS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fieldnames})
    return path


SUMMARY_KEYS = [
    "pl_name",
    "row_index",
    "noise_cache_target",
    "noise_cache_row_index",
    "model_cadence_seconds",
    "injection_index",
    "f_inj",
    "theta_inj_deg",
    "n_noise_realizations",
    "f_hat_median",
    "f_hat_p16",
    "f_hat_p84",
    "theta_hat_circular_median_deg",
    "f_abs_error_median",
    "f_abs_error_p16",
    "f_abs_error_p84",
    "theta_abs_error_deg_median",
    "theta_abs_error_deg_p16",
    "theta_abs_error_deg_p84",
    "recovery_distance_grid_steps_median",
    "recovery_distance_grid_steps_p16",
    "recovery_distance_grid_steps_p84",
    "coverage_68_fraction",
    "coverage_95_fraction",
    "outside_95_fraction",
    "f_ci68_width_median",
    "theta_ci68_width_deg_median",
    "reduced_chi2_median",
    "f_ci95_boundary_fraction",
    "is_best_median_recovery",
]


def _save_summary_csv(
    summaries: list[dict[str, Any]],
    csv_path: Path | str | None = None,
) -> Path:
    """Save one aggregated row per injected ``(f, theta)``."""
    root = _repo_root()
    path = (
        Path(csv_path)
        if csv_path is not None
        else root / "results" / str(BATCH_RUN_CONFIG["summary_csv_filename"])
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_KEYS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary[key] for key in SUMMARY_KEYS})
    return path


def main() -> None:
    realizations, summaries = run_batch(None)
    print("n_injection_grid_points:", len(summaries))
    print("n_realization_results:", len(realizations))
    if not realizations:
        return
    coverage_counts = {
        label: sum(1 for result in realizations if result["coverage"] == label)
        for label in ("inside_68", "inside_95", "outside_95")
    }
    print("joint confidence coverage across all realizations:", coverage_counts)
    print("planet:", realizations[0]["pl_name"])
    print(
        "noise cache target:",
        realizations[0]["noise_cache_target"],
        ", noise cache row_index =",
        realizations[0]["noise_cache_row_index"],
    )
    print(
        "full-transit sigma_flux =",
        realizations[0]["sigma_flux_full_transit"],
        ", sigma_frac_ppm =",
        realizations[0]["sigma_frac_ppm_full_transit"],
    )
    print(
        "per-cadence sigma_flux =",
        realizations[0]["sigma_flux_per_cadence"],
        ", sigma_frac_ppm =",
        realizations[0]["sigma_frac_ppm_per_cadence"],
        ", nint =",
        realizations[0]["noise_cache_nint"],
        ", cadence_s =",
        realizations[0]["model_cadence_seconds"],
    )
    best = next(summary for summary in summaries if summary["is_best_median_recovery"])
    print(
        "best median recovery across common noise realizations:",
        f"injected (f={best['f_inj']:.6g}, theta={best['theta_inj_deg']:.6g} deg),",
        f"median fitted (f={best['f_hat_median']:.6g}, "
        f"theta={best['theta_hat_circular_median_deg']:.6g} deg),",
        f"median distance={best['recovery_distance_grid_steps_median']:.6g} fine-grid steps",
    )
    print(
        "best-point recovery errors:",
        f"median |delta f|={best['f_abs_error_median']:.6g},",
        f"median angular error={best['theta_abs_error_deg_median']:.6g} deg,",
        f"95.4% coverage={best['coverage_95_fraction']:.3f}",
    )

    fig_path = _save_metric_figure(summaries)
    print("saved figure:", fig_path)
    if BATCH_RUN_CONFIG.get("save_results_npz"):
        p = _save_npz(realizations)
        print("saved npz:", p)
    if BATCH_RUN_CONFIG.get("save_realizations_csv"):
        p = _save_realizations_csv(realizations)
        print("saved realizations csv:", p)
    if BATCH_RUN_CONFIG.get("save_summary_csv"):
        p = _save_summary_csv(summaries)
        print("saved summary csv:", p)


if __name__ == "__main__":
    main()
