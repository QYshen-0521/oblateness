"""批量含噪声反演，并输出参数估计与 profile-χ² 置信区间。

噪声缓存中的 ``sigma_flux`` 视为整场凌星合并后的归一化光通量误差。当前时间轴约为一次积分
一个 cadence，因此用缓存中的 ``nint`` 近似换算
``sigma_flux_per_cadence = sigma_flux * sqrt(nint)``，再生成独立白色高斯噪声。

每次注入输出最佳拟合 ``f``、``theta``，以及一维 profile-χ² 的 68.3% 和 95.4% 非对称
置信区间。二维真值覆盖使用 Δχ²=2.30 和 6.18，仅作诊断，不再使用 A1 的严格命中判据。
批处理还会用细化网格步长归一化参数偏差，标出恢复值离各自注入真值最近的注入格点。

运行（需 ``pip install -e ".[squishy]"``）::

    python -m oblateness.noised_grid_recovery_single_planet_batch_v02

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

from oblateness.noiseless_recovery_tool_ingress_egress_sampling import build_lightcurve_time_array_days
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
BATCH_RUN_CONFIG: dict[str, Any] = {
    # None = 全部注入格点；设为例如 3 便于调试
    "max_injections": None,
    "summary_figure_filename": "batch_noised_confidence_ie60/batch_noised_f_theta_coverage.png",
    "save_results_npz": True,
    "results_npz_filename": "batch_noised_confidence_ie60/batch_noised_recovery.npz",
    "save_results_csv": True,
    "results_csv_filename": "batch_noised_confidence_ie60/batch_noised_recovery.csv",
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

    ts = pc.get("time_sampling")
    if not isinstance(ts, dict):
        ts = None
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


def run_batch(planet_cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """对 **单个** 行星跑完整注入网格；``planet_cfg`` 默认 ``BASE_PLANET_CONFIG``（见 ``PLANET_CONFIG['row_index']``）。"""
    pc = planet_cfg if planet_cfg is not None else BASE_PLANET_CONFIG
    root = _repo_root()
    csv_path = root / pc["csv_path"]
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    system, pl_name, _ = _build_system(pc)
    noise_row = _load_success_noise_for_target(root / str(NOISE_CONFIG["cache_csv_path"]), pl_name)
    sigma_flux_full_transit = float(noise_row["sigma_flux"])
    sigma_frac_ppm_full_transit = float(noise_row["sigma_frac_ppm"])
    try:
        nint = int(round(float(noise_row["nint"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"noise cache has invalid nint for {pl_name}: {noise_row.get('nint')!r}") from exc
    if nint <= 0:
        raise ValueError(f"noise cache nint must be positive for {pl_name}: {nint}")

    # The cache sigma is for all nint integrations combined. Approximate the uncertainty
    # of one cadence/integration before adding independent noise to the light-curve points.
    sigma_flux_per_cadence = sigma_flux_full_transit * np.sqrt(nint)
    sigma_frac_ppm_per_cadence = sigma_frac_ppm_full_transit * np.sqrt(nint)
    base_seed = int(NOISE_CONFIG["random_seed"])

    f_grid, theta_inj_grid = _injection_grid()
    pairs: list[tuple[float, float]] = [(float(f), float(th)) for f in f_grid for th in theta_inj_grid]

    br = BATCH_RUN_CONFIG
    max_n = br.get("max_injections")
    if max_n is not None:
        pairs = pairs[: int(max_n)]

    results: list[dict[str, Any]] = []
    iterator = enumerate(pairs)
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            total=len(pairs),
            desc=f"{pl_name} noised injections",
            unit="grid",
        )

    for injection_index, (f_inj, theta_inj_deg) in iterator:
        noise_seed = base_seed + injection_index
        r = run_single_injection(system, f_inj, theta_inj_deg, sigma_flux_per_cadence, noise_seed)
        r["pl_name"] = pl_name
        r["row_index"] = int(pc["row_index"])
        r["noise_cache_target"] = noise_row["target"]
        r["noise_cache_row_index"] = int(noise_row["row_index"])
        r["noise_cache_nint"] = nint
        r["sigma_flux_full_transit"] = sigma_flux_full_transit
        r["sigma_frac_ppm_full_transit"] = sigma_frac_ppm_full_transit
        r["sigma_frac_ppm_per_cadence"] = sigma_frac_ppm_per_cadence
        r["injection_index"] = int(injection_index)
        results.append(r)

    if results:
        closest_index = int(np.argmin([r["recovery_distance_grid_steps"] for r in results]))
        for index, result in enumerate(results):
            result["is_closest_recovery"] = index == closest_index
    return results


def _save_confidence_figure(
    results: list[dict[str, Any]],
    figure_path: Path | str | None = None,
) -> Path:
    matplotlib = __import__("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = {"inside_68": "#2ca02c", "inside_95": "#ffbf00", "outside_95": "#d62728"}
    label = {
        "inside_68": "truth inside 68.3% region",
        "inside_95": "truth inside 95.4% region",
        "outside_95": "truth outside 95.4% region",
    }
    fig, ax = plt.subplots(figsize=(9.0, 6.0), dpi=120)
    for coverage in ("inside_68", "inside_95", "outside_95"):
        pts = [r for r in results if r["coverage"] == coverage]
        if not pts:
            continue
        ax.scatter(
            [p["f_inj"] for p in pts],
            [p["theta_inj_deg"] for p in pts],
            c=color[coverage],
            s=55,
            marker="o",
            edgecolors="k",
            linewidths=0.3,
            label=label[coverage],
            zorder=3,
        )
    closest = next((r for r in results if r["is_closest_recovery"]), None)
    if closest is not None:
        ax.scatter(
            [closest["f_inj"]],
            [closest["theta_inj_deg"]],
            c="#ffd43b",
            s=260,
            marker="*",
            edgecolors="black",
            linewidths=0.9,
            label="closest recovery to injected truth",
            zorder=5,
        )
        ax.annotate(
            (
                f"closest recovery\n"
                f"inj=({closest['f_inj']:.3f}, {closest['theta_inj_deg']:.1f} deg)\n"
                f"fit=({closest['f_hat']:.3f}, {closest['theta_hat_deg']:.1f} deg)"
            ),
            xy=(closest["f_inj"], closest["theta_inj_deg"]),
            xytext=(10, 12),
            textcoords="offset points",
            fontsize=8,
            ha="left",
            va="bottom",
            bbox={"boxstyle": "square,pad=0.3", "facecolor": "white", "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "linewidth": 0.8},
            zorder=6,
        )
    ax.set_xlabel(r"injected $f_{\mathrm{inj}}$")
    ax.set_ylabel(r"injected $\theta_{\mathrm{inj}}$ (deg)")
    title = results[0]["pl_name"] if results else ""
    ax.set_title(f"{title} — joint profile-$\\chi^2$ coverage")
    ax.set_xlim(INJECTION_GRID_CONFIG["f_min"] - 0.005, INJECTION_GRID_CONFIG["f_max"] + 0.005)
    ax.set_ylim(
        INJECTION_GRID_CONFIG["theta_inj_deg_min"] - 5.0,
        INJECTION_GRID_CONFIG["theta_inj_deg_max"] + 5.0,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
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
    "noise_random_seed",
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


def _save_csv(
    results: list[dict[str, Any]],
    csv_path: Path | str | None = None,
) -> Path:
    """保存便于直接查看的参数估计和置信区间表。"""
    root = _repo_root()
    path = (
        Path(csv_path)
        if csv_path is not None
        else root / "results" / str(BATCH_RUN_CONFIG["results_csv_filename"])
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


def main() -> None:
    results = run_batch(None)
    print("n_results:", len(results))
    if not results:
        return
    coverage_counts = {
        label: sum(1 for r in results if r["coverage"] == label)
        for label in ("inside_68", "inside_95", "outside_95")
    }
    print("joint confidence coverage:", coverage_counts)
    print("planet:", results[0]["pl_name"])
    print("noise cache target:", results[0]["noise_cache_target"], ", noise cache row_index =", results[0]["noise_cache_row_index"])
    print(
        "full-transit sigma_flux =",
        results[0]["sigma_flux_full_transit"],
        ", sigma_frac_ppm =",
        results[0]["sigma_frac_ppm_full_transit"],
    )
    print(
        "per-cadence sigma_flux =",
        results[0]["sigma_flux_per_cadence"],
        ", sigma_frac_ppm =",
        results[0]["sigma_frac_ppm_per_cadence"],
        ", nint =",
        results[0]["noise_cache_nint"],
    )
    closest = next(r for r in results if r["is_closest_recovery"])
    print(
        "closest recovery to injected truth:",
        f"injected (f={closest['f_inj']:.6g}, theta={closest['theta_inj_deg']:.6g} deg),",
        f"fitted (f={closest['f_hat']:.6g}, theta={closest['theta_hat_deg']:.6g} deg),",
        f"distance={closest['recovery_distance_grid_steps']:.6g} refine-grid steps",
    )
    print(
        "closest fitted 68.3% intervals:",
        f"f={closest['f_hat']:.6g} -{closest['f_ci68_err_minus']:.6g} +{closest['f_ci68_err_plus']:.6g},",
        f"theta={closest['theta_hat_deg']:.6g} -{closest['theta_ci68_err_minus_deg']:.6g}"
        f" +{closest['theta_ci68_err_plus_deg']:.6g} deg",
    )

    fig_path = _save_confidence_figure(results)
    print("saved figure:", fig_path)
    if BATCH_RUN_CONFIG.get("save_results_npz"):
        p = _save_npz(results)
        print("saved npz:", p)
    if BATCH_RUN_CONFIG.get("save_results_csv"):
        p = _save_csv(results)
        print("saved csv:", p)


if __name__ == "__main__":
    main()
