"""单行星含噪声扁率探测完备度地图。

程序按 ``TARGET_NAME`` 从行星输入表和 calculate-noise 缓存中匹配目标。每个 cadence 注入的
独立白色高斯噪声按照 Wang & Winn (2026), arXiv:2607.09873v1 的经验噪声模型计算：
先由 K 星等计算 Equation (2) 的 1 分钟白光散布，再依据白噪声的平方根时间定律缩放到缓存
``t_exp_s`` 对应的实际 cadence。

程序先对 ``f=0`` 的球形注入做 Monte Carlo，并通过球形模型与最佳扁球模型之间的
``delta_chi2`` 分布标定经验假阳性阈值。随后用少量噪声 realization 扫描完整的粗注入网格，
只在探测概率跨过 50%/90% 或变化迅速的区域将 ``f`` 加密并增加噪声 realization 数量。
最终输出目标级 ``P_det(f, theta)`` 完备度地图，以及带 95% 置信区间的
``f50(theta)``、``f90(theta)`` 曲线。

运行（需 ``pip install -e ".[squishy]"``）::

    python -m oblateness.noised_grid_recovery_single_planet_batch_v04

默认运行自适应两阶段网格；调试可将 ``BATCH_RUN_CONFIG['max_injections']`` 设为小整数。
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

# 只需修改这一处即可切换目标；名称允许忽略大小写、空格和连字符。
TARGET_NAME = "TOI-2537 b"

# =============================================================================
# 行星 + LD + 时间轴（与单点反演一致；不含注入 —— 注入由 INJECTION_GRID_CONFIG 扫参）
# =============================================================================
BASE_PLANET_CONFIG: dict[str, Any] = {
    k: v
    for k, v in PLANET_CONFIG.items()
    if k not in ("injected_projected_f", "injected_projected_theta_deg")
}
BASE_PLANET_CONFIG["target_name"] = TARGET_NAME

# =============================================================================
# 第一阶段注入扫参网格：f 粗步长 0.01；theta 全局步长 5°
# =============================================================================
INJECTION_GRID_CONFIG: dict[str, Any] = {
    "f_min": 0.01,
    "f_max": 0.15,
    "f_step": 0.01,
    "theta_inj_deg_min": 0.0,
    "theta_inj_deg_max": 160.0,
    "theta_inj_deg_step": 5.0,
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
    # Wang & Winn (2026), Equation (2), is normalized to one-minute bins.
    "reference_cadence_seconds": 60.0,
    "valid_kmag_min": 7.0,
    "kmag_break": 9.5,
    "valid_kmag_max": 16.0,
}

DETECTION_CONFIG: dict[str, Any] = {
    "false_alarm_probability": 0.01,
    "detection_probability_threshold": 0.90,
    "n_null_realizations": 10000,
    "null_seed_offset": 1_000_000,
}

ADAPTIVE_COMPLETENESS_CONFIG: dict[str, Any] = {
    "enabled": True,
    # 粗地图每点先用较少的共同噪声 realization 定位概率过渡区域。
    "pilot_n_noise_realizations": 50,
    # 被选中的粗点和新增细点使用完整 realization 数。
    "final_n_noise_realizations": 500,
    "refined_f_step": 0.002,
    "probability_levels": (0.50, 0.90),
    # 即使没有恰好跨过 50%/90%，相邻点变化足够快也要加密。
    "probability_gradient_threshold": 0.20,
    # 在命中过渡条件的区间两侧再各保留一个粗区间，防止截断边界。
    "coarse_neighbor_intervals": 1,
    # 试跑阶段用约 68% Wilson 区间判断格点是否可能位于 50%/90% 边界附近。
    # 这不会把明显 0%/100% 的区域全部升级为完整 realization 数。
    "pilot_transition_interval_z": 1.0,
    # 最终 f50/f90 使用 95% Wilson 概率包络反推扁率置信区间。
    "detection_limit_ci_z": 1.959963984540054,
    "detection_limit_confidence_level": 0.95,
}

BATCH_RUN_CONFIG: dict[str, Any] = {
    # None = 全部注入格点；设为例如 3 便于调试
    "max_injections": None,
    "n_noise_realizations": 500,
    "summary_figure_filename": "batch_noised_completeness_v04_adaptive_ie60/{target_slug}/detection_completeness_map.png",
    "save_results_npz": True,
    "results_npz_filename": "batch_noised_completeness_v04_adaptive_ie60/{target_slug}/detection_realizations.npz",
    "save_realizations_csv": True,
    "realizations_csv_filename": "batch_noised_completeness_v04_adaptive_ie60/{target_slug}/detection_realizations.csv",
    "save_summary_csv": True,
    "summary_csv_filename": "batch_noised_completeness_v04_adaptive_ie60/{target_slug}/detection_probability_summary.csv",
    "save_detection_limits_csv": True,
    "detection_limits_csv_filename": "batch_noised_completeness_v04_adaptive_ie60/{target_slug}/f50_f90_curves.csv",
    "save_null_calibration_npz": True,
    "null_calibration_npz_filename": "batch_noised_completeness_v04_adaptive_ie60/{target_slug}/null_calibration.npz",
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

# Squishyplanet's implicit-to-parametric ellipse conversion takes a wrong
# zero-rotation branch at exactly 45 and 135 degrees. This internal offset is
# many orders of magnitude below the sampled angular resolution.
PROJECTED_THETA_SINGULARITY_OFFSET_DEG = 1.0e-7


def _target_name_key(name: str) -> str:
    """Normalize a target name for tolerant matching."""
    return "".join(character for character in str(name).casefold() if character.isalnum())


def _load_csv_target(path: Path, target_name: str) -> tuple[int, dict[str, str]]:
    """Return the table row index and row matching ``target_name``."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"planet input table is empty: {path}")
    if "pl_name" not in rows[0]:
        raise KeyError(f"planet input table has no pl_name column: {path}")

    requested = str(target_name).strip()
    exact = [
        (index, row)
        for index, row in enumerate(rows)
        if row["pl_name"].strip() == requested
    ]
    matches = exact
    if not matches:
        requested_key = _target_name_key(requested)
        matches = [
            (index, row)
            for index, row in enumerate(rows)
            if _target_name_key(row["pl_name"]) == requested_key
        ]
    if not matches:
        raise ValueError(f"target {target_name!r} is not in planet input table: {path}")
    if len(matches) > 1:
        names = ", ".join(row["pl_name"].strip() for _, row in matches)
        raise ValueError(f"target name {target_name!r} is ambiguous in planet input table: {names}")
    return matches[0]


def _target_slug(target_name: str) -> str:
    """Return a filesystem-friendly target label for per-target result folders."""
    slug = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(target_name).strip()
    )
    return slug.strip("_") or "target"


def _configured_output_path(config_key: str, target_name: str) -> Path:
    relative_path = str(BATCH_RUN_CONFIG[config_key]).format(
        target_slug=_target_slug(target_name)
    )
    return _repo_root() / "results" / relative_path


def _noise_sigma_from_cache(
    noise_row: dict[str, str],
    target_name: str,
    cadence_seconds: float,
) -> dict[str, float]:
    """Evaluate the Wang & Winn (2026) empirical 1-min model at the target cadence."""
    try:
        cache_sigma_ppm = float(noise_row["sigma_frac_ppm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"noise cache has invalid sigma_frac_ppm for {target_name}: "
            f"{noise_row.get('sigma_frac_ppm')!r}"
        ) from exc
    try:
        kmag = float(noise_row["kmag"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"noise cache has invalid kmag for {target_name}: "
            f"{noise_row.get('kmag')!r}"
        ) from exc

    reference_seconds = float(NOISE_INJECTION_CONFIG["reference_cadence_seconds"])
    kmag_min = float(NOISE_INJECTION_CONFIG["valid_kmag_min"])
    kmag_break = float(NOISE_INJECTION_CONFIG["kmag_break"])
    kmag_max = float(NOISE_INJECTION_CONFIG["valid_kmag_max"])
    if not np.isfinite(cache_sigma_ppm) or cache_sigma_ppm <= 0.0:
        raise ValueError(
            f"noise cache sigma_frac_ppm must be positive and finite for "
            f"{target_name}: {cache_sigma_ppm}"
        )
    if not np.isfinite(kmag):
        raise ValueError(f"noise cache kmag must be finite for {target_name}: {kmag}")
    if not np.isfinite(cadence_seconds) or cadence_seconds <= 0.0:
        raise ValueError(
            f"cadence_seconds must be positive and finite for "
            f"{target_name}: {cadence_seconds}"
        )
    if not np.isfinite(reference_seconds) or reference_seconds <= 0.0:
        raise ValueError(
            f"reference_cadence_seconds must be positive and finite: {reference_seconds}"
        )

    if kmag_min <= kmag <= kmag_break:
        sigma_ppm_one_minute = 63.0 * 10.0 ** (0.2 * (kmag - kmag_break)) + 14.0
    elif kmag_break < kmag < kmag_max:
        sigma_ppm_one_minute = 23.0 * 10.0 ** (0.2 * (kmag - kmag_break)) + 55.0
    else:
        raise ValueError(
            f"Wang & Winn (2026) empirical noise model is valid only for "
            f"{kmag_min} <= K < {kmag_max}; {target_name} has K={kmag}"
        )

    injected_sigma_ppm = sigma_ppm_one_minute * np.sqrt(
        reference_seconds / float(cadence_seconds)
    )
    return {
        "cache_sigma_ppm": cache_sigma_ppm,
        "kmag": kmag,
        "sigma_ppm_one_minute": float(sigma_ppm_one_minute),
        "injected_sigma_ppm": injected_sigma_ppm,
        "injected_sigma_flux": injected_sigma_ppm * 1e-6,
    }


def _theta_distance_deg(a_deg: float, b_deg: float) -> float:
    """Return the shortest angular distance on the 180-degree periodic domain."""
    a = float(a_deg) % 180.0
    b = float(b_deg) % 180.0
    distance = abs(a - b)
    return float(min(distance, 180.0 - distance))


def _safe_projected_theta_rad(theta_rad: float) -> float:
    """Nudge exact 45/135-degree ellipse-conversion singularities."""
    theta = float(theta_rad) % np.pi
    if np.isclose(np.cos(2.0 * theta), 0.0, rtol=0.0, atol=1.0e-14):
        theta += np.deg2rad(PROJECTED_THETA_SINGULARITY_OFFSET_DEG)
    return float(theta % np.pi)


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
                system.lightcurve(
                    params={
                        "projected_f": float(f),
                        "projected_theta": _safe_projected_theta_rad(float(th)),
                    }
                )
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
    th_inj_rad = _safe_projected_theta_rad(np.deg2rad(theta_inj_deg))
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
    return {
        **BASE_PLANET_CONFIG,
        "target_name": None,
        "row_index": int(row_index),
    }


def planet_config_for_target(target_name: str) -> dict[str, Any]:
    """Return a planet configuration that resolves the input row by target name."""
    return {
        **BASE_PLANET_CONFIG,
        "target_name": str(target_name),
    }


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
        system.lightcurve(
            params={
                "projected_f": float(fs[0]),
                "projected_theta": _safe_projected_theta_rad(
                    np.deg2rad(theta_deg[0])
                ),
            }
        ),
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
                    "projected_theta": _safe_projected_theta_rad(
                        np.deg2rad(theta_deg[i_theta])
                    ),
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


def _calibrate_null_detection_threshold(
    model_fluxes: np.ndarray,
    standard_noise: np.ndarray,
    sigma_flux_per_cadence: float,
) -> dict[str, Any]:
    """Calibrate the look-elsewhere-aware detection threshold with spherical injections."""
    false_alarm_probability = float(DETECTION_CONFIG["false_alarm_probability"])
    if not 0.0 < false_alarm_probability < 1.0:
        raise ValueError(
            "false_alarm_probability must lie strictly between 0 and 1: "
            f"{false_alarm_probability}"
        )

    spherical_flux = np.asarray(model_fluxes[0], dtype=np.float64)
    observed = spherical_flux[None, :] + float(sigma_flux_per_cadence) * standard_noise
    chi2_flat = _chi2_from_model_library(
        model_fluxes,
        observed,
        sigma_flux_per_cadence,
    )
    chi2_spherical = chi2_flat[0]
    chi2_best_oblate = np.min(chi2_flat, axis=0)
    delta_chi2_null = np.maximum(0.0, chi2_spherical - chi2_best_oblate)
    threshold = float(
        np.quantile(
            delta_chi2_null,
            1.0 - false_alarm_probability,
            method="higher",
        )
    )
    return {
        "delta_chi2_null": delta_chi2_null,
        "delta_chi2_threshold": threshold,
        "false_alarm_probability_target": false_alarm_probability,
        "false_alarm_probability_empirical": float(np.mean(delta_chi2_null > threshold)),
        "n_null_realizations": int(delta_chi2_null.size),
    }


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
    system: Any,
    model_fluxes: np.ndarray,
    fs: np.ndarray,
    theta_deg: np.ndarray,
    standard_noise: np.ndarray,
    f_inj: float,
    theta_inj_deg: float,
    sigma_flux_per_cadence: float,
    base_seed: int,
    detection_delta_chi2_threshold: float,
) -> list[dict[str, Any]]:
    """Fit all common-noise realizations for one injected ``(f, theta)``."""
    n_theta = int(theta_deg.size)
    clean_flux = np.asarray(
        system.lightcurve(
            params={
                "projected_f": float(f_inj),
                "projected_theta": _safe_projected_theta_rad(
                    np.deg2rad(theta_inj_deg)
                ),
            }
        ),
        dtype=np.float64,
    )
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
        chi2_spherical = float(chi2_values[0])
        delta_chi2_spherical = max(0.0, chi2_spherical - chi2_min)
        chi2_true = float(np.sum(standard_noise[realization_index] ** 2))
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
                "chi2_spherical": chi2_spherical,
                "delta_chi2_spherical": delta_chi2_spherical,
                "detection_delta_chi2_threshold": float(
                    detection_delta_chi2_threshold
                ),
                "is_detected": bool(
                    delta_chi2_spherical > float(detection_delta_chi2_threshold)
                ),
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


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return the two-sided Wilson interval for a binomial detection probability."""
    if total <= 0:
        return float("nan"), float("nan")
    probability = successes / total
    denominator = 1.0 + z**2 / total
    center = (probability + z**2 / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            probability * (1.0 - probability) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def _injection_pair_key(f_inj: float, theta_inj_deg: float) -> tuple[float, float]:
    """Return a stable key for merging coarse and adaptive injection points."""
    return round(float(f_inj), 10), round(float(theta_inj_deg), 8)


def _adaptive_refinement_plan(
    pilot_summaries: list[dict[str, Any]],
) -> tuple[set[tuple[float, float]], list[tuple[float, float]], dict[str, int]]:
    """Select coarse points to deepen and new f points to add near probability transitions."""
    cfg = ADAPTIVE_COMPLETENESS_CONFIG
    levels = tuple(float(value) for value in cfg["probability_levels"])
    if not levels or any(not 0.0 < value < 1.0 for value in levels):
        raise ValueError(f"invalid completeness probability levels: {levels}")
    refined_step = float(cfg["refined_f_step"])
    coarse_step = float(INJECTION_GRID_CONFIG["f_step"])
    if refined_step <= 0.0 or refined_step >= coarse_step:
        raise ValueError(
            "refined_f_step must be positive and smaller than the coarse f step: "
            f"{refined_step}, {coarse_step}"
        )

    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in pilot_summaries:
        grouped.setdefault(float(row["theta_inj_deg"]), []).append(row)

    selected_coarse: set[tuple[float, float]] = set()
    fine_pairs: set[tuple[float, float]] = set()
    marked_interval_count = 0
    gradient_threshold = float(cfg["probability_gradient_threshold"])
    neighbor_count = max(0, int(cfg["coarse_neighbor_intervals"]))
    false_alarm_probability = float(DETECTION_CONFIG["false_alarm_probability"])

    for theta_deg, unsorted_rows in grouped.items():
        rows = sorted(unsorted_rows, key=lambda row: float(row["f_inj"]))
        if not rows:
            continue
        interval_indices: set[int] = set()
        for index in range(len(rows) - 1):
            p_left = float(rows[index]["detection_probability"])
            p_right = float(rows[index + 1]["detection_probability"])
            p_low = min(p_left, p_right)
            p_high = max(p_left, p_right)
            crosses_level = any(p_low <= level <= p_high for level in levels)
            changes_quickly = abs(p_right - p_left) >= gradient_threshold
            if crosses_level or changes_quickly:
                interval_indices.add(index)

        transition_z = float(cfg["pilot_transition_interval_z"])
        for index, row in enumerate(rows):
            interval_low, interval_high = _wilson_interval(
                int(row["detected_count"]),
                int(row["n_noise_realizations"]),
                z=transition_z,
            )
            if any(
                interval_low <= level <= interval_high
                for level in levels
            ):
                if index > 0:
                    interval_indices.add(index - 1)
                if index < len(rows) - 1:
                    interval_indices.add(index)

        first_f = float(rows[0]["f_inj"])
        first_probability = float(rows[0]["detection_probability"])
        refine_below_first = any(
            false_alarm_probability <= level <= first_probability
            for level in levels
        ) or abs(first_probability - false_alarm_probability) >= gradient_threshold

        expanded_indices = set(interval_indices)
        for index in interval_indices:
            for offset in range(1, neighbor_count + 1):
                if index - offset >= 0:
                    expanded_indices.add(index - offset)
                if index + offset < len(rows) - 1:
                    expanded_indices.add(index + offset)

        for index in sorted(expanded_indices):
            marked_interval_count += 1
            left_f = float(rows[index]["f_inj"])
            right_f = float(rows[index + 1]["f_inj"])
            selected_coarse.add(_injection_pair_key(left_f, theta_deg))
            selected_coarse.add(_injection_pair_key(right_f, theta_deg))
            first_multiple = int(np.floor(left_f / refined_step + 1.0e-9)) + 1
            last_multiple = int(np.ceil(right_f / refined_step - 1.0e-9))
            for multiple in range(first_multiple, last_multiple):
                fine_f = multiple * refined_step
                if left_f < fine_f < right_f:
                    fine_pairs.add(_injection_pair_key(fine_f, theta_deg))

        if refine_below_first and first_f > refined_step:
            selected_coarse.add(_injection_pair_key(first_f, theta_deg))
            for multiple in range(1, int(np.ceil(first_f / refined_step))):
                fine_f = multiple * refined_step
                if fine_f < first_f:
                    fine_pairs.add(_injection_pair_key(fine_f, theta_deg))

    diagnostics = {
        "selected_coarse_points": len(selected_coarse),
        "new_fine_points": len(fine_pairs),
        "marked_intervals": marked_interval_count,
    }
    return selected_coarse, sorted(fine_pairs, key=lambda pair: (pair[1], pair[0])), diagnostics


def _weighted_isotonic_non_decreasing(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted pool-adjacent-violators fit for a non-decreasing probability curve."""
    y = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if y.ndim != 1 or w.shape != y.shape or np.any(w <= 0.0):
        raise ValueError("isotonic values and positive weights must be matching 1-D arrays")

    blocks: list[list[float | int]] = []
    for index, (value, weight) in enumerate(zip(y, w, strict=True)):
        blocks.append([index, index, float(weight), float(value * weight)])
        while len(blocks) >= 2:
            previous = blocks[-2]
            current = blocks[-1]
            previous_mean = float(previous[3]) / float(previous[2])
            current_mean = float(current[3]) / float(current[2])
            if previous_mean <= current_mean:
                break
            blocks[-2:] = [
                [
                    int(previous[0]),
                    int(current[1]),
                    float(previous[2]) + float(current[2]),
                    float(previous[3]) + float(current[3]),
                ]
            ]

    fitted = np.empty_like(y)
    for start, stop, total_weight, weighted_sum in blocks:
        fitted[int(start) : int(stop) + 1] = float(weighted_sum) / float(total_weight)
    return fitted


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
        detected_count = sum(bool(row["is_detected"]) for row in rows)
        detection_probability = detected_count / len(rows)
        detection_probability_ci95_low, detection_probability_ci95_high = _wilson_interval(
            detected_count,
            len(rows),
        )
        summary = {
            "pl_name": rows[0]["pl_name"],
            "row_index": int(rows[0]["row_index"]),
            "noise_cache_target": rows[0]["noise_cache_target"],
            "noise_cache_row_index": int(rows[0]["noise_cache_row_index"]),
            "model_cadence_seconds": float(rows[0]["model_cadence_seconds"]),
            "sigma_frac_ppm_full_transit": float(
                rows[0]["sigma_frac_ppm_full_transit"]
            ),
            "noise_model_kmag": float(rows[0]["noise_model_kmag"]),
            "sigma_frac_ppm_one_minute": float(
                rows[0]["sigma_frac_ppm_one_minute"]
            ),
            "sigma_frac_ppm_per_cadence": float(
                rows[0]["sigma_frac_ppm_per_cadence"]
            ),
            "sampling_stage": str(rows[0]["sampling_stage"]),
            "is_adaptive_refinement": bool(rows[0]["is_adaptive_refinement"]),
            "injection_index": injection_index,
            "f_inj": float(rows[0]["f_inj"]),
            "theta_inj_deg": float(rows[0]["theta_inj_deg"]),
            "n_noise_realizations": len(rows),
            "detected_count": detected_count,
            "detection_probability": detection_probability,
            "detection_probability_ci95_low": detection_probability_ci95_low,
            "detection_probability_ci95_high": detection_probability_ci95_high,
            "detection_probability_threshold": float(
                DETECTION_CONFIG["detection_probability_threshold"]
            ),
            "detection_delta_chi2_threshold": float(
                rows[0]["detection_delta_chi2_threshold"]
            ),
            "delta_chi2_spherical_median": float(
                np.median(values("delta_chi2_spherical"))
            ),
            "delta_chi2_spherical_p16": float(
                np.percentile(values("delta_chi2_spherical"), 16.0)
            ),
            "delta_chi2_spherical_p84": float(
                np.percentile(values("delta_chi2_spherical"), 84.0)
            ),
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
        }
        summaries.append(summary)
    return summaries


def _interpolate_probability_crossing(
    f_values: np.ndarray,
    probabilities: np.ndarray,
    probability_level: float,
) -> dict[str, Any]:
    """Locate the first interpolated crossing of a monotonic probability curve."""
    passing_indices = np.flatnonzero(probabilities >= probability_level)
    if passing_indices.size == 0:
        return {
            "f_detection_limit": float("nan"),
            "detection_probability_at_limit": float("nan"),
            "previous_f_inj": float(f_values[-1]),
            "previous_detection_probability": float(probabilities[-1]),
            "next_f_inj": float("nan"),
            "next_detection_probability": float("nan"),
            "limit_status": "not_reached",
        }

    first_index = int(passing_indices[0])
    if first_index == 0:
        return {
            "f_detection_limit": float(f_values[0]),
            "detection_probability_at_limit": probability_level,
            "previous_f_inj": float("nan"),
            "previous_detection_probability": float("nan"),
            "next_f_inj": float(f_values[0]),
            "next_detection_probability": float(probabilities[0]),
            "limit_status": "below_or_at_grid_min",
        }

    previous_index = first_index - 1
    previous_f = float(f_values[previous_index])
    previous_probability = float(probabilities[previous_index])
    next_f = float(f_values[first_index])
    next_probability = float(probabilities[first_index])
    probability_span = next_probability - previous_probability
    if probability_span <= 0.0:
        f_limit = next_f
    else:
        fraction = (probability_level - previous_probability) / probability_span
        f_limit = previous_f + float(np.clip(fraction, 0.0, 1.0)) * (
            next_f - previous_f
        )
    return {
        "f_detection_limit": f_limit,
        "detection_probability_at_limit": probability_level,
        "previous_f_inj": previous_f,
        "previous_detection_probability": previous_probability,
        "next_f_inj": next_f,
        "next_detection_probability": next_probability,
        "limit_status": "interpolated",
    }


def _derive_detection_limits(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive monotonic ``f50(theta)``/``f90(theta)`` curves and 95% intervals."""
    probability_levels = tuple(
        float(value)
        for value in ADAPTIVE_COMPLETENESS_CONFIG["probability_levels"]
    )
    confidence_z = float(
        ADAPTIVE_COMPLETENESS_CONFIG["detection_limit_ci_z"]
    )
    confidence_level = float(
        ADAPTIVE_COMPLETENESS_CONFIG["detection_limit_confidence_level"]
    )
    limits: list[dict[str, Any]] = []
    theta_values = sorted({float(row["theta_inj_deg"]) for row in summaries})
    for theta_deg in theta_values:
        rows = sorted(
            (
                row
                for row in summaries
                if np.isclose(float(row["theta_inj_deg"]), theta_deg)
            ),
            key=lambda row: float(row["f_inj"]),
        )
        if not rows:
            continue
        f_values = np.array([float(row["f_inj"]) for row in rows], dtype=np.float64)
        raw_probabilities = np.array(
            [float(row["detection_probability"]) for row in rows],
            dtype=np.float64,
        )
        weights = np.array(
            [float(row["n_noise_realizations"]) for row in rows],
            dtype=np.float64,
        )
        fitted_probabilities = _weighted_isotonic_non_decreasing(
            raw_probabilities,
            weights,
        )
        probability_intervals = np.array(
            [
                _wilson_interval(
                    int(row["detected_count"]),
                    int(row["n_noise_realizations"]),
                    z=confidence_z,
                )
                for row in rows
            ],
            dtype=np.float64,
        )
        fitted_probability_lows = _weighted_isotonic_non_decreasing(
            probability_intervals[:, 0],
            weights,
        )
        fitted_probability_highs = _weighted_isotonic_non_decreasing(
            probability_intervals[:, 1],
            weights,
        )
        for probability_level in probability_levels:
            estimate = _interpolate_probability_crossing(
                f_values,
                fitted_probabilities,
                probability_level,
            )
            # The upper probability envelope crosses first and therefore gives
            # the lower f bound; the lower envelope gives the upper f bound.
            lower_bound = _interpolate_probability_crossing(
                f_values,
                fitted_probability_highs,
                probability_level,
            )
            upper_bound = _interpolate_probability_crossing(
                f_values,
                fitted_probability_lows,
                probability_level,
            )
            limits.append(
                {
                    "pl_name": rows[0]["pl_name"],
                    "theta_inj_deg": theta_deg,
                    "probability_level": probability_level,
                    **estimate,
                    "f_detection_limit_ci95_low": float(
                        lower_bound["f_detection_limit"]
                    ),
                    "f_detection_limit_ci95_high": float(
                        upper_bound["f_detection_limit"]
                    ),
                    "f_detection_limit_ci95_low_status": str(
                        lower_bound["limit_status"]
                    ),
                    "f_detection_limit_ci95_high_status": str(
                        upper_bound["limit_status"]
                    ),
                    "confidence_level": confidence_level,
                    "confidence_method": "inverted_wilson_isotonic",
                }
            )
    return limits


def run_batch(
    planet_cfg: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Calibrate the spherical null, then run all injections with common Gaussian noise."""
    pc = planet_cfg if planet_cfg is not None else BASE_PLANET_CONFIG
    root = _repo_root()
    csv_path = root / pc["csv_path"]
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    configured_target_name = pc.get("target_name")
    if configured_target_name:
        row_index, input_row = _load_csv_target(csv_path, str(configured_target_name))
    else:
        row_index = int(pc["row_index"])
        input_row = _load_csv_row(csv_path, row_index)
    resolved_pc = {**pc, "row_index": row_index}
    pl_name = input_row["pl_name"].strip()
    noise_row = _load_success_noise_for_target(root / str(NOISE_CONFIG["cache_csv_path"]), pl_name)
    sigma_flux_full_transit = float(noise_row["sigma_flux"])
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
    noise_sigma = _noise_sigma_from_cache(
        noise_row,
        pl_name,
        cadence_seconds,
    )
    sigma_frac_ppm_full_transit = noise_sigma["cache_sigma_ppm"]

    system, system_name, _ = _build_system(
        resolved_pc,
        cadence_seconds=cadence_seconds,
    )
    if system_name != pl_name:
        raise ValueError(f"target name changed while building system: {pl_name!r} != {system_name!r}")

    sigma_frac_ppm_per_cadence = noise_sigma["injected_sigma_ppm"]
    sigma_flux_per_cadence = noise_sigma["injected_sigma_flux"]
    noise_model_kmag = noise_sigma["kmag"]
    sigma_frac_ppm_one_minute = noise_sigma["sigma_ppm_one_minute"]
    base_seed = int(NOISE_CONFIG["random_seed"])

    fs_model, theta_model_deg = _model_grid_axes()
    if fs_model.size == 0 or not np.isclose(fs_model[0], 0.0):
        raise ValueError(
            "MODEL_GRID_CONFIG must include f=0 as its first value for the spherical null"
        )
    model_fluxes = _precompute_model_library(system, fs_model, theta_model_deg)

    n_null_realizations = int(DETECTION_CONFIG["n_null_realizations"])
    if n_null_realizations <= 0:
        raise ValueError(f"n_null_realizations must be positive: {n_null_realizations}")
    null_seed = base_seed + int(DETECTION_CONFIG["null_seed_offset"])
    print(
        f"[null calibration] {n_null_realizations} spherical realizations, "
        f"target false-alarm probability="
        f"{float(DETECTION_CONFIG['false_alarm_probability']):.2%}"
    )
    null_standard_noise = np.random.default_rng(null_seed).normal(
        size=(n_null_realizations, model_fluxes.shape[1])
    )
    null_calibration = _calibrate_null_detection_threshold(
        model_fluxes,
        null_standard_noise,
        sigma_flux_per_cadence,
    )
    null_calibration["null_random_seed"] = null_seed
    null_calibration["pl_name"] = pl_name
    null_calibration["row_index"] = row_index
    null_calibration["sigma_frac_ppm_full_transit"] = sigma_frac_ppm_full_transit
    null_calibration["noise_model_kmag"] = noise_model_kmag
    null_calibration["sigma_frac_ppm_one_minute"] = sigma_frac_ppm_one_minute
    null_calibration["sigma_frac_ppm_per_cadence"] = sigma_frac_ppm_per_cadence
    print(
        "[null calibration] "
        f"delta_chi2_critical={float(null_calibration['delta_chi2_threshold']):.4f}, "
        f"empirical false-alarm probability="
        f"{float(null_calibration['false_alarm_probability_empirical']):.2%}"
    )

    adaptive_cfg = ADAPTIVE_COMPLETENESS_CONFIG
    adaptive_enabled = bool(adaptive_cfg["enabled"])
    pilot_n_realizations = (
        int(adaptive_cfg["pilot_n_noise_realizations"])
        if adaptive_enabled
        else int(BATCH_RUN_CONFIG["n_noise_realizations"])
    )
    final_n_realizations = (
        int(adaptive_cfg["final_n_noise_realizations"])
        if adaptive_enabled
        else int(BATCH_RUN_CONFIG["n_noise_realizations"])
    )
    if pilot_n_realizations <= 0 or final_n_realizations <= 0:
        raise ValueError(
            "pilot and final noise realization counts must be positive: "
            f"{pilot_n_realizations}, {final_n_realizations}"
        )
    if pilot_n_realizations > final_n_realizations:
        raise ValueError(
            "pilot_n_noise_realizations cannot exceed final_n_noise_realizations: "
            f"{pilot_n_realizations} > {final_n_realizations}"
        )
    standard_noise = np.stack(
        [
            np.random.default_rng(base_seed + realization_index).normal(size=model_fluxes.shape[1])
            for realization_index in range(final_n_realizations)
        ],
        axis=0,
    )

    f_grid, theta_inj_grid = _injection_grid()
    coarse_pairs: list[tuple[float, float]] = [
        (float(f), float(th))
        for th in theta_inj_grid
        for f in f_grid
    ]

    br = BATCH_RUN_CONFIG
    max_n = br.get("max_injections")
    if max_n is not None:
        coarse_pairs = coarse_pairs[: int(max_n)]

    def evaluate_pairs(
        pairs: list[tuple[float, float]],
        noise_count: int,
        sampling_stage: str,
        is_adaptive_refinement: bool,
    ) -> dict[tuple[float, float], list[dict[str, Any]]]:
        evaluated: dict[tuple[float, float], list[dict[str, Any]]] = {}
        iterator: Any = enumerate(pairs)
        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=len(pairs),
                desc=f"{pl_name} {sampling_stage} x {noise_count} noises",
                unit="injection",
            )
        for temporary_index, (f_inj, theta_inj_deg) in iterator:
            injection_results = _run_injection_realizations(
                system,
                model_fluxes,
                fs_model,
                theta_model_deg,
                standard_noise[:noise_count],
                f_inj,
                theta_inj_deg,
                sigma_flux_per_cadence,
                base_seed,
                float(null_calibration["delta_chi2_threshold"]),
            )
            for result in injection_results:
                result["pl_name"] = pl_name
                result["row_index"] = row_index
                result["noise_cache_target"] = noise_row["target"]
                result["noise_cache_row_index"] = int(noise_row["row_index"])
                result["noise_cache_nint"] = nint
                result["model_cadence_seconds"] = cadence_seconds
                result["sigma_flux_full_transit"] = sigma_flux_full_transit
                result["sigma_frac_ppm_full_transit"] = sigma_frac_ppm_full_transit
                result["noise_model_kmag"] = noise_model_kmag
                result["sigma_frac_ppm_one_minute"] = sigma_frac_ppm_one_minute
                result["sigma_frac_ppm_per_cadence"] = sigma_frac_ppm_per_cadence
                result["sampling_stage"] = sampling_stage
                result["is_adaptive_refinement"] = bool(
                    is_adaptive_refinement
                )
                result["injection_index"] = int(temporary_index)
            evaluated[_injection_pair_key(f_inj, theta_inj_deg)] = injection_results
        return evaluated

    if adaptive_enabled:
        print(
            "[adaptive completeness] "
            f"pilot grid={len(coarse_pairs)} points, "
            f"theta step={float(INJECTION_GRID_CONFIG['theta_inj_deg_step']):g} deg, "
            f"coarse f step={float(INJECTION_GRID_CONFIG['f_step']):g}, "
            f"pilot noises={pilot_n_realizations}"
        )
        pilot_by_pair = evaluate_pairs(
            coarse_pairs,
            pilot_n_realizations,
            "pilot_coarse",
            False,
        )
        pilot_realizations = [
            result
            for pair_results in pilot_by_pair.values()
            for result in pair_results
        ]
        pilot_summaries = _aggregate_realizations(pilot_realizations)
        selected_coarse, fine_pairs, refinement_diagnostics = (
            _adaptive_refinement_plan(pilot_summaries)
        )
        selected_coarse_pairs = [
            pair
            for pair in coarse_pairs
            if _injection_pair_key(*pair) in selected_coarse
        ]
        print(
            "[adaptive completeness] "
            f"marked intervals={refinement_diagnostics['marked_intervals']}, "
            f"coarse points promoted to {final_n_realizations} noises="
            f"{len(selected_coarse_pairs)}, "
            f"new f={float(adaptive_cfg['refined_f_step']):g} points="
            f"{len(fine_pairs)}"
        )
        final_by_pair = dict(pilot_by_pair)
        if selected_coarse_pairs:
            final_by_pair.update(
                evaluate_pairs(
                    selected_coarse_pairs,
                    final_n_realizations,
                    "final_coarse_transition",
                    True,
                )
            )
        if fine_pairs:
            final_by_pair.update(
                evaluate_pairs(
                    fine_pairs,
                    final_n_realizations,
                    "final_fine",
                    True,
                )
            )
    else:
        final_by_pair = evaluate_pairs(
            coarse_pairs,
            final_n_realizations,
            "uniform_grid",
            False,
        )

    realizations: list[dict[str, Any]] = []
    sorted_pairs = sorted(final_by_pair, key=lambda pair: (pair[1], pair[0]))
    for injection_index, pair in enumerate(sorted_pairs):
        pair_results = final_by_pair[pair]
        for result in pair_results:
            result["injection_index"] = int(injection_index)
        realizations.extend(pair_results)

    null_calibration["adaptive_completeness_enabled"] = adaptive_enabled
    null_calibration["pilot_n_noise_realizations"] = pilot_n_realizations
    null_calibration["final_n_noise_realizations"] = final_n_realizations
    null_calibration["n_coarse_injection_points"] = len(coarse_pairs)
    null_calibration["n_final_injection_points"] = len(sorted_pairs)
    return realizations, _aggregate_realizations(realizations), null_calibration


def _save_detection_figure(
    summaries: list[dict[str, Any]],
    detection_limits: list[dict[str, Any]],
    figure_path: Path | str | None = None,
) -> Path:
    matplotlib = __import__("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not summaries:
        raise ValueError("cannot plot empty detection summary")

    x = np.array([row["f_inj"] for row in summaries], dtype=np.float64)
    y = np.array([row["theta_inj_deg"] for row in summaries], dtype=np.float64)
    probability = np.array(
        [row["detection_probability"] for row in summaries],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(9.0, 6.0), dpi=120)

    points = ax.scatter(
        x,
        y,
        c=probability,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        s=55,
        marker="o",
        edgecolors="black",
        linewidths=0.3,
        zorder=3,
    )

    probability_levels = sorted(
        {float(row["probability_level"]) for row in detection_limits}
    )
    level_styles = {
        0.50: ("tab:blue", "-"),
        0.90: ("black", "--"),
    }
    has_curve_or_marker = False
    for level in probability_levels:
        color, linestyle = level_styles.get(level, ("tab:purple", ":"))
        level_rows = sorted(
            (
                row
                for row in detection_limits
                if np.isclose(float(row["probability_level"]), level)
            ),
            key=lambda row: float(row["theta_inj_deg"]),
        )
        band_theta = np.array(
            [float(row["theta_inj_deg"]) for row in level_rows],
            dtype=np.float64,
        )
        band_low = np.array(
            [float(row["f_detection_limit_ci95_low"]) for row in level_rows],
            dtype=np.float64,
        )
        band_high = np.array(
            [float(row["f_detection_limit_ci95_high"]) for row in level_rows],
            dtype=np.float64,
        )
        finite_band = np.isfinite(band_low) & np.isfinite(band_high)
        if np.count_nonzero(finite_band) >= 2:
            has_curve_or_marker = True
            ax.fill_betweenx(
                band_theta,
                band_low,
                band_high,
                where=finite_band,
                color=color,
                alpha=0.30,
                linewidth=0.0,
                label=fr"$f_{{{100.0 * level:.0f}}}$ 95% CI",
                zorder=4,
            )
        valid_limits = [
            row
            for row in level_rows
            if np.isfinite(float(row["f_detection_limit"]))
        ]
        if valid_limits:
            has_curve_or_marker = True
            ax.plot(
                [row["f_detection_limit"] for row in valid_limits],
                [row["theta_inj_deg"] for row in valid_limits],
                color=color,
                linewidth=1.2,
                linestyle=linestyle,
                marker="o",
                markersize=4.0,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.8,
                label=fr"$f_{{{100.0 * level:.0f}}}(\theta)$",
                zorder=5,
            )
        unresolved = [
            row for row in level_rows if row["limit_status"] == "not_reached"
        ]
        if unresolved:
            has_curve_or_marker = True
            ax.scatter(
                [INJECTION_GRID_CONFIG["f_max"]] * len(unresolved),
                [row["theta_inj_deg"] for row in unresolved],
                marker=">",
                s=45,
                facecolors="white",
                edgecolor=color,
                linewidth=0.8,
                label=fr"$f_{{{100.0 * level:.0f}}}$ above grid",
                zorder=5,
            )

    title = summaries[0]["pl_name"]
    realization_counts = [
        int(row["n_noise_realizations"]) for row in summaries
    ]
    ax.set_title(
        f"{title} — adaptive completeness map; "
        f"{min(realization_counts)}–{max(realization_counts)} noises/point; "
        fr"$\Delta\chi^2_{{\rm crit}}={summaries[0]['detection_delta_chi2_threshold']:.2f}$"
    )
    ax.set_xlabel(r"injected $f_{\mathrm{inj}}$")
    ax.set_ylabel(r"injected $\theta_{\mathrm{inj}}$ (deg)")
    ax.set_xlim(
        float(np.min(x)) - 0.003,
        INJECTION_GRID_CONFIG["f_max"] + 0.005,
    )
    ax.set_ylim(
        INJECTION_GRID_CONFIG["theta_inj_deg_min"] - 5.0,
        INJECTION_GRID_CONFIG["theta_inj_deg_max"] + 5.0,
    )
    ax.grid(True, alpha=0.3)
    if has_curve_or_marker:
        ax.legend(loc="best", fontsize=9)
    fig.colorbar(points, ax=ax, label=r"detection probability $P_{\rm det}$", pad=0.02)
    fig.tight_layout()
    out = (
        Path(figure_path)
        if figure_path is not None
        else _configured_output_path("summary_figure_filename", title)
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
    "chi2_spherical",
    "delta_chi2_spherical",
    "detection_delta_chi2_threshold",
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
    "noise_model_kmag",
    "sigma_frac_ppm_one_minute",
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
    "is_detected",
    "truth_better_than_grid",
    "f_ci95_touches_search_boundary",
    "is_closest_recovery",
    "is_adaptive_refinement",
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
    out["sampling_stage"] = np.array(
        [r["sampling_stage"] for r in results],
        dtype=object,
    )
    out["pl_name"] = results[0]["pl_name"] if results else ""
    if results and "row_index" in results[0]:
        out["row_index"] = int(results[0]["row_index"])
        out["noise_cache_target"] = results[0]["noise_cache_target"]
        out["noise_cache_row_index"] = int(results[0]["noise_cache_row_index"])
    path = (
        Path(npz_path)
        if npz_path is not None
        else _configured_output_path(
            "results_npz_filename",
            results[0]["pl_name"] if results else TARGET_NAME,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return path


def _save_realizations_csv(
    results: list[dict[str, Any]],
    csv_path: Path | str | None = None,
) -> Path:
    """Save all injection-by-noise realization results."""
    path = (
        Path(csv_path)
        if csv_path is not None
        else _configured_output_path(
            "realizations_csv_filename",
            results[0]["pl_name"] if results else TARGET_NAME,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_keys = [
        "pl_name",
        "row_index",
        "noise_cache_target",
        "noise_cache_row_index",
        "coverage",
        "sampling_stage",
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
    "sigma_frac_ppm_full_transit",
    "noise_model_kmag",
    "sigma_frac_ppm_one_minute",
    "sigma_frac_ppm_per_cadence",
    "sampling_stage",
    "is_adaptive_refinement",
    "injection_index",
    "f_inj",
    "theta_inj_deg",
    "n_noise_realizations",
    "detected_count",
    "detection_probability",
    "detection_probability_ci95_low",
    "detection_probability_ci95_high",
    "detection_probability_threshold",
    "detection_delta_chi2_threshold",
    "delta_chi2_spherical_median",
    "delta_chi2_spherical_p16",
    "delta_chi2_spherical_p84",
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
]


def _save_summary_csv(
    summaries: list[dict[str, Any]],
    csv_path: Path | str | None = None,
) -> Path:
    """Save one aggregated row per injected ``(f, theta)``."""
    path = (
        Path(csv_path)
        if csv_path is not None
        else _configured_output_path(
            "summary_csv_filename",
            summaries[0]["pl_name"] if summaries else TARGET_NAME,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_KEYS)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary[key] for key in SUMMARY_KEYS})
    return path


DETECTION_LIMIT_KEYS = [
    "pl_name",
    "theta_inj_deg",
    "probability_level",
    "f_detection_limit",
    "f_detection_limit_ci95_low",
    "f_detection_limit_ci95_high",
    "f_detection_limit_ci95_low_status",
    "f_detection_limit_ci95_high_status",
    "confidence_level",
    "confidence_method",
    "detection_probability_at_limit",
    "previous_f_inj",
    "previous_detection_probability",
    "next_f_inj",
    "next_detection_probability",
    "limit_status",
]


def _save_detection_limits_csv(
    detection_limits: list[dict[str, Any]],
    csv_path: Path | str | None = None,
) -> Path:
    """Save the isotonic/interpolated ``f50(theta)`` and ``f90(theta)`` curves."""
    path = (
        Path(csv_path)
        if csv_path is not None
        else _configured_output_path(
            "detection_limits_csv_filename",
            detection_limits[0]["pl_name"] if detection_limits else TARGET_NAME,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETECTION_LIMIT_KEYS)
        writer.writeheader()
        for row in detection_limits:
            writer.writerow({key: row[key] for key in DETECTION_LIMIT_KEYS})
    return path


def _save_null_calibration_npz(
    null_calibration: dict[str, Any],
    npz_path: Path | str | None = None,
) -> Path:
    """Save the empirical spherical-null distribution and calibrated threshold."""
    path = (
        Path(npz_path)
        if npz_path is not None
        else _configured_output_path(
            "null_calibration_npz_filename",
            str(null_calibration["pl_name"]),
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        delta_chi2_null=np.asarray(
            null_calibration["delta_chi2_null"],
            dtype=np.float64,
        ),
        delta_chi2_threshold=float(null_calibration["delta_chi2_threshold"]),
        false_alarm_probability_target=float(
            null_calibration["false_alarm_probability_target"]
        ),
        false_alarm_probability_empirical=float(
            null_calibration["false_alarm_probability_empirical"]
        ),
        n_null_realizations=int(null_calibration["n_null_realizations"]),
        null_random_seed=int(null_calibration["null_random_seed"]),
        pl_name=str(null_calibration["pl_name"]),
        row_index=int(null_calibration["row_index"]),
        sigma_frac_ppm_full_transit=float(
            null_calibration["sigma_frac_ppm_full_transit"]
        ),
        noise_model_kmag=float(null_calibration["noise_model_kmag"]),
        sigma_frac_ppm_one_minute=float(
            null_calibration["sigma_frac_ppm_one_minute"]
        ),
        sigma_frac_ppm_per_cadence=float(
            null_calibration["sigma_frac_ppm_per_cadence"]
        ),
        adaptive_completeness_enabled=bool(
            null_calibration["adaptive_completeness_enabled"]
        ),
        pilot_n_noise_realizations=int(
            null_calibration["pilot_n_noise_realizations"]
        ),
        final_n_noise_realizations=int(
            null_calibration["final_n_noise_realizations"]
        ),
        n_coarse_injection_points=int(
            null_calibration["n_coarse_injection_points"]
        ),
        n_final_injection_points=int(
            null_calibration["n_final_injection_points"]
        ),
    )
    return path


def main() -> None:
    realizations, summaries, null_calibration = run_batch(None)
    detection_limits = _derive_detection_limits(summaries)
    print("n_injection_grid_points:", len(summaries))
    print("n_realization_results:", len(realizations))
    if not realizations:
        return
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
        "Wang & Winn (2026) empirical one-minute noise =",
        realizations[0]["sigma_frac_ppm_one_minute"],
        "ppm at K =",
        realizations[0]["noise_model_kmag"],
    )
    print(
        "injected sigma_flux per light-curve point =",
        realizations[0]["sigma_flux_per_cadence"],
        ", sigma_frac_ppm =",
        realizations[0]["sigma_frac_ppm_per_cadence"],
        ", nint =",
        realizations[0]["noise_cache_nint"],
        ", cadence_s =",
        realizations[0]["model_cadence_seconds"],
    )
    print(
        "spherical-null calibration:",
        f"n={null_calibration['n_null_realizations']},",
        f"target false-alarm probability="
        f"{null_calibration['false_alarm_probability_target']:.4f},",
        f"empirical false-alarm probability="
        f"{null_calibration['false_alarm_probability_empirical']:.4f},",
        f"delta_chi2 threshold={null_calibration['delta_chi2_threshold']:.6g}",
    )
    print(
        "adaptive sampling:",
        f"coarse points={null_calibration['n_coarse_injection_points']},",
        f"final points={null_calibration['n_final_injection_points']},",
        f"noises/point={null_calibration['pilot_n_noise_realizations']}"
        f"–{null_calibration['final_n_noise_realizations']}",
    )
    for probability_level in ADAPTIVE_COMPLETENESS_CONFIG["probability_levels"]:
        print(f"f{100.0 * float(probability_level):.0f}(theta):")
        for row in detection_limits:
            if not np.isclose(
                float(row["probability_level"]),
                float(probability_level),
            ):
                continue
            if np.isfinite(float(row["f_detection_limit"])):
                ci_low = float(row["f_detection_limit_ci95_low"])
                ci_high = float(row["f_detection_limit_ci95_high"])
                if np.isfinite(ci_low) and np.isfinite(ci_high):
                    interval_text = f", 95% CI=[{ci_low:.4f}, {ci_high:.4f}]"
                elif np.isfinite(ci_low):
                    interval_text = (
                        f", 95% CI lower={ci_low:.4f}, upper above grid"
                    )
                else:
                    interval_text = ", 95% CI unresolved on current grid"
                print(
                    f"  theta={row['theta_inj_deg']:.1f} deg: "
                    f"f={row['f_detection_limit']:.4f} "
                    f"({row['limit_status']}){interval_text}"
                )
            else:
                print(
                    f"  theta={row['theta_inj_deg']:.1f} deg: "
                    f"not reached by f={INJECTION_GRID_CONFIG['f_max']:.4f}"
                )

    fig_path = _save_detection_figure(summaries, detection_limits)
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
    if BATCH_RUN_CONFIG.get("save_detection_limits_csv"):
        p = _save_detection_limits_csv(detection_limits)
        print("saved detection limits csv:", p)
    if BATCH_RUN_CONFIG.get("save_null_calibration_npz"):
        p = _save_null_calibration_npz(null_calibration)
        print("saved null calibration npz:", p)


if __name__ == "__main__":
    main()
