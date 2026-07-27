"""偏心轨道的 ingress/egress 时间采样。

本版本使用与 ``squishyplanet.OblateSystem`` 相同的 Kepler 与 sky-position 引擎，结合
``e`` 和近星点幅角 ``omega`` 数值寻找 ``rho=1+k`` 与 ``rho=1-k`` 的四个接触点。
若输入缺失或几何上没有四个接触点，则使用 ``pl_trandur`` 构造覆盖整场凌星的安全时间窗；
不再因 ``e>0.05`` 回退到固定的 ``t0 +/- 0.08 day``。
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np

SECONDS_PER_DAY = 86400.0

DEFAULT_TIME_SAMPLING: dict[str, Any] = {
    "mode": "ingress_egress",
    "cadence_seconds": 60.0,
    "edge_pad_cadences": 3,
    "n_ie_segment_max": 2500,
    "exact_search_n": 4097,
    "exact_search_duration_scale": 1.5,
    "uniform_fallback_half_width_days": 0.08,
}


def _linear_crossings(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Linearly interpolate every zero crossing in a densely sampled series."""
    roots: list[float] = []
    for index in np.flatnonzero(values[:-1] * values[1:] <= 0.0):
        t_left = float(times[index])
        t_right = float(times[index + 1])
        y_left = float(values[index])
        y_right = float(values[index + 1])
        if y_left == 0.0:
            root = t_left
        elif y_right == 0.0:
            root = t_right
        elif y_right == y_left:
            root = 0.5 * (t_left + t_right)
        else:
            root = t_left - y_left * (t_right - t_left) / (y_right - y_left)
        if not roots or not np.isclose(root, roots[-1], rtol=0.0, atol=1e-12):
            roots.append(root)
    return np.asarray(roots, dtype=np.float64)


def _bracketing_roots(roots: np.ndarray, t0_days: float) -> tuple[float, float] | None:
    left = roots[roots < float(t0_days)]
    right = roots[roots > float(t0_days)]
    if left.size == 0 or right.size == 0:
        return None
    return float(np.max(left)), float(np.min(right))


def contact_times_eccentric(
    t0_days: float,
    pl_ratdor: float,
    period_days: float,
    k_rp_over_rs: float,
    inc_deg: float,
    *,
    ecc: float,
    omega_deg: float | None,
    transit_duration_hours: float | None,
    time_sampling: dict[str, Any] | None = None,
) -> tuple[float, float, float, float] | None:
    """Find four contacts from the exact eccentric sky-projected separation."""
    cfg = {**DEFAULT_TIME_SAMPLING, **(time_sampling or {})}
    e = float(ecc)
    if not np.isfinite(e) or e < 0.0 or e >= 1.0:
        return None
    if omega_deg is None or not np.isfinite(float(omega_deg)):
        if np.isclose(e, 0.0):
            omega_deg = 0.0
        else:
            return None

    cadence_days = float(cfg["cadence_seconds"]) / SECONDS_PER_DAY
    duration_days = None
    if transit_duration_hours is not None:
        candidate = float(transit_duration_hours) / 24.0
        if np.isfinite(candidate) and candidate > 0.0:
            duration_days = candidate
    fallback_half = float(cfg["uniform_fallback_half_width_days"])
    if duration_days is None:
        half_window = fallback_half
    else:
        half_window = max(
            fallback_half,
            0.5 * duration_days * float(cfg["exact_search_duration_scale"]),
        )
    half_window = max(half_window, 10.0 * cadence_days)
    n_search = max(513, int(cfg["exact_search_n"]))
    if n_search % 2 == 0:
        n_search += 1
    search_times = np.linspace(
        float(t0_days) - half_window,
        float(t0_days) + half_window,
        n_search,
        dtype=np.float64,
    )

    import jax.numpy as jnp
    from squishyplanet.engine.kepler import kepler, skypos, t0_to_t_peri

    inc_rad = float(np.deg2rad(inc_deg))
    omega_rad = float(np.deg2rad(float(omega_deg)))
    t_peri = float(
        np.asarray(
            t0_to_t_peri(
                e=e,
                i=inc_rad,
                omega=omega_rad,
                period=float(period_days),
                t0=float(t0_days),
            )
        )
    )
    mean_anomaly = 2.0 * np.pi * (search_times - t_peri) / float(period_days)
    true_anomaly = kepler(jnp.asarray(mean_anomaly), e)
    positions = np.asarray(
        skypos(
            a=float(pl_ratdor),
            e=e,
            f=true_anomaly,
            Omega=0.0,
            i=inc_rad,
            omega=omega_rad,
        )
    )
    rho = np.hypot(positions[0], positions[1])
    k = float(k_rp_over_rs)
    outer = _bracketing_roots(_linear_crossings(search_times, rho - (1.0 + k)), t0_days)
    inner = _bracketing_roots(_linear_crossings(search_times, rho - (1.0 - k)), t0_days)
    if outer is None or inner is None:
        return None
    t1, t4 = outer
    t2, t3 = inner
    if not (t1 <= t2 <= float(t0_days) <= t3 <= t4):
        return None
    return t1, t2, t3, t4


def _segment_grid(t_lo: float, t_hi: float, dt_days: float) -> np.ndarray:
    if t_hi < t_lo:
        t_lo, t_hi = t_hi, t_lo
    if dt_days <= 0.0:
        return np.array([t_lo, t_hi], dtype=np.float64)
    n = int(np.floor((t_hi - t_lo) / dt_days)) + 1
    if n < 2:
        return np.array([t_lo, t_hi], dtype=np.float64)
    # 包含右端点（浮点容差）
    return np.arange(t_lo, t_hi + 0.5 * dt_days, dt_days, dtype=np.float64)


def deterministic_subsample(times: np.ndarray, max_n: int) -> np.ndarray:
    if times.size <= max_n:
        return times
    idx = np.unique(np.round(np.linspace(0, times.size - 1, max_n)).astype(int))
    return times[idx]


def merge_ie_arrays(
    ingress: np.ndarray,
    egress: np.ndarray,
) -> np.ndarray:
    out = np.unique(np.concatenate([ingress, egress]))
    out.sort()
    return out


def build_times_full_transit_fallback(
    t0_days: float,
    transit_duration_hours: float | None,
    time_sampling: dict[str, Any],
) -> np.ndarray:
    """Build a cadence-matched full-transit window when exact contacts are unavailable."""
    cfg = {**DEFAULT_TIME_SAMPLING, **time_sampling}
    cadence_days = float(cfg["cadence_seconds"]) / SECONDS_PER_DAY
    pad_days = max(0, int(cfg["edge_pad_cadences"])) * cadence_days
    duration_days = None
    if transit_duration_hours is not None:
        candidate = float(transit_duration_hours) / 24.0
        if np.isfinite(candidate) and candidate > 0.0:
            duration_days = candidate
    half = (
        0.5 * duration_days
        if duration_days is not None
        else float(cfg["uniform_fallback_half_width_days"])
    )
    return _segment_grid(float(t0_days) - half - pad_days, float(t0_days) + half + pad_days, cadence_days)


def build_times_ingress_egress(
    t0_days: float,
    pl_ratdor: float,
    period_days: float,
    k_rp_over_rs: float,
    inc_deg: float,
    ecc: float,
    omega_deg: float | None,
    transit_duration_hours: float | None,
    time_sampling: dict[str, Any],
) -> np.ndarray:
    """生成仅 ingress∪egress 上的时刻（日），含端点外延与段内子采样。"""
    cfg = {**DEFAULT_TIME_SAMPLING, **time_sampling}
    cadence_days = float(cfg["cadence_seconds"]) / SECONDS_PER_DAY
    pad_n = max(0, int(cfg["edge_pad_cadences"]))
    pad_days = float(pad_n) * cadence_days
    max_seg = int(cfg["n_ie_segment_max"])

    contacts = contact_times_eccentric(
        t0_days,
        pl_ratdor,
        period_days,
        k_rp_over_rs,
        inc_deg,
        ecc=float(ecc),
        omega_deg=omega_deg,
        transit_duration_hours=transit_duration_hours,
        time_sampling=cfg,
    )
    if contacts is None:
        warnings.warn(
            "exact eccentric contacts unavailable; using a cadence-matched full-transit window",
            stacklevel=2,
        )
        return build_times_full_transit_fallback(t0_days, transit_duration_hours, cfg)

    t1, t2, t3, t4 = contacts
    ing = _segment_grid(t1 - pad_days, t2 + pad_days, cadence_days)
    egr = _segment_grid(t3 - pad_days, t4 + pad_days, cadence_days)
    ing = deterministic_subsample(ing, max_seg)
    egr = deterministic_subsample(egr, max_seg)
    return merge_ie_arrays(ing, egr)


def effective_time_sampling_mode(time_sampling: dict[str, Any]) -> str:
    """``OBLATE_TIME_SAMPLING_MODE`` 环境变量可覆盖配置中的 ``mode``。"""
    env = os.environ.get("OBLATE_TIME_SAMPLING_MODE", "").strip()
    if env:
        return env
    cfg = {**DEFAULT_TIME_SAMPLING, **time_sampling}
    return str(cfg.get("mode", "ingress_egress"))


def build_lightcurve_time_array_days(
    *,
    t0_days: float,
    period_days: float,
    pl_ratdor: float,
    k_rp_over_rs: float,
    inc_deg: float,
    ecc: float,
    time_half_width_days: float,
    n_time_uniform: int,
    time_sampling: dict[str, Any] | None,
    omega_deg: float | None = None,
    transit_duration_hours: float | None = None,
) -> np.ndarray:
    """
    按 ``time_sampling``（及环境变量）构造 ``times`` [d]，与 ``OblateSystem`` 的 ``t0`` 单位一致。
    """
    ts = {**DEFAULT_TIME_SAMPLING, **(time_sampling or {})}
    mode = effective_time_sampling_mode(ts)
    if mode == "uniform_symmetric":
        half = float(time_half_width_days)
        return np.linspace(
            float(t0_days) - half,
            float(t0_days) + half,
            int(n_time_uniform),
            dtype=np.float64,
        )
    return build_times_ingress_egress(
        float(t0_days),
        float(pl_ratdor),
        float(period_days),
        float(k_rp_over_rs),
        float(inc_deg),
        float(ecc),
        omega_deg,
        transit_duration_hours,
        ts,
    )
