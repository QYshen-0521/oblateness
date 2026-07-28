"""
凌星 **ingress / egress** 时间采样（见 ``worklog.md``「时间采样——60 s… I/E」）。

**假设**：圆轨道、小天体；投影速度
``v_sky = 2\\pi (a/R_*)/P \\cdot \\sin i``（单位 R_\\*/day）；
碰撞参数 ``b = (a/R_*)\\cos i``（NASA ``pl_orbincl`` 为度，90° 为边缘朝天）。

接触时刻（行星-星心投影距）：外侧 ``d_1=\\sqrt{(1+k)^2-b^2}``、内侧 ``d_2=\\sqrt{(1-k)^2-b^2}``（``k=R_p/R_*``）。
ingress 为 ``t_0-d_1/v`` 至 ``t_0-d_2/v``，egress 为 ``t_0+d_2/v`` 至 ``t_0+d_1/v``。
离心率大于 ``ecc_max_circular`` 或几何不合法时回退到均匀对称窗。

``pl_trandur`` 等档案列与「一/四接触」的全宽定义可能不一致，故**未**用 trandur 定界；若日后对照
squishyplanet 几何需改公式，仅改本模块。
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

SECONDS_PER_DAY = 86400.0

DEFAULT_TIME_SAMPLING: dict[str, Any] = {
    "mode": "ingress_egress",
    "cadence_seconds": 60.0,
    "edge_pad_cadences": 3,
    "n_ie_segment_max": 2500,
    "ecc_max_circular": 0.05,
    "uniform_fallback_half_width_days": 0.08,
    "uniform_fallback_n_time": 500,
}


def impact_parameter(pl_ratdor: float, inc_rad: float) -> float:
    return float(pl_ratdor * np.cos(inc_rad))


def sky_plane_speed_rs_per_day(pl_ratdor: float, period_days: float, inc_rad: float) -> float:
    return float(2.0 * np.pi * pl_ratdor / period_days * np.sin(inc_rad))


def contact_times_circular(
    t0_days: float,
    pl_ratdor: float,
    period_days: float,
    k_rp_over_rs: float,
    inc_deg: float,
    *,
    ecc: float,
    ecc_max_circular: float,
) -> tuple[float, float, float, float] | None:
    """
    圆轨道下四接触时刻（日）。若 ``ecc > ecc_max_circular``、``v<=0``、或几何上无完整 ingress/egress
    （如 ``b >= 1-k``），返回 ``None``。
    """
    if ecc > ecc_max_circular:
        return None
    inc = float(np.deg2rad(inc_deg))
    b = impact_parameter(pl_ratdor, inc)
    v = sky_plane_speed_rs_per_day(pl_ratdor, period_days, inc)
    if v <= 0.0:
        return None
    k = float(k_rp_over_rs)
    if (1.0 + k) ** 2 - b * b <= 0.0:
        return None
    if (1.0 - k) ** 2 - b * b <= 0.0:
        return None
    d1 = float(np.sqrt((1.0 + k) ** 2 - b * b))
    d2 = float(np.sqrt((1.0 - k) ** 2 - b * b))
    t1 = t0_days - d1 / v
    t2 = t0_days - d2 / v
    t3 = t0_days + d2 / v
    t4 = t0_days + d1 / v
    if not (t1 <= t2 <= t0_days <= t3 <= t4):
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


def build_times_ingress_egress(
    t0_days: float,
    pl_ratdor: float,
    period_days: float,
    k_rp_over_rs: float,
    inc_deg: float,
    ecc: float,
    time_sampling: dict[str, Any],
) -> np.ndarray:
    """生成仅 ingress∪egress 上的时刻（日），含端点外延与段内子采样。"""
    cfg = {**DEFAULT_TIME_SAMPLING, **time_sampling}
    cadence_days = float(cfg["cadence_seconds"]) / SECONDS_PER_DAY
    pad_n = max(0, int(cfg["edge_pad_cadences"]))
    pad_days = float(pad_n) * cadence_days
    max_seg = int(cfg["n_ie_segment_max"])

    contacts = contact_times_circular(
        t0_days,
        pl_ratdor,
        period_days,
        k_rp_over_rs,
        inc_deg,
        ecc=float(ecc),
        ecc_max_circular=float(cfg["ecc_max_circular"]),
    )
    if contacts is None:
        half = float(cfg["uniform_fallback_half_width_days"])
        nfb = int(cfg["uniform_fallback_n_time"])
        return np.linspace(t0_days - half, t0_days + half, nfb, dtype=np.float64)

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
        ts,
    )
