"""凌星 ingress/egress 时间构造（无 squishy 依赖）。"""

import numpy as np

from oblateness.transit_ie_sampling import (
    DEFAULT_TIME_SAMPLING,
    build_lightcurve_time_array_days,
    build_times_ingress_egress,
    contact_times_circular,
    deterministic_subsample,
)


def test_contact_times_ordering():
    t0 = 2460000.0
    c = contact_times_circular(
        t0,
        pl_ratdor=8.0,
        period_days=3.0,
        k_rp_over_rs=0.1,
        inc_deg=88.0,
        ecc=0.0,
        ecc_max_circular=0.05,
    )
    assert c is not None
    t1, t2, t3, t4 = c
    assert t1 < t2 <= t3 < t4
    assert np.isclose((t2 - t1), (t4 - t3))


def test_ie_times_union_has_gap_at_flat_bottom():
    """ingress 与 egress 两段之间应留空（不同时含平台段中点）。"""
    t0 = 2460000.0
    ts = build_times_ingress_egress(
        t0,
        pl_ratdor=8.0,
        period_days=3.0,
        k_rp_over_rs=0.1,
        inc_deg=88.0,
        ecc=0.0,
        time_sampling=DEFAULT_TIME_SAMPLING,
    )
    mid = 0.5 * (ts.max() + ts.min())
    assert (ts < mid).any() and (ts > mid).any()
    # 中点附近应存在间隙（无点），除非两段极短导致数值重叠
    center_window = (ts > t0 - 1e-7) & (ts < t0 + 1e-7)
    assert center_window.sum() <= 2


def test_deterministic_subsample_cap():
    t = np.arange(10000, dtype=np.float64)
    s = deterministic_subsample(t, 100)
    assert s.size <= 100
    assert s[0] == t[0] and s[-1] == t[-1]


def test_high_eccentricity_fallback_is_uniform():
    t = build_lightcurve_time_array_days(
        t0_days=2460000.0,
        period_days=3.0,
        pl_ratdor=8.0,
        k_rp_over_rs=0.1,
        inc_deg=88.0,
        ecc=0.2,
        time_half_width_days=0.05,
        n_time_uniform=101,
        time_sampling={"ecc_max_circular": 0.05},
    )
    assert t.size == int(DEFAULT_TIME_SAMPLING["uniform_fallback_n_time"])


def test_uniform_mode_env_like():
    t = build_lightcurve_time_array_days(
        t0_days=2460000.0,
        period_days=3.0,
        pl_ratdor=8.0,
        k_rp_over_rs=0.1,
        inc_deg=88.0,
        ecc=0.0,
        time_half_width_days=0.1,
        n_time_uniform=50,
        time_sampling={"mode": "uniform_symmetric"},
    )
    assert t.size == 50
    assert np.isclose(t[0], 2460000.0 - 0.1)
    assert np.isclose(t[-1], 2460000.0 + 0.1)
