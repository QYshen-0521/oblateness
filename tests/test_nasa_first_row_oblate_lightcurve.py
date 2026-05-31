"""
使用 NASA `ps_tran_oblate_inputs_valid_*.csv` 中**一行**数据，调用 squishyplanet
`OblateSystem` 生成两条理论凌星光变：基准（球形/圆投影）与注入投影椭率 (f, theta)。

凌星两侧恒星边缘昏暗（LD）均由多项式系数描述；优先用 ExoTiC-LD（`exotic_ld`）按
`st_teff, st_logg, st_met` 与 **JWST** 仪器通带（默认 `JWST_NIRSpec_Prism`）生成 quadratic
的 :math:`u_1,u_2`，与 squishyplanet 所用 Agol 等 (2020) 基底一致。`exotic_ld_mode` 与
`exotic_ld_wavelength_range_angstrom` 须在 ExoTiC-LD 支持的通带范围内成对自洽（可改例如
`JWST_NIRCam_F444` 并调整波长窗）。

**可调参数**：仅修改下方 ``USER_CONFIG`` 字典（及必要时 ``USER_CONFIG['csv_path']``）。

参见黑板 `worklog.md` 中 OblateSystem ↔ (f,θ) ↔ NASA 列对照；
``parameterize_with_projected_ellipse=True`` 时 **必须** ``tidally_locked=False``
（squishyplanet API 约束）。
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pytest

from oblateness.transit_ie_sampling import build_lightcurve_time_array_days

# ---------------------------------------------------------------------------
# 所有可改参数集中在此处（与 ``noiseless_grid_chi2_inversion.PLANET_CONFIG`` 的 csv 约定一致）
# ---------------------------------------------------------------------------
USER_CONFIG: dict = {
    # 相对仓库根目录
    # "csv_path": "data/nasa_archive/ps_tran_oblate_inputs_valid_20260416.csv",  # 原 578 行 Tier A
    "csv_path": "data/nasa_archive/ps_tran_oblate_inputs_tier_b_derived_20260416.csv",  # Tier B 新源；或设 OBLATE_CSV_PATH
    # 表中第几行行星（0 = 第一个数据行，表头下一行）
    "row_index": 0,
    # 时间轴：以 pl_tranmid 为中心的半天窗口、采样点数
    "time_half_width_days": 0.08,
    "n_time": 500,
    # 投影椭球参数（与 worklog 中「投影 f、θ」一致；θ 用角度填入，代码内转弧度）
    # 基准曲线：projected_f=0（圆），theta 不影响形状
    "injected_projected_f": 0.12,
    "injected_projected_theta_deg": 40.0,
    # 极限昏暗：ExoTiC-LD 模型网格与 JWST 仪器模式（见 exotic_ld 文档中的 mode 列表）
    "ld_source": "exotic",  # "exotic" | "fixed"
    "exotic_ld_stellar_model": "mps1",  # 与 ExoTiC-LD 文档一致
    # 凌星/光谱常用：NIRSpec Prism；成像可改用 "JWST_NIRCam_F444" 等并收窄/平移波长窗
    "exotic_ld_mode": "JWST_NIRSpec_Prism",
    "exotic_ld_wavelength_range_angstrom": (20000.0, 30000.0),
    # 首次运行会下载网格到该目录（可写）
    "exotic_ld_data_path": "data/exotic_ld_data",
    # exotic 失败或未装依赖时使用（quadratic: I/I0 = 1 - u1(1-mu) - u2(1-mu)^2）
    "fixed_ld_u_coeffs": (0.45, 0.20),
    # Liborbital / 轨道
    "tidally_locked": False,  # 投影椭率模式下 API 要求 False
    "Omega_rad": float(np.pi),  # 文档默认
    "time_sampling": {
        "mode": "ingress_egress",
        "cadence_seconds": 60.0,
        "edge_pad_cadences": 3,
        "n_ie_segment_max": 2500,
        "ecc_max_circular": 0.05,
        "uniform_fallback_half_width_days": 0.08,
        "uniform_fallback_n_time": 500,
    },
    # 出图
    "save_figure": True,
    "figure_filename": "nasa_first_row_oblate_theory_compare.png",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_csv_row(path: Path, index: int) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if index < 0 or index >= len(rows):
        raise IndexError(f"row_index={index} out of range (n={len(rows)})")
    return rows[index]


def _f(name: str, row: dict[str, str]) -> float:
    v = row[name].strip()
    if not v:
        raise ValueError(f"missing column {name}")
    return float(v)


def _ld_coeffs_from_exotic(cfg: dict, teff: int, logg: float, m_h: float) -> np.ndarray:
    from exotic_ld import StellarLimbDarkening

    p = _repo_root() / cfg["exotic_ld_data_path"]
    p.mkdir(parents=True, exist_ok=True)
    sld = StellarLimbDarkening(
        M_H=m_h,
        Teff=int(teff),
        logg=float(logg),
        ld_model=cfg["exotic_ld_stellar_model"],
        ld_data_path=str(p),
        verbose=0,
    )
    wl = np.array(cfg["exotic_ld_wavelength_range_angstrom"], dtype=float)
    u1, u2 = sld.compute_quadratic_ld_coeffs(wavelength_range=wl, mode=cfg["exotic_ld_mode"])
    return np.array([u1, u2], dtype=float)


def _compute_ld_u(cfg: dict, teff: int, logg: float, m_h: float) -> np.ndarray:
    if cfg["ld_source"] == "fixed":
        return np.array(cfg["fixed_ld_u_coeffs"], dtype=float)
    try:
        return _ld_coeffs_from_exotic(cfg, teff, logg, m_h)
    except Exception as exc:  # noqa: BLE001 — 测试脚本需可在无网环境下回退
        warnings.warn(f"ExoTiC-LD failed ({exc}); using fixed_ld_u_coeffs.", stacklevel=2)
        return np.array(cfg["fixed_ld_u_coeffs"], dtype=float)


@pytest.mark.integration
def test_nasa_csv_first_row_theory_vs_injected_oblate_lightcurve() -> None:
    pytest.importorskip("squishyplanet")
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from squishyplanet import OblateSystem

    cfg = USER_CONFIG
    root = _repo_root()
    csv_path = root / cfg["csv_path"]
    if not csv_path.is_file():
        pytest.skip(f"missing CSV: {csv_path}")

    row = _load_csv_row(csv_path, cfg["row_index"])
    name = row["pl_name"].strip()

    period = _f("pl_orbper", row)
    a = _f("pl_ratdor", row)
    rp = _f("pl_ratror", row)
    inc_deg = _f("pl_orbincl", row)
    t0 = _f("pl_tranmid", row)
    ecc = _f("pl_orbeccen", row)
    teff = int(round(_f("st_teff", row)))
    logg = _f("st_logg", row)
    met = _f("st_met", row)

    inc = float(np.deg2rad(inc_deg))
    omega = 0.0 if ecc == 0.0 else 0.0  # 近心点角；圆轨道固定 0

    ld_u = _compute_ld_u(cfg, teff, logg, met)
    ld_j = jnp.array(ld_u, dtype=jnp.float64)

    ts = cfg.get("time_sampling")
    if not isinstance(ts, dict):
        ts = None
    times_np = build_lightcurve_time_array_days(
        t0_days=float(t0),
        period_days=float(period),
        pl_ratdor=float(a),
        k_rp_over_rs=float(rp),
        inc_deg=float(inc_deg),
        ecc=float(ecc),
        time_half_width_days=float(cfg["time_half_width_days"]),
        n_time_uniform=int(cfg["n_time"]),
        time_sampling=ts,
    )
    times = jnp.array(times_np, dtype=jnp.float64)

    common = dict(
        times=times,
        t0=float(t0),
        period=float(period),
        a=float(a),
        tidally_locked=bool(cfg["tidally_locked"]),
        e=float(ecc),
        i=inc,
        omega=float(omega),
        Omega=float(cfg["Omega_rad"]),
        obliq=0.0,
        prec=0.0,
        ld_u_coeffs=ld_j,
        parameterize_with_projected_ellipse=True,
        projected_effective_r=float(rp),
    )

    sys_sphere = OblateSystem(
        **common,
        projected_f=0.0,
        projected_theta=0.0,
    )
    sys_oblate = OblateSystem(
        **common,
        projected_f=float(cfg["injected_projected_f"]),
        projected_theta=float(np.deg2rad(cfg["injected_projected_theta_deg"])),
    )

    lc_sphere = np.array(sys_sphere.lightcurve())
    lc_oblate = np.array(sys_oblate.lightcurve())
    th = np.array(times)

    assert np.all(np.isfinite(lc_sphere)) and np.all(np.isfinite(lc_oblate))
    assert float(lc_sphere.max()) <= 1.0 + 1e-6 and float(lc_oblate.max()) <= 1.0 + 1e-6
    assert float(lc_sphere.min()) < 1.0 and float(lc_oblate.min()) < 1.0

    if cfg.get("save_figure"):
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
        x_h = (th - t0) * 24.0
        ax.plot(x_h, lc_sphere, label=r"spherical (projected $f=0$)", color="C0", lw=1.5)
        ax.plot(
            x_h,
            lc_oblate,
            label=rf"oblate $f={cfg['injected_projected_f']}$, "
            rf"$\theta={cfg['injected_projected_theta_deg']:.1f}^\circ$",
            color="C1",
            lw=1.5,
            alpha=0.9,
        )
        ax.set_xlabel(r"$t - t_0$ (hours)")
        ax.set_ylabel("relative flux")
        ax.set_title(f"{name} — transit + LD ({cfg['exotic_ld_mode']})")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = root / "results" / str(cfg["figure_filename"])
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)

    # 注入扁率后曲线应与基准不完全相同（若数值完全相同说明参数退化或窗口内无差异）
    assert not np.allclose(lc_sphere, lc_oblate, rtol=0, atol=1e-7)


if __name__ == "__main__":
    test_nasa_csv_first_row_theory_vs_injected_oblate_lightcurve()
    print("OK — see results/", USER_CONFIG["figure_filename"], "if save_figure=True.")
