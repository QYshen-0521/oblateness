"""
无噪声反演：二维网格 :math:`\\chi^2`（见根目录 ``worklog.md``「无噪声反演」节）。

代数上等价写法是 :math:`D=F_0-F_{\\mathrm{inj}}`、:math:`m=F_0-F_{\\mathrm{obl}}`，
:math:`\\chi^2=\\sum_k(D_k-m_k)^2`，其中 :math:`F_0` 会抵消，故**实现上不算** :math:`F_0`，只算：

1. 注入 :math:`(f_{\\mathrm{inj}},\\theta_{\\mathrm{inj}})` 的流量 :math:`F_{\\mathrm{inj}}(t)`；
2. 网格上每个 :math:`(f,\\theta)` 的 :math:`F_{\\mathrm{obl}}(t;f,\\theta)`；

取 :math:`\\sigma\\equiv 1` 时
:math:`\\chi^2(f,\\theta)=\\sum_k\\bigl(F_{\\mathrm{obl},k}-F_{\\mathrm{inj},k}\\bigr)^2`。

行星、LD、时间窗与 **注入的** ``projected_f`` / ``projected_theta`` 与
``tests/test_nasa_first_row_oblate_lightcurve.py`` 中 ``USER_CONFIG`` 保持一致（首行 CSV、
JWST NIRSpec Prism LD 等）。时间轴默认 **ingress/egress + 60 s 步长**（``PLANET_CONFIG['time_sampling']``），
新结果写入 ``results/noiseless_ie60/``。环境变量 ``OBLATE_TIME_SAMPLING_MODE=uniform_symmetric``
可恢复对称均匀窗。网格范围与分辨率见 ``GRID_CONFIG``。

运行::

    pip install -e ".[squishy]"
    python -m oblateness.noiseless_grid_chi2_inversion

图保存到 ``results/``（文件名见 ``GRID_CONFIG['figure_filename']``）。
"""

from __future__ import annotations

import csv
import os
import warnings
from pathlib import Path

import numpy as np

from oblateness.transit_ie_sampling import build_lightcurve_time_array_days
# NASA 凌星可建模输入表（相对仓库根；``batch_noiseless_recovery`` 继承此 ``PLANET_CONFIG``）
# - 原 Tier A 全量：  data/nasa_archive/ps_tran_oblate_inputs_valid_20260416.csv  （578 行）
# - Tier B 派生新源： data/nasa_archive/ps_tran_oblate_inputs_tier_b_derived_20260416.csv （~164 行）
# 超算补跑 Tier B 时**默认**用 B 表；重跑 578 可 ``export OBLATE_CSV_PATH=.../ps_tran_oblate_inputs_valid_20260416.csv`` 或改下段默认常数
# ---------------------------------------------------------------------------
# "csv_path": "data/nasa_archive/ps_tran_oblate_inputs_valid_20260416.csv",  # 原 578 行表（未删除，仅作注释备查）
_CSV_TIER_B_DERIVED = "data/nasa_archive/ps_tran_oblate_inputs_tier_b_derived_20260416.csv"


def _oblate_input_csv_path() -> str:
    """由环境变量 ``OBLATE_CSV_PATH`` 覆盖，否则为 Tier B 派生表。"""
    return os.environ.get("OBLATE_CSV_PATH", _CSV_TIER_B_DERIVED)


# ---------------------------------------------------------------------------
# 与 tests/test_nasa_first_row_oblate_lightcurve.py 中 USER_CONFIG 对齐（恢复注入）
# ---------------------------------------------------------------------------
PLANET_CONFIG: dict = {
    "csv_path": _oblate_input_csv_path(),
    "row_index": 0,
    "time_half_width_days": 0.08,
    "n_time": 500,
    "injected_projected_f": 0.12,
    "injected_projected_theta_deg": 40.0,
    "ld_source": "exotic",
    "exotic_ld_stellar_model": "mps1",
    "exotic_ld_mode": "JWST_NIRSpec_Prism",
    "exotic_ld_wavelength_range_angstrom": (20000.0, 30000.0),
    "exotic_ld_data_path": "data/exotic_ld_data",
    "fixed_ld_u_coeffs": (0.45, 0.20),
    "tidally_locked": False,
    "Omega_rad": float(np.pi),
    # I/E + 60 s：见 ``transit_ie_sampling``；``OBLATE_TIME_SAMPLING_MODE=uniform_symmetric`` 可回退旧均匀窗
    "time_sampling": {
        "mode": "ingress_egress",
        "cadence_seconds": 60.0,
        "edge_pad_cadences": 3,
        "n_ie_segment_max": 2500,
        "ecc_max_circular": 0.05,
        "uniform_fallback_half_width_days": 0.08,
        "uniform_fallback_n_time": 500,
    },
}

# ---------------------------------------------------------------------------
# 网格 : (f, projected_theta)；theta 用弧度，范围 [0, π) 避免等价重复
# ---------------------------------------------------------------------------
GRID_CONFIG: dict = {
    "f_min": 0.0,
    "f_max": 0.25,
    "n_f": 41,
    "theta_min_rad": 0.0,
    "theta_max_rad": np.pi * (1.0 - 1e-9),
    "n_theta": 37,
    # 新目录，避免覆盖旧 ``results/noiseless_chi2_grid*``
    "figure_filename": "noiseless_ie60/noiseless_chi2_grid_f_theta.png",
    "save_chi2_npy": True,
    "chi2_npy_filename": "noiseless_ie60/noiseless_chi2_grid.npz",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_csv_row(path: Path, index: int) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if index < 0 or index >= len(rows):
        raise IndexError(f"row_index={index} out of range (n={len(rows)})")
    return rows[index]


def _col(name: str, row: dict[str, str]) -> float:
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
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"ExoTiC-LD failed ({exc}); using fixed_ld_u_coeffs.", stacklevel=2)
        return np.array(cfg["fixed_ld_u_coeffs"], dtype=float)


def run_grid_inversion() -> dict:
    """计算 :math:`\\chi^2(f,\\theta)=\\sum_k(F_{\\mathrm{obl},k}-F_{\\mathrm{inj},k})^2` 网格并返回最优格点等。"""
    import jax.numpy as jnp
    from squishyplanet import OblateSystem

    pc = PLANET_CONFIG
    gc = GRID_CONFIG
    root = _repo_root()
    csv_path = root / pc["csv_path"]
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

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
    omega = 0.0

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
        omega=float(omega),
        Omega=float(pc["Omega_rad"]),
        obliq=0.0,
        prec=0.0,
        ld_u_coeffs=ld_j,
        parameterize_with_projected_ellipse=True,
        projected_effective_r=float(rp),
    )

    system = OblateSystem(**common, projected_f=0.0, projected_theta=0.0)

    f_inj = float(pc["injected_projected_f"])
    th_inj = float(np.deg2rad(pc["injected_projected_theta_deg"]))

    f_inj_arr = np.array(system.lightcurve(params={"projected_f": f_inj, "projected_theta": th_inj}))

    fs = np.linspace(gc["f_min"], gc["f_max"], int(gc["n_f"]))
    ths = np.linspace(gc["theta_min_rad"], gc["theta_max_rad"], int(gc["n_theta"]))
    chi2 = np.empty((len(fs), len(ths)), dtype=np.float64)

    for i, f in enumerate(fs):
        for j, th in enumerate(ths):
            f_obl = np.array(system.lightcurve(params={"projected_f": float(f), "projected_theta": float(th)}))
            chi2[i, j] = float(np.sum((f_obl - f_inj_arr) ** 2))

    idx = np.argmin(chi2)
    ii, jj = np.unravel_index(idx, chi2.shape)
    f_hat, th_hat = fs[ii], ths[jj]
    chi2_min = float(chi2[ii, jj])

    # 注入格点若恰在网格上，报告残差
    i_inj = int(np.argmin(np.abs(fs - f_inj)))
    j_inj = int(np.argmin(np.abs(ths - th_inj)))
    chi2_at_inj = float(chi2[i_inj, j_inj])

    out = {
        "pl_name": name,
        "f_grid": fs,
        "theta_grid_rad": ths,
        "chi2": chi2,
        "f_hat": f_hat,
        "theta_hat_rad": th_hat,
        "chi2_min": chi2_min,
        "f_injected": f_inj,
        "theta_injected_rad": th_inj,
        "chi2_at_injected_grid_nearest": chi2_at_inj,
        "f_inj": f_inj_arr,
    }
    return out


def _save_figure(result: dict) -> Path:
    matplotlib = __import__("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gc = GRID_CONFIG
    root = _repo_root()
    fs = result["f_grid"]
    ths = result["theta_grid_rad"]
    chi2 = result["chi2"]
    z = np.log10(chi2 + 1e-30)

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=120)
    cf = ax.pcolormesh(
        ths * 180.0 / np.pi,
        fs,
        z,
        shading="auto",
        cmap="viridis",
    )
    plt.colorbar(cf, ax=ax, label=r"$\log_{10}(\chi^2)$")
    ax.scatter(
        [result["theta_injected_rad"] * 180.0 / np.pi],
        [result["f_injected"]],
        marker="*",
        s=200,
        c="red",
        label="injected",
        zorder=5,
    )
    ax.scatter(
        [result["theta_hat_rad"] * 180.0 / np.pi],
        [result["f_hat"]],
        marker="+",
        s=200,
        c="white",
        label=r"grid min $\hat\theta,\hat f$",
        zorder=5,
    )
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel(r"projected $f$")
    ax.set_title(f"{result['pl_name']} — noiseless $\\chi^2(f,\\theta)$ (JWST LD)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = root / "results" / gc["figure_filename"]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    result = run_grid_inversion()
    print("planet:", result["pl_name"])
    print("injected: f =", result["f_injected"], ", theta =", np.rad2deg(result["theta_injected_rad"]), "deg")
    print("grid best: f_hat =", result["f_hat"], ", theta_hat =", np.rad2deg(result["theta_hat_rad"]), "deg")
    print("min chi2 =", result["chi2_min"])
    print("chi2 at nearest grid point to injected =", result["chi2_at_injected_grid_nearest"])

    root = _repo_root()
    fig_path = _save_figure(result)
    print("saved figure:", fig_path)

    if GRID_CONFIG.get("save_chi2_npy"):
        npy = root / "results" / str(GRID_CONFIG["chi2_npy_filename"])
        np.savez_compressed(
            npy,
            f_grid=result["f_grid"],
            theta_grid_rad=result["theta_grid_rad"],
            chi2=result["chi2"],
            f_injected=result["f_injected"],
            theta_injected_rad=result["theta_injected_rad"],
        )
        print("saved array:", npy)


if __name__ == "__main__":
    main()
