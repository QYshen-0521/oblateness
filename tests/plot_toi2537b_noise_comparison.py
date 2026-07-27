"""Plot the same injected TOI-2537 b light curve at two noise levels.

Run from the repository root:

    .venv/bin/python tests/plot_toi2537b_noise_comparison.py

The two figures are written beside this script in ``tests/``.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oblateness-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oblateness.noised_grid_recovery_single_injection import (
    NOISE_CONFIG,
    PLANET_CONFIG,
    _load_csv_row,
    _load_success_noise_for_target,
    _repo_root,
)
from oblateness.noised_grid_recovery_single_planet_batch_v03 import (
    BASE_PLANET_CONFIG,
    _build_system,
)


OUTPUT_DIR = Path(__file__).resolve().parent


def _plot_noisy_curve(
    *,
    times_days: np.ndarray,
    t0_days: float,
    clean_flux: np.ndarray,
    standard_noise: np.ndarray,
    sigma_flux: float,
    sigma_label: str,
    output_path: Path,
    cadence_seconds: float,
    nint: int,
) -> None:
    time_minutes = (times_days - float(t0_days)) * 24.0 * 60.0
    residual_ppm = standard_noise * float(sigma_flux) * 1e6
    noisy_flux = clean_flux + residual_ppm * 1e-6

    # Ingress and egress are separate sampled segments; do not connect across the gap.
    split_index = int(np.argmax(np.diff(time_minutes))) + 1
    segments = (slice(0, split_index), slice(split_index, None))

    fig, (ax_flux, ax_residual) = plt.subplots(
        2,
        1,
        figsize=(10.0, 6.8),
        dpi=140,
        sharex=True,
        gridspec_kw={"height_ratios": (2.1, 1.0), "hspace": 0.08},
    )
    clean_handle = None
    for segment in segments:
        line = ax_flux.plot(
            time_minutes[segment],
            clean_flux[segment],
            color="black",
            linewidth=1.4,
            zorder=2,
        )[0]
        if clean_handle is None:
            clean_handle = line
    noisy_handle = ax_flux.scatter(
        time_minutes,
        noisy_flux,
        s=24,
        color="#2678b2",
        edgecolor="white",
        linewidth=0.35,
        label="Gaussian-noised samples",
        zorder=3,
    )
    ax_flux.set_ylabel("normalized flux")
    ax_flux.grid(True, alpha=0.22)
    ax_flux.legend(
        [clean_handle, noisy_handle],
        ["clean injected light curve", "Gaussian-noised samples"],
        loc="lower center",
        frameon=True,
    )

    ax_residual.axhline(0.0, color="black", linewidth=1.0)
    ax_residual.axhline(sigma_flux * 1e6, color="#d1495b", linestyle="--", linewidth=1.0)
    ax_residual.axhline(-sigma_flux * 1e6, color="#d1495b", linestyle="--", linewidth=1.0)
    ax_residual.scatter(
        time_minutes,
        residual_ppm,
        s=22,
        color="#2678b2",
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    ax_residual.set_xlabel("time from mid-transit (minutes)")
    ax_residual.set_ylabel("noise residual (ppm)")
    ax_residual.grid(True, alpha=0.22)

    f_inj = float(PLANET_CONFIG["injected_projected_f"])
    theta_inj = float(PLANET_CONFIG["injected_projected_theta_deg"])
    fig.suptitle(
        f"TOI-2537 b: Gaussian noise sigma = {sigma_flux * 1e6:.2f} ppm ({sigma_label})\n"
        f"injected f={f_inj:.2f}, theta={theta_inj:.0f} deg; "
        f"{times_days.size} points at {cadence_seconds:.3f} s cadence; nint={nint}"
    )
    fig.subplots_adjust(top=0.86, left=0.11, right=0.98, bottom=0.10)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    root = _repo_root()
    input_path = root / str(PLANET_CONFIG["csv_path"])
    input_row = _load_csv_row(input_path, int(PLANET_CONFIG["row_index"]))
    target = input_row["pl_name"].strip()
    if target != "TOI-2537 b":
        raise ValueError(f"this diagnostic expects TOI-2537 b, got {target!r}")

    noise_path = root / str(NOISE_CONFIG["cache_csv_path"])
    noise_row = _load_success_noise_for_target(noise_path, target)
    sigma_full_transit = float(noise_row["sigma_flux"])
    nint = int(round(float(noise_row["nint"])))
    cadence_seconds = float(noise_row["t_exp_s"])
    sigma_per_integration = sigma_full_transit * np.sqrt(nint)

    system, _, times_days = _build_system(
        BASE_PLANET_CONFIG,
        cadence_seconds=cadence_seconds,
    )
    clean_flux = np.asarray(
        system.lightcurve(
            params={
                "projected_f": float(PLANET_CONFIG["injected_projected_f"]),
                "projected_theta": np.deg2rad(
                    float(PLANET_CONFIG["injected_projected_theta_deg"])
                ),
            }
        ),
        dtype=np.float64,
    )
    standard_noise = np.random.default_rng(int(NOISE_CONFIG["random_seed"])).normal(
        size=clean_flux.shape
    )
    t0_days = float(input_row["pl_tranmid"])

    outputs = (
        (
            sigma_per_integration,
            "one integration, inferred from the full-transit uncertainty",
            OUTPUT_DIR / "toi2537b_noisy_lightcurve_sigma_1809ppm.png",
        ),
        (
            sigma_full_transit,
            "full-transit combined uncertainty applied to every point",
            OUTPUT_DIR / "toi2537b_noisy_lightcurve_sigma_106ppm.png",
        ),
    )
    for sigma_flux, label, output_path in outputs:
        _plot_noisy_curve(
            times_days=times_days,
            t0_days=t0_days,
            clean_flux=clean_flux,
            standard_noise=standard_noise,
            sigma_flux=sigma_flux,
            sigma_label=label,
            output_path=output_path,
            cadence_seconds=cadence_seconds,
            nint=nint,
        )
        print(output_path)

    print(f"sigma_full_transit_ppm={sigma_full_transit * 1e6:.6f}")
    print(f"sigma_per_integration_ppm={sigma_per_integration * 1e6:.6f}")


if __name__ == "__main__":
    main()
