"""用 v04 完备度边界验证普通凌星参数微调后的扁率探测能力。

程序按 ``TARGET_NAME`` 从行星输入表和 calculate-noise 缓存中匹配目标。每个 cadence 注入的
独立白色高斯噪声按照 Wang & Winn (2026), arXiv:2607.09873v1 的经验噪声模型计算：
先由 K 星等计算 Equation (2) 的 1 分钟白光散布，再依据白噪声的平方根时间定律缩放到缓存
``t_exp_s`` 对应的实际 cadence。

程序先对 ``f=0`` 的球形注入做 Monte Carlo，并通过球形模型与最佳扁球模型之间的
``delta_chi2`` 分布标定经验假阳性阈值。每一条含噪光变都会让球形模型和全部扁球格点
按完全相同的规则微调投影等效半径、中凌时刻和加性流量基线。半径与时刻采用带高斯先验
的局部线性 profile 拟合，避免每条光变都重新运行昂贵的非线性优化。

程序不再遍历完整注入地图，而是读取 v04 已保存的 ``f50_f90_curves.csv``，只在每个
``theta`` 的 f50/f90 及其置信区间附近生成验证点。v05 不会调用或重跑 v04；v04 结果文件
缺失、目标不匹配或字段不完整时会在昂贵计算开始前给出清晰错误。

验证结束后输出微调模型得到的 f50/f90、95% 置信区间，以及它们相对 v04 边界的移动量。
若中心边界或 95% 上界尚未在初始窗口内闭合，程序只为受影响的 theta 自动追加更高 f，
并复用同一批噪声 realization，直到闭合、达到轮数限制或触及配置的 f 上限。

运行（需 ``pip install -e ".[squishy]"``）::

    python -m oblateness.noised_grid_recovery_single_planet_batch_v05

默认每个验证点运行 100 次共同噪声 realization；调试可将
``BATCH_RUN_CONFIG['max_injections']`` 设为小整数。
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
# 行星 + LD + 时间轴（与单点反演一致；注入点来自 v04 边界结果）
# =============================================================================
BASE_PLANET_CONFIG: dict[str, Any] = {
    k: v
    for k, v in PLANET_CONFIG.items()
    if k not in ("injected_projected_f", "injected_projected_theta_deg")
}
BASE_PLANET_CONFIG["target_name"] = TARGET_NAME

# =============================================================================
# v04 边界结果与 v05 验证点
# =============================================================================
V04_BOUNDARY_CONFIG: dict[str, Any] = {
    "results_csv_filename": (
        "batch_noised_completeness_v04_adaptive_ie60/"
        "{target_slug}/f50_f90_curves.csv"
    ),
    "probability_levels": (0.50, 0.90),
    # 每条 v04 边界除中心值、95%上下界及原始夹点外，再验证这些相对 f 偏移。
    # 正方向留得更宽，因为普通参数微调通常会把探测边界推向更大的 f。
    "relative_f_offsets": (-0.005, 0.0, 0.010, 0.020),
    "f_min": 0.0,
    "f_max": 0.35,
    "dedupe_ndigits": 10,
    # 若中心边界或 95% 上界仍未在初始窗口内闭合，只给对应 theta 向高 f 补点。
    "auto_extend_high_f": True,
    "extension_f_step": 0.010,
    "extension_points_per_round": 2,
    "max_extension_rounds": 3,
    "detection_limit_ci_z": 1.959963984540054,
    "detection_limit_confidence_level": 0.95,
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
    "n_null_realizations": 1000,
    "null_seed_offset": 1_000_000,
}

# =============================================================================
# 每条含噪光变的普通参数微调
# =============================================================================
NUISANCE_FIT_CONFIG: dict[str, Any] = {
    "fit_flux_baseline": True,
    "fit_projected_effective_r": True,
    "fit_t0": True,
    # 半径先验优先采用输入表 pl_ratrorerr1/2；缺失时使用名义半径的 1%。
    "radius_prior_sigma_fraction_fallback": 0.01,
    # 这是局部线性拟合，过宽的半径先验限制到名义半径的 5%。
    "radius_prior_sigma_fraction_max": 0.05,
    # 中凌时刻允许在每条光变上微调；1 个 cadence 是局部拟合的 1-sigma 尺度。
    "t0_prior_sigma_cadences": 1.0,
    # 中心有限差分计算半径导数时，步长取半径先验的这一比例。
    "radius_derivative_step_fraction_of_prior": 0.25,
    "radius_derivative_min_fraction": 1.0e-4,
    # 大批 null realization 分块求解，控制峰值内存。
    "realization_chunk_size": 32,
}

BATCH_RUN_CONFIG: dict[str, Any] = {
    # None = 全部 v04 边界验证点；设为例如 3 便于调试
    "max_injections": None,
    "n_noise_realizations": 100,
    "summary_figure_filename": "batch_noised_detection_v05_boundary_profiled/{target_slug}/boundary_validation.png",
    "save_results_npz": True,
    "results_npz_filename": "batch_noised_detection_v05_boundary_profiled/{target_slug}/detection_realizations.npz",
    "save_realizations_csv": True,
    "realizations_csv_filename": "batch_noised_detection_v05_boundary_profiled/{target_slug}/detection_realizations.csv",
    "save_summary_csv": True,
    "summary_csv_filename": "batch_noised_detection_v05_boundary_profiled/{target_slug}/detection_probability_summary.csv",
    "save_detection_limits_csv": True,
    "detection_limits_csv_filename": "batch_noised_detection_v05_boundary_profiled/{target_slug}/f50_f90_profiled.csv",
    "save_null_calibration_npz": True,
    "null_calibration_npz_filename": "batch_noised_detection_v05_boundary_profiled/{target_slug}/null_calibration.npz",
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


def _configured_v04_results_path(target_name: str) -> Path:
    relative_path = str(V04_BOUNDARY_CONFIG["results_csv_filename"]).format(
        target_slug=_target_slug(target_name)
    )
    return _repo_root() / "results" / relative_path


def _optional_float(row: dict[str, str], key: str) -> float:
    text = str(row.get(key, "")).strip()
    if not text:
        return float("nan")
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"v04 boundary column {key!r} is not numeric: {text!r}") from exc
    return float(value)


def _load_v04_boundary_rows(
    path: Path,
    target_name: str,
) -> list[dict[str, Any]]:
    """Load and validate the saved v04 f50/f90 boundary table."""
    if not path.is_file():
        raise FileNotFoundError(
            f"v04 boundary result not found for {target_name!r}: {path}. "
            "Run v04 for this target before running v05."
        )
    with path.open(newline="", encoding="utf-8-sig") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError(f"v04 boundary result is empty: {path}")

    required = {
        "pl_name",
        "theta_inj_deg",
        "probability_level",
        "f_detection_limit",
        "previous_f_inj",
        "next_f_inj",
        "limit_status",
    }
    missing = sorted(required.difference(raw_rows[0]))
    if missing:
        raise KeyError(
            f"v04 boundary result {path} is missing required columns: "
            f"{', '.join(missing)}"
        )

    requested_key = _target_name_key(target_name)
    selected_levels = tuple(
        float(value) for value in V04_BOUNDARY_CONFIG["probability_levels"]
    )
    parsed: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    for raw in raw_rows:
        source_name = str(raw["pl_name"]).strip()
        if _target_name_key(source_name) != requested_key:
            raise ValueError(
                f"v04 boundary target mismatch: requested {target_name!r}, "
                f"but {path} contains {source_name!r}"
            )
        theta_deg = _optional_float(raw, "theta_inj_deg")
        probability_level = _optional_float(raw, "probability_level")
        if not np.isfinite(theta_deg) or not np.isfinite(probability_level):
            raise ValueError(
                f"v04 boundary has invalid theta/probability row: {raw}"
            )
        if not any(
            np.isclose(probability_level, level)
            for level in selected_levels
        ):
            continue
        key = (round(theta_deg, 8), round(probability_level, 8))
        if key in seen:
            raise ValueError(
                f"v04 boundary contains duplicate theta/probability row: {key}"
            )
        seen.add(key)
        f_limit = _optional_float(raw, "f_detection_limit")
        ci_low = _optional_float(raw, "f_detection_limit_ci95_low")
        ci_high = _optional_float(raw, "f_detection_limit_ci95_high")
        has_confidence_interval = np.isfinite(ci_low) and np.isfinite(ci_high)
        if not np.isfinite(ci_low):
            ci_low = f_limit
        if not np.isfinite(ci_high):
            ci_high = f_limit
        parsed.append(
            {
                "pl_name": source_name,
                "theta_inj_deg": theta_deg,
                "probability_level": probability_level,
                "f_detection_limit": f_limit,
                "f_detection_limit_ci95_low": ci_low,
                "f_detection_limit_ci95_high": ci_high,
                "has_confidence_interval": has_confidence_interval,
                "previous_f_inj": _optional_float(raw, "previous_f_inj"),
                "next_f_inj": _optional_float(raw, "next_f_inj"),
                "limit_status": str(raw["limit_status"]).strip(),
            }
        )
    if not parsed:
        raise ValueError(
            f"v04 boundary result {path} contains none of the requested "
            f"probability levels {selected_levels}"
        )
    return sorted(
        parsed,
        key=lambda row: (
            float(row["theta_inj_deg"]),
            float(row["probability_level"]),
        ),
    )


def _build_v04_validation_pairs(
    boundary_rows: list[dict[str, Any]],
) -> tuple[
    list[tuple[float, float]],
    dict[float, dict[str, Any]],
    dict[str, int],
]:
    """Generate deduplicated injection points around the saved v04 boundaries."""
    cfg = V04_BOUNDARY_CONFIG
    f_min = float(cfg["f_min"])
    f_max = float(cfg["f_max"])
    if not 0.0 <= f_min < f_max:
        raise ValueError(f"invalid v04 validation f range: {f_min}, {f_max}")
    offsets = tuple(float(value) for value in cfg["relative_f_offsets"])
    ndigits = int(cfg["dedupe_ndigits"])

    references: dict[float, dict[str, Any]] = {}
    pair_set: set[tuple[float, float]] = set()
    unresolved_count = 0
    for row in boundary_rows:
        theta_deg = float(row["theta_inj_deg"])
        theta_key = round(theta_deg, 8)
        level = float(row["probability_level"])
        level_label = int(round(100.0 * level))
        reference = references.setdefault(
            theta_key,
            {
                "v04_f50": float("nan"),
                "v04_f50_ci95_low": float("nan"),
                "v04_f50_ci95_high": float("nan"),
                "v04_f50_status": "missing",
                "v04_f90": float("nan"),
                "v04_f90_ci95_low": float("nan"),
                "v04_f90_ci95_high": float("nan"),
                "v04_f90_status": "missing",
            },
        )
        reference[f"v04_f{level_label}"] = float(row["f_detection_limit"])
        reference[f"v04_f{level_label}_ci95_low"] = float(
            row["f_detection_limit_ci95_low"]
        )
        reference[f"v04_f{level_label}_ci95_high"] = float(
            row["f_detection_limit_ci95_high"]
        )
        reference[f"v04_f{level_label}_status"] = str(row["limit_status"])

        center = float(row["f_detection_limit"])
        if not np.isfinite(center):
            unresolved_count += 1
            continue
        candidates = [
            center + offset
            for offset in offsets
        ]
        if bool(row["has_confidence_interval"]):
            candidates.extend(
                [
                    float(row["f_detection_limit_ci95_low"]),
                    float(row["f_detection_limit_ci95_high"]),
                ]
            )
        else:
            candidates.extend(
                [
                    float(row["previous_f_inj"]),
                    float(row["next_f_inj"]),
                ]
            )
        for f_value in candidates:
            if not np.isfinite(f_value):
                continue
            clipped_f = float(np.clip(f_value, f_min, f_max))
            pair_set.add(
                (
                    round(clipped_f, ndigits),
                    theta_key,
                )
            )

    if not pair_set:
        raise ValueError(
            "v04 has no resolved f50/f90 boundary from which v05 can build "
            "validation points"
        )
    pairs = sorted(pair_set, key=lambda pair: (pair[1], pair[0]))
    diagnostics = {
        "n_v04_boundary_rows": len(boundary_rows),
        "n_v04_unresolved_rows": unresolved_count,
        "n_validation_points": len(pairs),
        "n_validation_theta": len(references),
    }
    return pairs, references, diagnostics


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
    """Avoid Squishyplanet's exact 45/135-degree ellipse-conversion branch."""
    theta = float(theta_rad) % np.pi
    if np.isclose(np.cos(2.0 * theta), 0.0, rtol=0.0, atol=1.0e-14):
        theta += np.deg2rad(PROJECTED_THETA_SINGULARITY_OFFSET_DEG)
    return float(theta % np.pi)


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


def _state_scalar(system: Any, key: str) -> float:
    """Return one scalar model-state value with a clear shape check."""
    value = np.asarray(system.state[key], dtype=np.float64)
    if value.size != 1:
        raise ValueError(f"system.state[{key!r}] is not scalar: shape={value.shape}")
    return float(value.reshape(-1)[0])


def _symmetric_row_uncertainty(
    row: dict[str, str],
    upper_key: str,
    lower_key: str,
) -> float | None:
    """Average the finite positive magnitudes of asymmetric table errors."""
    values: list[float] = []
    for key in (upper_key, lower_key):
        text = str(row.get(key, "")).strip()
        if not text:
            continue
        try:
            value = abs(float(text))
        except ValueError:
            continue
        if np.isfinite(value) and value > 0.0:
            values.append(value)
    return float(np.mean(values)) if values else None


def _precompute_nuisance_profile(
    system: Any,
    model_fluxes: np.ndarray,
    fs: np.ndarray,
    theta_deg: np.ndarray,
    times_days: np.ndarray,
    input_row: dict[str, str],
    sigma_flux_per_cadence: float,
    cadence_seconds: float,
) -> dict[str, Any]:
    """Prepare local derivatives and normal matrices for nuisance profiling.

    The same design matrix is used for the spherical row and every oblate grid
    point. Radius and t0 shifts have Gaussian priors; the additive flux baseline
    is unpenalized.
    """
    cfg = NUISANCE_FIT_CONFIG
    models = np.asarray(model_fluxes, dtype=np.float64)
    times = np.asarray(times_days, dtype=np.float64)
    if models.ndim != 2 or times.ndim != 1 or models.shape[1] != times.size:
        raise ValueError(
            "model_fluxes and times_days have incompatible shapes: "
            f"{models.shape}, {times.shape}"
        )
    if times.size < 3 or np.any(np.diff(times) <= 0.0):
        raise ValueError("times_days must contain at least three strictly increasing values")

    sigma = float(sigma_flux_per_cadence)
    nominal_radius = _state_scalar(system, "projected_effective_r")
    nominal_t0 = _state_scalar(system, "t0")
    radius_table_sigma = _symmetric_row_uncertainty(
        input_row,
        "pl_ratrorerr1",
        "pl_ratrorerr2",
    )
    radius_fallback_sigma = (
        nominal_radius * float(cfg["radius_prior_sigma_fraction_fallback"])
    )
    radius_prior_sigma = (
        radius_table_sigma
        if radius_table_sigma is not None
        else radius_fallback_sigma
    )
    radius_prior_sigma = min(
        radius_prior_sigma,
        nominal_radius * float(cfg["radius_prior_sigma_fraction_max"]),
    )
    if not np.isfinite(radius_prior_sigma) or radius_prior_sigma <= 0.0:
        raise ValueError(f"invalid projected-radius prior sigma: {radius_prior_sigma}")

    t0_prior_sigma_days = (
        float(cfg["t0_prior_sigma_cadences"]) * float(cadence_seconds) / 86400.0
    )
    if not np.isfinite(t0_prior_sigma_days) or t0_prior_sigma_days <= 0.0:
        raise ValueError(f"invalid t0 prior sigma: {t0_prior_sigma_days} days")

    derivative_names: list[str] = []
    derivative_columns: list[np.ndarray] = []
    prior_sigmas: list[float] = []
    radius_derivative_step = float("nan")

    if bool(cfg["fit_flux_baseline"]):
        derivative_names.append("flux_baseline")
        derivative_columns.append(np.ones_like(models))
        prior_sigmas.append(float("inf"))

    if bool(cfg["fit_projected_effective_r"]):
        radius_derivative_step = max(
            radius_prior_sigma
            * float(cfg["radius_derivative_step_fraction_of_prior"]),
            nominal_radius * float(cfg["radius_derivative_min_fraction"]),
        )
        if radius_derivative_step >= nominal_radius:
            raise ValueError(
                "radius finite-difference step must be smaller than the nominal radius: "
                f"{radius_derivative_step} >= {nominal_radius}"
            )
        radius_derivative = np.empty_like(models)
        iterator = range(models.shape[0])
        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=models.shape[0],
                desc="precompute radius derivative",
                unit="model",
            )
        n_theta = int(theta_deg.size)
        for flat_index in iterator:
            i_f, i_theta = divmod(flat_index, n_theta)
            if i_f == 0 and i_theta > 0:
                radius_derivative[flat_index] = radius_derivative[0]
                continue
            shape_params = {
                "projected_f": float(fs[i_f]),
                "projected_theta": _safe_projected_theta_rad(
                    np.deg2rad(theta_deg[i_theta])
                ),
            }
            flux_plus = np.asarray(
                system.lightcurve(
                    params={
                        **shape_params,
                        "projected_effective_r": nominal_radius
                        + radius_derivative_step,
                    }
                ),
                dtype=np.float64,
            )
            flux_minus = np.asarray(
                system.lightcurve(
                    params={
                        **shape_params,
                        "projected_effective_r": nominal_radius
                        - radius_derivative_step,
                    }
                ),
                dtype=np.float64,
            )
            radius_derivative[flat_index] = (
                flux_plus - flux_minus
            ) / (2.0 * radius_derivative_step)
        derivative_names.append("projected_effective_r")
        derivative_columns.append(radius_derivative)
        prior_sigmas.append(radius_prior_sigma)

    if bool(cfg["fit_t0"]):
        # Changing t0 translates the local transit profile: dF/dt0 = -dF/dt.
        t0_derivative = -np.gradient(
            models,
            times,
            axis=1,
            edge_order=2,
        )
        derivative_names.append("t0")
        derivative_columns.append(t0_derivative)
        prior_sigmas.append(t0_prior_sigma_days)

    if not derivative_columns:
        raise ValueError("NUISANCE_FIT_CONFIG disables every nuisance parameter")

    derivatives = np.stack(derivative_columns, axis=2)
    normal_matrices = np.einsum(
        "mtk,mtl->mkl",
        derivatives,
        derivatives,
        optimize=True,
    )
    for parameter_index, prior_sigma in enumerate(prior_sigmas):
        if np.isfinite(prior_sigma):
            normal_matrices[:, parameter_index, parameter_index] += (
                sigma**2 / prior_sigma**2
            )
    normal_matrix_inverse = np.linalg.pinv(normal_matrices)
    return {
        "derivative_names": tuple(derivative_names),
        "derivatives": derivatives,
        "normal_matrix_inverse": normal_matrix_inverse,
        "nominal_projected_effective_r": nominal_radius,
        "nominal_t0": nominal_t0,
        "radius_table_sigma": (
            float(radius_table_sigma)
            if radius_table_sigma is not None
            else float("nan")
        ),
        "radius_prior_sigma": radius_prior_sigma,
        "t0_prior_sigma_days": t0_prior_sigma_days,
        "radius_derivative_step": radius_derivative_step,
    }


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


def _profiled_chi2_from_model_library(
    model_fluxes: np.ndarray,
    observed_fluxes: np.ndarray,
    sigma_flux_per_cadence: float,
    nuisance_profile: dict[str, Any],
    *,
    return_coefficients: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Profile χ² over the local nuisance parameters in realization chunks."""
    models = np.asarray(model_fluxes, dtype=np.float64)
    observations = np.asarray(observed_fluxes, dtype=np.float64)
    if observations.ndim == 1:
        observations = observations[None, :]
    if (
        models.ndim != 2
        or observations.ndim != 2
        or models.shape[1] != observations.shape[1]
    ):
        raise ValueError(
            "model and observed flux arrays have incompatible shapes: "
            f"{models.shape}, {observations.shape}"
        )

    derivatives = np.asarray(nuisance_profile["derivatives"], dtype=np.float64)
    inverse = np.asarray(
        nuisance_profile["normal_matrix_inverse"],
        dtype=np.float64,
    )
    if derivatives.shape[:2] != models.shape or inverse.shape[0] != models.shape[0]:
        raise ValueError("nuisance profile does not match the supplied model library")

    n_models = models.shape[0]
    n_realizations = observations.shape[0]
    n_parameters = derivatives.shape[2]
    chi2_profiled = np.empty((n_models, n_realizations), dtype=np.float64)
    coefficients = (
        np.empty((n_models, n_realizations, n_parameters), dtype=np.float64)
        if return_coefficients
        else None
    )
    model_projection = np.einsum(
        "mtk,mt->mk",
        derivatives,
        models,
        optimize=True,
    )
    sigma_squared = float(sigma_flux_per_cadence) ** 2
    chunk_size = max(1, int(NUISANCE_FIT_CONFIG["realization_chunk_size"]))
    for start in range(0, n_realizations, chunk_size):
        stop = min(start + chunk_size, n_realizations)
        observed_chunk = observations[start:stop]
        base_chi2 = _chi2_from_model_library(
            models,
            observed_chunk,
            sigma_flux_per_cadence,
        )
        right_hand_side = (
            np.einsum(
                "mtk,rt->mrk",
                derivatives,
                observed_chunk,
                optimize=True,
            )
            - model_projection[:, None, :]
        )
        beta = np.einsum(
            "mkl,mrl->mrk",
            inverse,
            right_hand_side,
            optimize=True,
        )
        improvement = np.einsum(
            "mrk,mrk->mr",
            right_hand_side,
            beta,
            optimize=True,
        ) / sigma_squared
        chi2_profiled[:, start:stop] = np.maximum(base_chi2 - improvement, 0.0)
        if coefficients is not None:
            coefficients[:, start:stop, :] = beta
    return chi2_profiled, coefficients


def _nuisance_coefficient(
    nuisance_profile: dict[str, Any],
    coefficients: np.ndarray,
    model_index: int,
    realization_index: int,
    parameter_name: str,
) -> float:
    """Read one fitted nuisance shift, returning zero when that fit is disabled."""
    names = nuisance_profile["derivative_names"]
    if parameter_name not in names:
        return 0.0
    parameter_index = names.index(parameter_name)
    return float(coefficients[model_index, realization_index, parameter_index])


def _calibrate_null_detection_threshold(
    model_fluxes: np.ndarray,
    standard_noise: np.ndarray,
    sigma_flux_per_cadence: float,
    nuisance_profile: dict[str, Any],
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
    chi2_flat, _ = _profiled_chi2_from_model_library(
        model_fluxes,
        observed,
        sigma_flux_per_cadence,
        nuisance_profile,
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
    nuisance_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Fit all common-noise realizations, profiling ordinary parameters each time."""
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
    chi2_flat, nuisance_coefficients = _profiled_chi2_from_model_library(
        model_fluxes,
        observed,
        sigma_flux_per_cadence,
        nuisance_profile,
        return_coefficients=True,
    )
    if nuisance_coefficients is None:
        raise RuntimeError("nuisance coefficients were requested but not returned")
    chi2_cube = chi2_flat.reshape(fs.size, theta_deg.size, standard_noise.shape[0])

    n_data = int(clean_flux.size)
    n_nuisance_parameters = len(nuisance_profile["derivative_names"])
    degrees_of_freedom = max(1, n_data - 2 - n_nuisance_parameters)
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
        best_baseline_shift = _nuisance_coefficient(
            nuisance_profile,
            nuisance_coefficients,
            best_flat_index,
            realization_index,
            "flux_baseline",
        )
        best_radius_shift = _nuisance_coefficient(
            nuisance_profile,
            nuisance_coefficients,
            best_flat_index,
            realization_index,
            "projected_effective_r",
        )
        best_t0_shift_days = _nuisance_coefficient(
            nuisance_profile,
            nuisance_coefficients,
            best_flat_index,
            realization_index,
            "t0",
        )
        spherical_baseline_shift = _nuisance_coefficient(
            nuisance_profile,
            nuisance_coefficients,
            0,
            realization_index,
            "flux_baseline",
        )
        spherical_radius_shift = _nuisance_coefficient(
            nuisance_profile,
            nuisance_coefficients,
            0,
            realization_index,
            "projected_effective_r",
        )
        spherical_t0_shift_days = _nuisance_coefficient(
            nuisance_profile,
            nuisance_coefficients,
            0,
            realization_index,
            "t0",
        )
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
                "nuisance_nominal_projected_effective_r": float(
                    nuisance_profile["nominal_projected_effective_r"]
                ),
                "nuisance_radius_prior_sigma": float(
                    nuisance_profile["radius_prior_sigma"]
                ),
                "nuisance_radius_derivative_step": float(
                    nuisance_profile["radius_derivative_step"]
                ),
                "nuisance_t0_prior_sigma_days": float(
                    nuisance_profile["t0_prior_sigma_days"]
                ),
                "best_flux_baseline_shift": best_baseline_shift,
                "best_projected_effective_r_shift": best_radius_shift,
                "best_projected_effective_r": float(
                    nuisance_profile["nominal_projected_effective_r"]
                )
                + best_radius_shift,
                "best_t0_shift_days": best_t0_shift_days,
                "best_t0_shift_seconds": best_t0_shift_days * 86400.0,
                "spherical_flux_baseline_shift": spherical_baseline_shift,
                "spherical_projected_effective_r_shift": spherical_radius_shift,
                "spherical_projected_effective_r": float(
                    nuisance_profile["nominal_projected_effective_r"]
                )
                + spherical_radius_shift,
                "spherical_t0_shift_days": spherical_t0_shift_days,
                "spherical_t0_shift_seconds": spherical_t0_shift_days * 86400.0,
                "sigma_flux_per_cadence": float(sigma_flux_per_cadence),
                "noise_random_seed": int(base_seed + realization_index),
                "noise_realization_index": int(realization_index),
                "n_data": n_data,
                "degrees_of_freedom": degrees_of_freedom,
                "n_nuisance_parameters": n_nuisance_parameters,
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
            "nuisance_radius_prior_sigma": float(
                rows[0]["nuisance_radius_prior_sigma"]
            ),
            "nuisance_t0_prior_sigma_days": float(
                rows[0]["nuisance_t0_prior_sigma_days"]
            ),
            "v04_results_csv": str(rows[0]["v04_results_csv"]),
            "v04_f50": float(rows[0]["v04_f50"]),
            "v04_f50_ci95_low": float(rows[0]["v04_f50_ci95_low"]),
            "v04_f50_ci95_high": float(rows[0]["v04_f50_ci95_high"]),
            "v04_f50_status": str(rows[0]["v04_f50_status"]),
            "v04_f90": float(rows[0]["v04_f90"]),
            "v04_f90_ci95_low": float(rows[0]["v04_f90_ci95_low"]),
            "v04_f90_ci95_high": float(rows[0]["v04_f90_ci95_high"]),
            "v04_f90_status": str(rows[0]["v04_f90_status"]),
            "validation_stage": str(rows[0]["validation_stage"]),
            "is_auto_extension": bool(rows[0]["is_auto_extension"]),
            "auto_extension_round": int(rows[0]["auto_extension_round"]),
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


def _interpolate_probability_crossing(
    f_values: np.ndarray,
    probabilities: np.ndarray,
    probability_level: float,
) -> dict[str, Any]:
    """Locate the first interpolated crossing inside the v05 validation window."""
    passing_indices = np.flatnonzero(probabilities >= probability_level)
    if passing_indices.size == 0:
        return {
            "f_detection_limit": float("nan"),
            "detection_probability_at_limit": float("nan"),
            "previous_f_inj": float(f_values[-1]),
            "previous_detection_probability": float(probabilities[-1]),
            "next_f_inj": float("nan"),
            "next_detection_probability": float("nan"),
            "limit_status": "not_reached_in_validation_window",
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
            "limit_status": "below_or_at_validation_min",
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
        "limit_status": "interpolated_in_validation_window",
    }


def _derive_detection_limits(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive profiled f50/f90 near v04 and compare the two stages."""
    probability_levels = tuple(
        float(value) for value in V04_BOUNDARY_CONFIG["probability_levels"]
    )
    confidence_z = float(V04_BOUNDARY_CONFIG["detection_limit_ci_z"])
    confidence_level = float(
        V04_BOUNDARY_CONFIG["detection_limit_confidence_level"]
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
        f_values = np.array(
            [float(row["f_inj"]) for row in rows],
            dtype=np.float64,
        )
        probabilities = np.array(
            [float(row["detection_probability"]) for row in rows],
            dtype=np.float64,
        )
        weights = np.array(
            [float(row["n_noise_realizations"]) for row in rows],
            dtype=np.float64,
        )
        fitted_probabilities = _weighted_isotonic_non_decreasing(
            probabilities,
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
            level_label = int(round(100.0 * probability_level))
            v04_limit = float(rows[0][f"v04_f{level_label}"])
            if not np.isfinite(v04_limit):
                limits.append(
                    {
                        "pl_name": rows[0]["pl_name"],
                        "theta_inj_deg": theta_deg,
                        "probability_level": probability_level,
                        "f_detection_limit": float("nan"),
                        "f_detection_limit_ci95_low": float("nan"),
                        "f_detection_limit_ci95_high": float("nan"),
                        "f_detection_limit_ci95_low_status": "not_validated",
                        "f_detection_limit_ci95_high_status": "not_validated",
                        "confidence_level": confidence_level,
                        "confidence_method": "inverted_wilson_isotonic",
                        "detection_probability_at_limit": float("nan"),
                        "previous_f_inj": float("nan"),
                        "previous_detection_probability": float("nan"),
                        "next_f_inj": float("nan"),
                        "next_detection_probability": float("nan"),
                        "limit_status": "v04_unresolved_not_validated",
                        "validation_f_min": float(f_values[0]),
                        "validation_f_max": float(f_values[-1]),
                        "v04_f_detection_limit": float("nan"),
                        "v04_f_detection_limit_ci95_low": float(
                            rows[0][f"v04_f{level_label}_ci95_low"]
                        ),
                        "v04_f_detection_limit_ci95_high": float(
                            rows[0][f"v04_f{level_label}_ci95_high"]
                        ),
                        "v04_limit_status": str(
                            rows[0][f"v04_f{level_label}_status"]
                        ),
                        "f_detection_limit_shift_from_v04": float("nan"),
                        "v04_results_csv": str(rows[0]["v04_results_csv"]),
                    }
                )
                continue
            estimate = _interpolate_probability_crossing(
                f_values,
                fitted_probabilities,
                probability_level,
            )
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
            v05_limit = float(estimate["f_detection_limit"])
            shift = (
                v05_limit - v04_limit
                if np.isfinite(v05_limit) and np.isfinite(v04_limit)
                else float("nan")
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
                    "validation_f_min": float(f_values[0]),
                    "validation_f_max": float(f_values[-1]),
                    "v04_f_detection_limit": v04_limit,
                    "v04_f_detection_limit_ci95_low": float(
                        rows[0][f"v04_f{level_label}_ci95_low"]
                    ),
                    "v04_f_detection_limit_ci95_high": float(
                        rows[0][f"v04_f{level_label}_ci95_high"]
                    ),
                    "v04_limit_status": str(
                        rows[0][f"v04_f{level_label}_status"]
                    ),
                    "f_detection_limit_shift_from_v04": shift,
                    "v04_results_csv": str(rows[0]["v04_results_csv"]),
                }
            )
    return limits


def _plan_high_f_extension(
    detection_limits: list[dict[str, Any]],
    evaluated_pairs: set[tuple[float, float]],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return new high-f points and the ``(theta, probability)`` rows that need them."""
    cfg = V04_BOUNDARY_CONFIG
    if not bool(cfg["auto_extend_high_f"]):
        return [], []
    step = float(cfg["extension_f_step"])
    points_per_round = int(cfg["extension_points_per_round"])
    f_max = float(cfg["f_max"])
    ndigits = int(cfg["dedupe_ndigits"])
    if step <= 0.0 or points_per_round <= 0:
        raise ValueError(
            "v05 high-f extension requires a positive step and points_per_round: "
            f"{step}, {points_per_round}"
        )

    unresolved_rows: list[tuple[float, float]] = []
    theta_maxima: dict[float, float] = {}
    for row in detection_limits:
        if row["limit_status"] == "v04_unresolved_not_validated":
            continue
        center_unresolved = (
            row["limit_status"] == "not_reached_in_validation_window"
        )
        upper_ci_unresolved = (
            row["f_detection_limit_ci95_high_status"]
            == "not_reached_in_validation_window"
        )
        if not center_unresolved and not upper_ci_unresolved:
            continue
        theta_deg = round(float(row["theta_inj_deg"]), 8)
        probability_level = float(row["probability_level"])
        unresolved_rows.append((theta_deg, probability_level))
        theta_maxima[theta_deg] = max(
            theta_maxima.get(theta_deg, float("-inf")),
            float(row["validation_f_max"]),
        )

    new_pairs: set[tuple[float, float]] = set()
    for theta_deg, current_max in theta_maxima.items():
        for point_index in range(1, points_per_round + 1):
            candidate_f = current_max + point_index * step
            if candidate_f > f_max + 1.0e-12:
                continue
            pair = (
                round(min(candidate_f, f_max), ndigits),
                theta_deg,
            )
            if pair not in evaluated_pairs:
                new_pairs.add(pair)
    return (
        sorted(new_pairs, key=lambda pair: (pair[1], pair[0])),
        sorted(set(unresolved_rows)),
    )


def run_batch(
    planet_cfg: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Read v04 boundaries, calibrate the profiled null, and validate those boundaries."""
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
    v04_results_path = _configured_v04_results_path(pl_name)
    v04_boundary_rows = _load_v04_boundary_rows(v04_results_path, pl_name)
    pairs, v04_references, boundary_diagnostics = _build_v04_validation_pairs(
        v04_boundary_rows
    )
    max_n = BATCH_RUN_CONFIG.get("max_injections")
    if max_n is not None:
        pairs = pairs[: int(max_n)]
    if not pairs:
        raise ValueError("v05 validation-point list is empty after max_injections")
    print(
        "[v04 boundary input] "
        f"path={v04_results_path}; "
        f"rows={boundary_diagnostics['n_v04_boundary_rows']}, "
        f"theta={boundary_diagnostics['n_validation_theta']}, "
        f"unresolved rows skipped={boundary_diagnostics['n_v04_unresolved_rows']}, "
        f"validation points={len(pairs)}"
    )

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

    system, system_name, model_times = _build_system(
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
    nuisance_profile = _precompute_nuisance_profile(
        system,
        model_fluxes,
        fs_model,
        theta_model_deg,
        model_times,
        input_row,
        sigma_flux_per_cadence,
        cadence_seconds,
    )
    print(
        "[nuisance profile] "
        f"parameters={', '.join(nuisance_profile['derivative_names'])}; "
        f"sigma(Rp/R*)={float(nuisance_profile['radius_prior_sigma']):.6g}, "
        f"sigma(t0)={86400.0 * float(nuisance_profile['t0_prior_sigma_days']):.3f} s"
    )

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
        nuisance_profile,
    )
    null_calibration["null_random_seed"] = null_seed
    null_calibration["pl_name"] = pl_name
    null_calibration["row_index"] = row_index
    null_calibration["sigma_frac_ppm_full_transit"] = sigma_frac_ppm_full_transit
    null_calibration["noise_model_kmag"] = noise_model_kmag
    null_calibration["sigma_frac_ppm_one_minute"] = sigma_frac_ppm_one_minute
    null_calibration["sigma_frac_ppm_per_cadence"] = sigma_frac_ppm_per_cadence
    null_calibration["nuisance_radius_prior_sigma"] = float(
        nuisance_profile["radius_prior_sigma"]
    )
    null_calibration["nuisance_t0_prior_sigma_days"] = float(
        nuisance_profile["t0_prior_sigma_days"]
    )
    null_calibration["nuisance_radius_derivative_step"] = float(
        nuisance_profile["radius_derivative_step"]
    )
    null_calibration["v04_results_csv"] = str(v04_results_path)
    null_calibration["n_v04_boundary_rows"] = int(
        boundary_diagnostics["n_v04_boundary_rows"]
    )
    null_calibration["n_v04_unresolved_rows"] = int(
        boundary_diagnostics["n_v04_unresolved_rows"]
    )
    null_calibration["n_validation_points"] = len(pairs)
    print(
        "[null calibration] "
        f"delta_chi2_critical={float(null_calibration['delta_chi2_threshold']):.4f}, "
        f"empirical false-alarm probability="
        f"{float(null_calibration['false_alarm_probability_empirical']):.2%}"
    )

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

    def evaluate_pairs(
        batch_pairs: list[tuple[float, float]],
        validation_stage: str,
        extension_round: int,
    ) -> dict[tuple[float, float], list[dict[str, Any]]]:
        evaluated: dict[tuple[float, float], list[dict[str, Any]]] = {}
        iterator: Any = enumerate(batch_pairs)
        if tqdm is not None:
            iterator = tqdm(
                iterator,
                total=len(batch_pairs),
                desc=(
                    f"{pl_name} {validation_stage} "
                    f"x {n_realizations} noises"
                ),
                unit="injection",
            )
        for temporary_index, (f_inj, theta_inj_deg) in iterator:
            reference = v04_references[round(float(theta_inj_deg), 8)]
            injection_results = _run_injection_realizations(
                system,
                model_fluxes,
                fs_model,
                theta_model_deg,
                standard_noise,
                f_inj,
                theta_inj_deg,
                sigma_flux_per_cadence,
                base_seed,
                float(null_calibration["delta_chi2_threshold"]),
                nuisance_profile,
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
                result["injection_index"] = int(temporary_index)
                result["v04_results_csv"] = str(v04_results_path)
                result["validation_stage"] = validation_stage
                result["is_auto_extension"] = bool(extension_round > 0)
                result["auto_extension_round"] = int(extension_round)
                result.update(reference)
            pair_key = (
                round(float(f_inj), int(V04_BOUNDARY_CONFIG["dedupe_ndigits"])),
                round(float(theta_inj_deg), 8),
            )
            evaluated[pair_key] = injection_results
        return evaluated

    def flatten_evaluated(
        evaluated: dict[tuple[float, float], list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        flattened: list[dict[str, Any]] = []
        for injection_index, pair in enumerate(
            sorted(evaluated, key=lambda item: (item[1], item[0]))
        ):
            pair_results = evaluated[pair]
            for result in pair_results:
                result["injection_index"] = int(injection_index)
            flattened.extend(pair_results)
        return flattened

    evaluated_by_pair = evaluate_pairs(
        pairs,
        "initial_v04_boundary_window",
        0,
    )
    initial_point_count = len(evaluated_by_pair)
    max_extension_rounds = int(V04_BOUNDARY_CONFIG["max_extension_rounds"])
    if max_extension_rounds < 0:
        raise ValueError(
            f"max_extension_rounds cannot be negative: {max_extension_rounds}"
        )
    completed_extension_rounds = 0
    last_unresolved_rows: list[tuple[float, float]] = []
    for extension_round in range(1, max_extension_rounds + 1):
        current_realizations = flatten_evaluated(evaluated_by_pair)
        current_summaries = _aggregate_realizations(current_realizations)
        current_limits = _derive_detection_limits(current_summaries)
        extension_pairs, unresolved_rows = _plan_high_f_extension(
            current_limits,
            set(evaluated_by_pair),
        )
        last_unresolved_rows = unresolved_rows
        if not unresolved_rows:
            print(
                "[v05 auto-extension] all requested f50/f90 upper bounds "
                "closed inside the validation window"
            )
            break
        if not extension_pairs:
            print(
                "[v05 auto-extension] unresolved boundaries remain, but no "
                "new point can be added before f_max="
                f"{float(V04_BOUNDARY_CONFIG['f_max']):.4f}"
            )
            break
        trigger_text = ", ".join(
            f"(theta={theta:g}, P={probability:.0%})"
            for theta, probability in unresolved_rows
        )
        print(
            f"[v05 auto-extension] round {extension_round}: "
            f"adding {len(extension_pairs)} high-f points for {trigger_text}"
        )
        evaluated_by_pair.update(
            evaluate_pairs(
                extension_pairs,
                f"auto_extend_high_f_round_{extension_round}",
                extension_round,
            )
        )
        completed_extension_rounds = extension_round

    realizations = flatten_evaluated(evaluated_by_pair)
    summaries = _aggregate_realizations(realizations)
    final_limits = _derive_detection_limits(summaries)
    _, final_unresolved_rows = _plan_high_f_extension(
        final_limits,
        set(evaluated_by_pair),
    )
    if not bool(V04_BOUNDARY_CONFIG["auto_extend_high_f"]):
        final_unresolved_rows = []

    null_calibration["n_initial_validation_points"] = initial_point_count
    null_calibration["n_auto_extension_points"] = (
        len(evaluated_by_pair) - initial_point_count
    )
    null_calibration["n_validation_points"] = len(evaluated_by_pair)
    null_calibration["auto_extension_rounds_completed"] = (
        completed_extension_rounds
    )
    null_calibration["n_unresolved_boundaries_after_extension"] = len(
        final_unresolved_rows
    )
    null_calibration["auto_extension_last_trigger_count"] = len(
        last_unresolved_rows
    )
    return realizations, summaries, null_calibration


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
    has_overlay = False
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
        theta_values = np.array(
            [float(row["theta_inj_deg"]) for row in level_rows],
            dtype=np.float64,
        )
        v04_values = np.array(
            [float(row["v04_f_detection_limit"]) for row in level_rows],
            dtype=np.float64,
        )
        valid_v04 = np.isfinite(v04_values)
        if np.count_nonzero(valid_v04) >= 1:
            has_overlay = True
            ax.plot(
                v04_values[valid_v04],
                theta_values[valid_v04],
                color=color,
                linewidth=1.0,
                linestyle=":",
                alpha=0.75,
                label=fr"v04 $f_{{{100.0 * level:.0f}}}(\theta)$",
                zorder=4,
            )

        ci_low = np.array(
            [float(row["f_detection_limit_ci95_low"]) for row in level_rows],
            dtype=np.float64,
        )
        ci_high = np.array(
            [float(row["f_detection_limit_ci95_high"]) for row in level_rows],
            dtype=np.float64,
        )
        finite_band = np.isfinite(ci_low) & np.isfinite(ci_high)
        if np.count_nonzero(finite_band) >= 2:
            has_overlay = True
            ax.fill_betweenx(
                theta_values,
                ci_low,
                ci_high,
                where=finite_band,
                color=color,
                alpha=0.30,
                linewidth=0.0,
                label=fr"v05 $f_{{{100.0 * level:.0f}}}$ 95% CI",
                zorder=4,
            )

        valid_v05 = [
            row
            for row in level_rows
            if np.isfinite(float(row["f_detection_limit"]))
        ]
        if valid_v05:
            has_overlay = True
            ax.plot(
                [float(row["f_detection_limit"]) for row in valid_v05],
                [float(row["theta_inj_deg"]) for row in valid_v05],
                color=color,
                linewidth=1.3,
                linestyle=linestyle,
                marker="o",
                markersize=4.0,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.8,
                label=fr"v05 profiled $f_{{{100.0 * level:.0f}}}(\theta)$",
                zorder=5,
            )

        unresolved = [
            row
            for row in level_rows
            if row["limit_status"] == "not_reached_in_validation_window"
        ]
        if unresolved:
            has_overlay = True
            ax.scatter(
                [float(row["validation_f_max"]) for row in unresolved],
                [float(row["theta_inj_deg"]) for row in unresolved],
                marker=">",
                s=45,
                facecolors="white",
                edgecolor=color,
                linewidth=0.8,
                label=fr"v05 $f_{{{100.0 * level:.0f}}}$ above validation window",
                zorder=5,
            )

    title = summaries[0]["pl_name"]
    ax.set_title(
        f"{title} — v05 profiled validation of v04 boundaries; "
        f"{summaries[0]['n_noise_realizations']} noises/point; "
        fr"$\Delta\chi^2_{{\rm crit}}={summaries[0]['detection_delta_chi2_threshold']:.2f}$"
    )
    ax.set_xlabel(r"injected $f_{\mathrm{inj}}$")
    ax.set_ylabel(r"injected $\theta_{\mathrm{inj}}$ (deg)")
    ax.set_xlim(float(np.min(x)) - 0.005, float(np.max(x)) + 0.005)
    ax.set_ylim(float(np.min(y)) - 5.0, float(np.max(y)) + 5.0)
    ax.grid(True, alpha=0.3)
    if has_overlay:
        ax.legend(loc="best", fontsize=8)
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
    "nuisance_nominal_projected_effective_r",
    "nuisance_radius_prior_sigma",
    "nuisance_radius_derivative_step",
    "nuisance_t0_prior_sigma_days",
    "v04_f50",
    "v04_f50_ci95_low",
    "v04_f50_ci95_high",
    "v04_f90",
    "v04_f90_ci95_low",
    "v04_f90_ci95_high",
    "best_flux_baseline_shift",
    "best_projected_effective_r_shift",
    "best_projected_effective_r",
    "best_t0_shift_days",
    "best_t0_shift_seconds",
    "spherical_flux_baseline_shift",
    "spherical_projected_effective_r_shift",
    "spherical_projected_effective_r",
    "spherical_t0_shift_days",
    "spherical_t0_shift_seconds",
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
    "auto_extension_round",
    "injection_index",
    "n_data",
    "degrees_of_freedom",
    "n_nuisance_parameters",
    "n_chi2_evaluated",
    "n_chi2_unique",
]

RESULT_BOOLEAN_KEYS = [
    "is_detected",
    "truth_better_than_grid",
    "f_ci95_touches_search_boundary",
    "is_closest_recovery",
    "is_auto_extension",
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
    out["v04_f50_status"] = np.array(
        [r["v04_f50_status"] for r in results],
        dtype=object,
    )
    out["v04_f90_status"] = np.array(
        [r["v04_f90_status"] for r in results],
        dtype=object,
    )
    out["validation_stage"] = np.array(
        [r["validation_stage"] for r in results],
        dtype=object,
    )
    out["pl_name"] = results[0]["pl_name"] if results else ""
    if results and "row_index" in results[0]:
        out["row_index"] = int(results[0]["row_index"])
        out["noise_cache_target"] = results[0]["noise_cache_target"]
        out["noise_cache_row_index"] = int(results[0]["noise_cache_row_index"])
        out["v04_results_csv"] = results[0]["v04_results_csv"]
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
        "v04_results_csv",
        "v04_f50_status",
        "v04_f90_status",
        "validation_stage",
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
    "nuisance_radius_prior_sigma",
    "nuisance_t0_prior_sigma_days",
    "v04_results_csv",
    "v04_f50",
    "v04_f50_ci95_low",
    "v04_f50_ci95_high",
    "v04_f50_status",
    "v04_f90",
    "v04_f90_ci95_low",
    "v04_f90_ci95_high",
    "v04_f90_status",
    "validation_stage",
    "is_auto_extension",
    "auto_extension_round",
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
    "validation_f_min",
    "validation_f_max",
    "v04_f_detection_limit",
    "v04_f_detection_limit_ci95_low",
    "v04_f_detection_limit_ci95_high",
    "v04_limit_status",
    "f_detection_limit_shift_from_v04",
    "v04_results_csv",
    "limit_status",
]


def _save_detection_limits_csv(
    detection_limits: list[dict[str, Any]],
    csv_path: Path | str | None = None,
) -> Path:
    """Save profiled v05 f50/f90 and their shifts relative to v04."""
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
        nuisance_radius_prior_sigma=float(
            null_calibration["nuisance_radius_prior_sigma"]
        ),
        nuisance_t0_prior_sigma_days=float(
            null_calibration["nuisance_t0_prior_sigma_days"]
        ),
        nuisance_radius_derivative_step=float(
            null_calibration["nuisance_radius_derivative_step"]
        ),
        v04_results_csv=str(null_calibration["v04_results_csv"]),
        n_v04_boundary_rows=int(null_calibration["n_v04_boundary_rows"]),
        n_v04_unresolved_rows=int(null_calibration["n_v04_unresolved_rows"]),
        n_validation_points=int(null_calibration["n_validation_points"]),
        n_initial_validation_points=int(
            null_calibration["n_initial_validation_points"]
        ),
        n_auto_extension_points=int(
            null_calibration["n_auto_extension_points"]
        ),
        auto_extension_rounds_completed=int(
            null_calibration["auto_extension_rounds_completed"]
        ),
        n_unresolved_boundaries_after_extension=int(
            null_calibration["n_unresolved_boundaries_after_extension"]
        ),
    )
    return path


def main() -> None:
    realizations, summaries, null_calibration = run_batch(None)
    detection_limits = _derive_detection_limits(summaries)
    print("n_v04_boundary_validation_points:", len(summaries))
    print("n_realization_results:", len(realizations))
    if not realizations:
        return
    print("planet:", realizations[0]["pl_name"])
    print("v04 boundary source:", realizations[0]["v04_results_csv"])
    print(
        "v05 validation sampling:",
        f"initial points={null_calibration['n_initial_validation_points']},",
        f"auto-extension points={null_calibration['n_auto_extension_points']},",
        f"extension rounds={null_calibration['auto_extension_rounds_completed']},",
        f"unresolved boundaries="
        f"{null_calibration['n_unresolved_boundaries_after_extension']}",
    )
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
    for probability_level in V04_BOUNDARY_CONFIG["probability_levels"]:
        print(f"profiled f{100.0 * float(probability_level):.0f}(theta):")
        for row in detection_limits:
            if not np.isclose(
                float(row["probability_level"]),
                float(probability_level),
            ):
                continue
            v04_limit = float(row["v04_f_detection_limit"])
            v05_limit = float(row["f_detection_limit"])
            if row["limit_status"] == "v04_unresolved_not_validated":
                print(
                    f"  theta={row['theta_inj_deg']:.1f} deg: "
                    "v04 boundary unresolved; v05 skipped this level"
                )
                continue
            if np.isfinite(v05_limit):
                ci_low = float(row["f_detection_limit_ci95_low"])
                ci_high = float(row["f_detection_limit_ci95_high"])
                if np.isfinite(ci_low) and np.isfinite(ci_high):
                    ci_text = f"[{ci_low:.4f}, {ci_high:.4f}]"
                elif np.isfinite(ci_low):
                    ci_text = f"[{ci_low:.4f}, above validation window]"
                else:
                    ci_text = "unresolved"
                shift = float(row["f_detection_limit_shift_from_v04"])
                print(
                    f"  theta={row['theta_inj_deg']:.1f} deg: "
                    f"v04={v04_limit:.4f}, v05={v05_limit:.4f}, "
                    f"shift={shift:+.4f}, 95% CI={ci_text} "
                    f"({row['limit_status']})"
                )
            else:
                print(
                    f"  theta={row['theta_inj_deg']:.1f} deg: "
                    f"v04={v04_limit:.4f}, v05 not reached in "
                    f"[{row['validation_f_min']:.4f}, "
                    f"{row['validation_f_max']:.4f}]"
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
