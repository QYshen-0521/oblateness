"""Run the v04 noisy completeness pipeline for multiple planet-table rows.

Each selected row is passed to
``noised_grid_recovery_single_planet_batch_v04.run_batch``. Results are stored
in one target directory per planet.

Run from the repository root::

    .venv/bin/python -m oblateness.noised_grid_recovery_multi_planet_batch \
        --rows 0,1,2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import oblateness.noised_grid_recovery_single_planet_batch_v04 as v04
from oblateness.noised_grid_recovery_single_planet_batch_v04 import (
    BATCH_RUN_CONFIG,
    _derive_detection_limits,
    _save_detection_figure,
    _save_detection_limits_csv,
    _save_null_calibration_npz,
    _save_npz,
    _save_realizations_csv,
    _save_summary_csv,
    _target_name_key,
    _target_slug,
    run_batch,
    planet_config_for_row,
)
from oblateness.noised_grid_recovery_single_injection import _repo_root

# =============================================================================
# 默认要跑的系统（CSV 行号，0 = 首条数据行）；可用 CLI / 环境变量覆盖
# =============================================================================
MULTI_SYSTEM_CONFIG: dict[str, Any] = {
    "row_indices": None,
    "output_subdir": "batch_noised_completeness_v04_adaptive_ie60",
    "summary_csv": "multi_system_summary.csv",
    "completion_marker": "run_complete.json",
    "failure_subdir": "_failures",
}

SUMMARY_FIELDS = [
    "row_index",
    "pl_name",
    "status",
    "n_injection_grid_points",
    "n_realization_results",
    "delta_chi2_threshold",
    "figure_path",
    "npz_path",
    "realizations_csv_path",
    "summary_csv_path",
    "detection_limits_csv_path",
    "null_calibration_npz_path",
    "completion_marker_path",
    "config_fingerprint",
    "error",
]


def _path_for_summary(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_planet_rows(root: Path) -> tuple[Path, list[dict[str, str]]]:
    path = root / str(v04.BASE_PLANET_CONFIG["csv_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"planet input table is empty: {path}")
    if "pl_name" not in rows[0]:
        raise KeyError(f"planet input table has no pl_name column: {path}")
    return path, rows


def _load_noise_rows(root: Path) -> tuple[Path, dict[str, dict[str, str]]]:
    path = root / str(v04.NOISE_CONFIG["cache_csv_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    by_target: dict[str, dict[str, str]] = {}
    for row in rows:
        key = _target_name_key(row.get("target", ""))
        if not key:
            continue
        previous = by_target.get(key)
        if previous is None or (
            previous.get("status", "").strip().casefold() != "success"
            and row.get("status", "").strip().casefold() == "success"
        ):
            by_target[key] = row
    return path, by_target


def _v04_config_fingerprint() -> str:
    base_planet_config = {
        key: value
        for key, value in v04.BASE_PLANET_CONFIG.items()
        if key != "target_name"
    }
    payload = {
        "pipeline": "noised_grid_recovery_single_planet_batch_v04",
        "BASE_PLANET_CONFIG": base_planet_config,
        "NOISE_CONFIG": v04.NOISE_CONFIG,
        "INJECTION_GRID_CONFIG": v04.INJECTION_GRID_CONFIG,
        "COARSE_GRID_CONFIG": v04.COARSE_GRID_CONFIG,
        "REFINE_GRID_CONFIG": v04.REFINE_GRID_CONFIG,
        "CONFIDENCE_CONFIG": v04.CONFIDENCE_CONFIG,
        "NOISE_INJECTION_CONFIG": v04.NOISE_INJECTION_CONFIG,
        "DETECTION_CONFIG": v04.DETECTION_CONFIG,
        "ADAPTIVE_COMPLETENESS_CONFIG": v04.ADAPTIVE_COMPLETENESS_CONFIG,
        "BATCH_RUN_CONFIG": v04.BATCH_RUN_CONFIG,
        "MODEL_GRID_CONFIG": v04.MODEL_GRID_CONFIG,
        "PROJECTED_THETA_SINGULARITY_OFFSET_DEG": (
            v04.PROJECTED_THETA_SINGULARITY_OFFSET_DEG
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds_part:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds_part:02d}s"
    return f"{seconds_part:d}s"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _load_summary_rows(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    loaded: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            loaded[int(row["row_index"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return loaded


def _write_summary_rows(
    path: Path,
    rows_by_index: dict[int, dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=SUMMARY_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row_index in sorted(rows_by_index):
            row = rows_by_index[row_index]
            writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})
    temporary_path.replace(path)


def _failure_marker_path(out_dir: Path, row_index: int) -> Path:
    return (
        out_dir
        / str(MULTI_SYSTEM_CONFIG["failure_subdir"])
        / f"row{int(row_index):04d}.json"
    )


def _read_failure_marker(
    path: Path,
    config_fingerprint: str,
) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if (
        marker.get("pipeline")
        != "noised_grid_recovery_single_planet_batch_v04"
        or marker.get("config_fingerprint") != config_fingerprint
    ):
        return {}
    return marker


def _write_failure_marker(
    path: Path,
    row_index: int,
    pl_name: str,
    error: str,
    traceback_text: str,
    status: str,
    config_fingerprint: str,
) -> None:
    _atomic_write_json(
        path,
        {
            "schema_version": 1,
            "pipeline": "noised_grid_recovery_single_planet_batch_v04",
            "row_index": int(row_index),
            "pl_name": str(pl_name),
            "status": str(status),
            "config_fingerprint": str(config_fingerprint),
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": str(error),
            "traceback": str(traceback_text),
        },
    )


def _target_output_paths(out_dir: Path, pl_name: str) -> dict[str, Path]:
    target_dir = out_dir / _target_slug(pl_name)
    return {
        "target_dir": target_dir,
        "figure": target_dir / "detection_completeness_map.png",
        "npz": target_dir / "detection_realizations.npz",
        "realizations_csv": target_dir / "detection_realizations.csv",
        "summary_csv": target_dir / "detection_probability_summary.csv",
        "detection_limits_csv": target_dir / "f50_f90_curves.csv",
        "null_calibration_npz": target_dir / "null_calibration.npz",
        "completion_marker": target_dir
        / str(MULTI_SYSTEM_CONFIG["completion_marker"]),
    }


def _required_output_keys() -> list[str]:
    keys = ["figure"]
    if BATCH_RUN_CONFIG.get("save_results_npz", True):
        keys.append("npz")
    if BATCH_RUN_CONFIG.get("save_realizations_csv", True):
        keys.append("realizations_csv")
    if BATCH_RUN_CONFIG.get("save_summary_csv", True):
        keys.append("summary_csv")
    if BATCH_RUN_CONFIG.get("save_detection_limits_csv", True):
        keys.append("detection_limits_csv")
    if BATCH_RUN_CONFIG.get("save_null_calibration_npz", True):
        keys.append("null_calibration_npz")
    return keys


def _first_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV has no data rows: {path}") from exc


def _validate_target_outputs(
    paths: dict[str, Path],
    expected_name: str,
    expected_row_index: int,
) -> tuple[bool, str]:
    for key in _required_output_keys():
        path = paths[key]
        if not path.is_file() or path.stat().st_size <= 0:
            return False, f"missing or empty {key}: {path}"

    try:
        if paths["figure"].read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            return False, f"invalid PNG header: {paths['figure']}"

        for key in ("realizations_csv", "summary_csv"):
            if key not in _required_output_keys():
                continue
            row = _first_csv_row(paths[key])
            if row.get("pl_name", "").strip() != expected_name:
                return False, f"{key} target mismatch"
            if int(row["row_index"]) != expected_row_index:
                return False, f"{key} row_index mismatch"

        if "detection_limits_csv" in _required_output_keys():
            row = _first_csv_row(paths["detection_limits_csv"])
            if row.get("pl_name", "").strip() != expected_name:
                return False, "detection_limits_csv target mismatch"

        for key in ("npz", "null_calibration_npz"):
            if key not in _required_output_keys():
                continue
            with np.load(paths[key], allow_pickle=True) as archive:
                stored_name = str(np.asarray(archive["pl_name"]).item())
                stored_row_index = int(np.asarray(archive["row_index"]).item())
                if key == "null_calibration_npz":
                    stored_n_null = int(
                        np.asarray(archive["n_null_realizations"]).item()
                    )
                    stored_pilot_n = int(
                        np.asarray(
                            archive["pilot_n_noise_realizations"]
                        ).item()
                    )
                    stored_final_n = int(
                        np.asarray(
                            archive["final_n_noise_realizations"]
                        ).item()
                    )
            if stored_name != expected_name:
                return False, f"{key} target mismatch"
            if stored_row_index != expected_row_index:
                return False, f"{key} row_index mismatch"
            if key == "null_calibration_npz":
                expected_n_null = int(
                    v04.DETECTION_CONFIG["n_null_realizations"]
                )
                expected_pilot_n = int(
                    v04.ADAPTIVE_COMPLETENESS_CONFIG[
                        "pilot_n_noise_realizations"
                    ]
                )
                expected_final_n = int(
                    v04.ADAPTIVE_COMPLETENESS_CONFIG[
                        "final_n_noise_realizations"
                    ]
                )
                stored_counts = (
                    stored_n_null,
                    stored_pilot_n,
                    stored_final_n,
                )
                expected_counts = (
                    expected_n_null,
                    expected_pilot_n,
                    expected_final_n,
                )
                if stored_counts != expected_counts:
                    return (
                        False,
                        "null/pilot/final realization counts mismatch: "
                        f"stored={stored_counts}, expected={expected_counts}",
                    )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return False, f"{type(exc).__name__}: {exc}"

    return True, "complete"


def _write_completion_marker(
    paths: dict[str, Path],
    row_index: int,
    pl_name: str,
    n_injection_grid_points: int | None,
    n_realization_results: int | None,
    delta_chi2_threshold: float | None,
    adopted_existing: bool,
    config_fingerprint: str,
) -> None:
    marker = {
        "schema_version": 1,
        "pipeline": "noised_grid_recovery_single_planet_batch_v04",
        "row_index": int(row_index),
        "pl_name": str(pl_name),
        "config_fingerprint": str(config_fingerprint),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "adopted_existing_outputs": bool(adopted_existing),
        "n_injection_grid_points": n_injection_grid_points,
        "n_realization_results": n_realization_results,
        "delta_chi2_threshold": delta_chi2_threshold,
        "required_outputs": [
            paths[key].name for key in _required_output_keys()
        ],
    }
    marker_path = paths["completion_marker"]
    _atomic_write_json(marker_path, marker)


def _resume_if_complete(
    paths: dict[str, Path],
    row_index: int,
    pl_name: str,
    config_fingerprint: str,
) -> tuple[bool, dict[str, Any]]:
    valid, _ = _validate_target_outputs(paths, pl_name, row_index)
    if not valid:
        return False, {}

    marker: dict[str, Any] = {}
    marker_path = paths["completion_marker"]
    marker_exists = marker_path.is_file()
    if marker_exists:
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            marker = {}
    marker_matches = (
        marker.get("pipeline")
        == "noised_grid_recovery_single_planet_batch_v04"
        and int(marker.get("row_index", -1)) == row_index
        and str(marker.get("pl_name", "")) == pl_name
        and marker.get("config_fingerprint") == config_fingerprint
    )
    if marker_exists and not marker_matches:
        return False, {}
    if not marker_exists:
        _write_completion_marker(
            paths,
            row_index,
            pl_name,
            None,
            None,
            None,
            adopted_existing=True,
            config_fingerprint=config_fingerprint,
        )
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    return True, marker


def run_multi_system(
    row_indices: list[int],
    output_subdir: str | None = None,
    output_dir: Path | str | None = None,
    write_summary: bool = True,
    resume: bool = True,
    retry_failed: bool = False,
) -> list[dict[str, Any]]:
    """
    Run the v04 completeness pipeline for every selected table row.

    ``output_dir`` is the root containing one subdirectory per target.
    """
    root = _repo_root()
    if output_dir is not None:
        out_dir = Path(output_dir).expanduser().resolve()
    else:
        sub = output_subdir or str(MULTI_SYSTEM_CONFIG["output_subdir"])
        out_dir = (root / "results" / sub).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    planet_table_path, input_rows = _load_planet_rows(root)
    noise_cache_path, noise_rows_by_target = _load_noise_rows(root)
    config_fingerprint = _v04_config_fingerprint()
    total_targets = len(row_indices)
    n_success_noise_targets = sum(
        row.get("status", "").strip().casefold() == "success"
        for row in noise_rows_by_target.values()
    )
    print(f"planet table: {planet_table_path} ({len(input_rows)} rows)")
    print(
        f"noise cache: {noise_cache_path} "
        f"({n_success_noise_targets} successful targets)"
    )
    print(f"v04 config fingerprint: {config_fingerprint}")

    summary_name = str(MULTI_SYSTEM_CONFIG.get("summary_csv", "multi_system_summary.csv"))
    summary_path = out_dir / summary_name
    summary_rows_by_index = (
        _load_summary_rows(summary_path) if write_summary else {}
    )
    summary_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    t0 = time.perf_counter()

    def record_progress(
        record: dict[str, Any],
        target_number: int,
    ) -> None:
        summary_rows.append(record)
        if write_summary:
            summary_rows_by_index[int(record["row_index"])] = record
            _write_summary_rows(summary_path, summary_rows_by_index)
        status_counts[str(record["status"])] += 1
        elapsed = time.perf_counter() - t0
        average_seconds = elapsed / max(target_number, 1)
        eta_seconds = average_seconds * max(total_targets - target_number, 0)
        counts_text = ", ".join(
            f"{status}={count}"
            for status, count in sorted(status_counts.items())
        )
        print(
            f"[overall {target_number}/{total_targets}] "
            f"elapsed={_format_duration(elapsed)}, "
            f"ETA~{_format_duration(eta_seconds)}; {counts_text}"
        )

    for target_number, ri in enumerate(row_indices, start=1):
        failure_path = _failure_marker_path(out_dir, ri)
        rec: dict[str, Any] = {
            "row_index": ri,
            "pl_name": "",
            "status": "",
            "n_injection_grid_points": "",
            "n_realization_results": "",
            "delta_chi2_threshold": "",
            "figure_path": "",
            "npz_path": "",
            "realizations_csv_path": "",
            "summary_csv_path": "",
            "detection_limits_csv_path": "",
            "null_calibration_npz_path": "",
            "completion_marker_path": "",
            "config_fingerprint": config_fingerprint,
            "error": "",
        }
        try:
            if ri < 0 or ri >= len(input_rows):
                raise IndexError(
                    f"row_index {ri} is outside input table range "
                    f"0..{len(input_rows) - 1}"
                )
            pc = planet_config_for_row(ri)
            input_row = input_rows[ri]
            expected_name = input_row["pl_name"].strip()
            print(
                f"[target {target_number}/{total_targets}] "
                f"row_index={ri}, target={expected_name}"
            )
            paths = _target_output_paths(out_dir, expected_name)
            rec["pl_name"] = expected_name
            rec["figure_path"] = _path_for_summary(paths["figure"], root)
            rec["npz_path"] = _path_for_summary(paths["npz"], root)
            rec["realizations_csv_path"] = _path_for_summary(
                paths["realizations_csv"],
                root,
            )
            rec["summary_csv_path"] = _path_for_summary(
                paths["summary_csv"],
                root,
            )
            rec["detection_limits_csv_path"] = _path_for_summary(
                paths["detection_limits_csv"],
                root,
            )
            rec["null_calibration_npz_path"] = _path_for_summary(
                paths["null_calibration_npz"],
                root,
            )
            rec["completion_marker_path"] = _path_for_summary(
                paths["completion_marker"],
                root,
            )

            already_complete, marker = (
                _resume_if_complete(
                    paths,
                    ri,
                    expected_name,
                    config_fingerprint,
                )
                if resume
                else (False, {})
            )
            if already_complete:
                rec["status"] = "skipped_complete"
                rec["n_injection_grid_points"] = (
                    marker.get("n_injection_grid_points") or ""
                )
                rec["n_realization_results"] = (
                    marker.get("n_realization_results") or ""
                )
                rec["delta_chi2_threshold"] = (
                    marker.get("delta_chi2_threshold") or ""
                )
                print(
                    f"[resume] row_index={ri}, target={expected_name}: "
                    "complete, skipped"
                )
                failure_path.unlink(missing_ok=True)
                record_progress(rec, target_number)
                continue

            noise_row = noise_rows_by_target.get(
                _target_name_key(expected_name)
            )
            noise_status = (
                noise_row.get("status", "").strip().casefold()
                if noise_row is not None
                else ""
            )
            if noise_status != "success":
                if noise_row is None:
                    reason = "target is absent from the calculate-noise cache"
                else:
                    detail = noise_row.get("error_message", "").strip()
                    reason = (
                        f"calculate-noise cache status is "
                        f"{noise_row.get('status', '')!r}"
                    )
                    if detail:
                        reason = f"{reason}: {detail}"
                rec["status"] = "skipped_no_success_noise"
                rec["error"] = reason
                _write_failure_marker(
                    failure_path,
                    ri,
                    expected_name,
                    reason,
                    "",
                    status=rec["status"],
                    config_fingerprint=config_fingerprint,
                )
                print(
                    f"[skip noise] row_index={ri}, target={expected_name}: "
                    f"{reason}"
                )
                record_progress(rec, target_number)
                continue

            previous_failure = (
                _read_failure_marker(failure_path, config_fingerprint)
                if resume and not retry_failed
                else {}
            )
            if previous_failure.get("status") == "skipped_no_success_noise":
                failure_path.unlink(missing_ok=True)
                previous_failure = {}
            if previous_failure:
                rec["status"] = "skipped_failed"
                rec["error"] = str(
                    previous_failure.get("error", "previous run failed")
                )
                print(
                    f"[resume] row_index={ri}, target={expected_name}: "
                    "previous failure, skipped; use --retry-failed to retry"
                )
                record_progress(rec, target_number)
                continue

            paths["completion_marker"].unlink(missing_ok=True)
            realizations, summaries, null_calibration = run_batch(pc)
            if not realizations or not summaries:
                raise RuntimeError("v04 returned empty results")
            detection_limits = _derive_detection_limits(summaries)
            pl_name = str(realizations[0]["pl_name"])
            if pl_name != expected_name:
                raise ValueError(
                    f"target changed during run: {expected_name!r} != {pl_name!r}"
                )
            rec["pl_name"] = pl_name
            rec["status"] = "completed"
            rec["n_injection_grid_points"] = len(summaries)
            rec["n_realization_results"] = len(realizations)
            rec["delta_chi2_threshold"] = float(
                null_calibration["delta_chi2_threshold"]
            )

            target_dir = paths["target_dir"]
            target_dir.mkdir(parents=True, exist_ok=True)
            fig_path = paths["figure"]
            npz_path = paths["npz"]
            realizations_csv_path = paths["realizations_csv"]
            summary_csv_path = paths["summary_csv"]
            detection_limits_csv_path = paths["detection_limits_csv"]
            null_calibration_npz_path = paths["null_calibration_npz"]

            _save_detection_figure(summaries, detection_limits, fig_path)
            rec["figure_path"] = _path_for_summary(fig_path, root)
            if BATCH_RUN_CONFIG.get("save_results_npz", True):
                _save_npz(realizations, npz_path)
                rec["npz_path"] = _path_for_summary(npz_path, root)
            if BATCH_RUN_CONFIG.get("save_realizations_csv", True):
                _save_realizations_csv(realizations, realizations_csv_path)
                rec["realizations_csv_path"] = _path_for_summary(
                    realizations_csv_path,
                    root,
                )
            if BATCH_RUN_CONFIG.get("save_summary_csv", True):
                _save_summary_csv(summaries, summary_csv_path)
                rec["summary_csv_path"] = _path_for_summary(
                    summary_csv_path,
                    root,
                )
            if BATCH_RUN_CONFIG.get("save_detection_limits_csv", True):
                _save_detection_limits_csv(
                    detection_limits,
                    detection_limits_csv_path,
                )
                rec["detection_limits_csv_path"] = _path_for_summary(
                    detection_limits_csv_path,
                    root,
                )
            if BATCH_RUN_CONFIG.get("save_null_calibration_npz", True):
                _save_null_calibration_npz(
                    null_calibration,
                    null_calibration_npz_path,
                )
                rec["null_calibration_npz_path"] = _path_for_summary(
                    null_calibration_npz_path,
                    root,
                )

            outputs_valid, validation_message = _validate_target_outputs(
                paths,
                pl_name,
                ri,
            )
            if not outputs_valid:
                raise RuntimeError(
                    "saved v04 outputs failed completion validation: "
                    f"{validation_message}"
                )
            _write_completion_marker(
                paths,
                ri,
                pl_name,
                len(summaries),
                len(realizations),
                float(null_calibration["delta_chi2_threshold"]),
                adopted_existing=False,
                config_fingerprint=config_fingerprint,
            )
            print(
                f"[complete] row_index={ri}, target={pl_name}: "
                f"{paths['completion_marker']}"
            )
            failure_path.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 — 记录 skip，不静默
            rec["status"] = "failed"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            try:
                _write_failure_marker(
                    failure_path,
                    ri,
                    str(rec.get("pl_name", "")),
                    rec["error"],
                    traceback.format_exc(),
                    status=rec["status"],
                    config_fingerprint=config_fingerprint,
                )
            except OSError as marker_exc:
                print(
                    f"[WARN] could not write failure marker {failure_path}: "
                    f"{marker_exc}",
                    file=sys.stderr,
                )
            print(f"[SKIP] row_index={ri}: {rec['error']}", file=sys.stderr)
            traceback.print_exc()
        record_progress(rec, target_number)

    if write_summary and summary_rows:
        print("summary:", summary_path)

    elapsed = time.perf_counter() - t0
    timing_path = out_dir / "run_timing.txt"
    timing_lines = [
        f"wall_seconds={elapsed:.6f}",
        f"n_row_indices={len(row_indices)}",
    ]
    jid = os.environ.get("SLURM_JOB_ID", "").strip()
    if jid:
        timing_lines.append(f"slurm_job_id={jid}")
    timing_path.write_text("\n".join(timing_lines) + "\n", encoding="utf-8")
    print(f"timing: wall_seconds={elapsed:.6f} -> {timing_path}")

    return summary_rows


def _parse_row_indices_from_env_and_args(
    args: argparse.Namespace,
    n_table_rows: int,
) -> list[int]:
    configured = getattr(args, "rows", None)
    if not configured:
        configured = os.environ.get("OBLATE_ROW_INDICES", "").strip()
    if configured:
        parsed = [
            int(value.strip())
            for value in configured.split(",")
            if value.strip()
        ]
    else:
        parsed = list(range(n_table_rows))

    row_indices = list(dict.fromkeys(parsed))
    invalid = [
        row_index
        for row_index in row_indices
        if row_index < 0 or row_index >= n_table_rows
    ]
    if invalid:
        raise ValueError(
            f"row indices outside valid range 0..{n_table_rows - 1}: "
            f"{invalid}"
        )
    return row_indices


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run the v04 noisy completeness pipeline for multiple planets."
    )
    p.add_argument(
        "--rows",
        type=str,
        default=None,
        help=(
            "Comma-separated CSV row indices, e.g. 0,1,5. "
            "Omit to process every row in the planet table."
        ),
    )
    p.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help="Under repo results/, default from MULTI_SYSTEM_CONFIG. Ignored if --output-dir is set.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Absolute directory for png/npz/summary (overrides --output-subdir). Env: OBLATE_MULTI_OUTPUT_DIR.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute selected targets even when complete outputs already exist.",
    )
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry targets recorded as failed instead of skipping them.",
    )
    args = p.parse_args()
    try:
        _, input_rows = _load_planet_rows(_repo_root())
        rows = _parse_row_indices_from_env_and_args(args, len(input_rows))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        p.error(str(exc))
    if not rows:
        print("No row indices selected.")
        return
    od_env = os.environ.get("OBLATE_MULTI_OUTPUT_DIR", "").strip()
    output_dir = args.output_dir or od_env or None
    if len(rows) <= 20:
        print("row_indices:", rows)
    else:
        print(
            f"row_indices: all {len(rows)} selected "
            f"({rows[0]}..{rows[-1]})"
        )
    if output_dir:
        print("output_dir:", Path(output_dir).expanduser().resolve())
    run_multi_system(
        rows,
        output_subdir=args.output_subdir,
        output_dir=output_dir,
        resume=not args.force,
        retry_failed=args.retry_failed,
    )


if __name__ == "__main__":
    main()
