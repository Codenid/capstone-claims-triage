"""Evaluate predictions from weekly count models."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.models.weekly_counts.contracts import INTERVALS, SPLITS

CONVERGENCE_CHECKS = {"rhat", "ess_bulk", "ess_tail", "divergences"}
COVERAGE_CHECKS = {"coverage_80", "coverage_95"}
# Promotion rule frozen on 2026-09-27 (models_plan.md §12.4).
BOOTSTRAP_RULE = "best_baseline_bootstrap_v1"


def weighted_interval_score(
    observed: np.ndarray,
    median: np.ndarray,
    intervals: list[tuple[float, np.ndarray, np.ndarray]],
) -> np.ndarray:
    score = 0.5 * np.abs(observed - median)
    for alpha, lower, upper in intervals:
        interval_score = upper - lower
        interval_score += (2 / alpha) * (lower - observed) * (observed < lower)
        interval_score += (2 / alpha) * (observed - upper) * (observed > upper)
        score += (alpha / 2) * interval_score
    return score / (len(intervals) + 0.5)


def interval_bounds(
    frame: pd.DataFrame,
    prefix: str,
    intervals: tuple[tuple[float, str, str], ...] = INTERVALS,
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    return [
        (
            alpha,
            frame[f"{prefix}_{lower}"].to_numpy(dtype=float),
            frame[f"{prefix}_{upper}"].to_numpy(dtype=float),
        )
        for alpha, lower, upper in intervals
    ]


def predictive_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    observed = frame["complaint_count"].to_numpy(dtype=float)
    median = frame[f"{prefix}_p50"].to_numpy(dtype=float)
    intervals = interval_bounds(frame, prefix)
    wis = weighted_interval_score(observed, median, intervals)
    errors = median - observed
    denominator = float(observed.sum())
    metrics = {
        "wis": float(wis.mean()),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "wape": float(np.abs(errors).sum() / denominator),
        "relative_bias": float(errors.sum() / denominator),
        "coverage_50": float(
            ((observed >= intervals[0][1]) & (observed <= intervals[0][2])).mean()
        ),
        "coverage_80": float(
            ((observed >= intervals[1][1]) & (observed <= intervals[1][2])).mean()
        ),
        "coverage_95": float(
            ((observed >= intervals[2][1]) & (observed <= intervals[2][2])).mean()
        ),
        "width_50": float((intervals[0][2] - intervals[0][1]).mean()),
        "width_80": float((intervals[1][2] - intervals[1][1]).mean()),
        "width_95": float((intervals[2][2] - intervals[2][1]).mean()),
        "upper_95_excess_fraction": float((observed > intervals[2][2]).mean()),
        "impossible_probability": float(
            frame[f"{prefix}_impossible_probability"].mean()
        ),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("M9 predictive metrics contain non-finite values.")
    return metrics


def evaluate_predictions(
    predictions: pd.DataFrame,
    splits: tuple[str, ...] = SPLITS,
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, float]]]]:
    rows = []
    report: dict[str, dict[str, dict[str, float]]] = {}
    for split in splits:
        split_frame = predictions.loc[predictions["split"] == split]
        report[split] = {}
        for model in ("model", "baseline"):
            global_metrics = predictive_metrics(split_frame, model)
            report[split][model] = global_metrics
            rows.append(
                {
                    "split": split,
                    "model": model,
                    "cluster_id": -1,
                    **global_metrics,
                }
            )
            for cluster, group in split_frame.groupby("cluster_id", observed=True):
                rows.append(
                    {
                        "split": split,
                        "model": model,
                        "cluster_id": int(cluster),
                        **predictive_metrics(group, model),
                    }
                )
        model_metrics = report[split]["model"]
        baseline_metrics = report[split]["baseline"]
        model_metrics["wis_skill"] = float(
            1 - model_metrics["wis"] / baseline_metrics["wis"]
        )
        model_metrics["wape_skill"] = float(
            1 - model_metrics["wape"] / baseline_metrics["wape"]
        )
    return pd.DataFrame(rows), report


def convergence_checks(
    diagnostics: dict[str, float | int],
    acceptance: dict[str, Any],
) -> dict[str, bool]:
    return {
        "rhat": diagnostics["rhat_max"] <= acceptance["maximum_rhat"],
        "ess_bulk": diagnostics["ess_bulk_min"] >= acceptance["minimum_ess"],
        "ess_tail": diagnostics["ess_tail_min"] >= acceptance["minimum_ess"],
        "divergences": diagnostics["divergences"] == 0,
    }


def coverage_checks(
    model: dict[str, float],
    acceptance: dict[str, Any],
) -> dict[str, bool]:
    return {
        "coverage_80": (
            acceptance["coverage_80_minimum"]
            <= model["coverage_80"]
            <= acceptance["coverage_80_maximum"]
        ),
        "coverage_95": (
            acceptance["coverage_95_minimum"]
            <= model["coverage_95"]
            <= acceptance["coverage_95_maximum"]
        ),
    }


def calibration_acceptance(
    diagnostics: dict[str, float | int],
    calibration: dict[str, dict[str, float]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Rule frozen with B1, NB-V1, NB-V2 and NB-V3: beat the fixed Poisson."""
    acceptance = settings["acceptance"]
    model = calibration["model"]
    baseline = calibration["baseline"]
    checks = {
        **convergence_checks(diagnostics, acceptance),
        "wis": model["wis"] < baseline["wis"],
        "wape": model["wape"] <= baseline["wape"],
        **coverage_checks(model, acceptance),
    }
    return {"accepted": all(checks.values()), "checks": checks}


def bootstrap_acceptance(
    diagnostics: dict[str, float | int],
    calibration: dict[str, dict[str, float]],
    comparison: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Beat the best baseline by the frozen WIS gain without a clear loss."""
    acceptance = settings["acceptance"]
    difference = comparison["difference"]
    checks = {
        **convergence_checks(diagnostics, acceptance),
        "wis_gain": comparison["wis_gain"] >= acceptance["minimum_wis_gain"],
        "wis_bootstrap": difference["wis"]["p975"] < 0,
        # WAPE and MAE fail only when the whole interval shows a degradation.
        "wape": difference["wape"]["p025"] <= 0,
        "mae": difference["mae"]["p025"] <= 0,
        **coverage_checks(calibration["model"], acceptance),
    }
    return {"accepted": all(checks.values()), "checks": checks}


def run_acceptance(
    diagnostics: dict[str, float | int],
    calibration: dict[str, dict[str, float]],
    comparison: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Configs without acceptance.rule keep the rule they were frozen with."""
    if settings["acceptance"].get("rule") == BOOTSTRAP_RULE:
        return bootstrap_acceptance(diagnostics, calibration, comparison, settings)
    return calibration_acceptance(diagnostics, calibration, settings)


def failed_checks(acceptance: dict[str, Any]) -> list[str]:
    return [name for name, passed in acceptance["checks"].items() if not passed]


def candidate_status(run_mode: str, acceptance: dict[str, Any]) -> str:
    """Map automatic gates to the states allowed by models_plan.md."""
    if run_mode == "pilot":
        return "pilot_only"
    failed = set(failed_checks(acceptance))
    if failed & CONVERGENCE_CHECKS:
        return "rejected_convergence"
    if failed & COVERAGE_CHECKS:
        return "rejected_predictive"
    if failed:
        # The remaining checks compare accuracy with the baselines.
        return "rejected_no_practical_gain"
    return "accepted"


def rejection_reason(run_mode: str, acceptance: dict[str, Any]) -> str:
    failed = failed_checks(acceptance)
    if run_mode == "pilot" or not failed:
        return "none"
    return ",".join(failed)
