"""Evaluate predictions from weekly count models."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.models.weekly_counts.contracts import INTERVALS, SPLITS


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


def predictive_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    observed = frame["complaint_count"].to_numpy(dtype=float)
    median = frame[f"{prefix}_p50"].to_numpy(dtype=float)
    intervals = [
        (
            alpha,
            frame[f"{prefix}_{lower}"].to_numpy(dtype=float),
            frame[f"{prefix}_{upper}"].to_numpy(dtype=float),
        )
        for alpha, lower, upper in INTERVALS
    ]
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
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, float]]]]:
    rows = []
    report: dict[str, dict[str, dict[str, float]]] = {}
    for split in SPLITS:
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


def calibration_acceptance(
    diagnostics: dict[str, float | int],
    calibration: dict[str, dict[str, float]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    acceptance = settings["acceptance"]
    model = calibration["model"]
    baseline = calibration["baseline"]
    checks = {
        "rhat": diagnostics["rhat_max"] <= acceptance["maximum_rhat"],
        "ess_bulk": diagnostics["ess_bulk_min"] >= acceptance["minimum_ess"],
        "ess_tail": diagnostics["ess_tail_min"] >= acceptance["minimum_ess"],
        "divergences": diagnostics["divergences"] == 0,
        "wis": model["wis"] < baseline["wis"],
        "wape": model["wape"] <= baseline["wape"],
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
    return {"accepted": all(checks.values()), "checks": checks}
