"""Paired weekly bootstrap of a candidate against its best baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.weekly_counts.contracts import INTERVALS
from src.models.weekly_counts.metrics import interval_bounds, weighted_interval_score

# Frozen in models_plan.md §12.5.
BOOTSTRAP_DRAWS = 2000
DIFFERENCE_QUANTILES = {"p025": 0.025, "p50": 0.50, "p975": 0.975}


def weekly_sums(
    frame: pd.DataFrame,
    prefix: str,
    intervals: tuple[tuple[float, str, str], ...] = INTERVALS,
) -> pd.DataFrame:
    """Per-week sums that rebuild WIS, WAPE and MAE for one predictor."""
    observed = frame["complaint_count"].to_numpy(dtype=float)
    median = frame[f"{prefix}_p50"].to_numpy(dtype=float)
    scores = pd.DataFrame(
        {
            "week": frame["week"].to_numpy(),
            "wis": weighted_interval_score(
                observed, median, interval_bounds(frame, prefix, intervals)
            ),
            "absolute_error": np.abs(median - observed),
            "observed": observed,
            "rows": 1.0,
        }
    )
    return scores.groupby("week").sum()


def accuracy(sums: dict[str, Any]) -> dict[str, Any]:
    """WIS, WAPE and MAE from summed scores; works with scalars or arrays."""
    return {
        "wis": sums["wis"] / sums["rows"],
        "wape": sums["absolute_error"] / sums["observed"],
        "mae": sums["absolute_error"] / sums["rows"],
    }


def resampled_accuracy(weekly: pd.DataFrame, index: np.ndarray) -> dict[str, Any]:
    return accuracy(
        {column: weekly[column].to_numpy()[index].sum(axis=1) for column in weekly}
    )


def paired_bootstrap(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    draws: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Differences candidate - baseline over the same resampled weeks."""
    if not candidate.index.equals(baseline.index):
        raise ValueError("M9 bootstrap needs the same weeks for both predictors.")
    weeks = len(candidate)
    index = np.random.default_rng(seed).integers(weeks, size=(draws, weeks))
    point = {
        name: value - accuracy(baseline.sum().to_dict())[name]
        for name, value in accuracy(candidate.sum().to_dict()).items()
    }
    candidate_draws = resampled_accuracy(candidate, index)
    baseline_draws = resampled_accuracy(baseline, index)
    result = {}
    for name, estimate in point.items():
        differences = candidate_draws[name] - baseline_draws[name]
        quantiles = np.quantile(differences, list(DIFFERENCE_QUANTILES.values()))
        result[name] = {
            "estimate": float(estimate),
            **{
                label: float(value)
                for label, value in zip(DIFFERENCE_QUANTILES, quantiles)
            },
        }
    return result


def compare_with_baselines(
    predictions: pd.DataFrame,
    baselines: tuple[str, ...],
    seed: int,
    candidate: str = "model",
    split: str = "calibration",
    intervals: tuple[tuple[float, str, str], ...] = INTERVALS,
) -> dict[str, Any]:
    """Pick the lowest-WIS baseline in `split` and bootstrap the candidate against it."""
    rows = predictions.loc[predictions["split"] == split]
    weekly = {
        name: weekly_sums(rows, name, intervals) for name in (candidate, *baselines)
    }
    metrics = {
        name: {key: float(value) for key, value in accuracy(sums.sum().to_dict()).items()}
        for name, sums in weekly.items()
    }
    best = min(baselines, key=lambda name: metrics[name]["wis"])
    return {
        "split": split,
        "weeks": len(weekly[candidate]),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "candidate": candidate,
        "best_baseline": best,
        "metrics": metrics,
        "wis_gain": 1 - metrics[candidate]["wis"] / metrics[best]["wis"],
        "difference": paired_bootstrap(
            weekly[candidate], weekly[best], BOOTSTRAP_DRAWS, seed
        ),
    }
