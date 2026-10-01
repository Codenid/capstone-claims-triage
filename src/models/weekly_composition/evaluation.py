"""Metrics and the M10 promotion rule (models_plan.md §12.3)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.weekly_composition.data import select_weeks
from src.models.weekly_counts.comparison import BOOTSTRAP_DRAWS, DIFFERENCE_QUANTILES
from src.models.weekly_counts.metrics import (
    convergence_checks,
    coverage_checks,
    predictive_metrics,
)
from src.models.weekly_counts.prediction import summarize_draws

# Promotion rule frozen on 2026-09-27 (models_plan.md §12.3).
COMPOSITION_RULE = "best_joint_baseline_bootstrap_v1"


def long_predictions(
    panel: dict[str, np.ndarray],
    draws: dict[str, np.ndarray],
) -> pd.DataFrame:
    """One row per week and cluster with each predictor's marginal quantiles."""
    weeks, clusters = panel["counts"].shape
    totals = np.repeat(panel["weekly_total"], clusters)
    parts = [
        pd.DataFrame(
            {
                "split": np.repeat(panel["split"], clusters),
                "week": np.repeat(panel["week"], clusters),
                "cluster_id": np.tile(np.arange(clusters), weeks),
                "complaint_count": panel["counts"].reshape(-1),
                "weekly_total": totals,
            }
        )
    ]
    for name, values in draws.items():
        flat = values.reshape(len(values), -1)
        summary = summarize_draws(flat, name)
        summary[f"{name}_impossible_probability"] = (flat > totals).mean(axis=0)
        parts.append(summary)
    return pd.concat(parts, axis=1)


def total_variation(panel: dict[str, np.ndarray], draws: np.ndarray) -> float:
    """Mean distance between observed and predicted mean shares; 0 means equal."""
    observed = panel["counts"] / panel["weekly_total"][:, None]
    predicted = draws.mean(axis=0) / panel["weekly_total"][:, None]
    return float(0.5 * np.abs(observed - predicted).sum(axis=1).mean())


def split_metrics(
    panel: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    draws: dict[str, np.ndarray],
    split: str,
) -> dict[str, dict[str, float]]:
    """Joint log score, composition distance and marginal metrics of one split."""
    rows = panel["split"] == split
    weeks = select_weeks(panel, rows)
    marginal = long_predictions(
        weeks, {name: values[:, rows] for name, values in draws.items()}
    )
    return {
        name: {
            "log_score": float(scores[name][rows].mean()),
            "total_variation": total_variation(weeks, draws[name][:, rows]),
            **predictive_metrics(marginal, name),
        }
        for name in scores
    }


def compare_with_baselines(
    scores: dict[str, np.ndarray],
    baselines: tuple[str, ...],
    seed: int,
    candidate: str = "model",
) -> dict[str, Any]:
    """Best baseline by mean weekly log score, with a paired weekly bootstrap."""
    means = {name: float(values.mean()) for name, values in scores.items()}
    best = max(baselines, key=lambda name: means[name])
    difference = scores[candidate] - scores[best]
    weeks = len(difference)
    index = np.random.default_rng(seed).integers(weeks, size=(BOOTSTRAP_DRAWS, weeks))
    quantiles = np.quantile(
        difference[index].mean(axis=1), list(DIFFERENCE_QUANTILES.values())
    )
    interval = {
        label: float(value) for label, value in zip(DIFFERENCE_QUANTILES, quantiles)
    }
    return {
        "weeks": weeks,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "candidate": candidate,
        "best_baseline": best,
        "log_score": means,
        "difference": {"estimate": float(difference.mean()), **interval},
    }


def composition_acceptance(
    diagnostics: dict[str, float | int],
    calibration: dict[str, float],
    comparison: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Beat the best joint baseline with a bootstrap interval above zero."""
    acceptance = settings["acceptance"]
    if acceptance["rule"] != COMPOSITION_RULE:
        raise ValueError(f"M10 configs must use the {COMPOSITION_RULE} rule.")
    difference = comparison["difference"]
    checks = {
        **convergence_checks(diagnostics, acceptance),
        "log_score_gain": difference["estimate"] > 0,
        "log_score_bootstrap": difference["p025"] > 0,
        **coverage_checks(calibration, acceptance),
    }
    return {"accepted": all(checks.values()), "checks": checks}
