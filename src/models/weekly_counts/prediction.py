"""Prediction summaries shared by weekly count models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.weekly_counts.contracts import PREDICTIVE_QUANTILES


def _quantile_label(value: float) -> str:
    labels = {
        0.025: "p025",
        0.10: "p10",
        0.25: "p25",
        0.50: "p50",
        0.75: "p75",
        0.90: "p90",
        0.975: "p975",
    }
    return labels[value]


def summarize_draws(draws: np.ndarray, prefix: str) -> pd.DataFrame:
    quantiles = np.quantile(draws, PREDICTIVE_QUANTILES, axis=0)
    result = {f"{prefix}_mean": draws.mean(axis=0)}
    for value, rows in zip(PREDICTIVE_QUANTILES, quantiles):
        result[f"{prefix}_{_quantile_label(value)}"] = rows
    return pd.DataFrame(result)


def build_predictions(
    frame: pd.DataFrame,
    expected_draws: np.ndarray,
    predictive_draws: np.ndarray,
    reference_draws: np.ndarray,
) -> pd.DataFrame:
    observed = frame["complaint_count"].to_numpy(dtype=np.int64)
    weekly_total = frame["weekly_total"].to_numpy(dtype=float)
    columns = [
        "split",
        "week",
        "cluster_id",
        "complaint_count",
        "weekly_total",
        "time_years",
    ]
    result = frame[columns].reset_index(drop=True).copy()
    result = pd.concat(
        [
            result,
            summarize_draws(expected_draws, "expected"),
            summarize_draws(predictive_draws, "model"),
            summarize_draws(reference_draws, "baseline"),
        ],
        axis=1,
    )
    result["model_upper_tail_probability"] = (
        predictive_draws >= observed[None, :]
    ).mean(axis=0)
    result["baseline_upper_tail_probability"] = (
        reference_draws >= observed[None, :]
    ).mean(axis=0)
    result["model_impossible_probability"] = (
        predictive_draws > weekly_total[None, :]
    ).mean(axis=0)
    result["baseline_impossible_probability"] = (
        reference_draws > weekly_total[None, :]
    ).mean(axis=0)
    return result
