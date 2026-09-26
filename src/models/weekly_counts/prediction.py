"""Generate predictions for the weekly Negative Binomial model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.weekly_counts.contracts import PREDICTIVE_QUANTILES
from src.models.weekly_counts.negative_binomial import (
    expected_counts as negative_binomial_expected_counts,
    negative_binomial_draws,
)
from src.models.weekly_counts.poisson import (
    expected_counts as poisson_expected_counts,
    predictive_draws as poisson_predictive_draws,
)


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


def generate_predictions(
    frame: pd.DataFrame,
    log_rate_draws: np.ndarray,
    trend_draws: np.ndarray,
    alpha_draws: np.ndarray,
    baseline_rates: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    cluster_index = frame["cluster_id"].to_numpy(dtype=np.int64)
    weekly_total = frame["weekly_total"].to_numpy(dtype=float)
    time_years = frame["time_years"].to_numpy(dtype=float)
    observed = frame["complaint_count"].to_numpy(dtype=np.int64)

    mu = negative_binomial_expected_counts(
        log_rate_draws,
        trend_draws,
        cluster_index,
        weekly_total,
        time_years,
    )
    alpha = alpha_draws[:, cluster_index]
    predictive = negative_binomial_draws(mu, alpha, seed)

    baseline_mu = poisson_expected_counts(
        cluster_index,
        weekly_total,
        baseline_rates,
    )
    baseline_predictive = poisson_predictive_draws(
        baseline_mu,
        draws=mu.shape[0],
        seed=seed + 1,
    )

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
            summarize_draws(mu, "expected"),
            summarize_draws(predictive, "model"),
            summarize_draws(baseline_predictive, "baseline"),
        ],
        axis=1,
    )
    result["model_upper_tail_probability"] = (
        predictive >= observed[None, :]
    ).mean(axis=0)
    result["baseline_upper_tail_probability"] = (
        baseline_predictive >= observed[None, :]
    ).mean(axis=0)
    result["model_impossible_probability"] = (
        predictive > weekly_total[None, :]
    ).mean(axis=0)
    result["baseline_impossible_probability"] = (
        baseline_predictive > weekly_total[None, :]
    ).mean(axis=0)
    return result
