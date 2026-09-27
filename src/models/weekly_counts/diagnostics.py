"""Diagnostics for weekly count models."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

PRIOR_QUANTILES = {"p025": 0.025, "p50": 0.50, "p975": 0.975}
PRIOR_REFERENCE_MEANS = (1, 5, 20, 100)
VOLUME_GROUPS = ("small", "medium", "large")


def posterior_diagnostics(
    idata: Any,
    diagnostic_variables: tuple[str, ...],
    summary_variables: tuple[str, ...],
    max_treedepth: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    az = importlib.import_module("arviz")
    # ArviZ rounds R-hat to two decimals unless round_to is the string "none".
    summary = az.summary(
        idata,
        var_names=list(summary_variables),
        kind="all",
        round_to="none",
    )
    convergence = az.summary(
        idata,
        var_names=list(diagnostic_variables),
        kind="diagnostics",
        round_to="none",
    )
    if not isinstance(summary, pd.DataFrame) or not isinstance(
        convergence, pd.DataFrame
    ):
        raise TypeError("M9 posterior summaries must be pandas DataFrames.")
    diagnostics: dict[str, float | int] = {
        "rhat_max": float(convergence["r_hat"].max()),
        "ess_bulk_min": float(convergence["ess_bulk"].min()),
        "ess_tail_min": float(convergence["ess_tail"].min()),
        "divergences": int(idata.sample_stats["diverging"].sum().item()),
        "bfmi_min": float(np.min(az.bfmi(idata))),
    }
    if "tree_depth" in idata.sample_stats:
        observed_maximum = int(idata.sample_stats["tree_depth"].max().item())
        diagnostics["tree_depth_max"] = observed_maximum
        if max_treedepth is not None:
            diagnostics["tree_depth_limit_hits"] = int(
                (idata.sample_stats["tree_depth"] == max_treedepth).sum().item()
            )
    return summary, diagnostics


def prior_predictive_summary(
    prior_draws: np.ndarray,
    fit: pd.DataFrame,
) -> dict[str, float]:
    totals = fit["weekly_total"].to_numpy(dtype=float)
    return {
        "median": float(np.median(prior_draws)),
        "p99": float(np.quantile(prior_draws, 0.99)),
        "maximum": float(prior_draws.max()),
        "impossible_fraction": float((prior_draws > totals[None, :]).mean()),
    }


def quantile_summary(values: np.ndarray) -> dict[str, float]:
    points = np.quantile(values, list(PRIOR_QUANTILES.values()))
    return {label: float(point) for label, point in zip(PRIOR_QUANTILES, points)}


def week_cluster_array(values: np.ndarray, fit: pd.DataFrame) -> np.ndarray:
    """Arrange (draw, row) values as (draw, week, cluster)."""
    week_index = pd.factorize(fit["week"], sort=True)[0]
    cluster_index = fit["cluster_id"].to_numpy(dtype=np.int64)
    result = np.zeros((len(values), week_index.max() + 1, cluster_index.max() + 1))
    result[:, week_index, cluster_index] = values
    return result


def fit_volume_groups(fit_counts: np.ndarray) -> np.ndarray:
    """Label clusters as small, medium or large by terciles of fit volume."""
    totals = fit_counts.sum(axis=(0, 1))
    ranks = np.argsort(np.argsort(totals))
    return np.array(VOLUME_GROUPS)[ranks * len(VOLUME_GROUPS) // len(totals)]


def composition_summary(
    counts: np.ndarray,
    weekly_total: np.ndarray,
    volume_groups: np.ndarray,
) -> dict[str, Any]:
    shares = counts / weekly_total[None, :, None]
    return {
        "zero_fraction_by_cluster": quantile_summary((counts == 0).mean(axis=(0, 1))),
        "max_weekly_share": quantile_summary(shares.max(axis=2)),
        "share_by_fit_volume": {
            group: quantile_summary(shares[:, :, volume_groups == group])
            for group in VOLUME_GROUPS
            if (volume_groups == group).any()
        },
    }


def intervals_near_means(
    observed: np.ndarray,
    expected: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Summarize prior draws whose prior mean is within 25% of a reference."""
    result = {}
    for mean in PRIOR_REFERENCE_MEANS:
        values = observed[(expected >= mean / 1.25) & (expected <= mean * 1.25)]
        result[str(mean)] = {"rows": float(values.size)}
        if values.size:
            result[str(mean)].update(quantile_summary(values))
    return result


def dispersion_summary(alpha: np.ndarray) -> dict[str, Any]:
    """Summarize alpha and the implied variance-to-mean ratio 1 + mu / alpha."""
    values = alpha.reshape(-1)
    return {
        "alpha": quantile_summary(values),
        "variance_to_mean": {
            str(mean): quantile_summary(1 + mean / values)
            for mean in PRIOR_REFERENCE_MEANS
        },
    }


def prior_check_summary(
    observed: np.ndarray,
    expected: np.ndarray,
    alpha: np.ndarray | None,
    fit: pd.DataFrame,
) -> dict[str, Any]:
    """Compare prior predictive draws with the same statistics observed in fit."""
    counts = week_cluster_array(observed, fit)
    fit_counts = week_cluster_array(fit["complaint_count"].to_numpy()[None, :], fit)
    weekly_total = fit.groupby("week")["weekly_total"].first().to_numpy(dtype=float)
    volume_groups = fit_volume_groups(fit_counts)
    summary = {
        "counts": prior_predictive_summary(observed, fit),
        "negative_count_fraction": float((observed < 0).mean()),
        "draw_sum_relative_deviation": quantile_summary(
            counts.sum(axis=2) / weekly_total[None, :] - 1
        ),
        "intervals_near_mean": intervals_near_means(observed, expected),
        "prior": composition_summary(counts, weekly_total, volume_groups),
        "fit_observed": composition_summary(fit_counts, weekly_total, volume_groups),
    }
    if alpha is not None:
        summary["dispersion"] = dispersion_summary(alpha)
    return summary
