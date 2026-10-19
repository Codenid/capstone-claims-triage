"""Counts per period and the frozen predictive rules (models_plan.md §29).

Everything here is arithmetic on arrays: the posterior draws, discounts and
thresholds come from the frozen runs; nothing is re-estimated. The weekly
history is the frozen panel extended with the weeks of the new table; the
weeks that the new table covers replace the frozen counts of those weeks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.models.daily_change import DailyExpectation
from src.models.persistent_change import cusum, excess_scores, recent_shares
from src.models.weekly_composition.mixture_signal import week_surprise
from src.models.weekly_counts.discounted_reference import shares_from_counts

QUANTILES = {
    "p025": 0.025,
    "p10": 0.10,
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
    "p90": 0.90,
    "p975": 0.975,
}
WEEK_START = "W-SUN"


# ----------------------------------------------------------------- panels
def period_counts(
    assignments: pd.DataFrame,
    clusters: int,
    period: str,
) -> pd.DataFrame:
    """Complete grid of counts per period ("week" or "day") and pattern."""
    frame = assignments.copy()
    dates = pd.to_datetime(frame["Date received"])
    if period == "week":
        frame["period"] = dates.dt.to_period(WEEK_START).dt.start_time
    elif period == "day":
        frame["period"] = dates.dt.normalize()
    else:
        raise ValueError("period must be 'week' or 'day'")
    grouped = (
        frame.groupby(["period", "cluster_id"], observed=True)
        .agg(complaint_count=("Complaint ID", "size"), novel_count=("is_novel", "sum"))
        .reset_index()
    )
    periods = sorted(frame["period"].unique())
    index = pd.MultiIndex.from_product(
        [periods, range(clusters)], names=["period", "cluster_id"]
    )
    complete = grouped.set_index(["period", "cluster_id"]).reindex(index, fill_value=0)
    complete = complete.reset_index()
    complete[["complaint_count", "novel_count"]] = complete[
        ["complaint_count", "novel_count"]
    ].astype(np.int64)
    complete["total"] = complete.groupby("period")["complaint_count"].transform("sum")
    if period == "week":
        last = dates.max().normalize()
        complete["is_complete"] = complete["period"] + pd.Timedelta(days=6) <= last
    else:
        complete["is_complete"] = True
    return complete.sort_values(["period", "cluster_id"]).reset_index(drop=True)


def extend_history(
    history: pd.DataFrame,
    new: pd.DataFrame,
    clusters: int,
) -> pd.DataFrame:
    """Frozen history (period, cluster_id, complaint_count) plus the new periods.

    Periods present in `new` replace the frozen counts of those periods; the
    result is a complete, period-ordered grid with a `source` column.
    """
    base = history[["period", "cluster_id", "complaint_count"]].copy()
    base["period"] = pd.to_datetime(base["period"])
    base["source"] = "frozen"
    fresh = new[["period", "cluster_id", "complaint_count"]].copy()
    fresh["period"] = pd.to_datetime(fresh["period"])
    fresh["source"] = "new"
    fresh_periods = set(fresh["period"].tolist())
    base = base.loc[~base["period"].isin(fresh_periods)]
    combined = pd.concat([base, fresh], ignore_index=True)
    periods = sorted(set(combined["period"].tolist()))
    index = pd.MultiIndex.from_product(
        [periods, range(clusters)], names=["period", "cluster_id"]
    )
    keyed = combined.set_index(["period", "cluster_id"])
    if keyed.index.duplicated().any():
        raise ValueError("The extended history repeats a period-pattern key.")
    result = keyed.reindex(index).reset_index()
    result["complaint_count"] = result["complaint_count"].fillna(0).astype(np.int64)
    result["source"] = result["source"].fillna("frozen")
    result["total"] = result.groupby("period")["complaint_count"].transform("sum")
    return result.sort_values(["period", "cluster_id"]).reset_index(drop=True)


def matrix(panel: pd.DataFrame, column: str = "complaint_count") -> np.ndarray:
    """(periods, clusters) array of a complete, sorted panel."""
    periods = int(panel["period"].nunique())
    clusters = int(panel["cluster_id"].nunique())
    return panel[column].to_numpy().reshape((periods, clusters))


# ------------------------------------------------------- weekly predictive
def negative_binomial_draws(
    expected: np.ndarray,
    alpha: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """(draws, periods, clusters) counts from NB(mu, alpha) per draw."""
    mu = expected[None, :, :]
    shape = alpha[:, None, :]
    rate = rng.gamma(shape, mu / shape)
    return rng.poisson(rate)


def quantile_columns(draws: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    """Quantile and mean columns, flattened in (period, cluster) order."""
    flat = draws.reshape(draws.shape[0], -1)
    columns = {f"{prefix}_mean": flat.mean(axis=0)}
    for name, level in QUANTILES.items():
        columns[f"{prefix}_{name}"] = np.quantile(flat, level, axis=0)
    return columns


def weekly_predictive(
    panel: pd.DataFrame,
    alpha: np.ndarray,
    share: dict[str, Any],
    thresholds: dict[str, float],
    cusum_k: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """M9 expectation and intervals, M11 excess, CUSUM and alarms per week.

    `share` holds C-A's discount and cap; `thresholds` M11's `cusum` (h) and
    `weekly` (z*). Weeks without memory (the first one) are not scored.
    """
    counts = matrix(panel)
    totals = counts.sum(axis=1, keepdims=True)
    shares = shares_from_counts(
        counts.astype(float), share["discount"], share.get("cap")
    )
    expected = totals * shares
    scored = ~np.isnan(shares).any(axis=1)
    draws = negative_binomial_draws(np.nan_to_num(expected, nan=0.0), alpha, rng)
    columns = quantile_columns(draws, "model")
    clusters = counts.shape[1]
    z = np.full(counts.shape, np.nan)
    rows = np.flatnonzero(scored)
    for cluster in range(clusters):
        z[rows, cluster] = excess_scores(
            counts[rows, cluster], expected[rows, cluster], alpha[:, [cluster]]
        )
    statistic, alarms = cusum(np.nan_to_num(z), cusum_k, thresholds["cusum"])
    result = panel.copy()
    result["expected"] = expected.ravel()
    for name, values in columns.items():
        result[name] = values
    result["z"] = z.ravel()
    result["cusum"] = statistic.ravel()
    result["cusum_alarm"] = alarms.ravel() & scored.repeat(clusters)
    result["weekly_alarm"] = (z.ravel() > thresholds["weekly"]) & scored.repeat(
        clusters
    )
    result["scored"] = scored.repeat(clusters)
    return result


# ------------------------------------------------------ weekly composition
def weekly_mixture(
    panel: pd.DataFrame,
    kappa: np.ndarray,
    simulations: int,
    alert_p_value: float,
    seed: int,
) -> pd.DataFrame:
    """M10's log score and rare-mixture p-value per week (4-week shares)."""
    counts = matrix(panel)
    shares = recent_shares(counts)
    totals = counts.sum(axis=1)
    rng = np.random.default_rng(seed)
    periods = sorted(panel["period"].unique())
    rows = []
    for index, period in enumerate(periods):
        if np.isnan(shares[index]).any():
            rows.append(
                {
                    "period": period,
                    "log_score": np.nan,
                    "p_value": np.nan,
                    "rare_mixture": False,
                }
            )
            continue
        concentration = kappa[:, None] * shares[index][None, :]
        log_score, p_value = week_surprise(
            counts[index], int(totals[index]), concentration, simulations, rng
        )
        rows.append(
            {
                "period": period,
                "log_score": log_score,
                "p_value": p_value,
                "rare_mixture": p_value < alert_p_value,
            }
        )
    return pd.DataFrame(rows)


# -------------------------------------------------------- daily predictive
def daily_predictive(
    panel: pd.DataFrame,
    alpha: np.ndarray,
    model: DailyExpectation,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """D-D expectation and intervals per day; zero-inflated when the run is."""
    counts = matrix(panel)
    periods = sorted(panel["period"].unique())
    day_of_week = np.asarray(pd.DatetimeIndex(periods).dayofweek, dtype=np.int64)
    nb_mean = model.nb_mean(counts, day_of_week)
    expected = model.expected(counts, day_of_week)
    scored = ~np.isnan(nb_mean).any(axis=1)
    draws = negative_binomial_draws(np.nan_to_num(nb_mean, nan=0.0), alpha, rng)
    if model.zero is not None:
        extra = rng.random(draws.shape) < model.zero[None, None, :]
        draws = np.where(extra, 0, draws)
    columns = quantile_columns(draws, "model")
    clusters = counts.shape[1]
    result = panel.copy()
    result["expected"] = expected.ravel()
    for name, values in columns.items():
        result[name] = values
    result["scored"] = scored.repeat(clusters)
    return result
