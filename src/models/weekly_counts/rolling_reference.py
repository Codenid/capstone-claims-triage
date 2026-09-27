"""One-step-ahead rolling-share Poisson baselines B1-R4 and B1-R13."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.weekly_counts import fixed_poisson_reference
from src.models.weekly_counts.prediction import summarize_draws

# Column prefix -> number of previous complete weeks (models_plan.md §6).
ROLLING_WINDOWS = {"b1_rolling_4": 4, "b1_rolling_13": 13}
ROLLING_VERSION = "v1"
# The fixed Poisson reference already uses seed + 1.
ROLLING_SEED_OFFSET = 2


def rolling_shares(frame: pd.DataFrame, clusters: int, window: int) -> pd.DataFrame:
    """Shares of the previous `window` weeks with one pseudo-count per cluster.

    Weeks without `window` previous complete weeks stay NaN.
    """
    counts = (
        frame.pivot(index="week", columns="cluster_id", values="complaint_count")
        .sort_index()
        .reindex(columns=range(clusters))
    )
    previous = np.asarray(counts.shift(1).rolling(window).sum(), dtype=float)
    shares = (previous + 1) / (previous.sum(axis=1, keepdims=True) + clusters)
    return pd.DataFrame(shares, index=counts.index, columns=counts.columns)


def row_shares(frame: pd.DataFrame, clusters: int, window: int) -> np.ndarray:
    """Rolling share of each row's cluster; NaN before a full window."""
    shares = rolling_shares(frame, clusters, window)
    week_position = shares.index.get_indexer(frame["week"])
    return shares.to_numpy()[week_position, frame["cluster_id"].to_numpy()]


def expected_counts(frame: pd.DataFrame, clusters: int, window: int) -> np.ndarray:
    shares = row_shares(frame, clusters, window)
    return frame["weekly_total"].to_numpy(dtype=float) * shares


def rolling_columns(
    frame: pd.DataFrame,
    clusters: int,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    """Poisson predictive summaries per frame row; NaN before a full window."""
    weekly_total = frame["weekly_total"].to_numpy(dtype=float)
    columns = []
    for offset, (prefix, window) in enumerate(ROLLING_WINDOWS.items()):
        mu = expected_counts(frame, clusters, window)
        available = np.isfinite(mu)
        predictive = fixed_poisson_reference.predictive_draws(
            mu[available], draws, seed + offset
        )
        summary = summarize_draws(predictive, prefix)
        summary[f"{prefix}_impossible_probability"] = (
            predictive > weekly_total[available][None, :]
        ).mean(axis=0)
        summary.index = frame.index[available]
        columns.append(summary.reindex(frame.index))
    return pd.concat(columns, axis=1)
