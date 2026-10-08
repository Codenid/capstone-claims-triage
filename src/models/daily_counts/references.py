"""Shares that every daily model and baseline reads from the previous days.

- `discounted_daily_shares`: the decaying memory of C-A applied to days,
  r[c,d] = (sum_k delta^(k-1) * count[c,d-k] + 1)
           / (sum_k delta^(k-1) * total[d-k] + C).
- `windowed_daily_shares`: the plain share of the previous `window` days,
  used by the Poisson baseline.
- `static_daily_shares`: the share of each pattern over the whole fit period,
  used by the fixed Poisson reference.

Shares are computed per split in day order; the first day of a split has no
memory and is dropped by the models through `warmup_days`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.daily_counts.contracts import DAY_COLUMN, TOTAL_COLUMN
from src.models.weekly_counts.discounted_reference import shares_from_counts


def _pivot(frame: pd.DataFrame, clusters: int) -> pd.DataFrame:
    return (
        frame.pivot(index=DAY_COLUMN, columns="cluster_id", values="complaint_count")
        .sort_index()
        .reindex(columns=range(clusters))
    )


def discounted_daily_shares(
    frame: pd.DataFrame,
    clusters: int,
    discount: float,
) -> np.ndarray:
    """Share of each row's pattern from the previous days of its split."""
    shares = np.full(len(frame), np.nan)
    for split, rows in frame.groupby("split", observed=True):
        counts = _pivot(rows, clusters)
        values = shares_from_counts(counts.to_numpy(dtype=float), discount, None)
        position = counts.index.get_indexer(rows[DAY_COLUMN])
        shares[rows.index] = values[position, rows["cluster_id"].to_numpy()]
    return shares


def windowed_daily_shares(
    frame: pd.DataFrame,
    clusters: int,
    window: int,
) -> np.ndarray:
    """Share of the previous `window` days (+1 / +C smoothing), NaN before day 2."""
    shares = np.full(len(frame), np.nan)
    for split, rows in frame.groupby("split", observed=True):
        counts = _pivot(rows, clusters).to_numpy(dtype=float)
        values = np.full(counts.shape, np.nan)
        for day in range(1, len(counts)):
            recent = counts[max(0, day - window) : day]
            values[day] = (recent.sum(axis=0) + 1) / (recent.sum() + clusters)
        position = _pivot(rows, clusters).index.get_indexer(rows[DAY_COLUMN])
        shares[rows.index] = values[position, rows["cluster_id"].to_numpy()]
    return shares


def static_daily_shares(fit: pd.DataFrame, clusters: int) -> np.ndarray:
    """Average share of each pattern over the fit days, +1 / +C smoothing."""
    totals = fit.groupby("cluster_id", observed=True)["complaint_count"].sum()
    totals = totals.reindex(range(clusters), fill_value=0).to_numpy(dtype=float)
    return (totals + 1) / (totals.sum() + clusters)


def warm_rows(frame: pd.DataFrame, warmup_days: int) -> pd.Series:
    """Rows after the first `warmup_days` of their split, where shares exist."""
    first = frame.groupby("split", observed=True)[DAY_COLUMN].transform("min")
    return (frame[DAY_COLUMN] - first).dt.days >= warmup_days


def expected_from_shares(frame: pd.DataFrame, shares: np.ndarray) -> np.ndarray:
    return frame[TOTAL_COLUMN].to_numpy(dtype=float) * shares
