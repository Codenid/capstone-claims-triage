"""Poisson references of the daily block, computed without PyMC (§28.2).

- `baseline`: fixed Poisson with the average fit share of each pattern.
- `b1_rolling_7`: Poisson with the share of the previous 7 days (D-B1).

Both condition on the observed daily total, like every daily model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.daily_counts.references import (
    expected_from_shares,
    static_daily_shares,
    windowed_daily_shares,
)
from src.models.weekly_counts.prediction import summarize_draws

ROLLING_WINDOW = 7
ROLLING_NAME = "b1_rolling_7"
BASELINE_SEED_OFFSET = 1
ROLLING_SEED_OFFSET = 2


def poisson_draws(expected: np.ndarray, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.poisson(np.broadcast_to(expected, (draws, len(expected))))


def fixed_reference_draws(
    frame: pd.DataFrame,
    fit: pd.DataFrame,
    clusters: int,
    draws: int,
    seed: int,
) -> np.ndarray:
    shares = static_daily_shares(fit, clusters)[frame["cluster_id"].to_numpy()]
    return poisson_draws(expected_from_shares(frame, shares), draws, seed)


def rolling_columns(
    frame: pd.DataFrame,
    panel: pd.DataFrame,
    clusters: int,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    """Quantile columns of D-B1 for the rows of `frame`, named like a model.

    The shares come from the full panel, so the first evaluated days of each
    split still see the days a model dropped as warm-up.
    """
    keys = ["split", "day", "cluster_id"]
    on_panel = pd.Series(
        windowed_daily_shares(panel, clusters, ROLLING_WINDOW),
        index=pd.MultiIndex.from_frame(panel[keys]),
    )
    shares = on_panel.reindex(pd.MultiIndex.from_frame(frame[keys])).to_numpy()
    if np.isnan(shares).any():
        raise ValueError("The rolling baseline needs rows after the first day.")
    values = poisson_draws(expected_from_shares(frame, shares), draws, seed)
    summary = summarize_draws(values, ROLLING_NAME)
    totals = frame["daily_total"].to_numpy(dtype=float)
    impossible = (values > totals[None, :]).mean(axis=0)
    summary[f"{ROLLING_NAME}_impossible_probability"] = impossible
    return summary
