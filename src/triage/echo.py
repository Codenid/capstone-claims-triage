"""Echo of a burst in the 4-week reference of M9 and M11 (models_plan.md §26).

The frozen M11 accumulator recomputed without some patterns: they leave the
weekly total and the 4 previous weeks, and the rest keep M9's dispersion draws,
k and h. It is an exploratory diagnostic on data already used, not a model.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.models.persistent_change import excess_scores, recent_shares


def kept_columns(counts: pd.DataFrame, dropped: Iterable[int]) -> list[int]:
    removed = set(dropped)
    return [c for c in counts.columns if c not in removed]


def expected_without(counts: pd.DataFrame, dropped: Iterable[int]) -> pd.DataFrame:
    """M9's R4 expected count of the other patterns, from their own total and shares.

    `counts` has one row per week and one column per pattern. With nothing
    dropped it is M9's expected count.
    """
    part = counts[kept_columns(counts, dropped)]
    shares = recent_shares(part.to_numpy())
    totals = part.sum(axis=1).to_numpy()[:, None]
    return pd.DataFrame(shares * totals, index=counts.index, columns=part.columns)


def frozen_expected(
    counts: pd.DataFrame,
    dropped: Iterable[int],
    reference: list[pd.Timestamp],
) -> pd.DataFrame:
    """Expected counts with the shares of fixed reference weeks instead of the last 4."""
    part = counts[kept_columns(counts, dropped)]
    pooled = part.loc[reference].sum()
    shares = (pooled + 1) / (pooled.sum() + part.shape[1])
    return pd.DataFrame(
        np.outer(part.sum(axis=1), shares), index=counts.index, columns=part.columns
    )


def scores(counts: pd.DataFrame, expected: pd.DataFrame, alpha: np.ndarray) -> pd.DataFrame:
    """Normal score of the mid-PIT under M9 for every week with an expected count.

    `alpha` has one row per posterior draw and one column per pattern id.
    """
    valid = expected.dropna()
    observed = counts.loc[valid.index, valid.columns].to_numpy()
    patterns = np.tile(valid.columns.to_numpy(), len(valid))
    flat = excess_scores(observed.ravel(), valid.to_numpy().ravel(), alpha[:, patterns])
    return pd.DataFrame(flat.reshape(observed.shape), index=valid.index, columns=valid.columns)


def accumulate(
    z: pd.DataFrame,
    start: pd.Series,
    k: float,
    h: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Upper CUSUM from a given state, one row per week; it restarts at 0 after each alarm."""
    state = start.reindex(z.columns).to_numpy(dtype=float)
    statistic, alarms = [], []
    for values in z.to_numpy():
        state = np.maximum(0.0, state + values - k)
        alarm = state > h
        statistic.append(state.copy())
        alarms.append(alarm)
        state = np.where(alarm, 0.0, state)
    return (
        pd.DataFrame(statistic, index=z.index, columns=z.columns),
        pd.DataFrame(alarms, index=z.index, columns=z.columns),
    )


def state_after(cusum: pd.DataFrame, alarm: pd.DataFrame, week: pd.Timestamp) -> pd.Series:
    """The accumulator after a closed week: its value, or 0 when it alarmed and restarted."""
    return cusum.loc[week].where(~alarm.loc[week].astype(bool), 0.0)


def echo_factor(real_share: float, expected_share: float) -> float:
    """How many times larger the others' expected count is without the burst pattern."""
    return (1 - real_share) / (1 - expected_share)
