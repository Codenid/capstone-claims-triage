"""Poisson baseline helpers for weekly count models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fit_baseline_rates(
    frame: pd.DataFrame,
    clusters: int,
) -> np.ndarray:
    fit = frame.loc[frame["split"] == "fit"]
    denominator = float(
        fit.drop_duplicates("week")["weekly_total"].sum()
    )
    counts = (
        fit.groupby("cluster_id", observed=True)["complaint_count"]
        .sum()
        .reindex(range(clusters), fill_value=0)
        .to_numpy(dtype=float)
    )
    rates = counts / denominator
    if not np.isclose(rates.sum(), 1.0):
        raise ValueError("M9 baseline cluster rates must sum to one.")
    return rates


def expected_counts(
    cluster_index: np.ndarray,
    weekly_total: np.ndarray,
    rates: np.ndarray,
) -> np.ndarray:
    return weekly_total * rates[cluster_index]


def predictive_draws(
    mu: np.ndarray,
    draws: int,
    seed: int,
) -> np.ndarray:
    return np.random.default_rng(seed).poisson(
        mu[None, :],
        size=(draws, len(mu)),
    )
