"""Shares with a decaying memory and a cap on atypical weeks (C-A, §25.5).

    r[c,t] = (sum_k discount**(k-1) * capped[c,t-k] + 1)
             / (sum_k discount**(k-1) * capped_total[t-k] + clusters)
    capped[c,s] = min(count[c,s], cap * weekly_total[s] * r[c,s])

A week enters the memory with at most `cap` times what was expected for it,
so one burst does not inflate the following weeks (models_plan.md §24).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def discounted_shares(
    frame: pd.DataFrame,
    clusters: int,
    discount: float,
    cap: float | None,
) -> pd.DataFrame:
    """Share of each cluster from the previous weeks only; NaN in the first week."""
    if not 0 < discount <= 1:
        raise ValueError("The discount must be in (0, 1].")
    if cap is not None and cap < 1:
        raise ValueError("The cap must be at least 1 or None.")
    counts = (
        frame.pivot(index="week", columns="cluster_id", values="complaint_count")
        .sort_index()
        .reindex(columns=range(clusters))
    )
    values = counts.to_numpy(dtype=float)
    shares = np.full_like(values, np.nan)
    memory = np.zeros(clusters)
    for week, observed in enumerate(values):
        if week > 0:
            shares[week] = (memory + 1) / (memory.sum() + clusters)
            if cap is not None:
                observed = np.minimum(observed, cap * observed.sum() * shares[week])
        memory = discount * memory + observed
    return pd.DataFrame(shares, index=counts.index, columns=counts.columns)


def row_discounted_shares(
    frame: pd.DataFrame,
    clusters: int,
    discount: float,
    cap: float | None,
) -> np.ndarray:
    """Share of each row's cluster, in the order of `frame`."""
    shares = discounted_shares(frame, clusters, discount, cap)
    week_position = shares.index.get_indexer(frame["week"])
    return shares.to_numpy()[week_position, frame["cluster_id"].to_numpy()]
