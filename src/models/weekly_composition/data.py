"""Week x cluster matrices for the joint composition models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.weekly_counts.rolling_reference import rolling_shares

# The same window as B1-R4 and NB-R4-H v3 (models_plan.md §7.9).
RECENT_WEEKS = 4


def composition_panel(frame: pd.DataFrame, clusters: int) -> dict[str, np.ndarray]:
    """One row per complete week: split, total, counts by cluster and recent shares."""
    pivot = frame.pivot(index="week", columns="cluster_id", values="complaint_count")
    pivot = pivot.sort_index()
    if list(pivot.columns) != list(range(clusters)):
        raise ValueError("M10 needs the frozen cluster order 0..clusters-1.")
    weeks = frame.drop_duplicates("week").set_index("week").sort_index()
    counts = np.asarray(pivot, dtype=np.int64)
    totals = np.asarray(weeks["weekly_total"], dtype=np.int64)
    if not np.array_equal(counts.sum(axis=1), totals):
        raise ValueError("M10 cluster counts must sum to weekly_total.")
    return {
        "week": np.asarray(pivot.index),
        "split": np.asarray(weeks["split"], dtype=str),
        "weekly_total": totals,
        "counts": counts,
        "recent_share": np.asarray(rolling_shares(frame, clusters, RECENT_WEEKS)),
    }


def select_weeks(
    panel: dict[str, np.ndarray],
    rows: np.ndarray,
) -> dict[str, np.ndarray]:
    return {name: values[rows] for name, values in panel.items()}


def windowed_weeks(panel: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Weeks with a full recent window: every model is evaluated on these weeks."""
    complete = np.isfinite(panel["recent_share"]).all(axis=1)
    return select_weeks(panel, np.asarray(complete))


def split_weeks(panel: dict[str, np.ndarray], split: str) -> dict[str, np.ndarray]:
    return select_weeks(panel, panel["split"] == split)
