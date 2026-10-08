"""The daily panel: one row per split, day and pattern (models_plan.md §28.1).

It is built once from the frozen M8B assignments (every complaint keeps its
pattern, novel ones included, as the weekly counts do) and stored as
reports/modeling/daily_counts.csv with its SHA-256 frozen in each model
config. Every model reads the same validated panel.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.experiment import PROJECT_ROOT
from src.models.bge_sample import ID_COLUMN
from src.models.daily_counts.contracts import (
    DAY_COLUMN,
    REQUIRED_COLUMNS,
    SPLITS,
    TOTAL_COLUMN,
    WEEKLY_ALIASES,
)
from src.models.semantic_space import load_dvc_hash

DATE_COLUMN = "Date received"


def panel_from_assignments(
    assignments: pd.DataFrame,
    clusters: int,
    periods: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """Complete day x pattern grid per split, with zeros where nothing arrived."""
    frame = assignments.copy()
    frame[DAY_COLUMN] = pd.to_datetime(frame[DATE_COLUMN]).dt.normalize()
    grouped = (
        frame.groupby(["split", DAY_COLUMN, "cluster_id"], observed=True)
        .agg(complaint_count=(ID_COLUMN, "size"), novel_count=("is_novel", "sum"))
        .reset_index()
    )
    grids = []
    for split in SPLITS:
        period = periods[split]
        days = pd.date_range(period["start"], period["end"], freq="D")
        index = pd.MultiIndex.from_product(
            [[split], days, range(clusters)], names=["split", DAY_COLUMN, "cluster_id"]
        )
        part = grouped.loc[grouped["split"] == split]
        outside = (part[DAY_COLUMN] < days[0]) | (part[DAY_COLUMN] > days[-1])
        if outside.any():
            raise ValueError(f"Daily panel found a {split} date outside its period.")
        grids.append(
            part.set_index(["split", DAY_COLUMN, "cluster_id"])
            .reindex(index, fill_value=0)
            .reset_index()
        )
    result = pd.concat(grids, ignore_index=True)
    result[["complaint_count", "novel_count"]] = result[
        ["complaint_count", "novel_count"]
    ].astype(np.int64)
    result[TOTAL_COLUMN] = result.groupby(["split", DAY_COLUMN], observed=True)[
        "complaint_count"
    ].transform("sum")
    result["day_of_week"] = result[DAY_COLUMN].dt.dayofweek.astype(np.int64)
    ordered = result.sort_values(["split", DAY_COLUMN, "cluster_id"])
    return ordered.reset_index(drop=True)


def validate_daily_counts(
    frame: pd.DataFrame,
    clusters: int,
    expected_days: dict[str, int] | None = None,
) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Daily counts are missing columns: {sorted(missing)}")
    result = frame.copy()
    result[DAY_COLUMN] = pd.to_datetime(result[DAY_COLUMN], errors="raise")
    if result[["split", DAY_COLUMN, "cluster_id"]].duplicated().any():
        raise ValueError("Daily counts contain duplicate day-pattern rows.")
    if set(result["split"].unique()) != set(SPLITS):
        raise ValueError("Daily counts must contain the frozen three splits.")
    for name in ("cluster_id", "complaint_count", TOTAL_COLUMN):
        result[name] = pd.to_numeric(result[name], errors="raise").astype(np.int64)
    if (result["complaint_count"] < 0).any():
        raise ValueError("Daily counts must be non-negative.")
    grouped = result.groupby(["split", DAY_COLUMN], observed=True)
    if (grouped["cluster_id"].nunique() != clusters).any():
        raise ValueError("Every day must contain every frozen pattern exactly once.")
    sums = grouped["complaint_count"].sum().to_numpy()
    totals = grouped[TOTAL_COLUMN].first().to_numpy()
    if not np.array_equal(sums, totals):
        raise ValueError("Pattern counts must sum to the daily total.")
    if expected_days:
        by_split = result.groupby("split", observed=True)[DAY_COLUMN]
        actual = by_split.nunique().to_dict()
        if actual != expected_days:
            raise ValueError(f"Days per split do not match the contract: {actual}")
    result["day_of_week"] = result[DAY_COLUMN].dt.dayofweek.astype(np.int64)
    ordered = result.sort_values(["split", DAY_COLUMN, "cluster_id"])
    return ordered.reset_index(drop=True)


def load_frozen_panel(
    config: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, str, str]:
    """Load the daily panel only if its frozen hashes match the config."""
    path = PROJECT_ROOT / config["paths"]["daily_counts"]
    content = path.read_bytes()
    sha256 = hashlib.sha256(content).hexdigest()
    if sha256 != settings["daily_counts_sha256"]:
        raise ValueError("The daily counts SHA-256 is not frozen in this config.")
    source_dvc_hash = load_dvc_hash(
        PROJECT_ROOT / config["paths"]["weekly_patterns_artifacts_dvc"]
    )
    if source_dvc_hash != settings["source_dvc_hash"]:
        raise ValueError("The weekly patterns DVC hash is not frozen in this config.")
    panel = validate_daily_counts(
        pd.read_csv(path), settings["clusters"], settings["expected_days"]
    )
    return panel, sha256, source_dvc_hash


def fit_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["split"] == "fit"].reset_index(drop=True)


def as_weekly_view(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame with the column names the shared weekly helpers group by.

    The bootstrap and the prior checks of weekly_counts only use those names
    to group rows per period; here each period is a day.
    """
    return frame.rename(columns=WEEKLY_ALIASES)


def panel_arrays(frame: pd.DataFrame, clusters: int) -> dict[str, np.ndarray]:
    """(days, clusters) matrices of a complete, sorted panel of one split."""
    days = frame[DAY_COLUMN].drop_duplicates().sort_values()
    shape = (len(days), clusters)
    ordered = frame.sort_values([DAY_COLUMN, "cluster_id"])
    if len(ordered) != shape[0] * shape[1]:
        raise ValueError("Daily panel arrays need a complete day x pattern grid.")
    totals = ordered.groupby(DAY_COLUMN, sort=True)[TOTAL_COLUMN].first()
    return {
        "days": days.to_numpy(),
        "counts": ordered["complaint_count"].to_numpy(dtype=np.int64).reshape(shape),
        "totals": totals.to_numpy(dtype=float),
        "day_of_week": pd.DatetimeIndex(days).dayofweek.to_numpy(dtype=np.int64),
        "recent_share": ordered["recent_share"].to_numpy(dtype=float).reshape(shape)
        if "recent_share" in ordered
        else np.full(shape, np.nan),
    }


def write_panel(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out[DAY_COLUMN] = out[DAY_COLUMN].dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, lineterminator="\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()
