"""Data preparation for weekly count models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.weekly_counts.contracts import REQUIRED_COLUMNS, SPLITS


def _coerce_complete_week(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.lower().map({"true": True, "false": False})
    if normalized.isna().any():
        raise ValueError("M9 is_complete_week must contain only booleans.")
    return normalized.astype(bool)


def validate_weekly_counts(
    frame: pd.DataFrame,
    clusters: int,
    expected_complete_weeks: dict[str, int] | None = None,
) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"M9 weekly counts are missing columns: {names}")

    result = frame.copy()
    result["week"] = pd.to_datetime(result["week"], errors="raise")
    result["is_complete_week"] = _coerce_complete_week(result["is_complete_week"])
    if result[["split", "week", "cluster_id"]].duplicated().any():
        raise ValueError("M9 weekly counts contain duplicate week-cluster rows.")
    if set(result["split"].unique()) != set(SPLITS):
        raise ValueError("M9 weekly counts must contain the frozen three splits.")

    numeric_columns = ("cluster_id", "complaint_count", "weekly_total")
    for name in numeric_columns:
        if result[name].isna().any():
            raise ValueError(f"M9 {name} contains missing values.")
        numeric = pd.to_numeric(result[name], errors="raise")
        if not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"M9 {name} must contain integers.")
        result[name] = numeric.astype(np.int64)

    if (result["complaint_count"] < 0).any():
        raise ValueError("M9 complaint counts must be non-negative.")
    if (result["weekly_total"] <= 0).any():
        raise ValueError("M9 weekly totals must be positive.")

    expected_clusters = set(range(clusters))
    cluster_sets = result.groupby(["split", "week"], observed=True)["cluster_id"].agg(set)
    if any(values != expected_clusters for values in cluster_sets):
        raise ValueError("M9 each week must contain every frozen cluster exactly once.")

    grouped = result.groupby(["split", "week"], observed=True)
    total_counts = grouped["complaint_count"].sum()
    total_nunique = grouped["weekly_total"].nunique()
    recorded_totals = grouped["weekly_total"].first()
    if (total_nunique != 1).any():
        raise ValueError("M9 weekly_total must be constant within each week.")
    if not np.array_equal(total_counts.to_numpy(), recorded_totals.to_numpy()):
        raise ValueError("M9 cluster counts must sum to weekly_total.")

    complete = result.loc[result["is_complete_week"]].copy()
    if expected_complete_weeks:
        actual = complete.groupby("split", observed=True)["week"].nunique().to_dict()
        if actual != expected_complete_weeks:
            raise ValueError(
                f"M9 complete weeks do not match the frozen contract: {actual}"
            )
    return complete.sort_values(["week", "cluster_id"]).reset_index(drop=True)


def prepare_model_frame(
    frame: pd.DataFrame,
    clusters: int,
    expected_complete_weeks: dict[str, int],
    time_scale_days: float,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    complete = validate_weekly_counts(frame, clusters, expected_complete_weeks)
    fit_weeks = complete.loc[complete["split"] == "fit", "week"].drop_duplicates()
    fit_weeks = fit_weeks.sort_values().reset_index(drop=True)
    center = fit_weeks.iloc[0] + (fit_weeks.iloc[-1] - fit_weeks.iloc[0]) / 2
    complete["time_years"] = (
        (complete["week"] - center).dt.total_seconds() / (time_scale_days * 86400)
    )
    return complete, center


def fit_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the only rows that may inform a posterior."""
    return frame.loc[frame["split"] == "fit"].reset_index(drop=True)

