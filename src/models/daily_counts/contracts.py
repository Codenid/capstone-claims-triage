"""Contracts of the daily count models (models_plan.md §28).

The daily panel mirrors the weekly one with days instead of weeks: one row per
split, day and pattern, the day's total as the observed exposure, and the
same predictive quantiles and intervals, so the weekly metrics apply unchanged.
"""

from src.models.weekly_counts.contracts import INTERVALS, PREDICTIVE_QUANTILES, SPLITS

DAY_COLUMN = "day"
TOTAL_COLUMN = "daily_total"
REQUIRED_COLUMNS = {"split", DAY_COLUMN, "cluster_id", "complaint_count", TOTAL_COLUMN}
# The weekly helpers (bootstrap, prior checks) group by these names.
WEEKLY_ALIASES = {DAY_COLUMN: "week", TOTAL_COLUMN: "weekly_total"}
DAYS_PER_WEEK = 7

__all__ = [
    "DAYS_PER_WEEK",
    "DAY_COLUMN",
    "INTERVALS",
    "PREDICTIVE_QUANTILES",
    "REQUIRED_COLUMNS",
    "SPLITS",
    "TOTAL_COLUMN",
    "WEEKLY_ALIASES",
]
