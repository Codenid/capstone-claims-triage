"""Contracts for weekly count models."""

SPLITS = ("fit", "calibration", "validation")
REQUIRED_COLUMNS = {
    "split",
    "week",
    "cluster_id",
    "complaint_count",
    "is_complete_week",
    "weekly_total",
}
PREDICTIVE_QUANTILES = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)
INTERVALS = (
    (0.50, "p25", "p75"),
    (0.20, "p10", "p90"),
    (0.05, "p025", "p975"),
)
