"""Weekly status of the 40 patterns: CUSUM, level and composition (§25)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.experiment import PROJECT_ROOT
from src.evaluation.final_confirmation import M9_MODEL, M10_REPORT, load_settings
from src.models.persistent_change import (
    DESIGN_FILES,
    add_alarms,
    load_alpha,
    windowed_rows,
)
from src.models.representation_comparison import read_json
from src.models.weekly_composition.data import RECENT_WEEKS, composition_panel
from src.models.weekly_counts.run import load_frozen_frame

# About 3 months of complete weeks (models_plan.md §23).
LEVEL_LAG_WEEKS = 13
BAND_QUANTILES = {"level_low": 0.05, "level_high": 0.95}


def long_table(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """A week x pattern table as rows of week, cluster_id and `name`."""
    wide = frame.rename_axis(index="week", columns="cluster_id").reset_index()
    rows = wide.melt(id_vars="week", var_name="cluster_id", value_name=name)
    return rows.astype({"cluster_id": "int64"})


def level_changes(counts: pd.DataFrame) -> pd.DataFrame:
    """Δ: share of the last 4 weeks against the same weeks 13 weeks earlier."""
    window = pd.DataFrame(counts.rolling(RECENT_WEEKS).sum())
    share = window.div(window.sum(axis=1), axis=0)
    return share / share.shift(LEVEL_LAG_WEEKS) - 1


def level_percentiles(changes: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Share of each pattern's reference changes at or below its current change."""
    percentiles = {}
    for cluster in changes.columns:
        ranked = np.sort(reference[cluster].dropna().to_numpy())
        values = changes[cluster].to_numpy(dtype=float)
        ranks = np.searchsorted(ranked, values, side="right") / len(ranked)
        percentiles[cluster] = np.where(np.isnan(values), np.nan, ranks)
    return pd.DataFrame(percentiles, index=changes.index)


def active_alerts(rows: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """Whether the CUSUM alarmed in any of the last `weeks` closed weeks."""
    alarms = rows.pivot(index="week", columns="cluster_id", values="cusum_alarm")
    active = alarms.astype(float).rolling(weeks, min_periods=1).max() > 0
    return long_table(active, "active_alert")


def level_table(panel: dict[str, np.ndarray]) -> pd.DataFrame:
    """Δ, its percentile among the fit changes and the usual fit band."""
    counts = pd.DataFrame(panel["counts"], index=pd.DatetimeIndex(panel["week"]))
    changes = level_changes(counts)
    fit = changes.loc[panel["split"] == "fit"]
    level = long_table(changes, "level_change").merge(
        long_table(level_percentiles(changes, fit), "level_percentile"),
        on=["week", "cluster_id"],
    )
    bands = pd.DataFrame(
        {name: fit.quantile(quantile) for name, quantile in BAND_QUANTILES.items()}
    ).rename_axis("cluster_id").reset_index()
    return level.merge(bands, on="cluster_id")


def composition_context() -> pd.DataFrame:
    """Observed share and DM-R4's expected share with its 95% interval."""
    predictions = pd.read_csv(M10_REPORT / "predictions.csv", parse_dates=["week"])
    total = predictions["weekly_total"]
    return pd.DataFrame(
        {
            "week": predictions["week"],
            "cluster_id": predictions["cluster_id"],
            "share": predictions["complaint_count"] / total,
            "expected_share": predictions["model_mean"] / total,
            "share_low": predictions["model_p025"] / total,
            "share_high": predictions["model_p975"] / total,
        }
    )


def pattern_info(config: dict[str, Any]) -> pd.DataFrame:
    """Words, main product and whether a template dominates each pattern."""
    summary = pd.read_csv(PROJECT_ROOT / config["paths"]["cluster_summary"])
    threshold = config["triage"]["template_threshold"]
    dominated = summary["dominant_template_fraction"] > threshold
    return summary[["cluster_id", "main_product", "representative_terms"]].assign(
        template_dominated=dominated
    )


def pattern_history(config: dict[str, Any], seed: int) -> pd.DataFrame:
    """Every pattern and complete week with a full window, as the panel shows it."""
    m9 = load_settings("weekly_counts", M9_MODEL)
    frame, _, _, _ = load_frozen_frame(config, m9)
    panel = composition_panel(frame, m9["clusters"])
    design_dir = PROJECT_ROOT / config["paths"]["persistent_change_reports"]
    design = read_json(design_dir / DESIGN_FILES["results"])
    rows = windowed_rows(panel, load_alpha(m9, seed))
    rows = add_alarms(rows, design["thresholds"], design["settings"]["cusum_k"])
    history = rows.merge(
        active_alerts(rows, config["triage"]["alert_weeks"]),
        on=["week", "cluster_id"],
    )
    keys = ["week", "cluster_id"]
    history = history.merge(level_table(panel), on=keys, how="left")
    history = history.merge(composition_context(), on=keys, how="left")
    return history.merge(pattern_info(config), on="cluster_id", how="left")


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-ready rows: plain numbers, ISO dates and null for missing values."""
    dated = frame.assign(week=frame["week"].dt.strftime("%Y-%m-%d"))
    return json.loads(dated.to_json(orient="records") or "[]")


def weekly_panel(history: pd.DataFrame, week: pd.Timestamp) -> list[dict[str, Any]]:
    """The 40 patterns at the close of `week`, active alerts first."""
    rows = history.loc[history["week"] == week]
    if rows.empty:
        raise ValueError(f"No complete week starts on {week:%Y-%m-%d}.")
    return records(rows.sort_values(["active_alert", "cusum"], ascending=False))


def last_closed_week(history: pd.DataFrame, week: pd.Timestamp) -> pd.Timestamp:
    """The latest complete week before `week`: what a complaint received then sees."""
    earlier = history.loc[history["week"] < week, "week"]
    if earlier.empty:
        raise ValueError(f"No closed week before {week:%Y-%m-%d}.")
    return earlier.max()
