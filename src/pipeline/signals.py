"""Retraining signals measured on every score run (models_plan.md §29).

Default rule: M9, M10 and the alert thresholds are refitted every quarter
on a moving window; classifiers and patterns only when a signal asks for it.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import f1_score

NOVEL_FRACTION_LIMIT = 0.03
COVERAGE_80_FLOOR = 0.70
COVERAGE_WEEKS = 12
MACRO_F1_RELATIVE_DROP = 0.05


def novelty_signal(complaints: pd.DataFrame) -> dict[str, Any]:
    fraction = float(complaints["is_novel"].mean()) if len(complaints) else float("nan")
    return {
        "novel_fraction": fraction,
        "limit": NOVEL_FRACTION_LIMIT,
        "warning": bool(fraction > NOVEL_FRACTION_LIMIT),
    }


def coverage_signal(weekly: pd.DataFrame) -> dict[str, Any]:
    """80 % coverage of M9 over the last scored, complete weeks."""
    rows = weekly.loc[weekly["scored"] & weekly["is_complete"]]
    weeks = sorted(rows["period"].unique())[-COVERAGE_WEEKS:]
    recent = rows.loc[rows["period"].isin(weeks)]
    if recent.empty:
        return {
            "coverage_80": None,
            "weeks": 0,
            "floor": COVERAGE_80_FLOOR,
            "warning": False,
        }
    inside = (recent["complaint_count"] >= recent["model_p10"]) & (
        recent["complaint_count"] <= recent["model_p90"]
    )
    coverage = float(inside.mean())
    return {
        "coverage_80": coverage,
        "weeks": len(weeks),
        "floor": COVERAGE_80_FLOOR,
        "warning": bool(coverage < COVERAGE_80_FLOOR),
    }


def mixture_signal(mixture: pd.DataFrame) -> dict[str, Any]:
    flagged = mixture.loc[mixture["rare_mixture"].fillna(False).astype(bool), "period"]
    return {
        "rare_weeks": [str(pd.Timestamp(p).date()) for p in flagged],
        "scored_weeks": int(mixture["p_value"].notna().sum()),
    }


def macro_f1_signal(
    complaints: pd.DataFrame,
    frozen_macro_f1: float,
) -> dict[str, Any]:
    """Macro-F1 of the top-1 issue when the table brings its canonical issue."""
    unlabeled = "Issue canonical" not in complaints or bool(
        complaints["Issue canonical"].isna().all()
    )
    if unlabeled:
        return {
            "macro_f1": None,
            "frozen": frozen_macro_f1,
            "rows": 0,
            "warning": False,
        }
    labeled = complaints.loc[complaints["Issue canonical"].notna()]
    truth = labeled["Issue canonical"].astype(str).to_numpy()
    predicted = labeled["issue_top1"].astype(str).to_numpy()
    score = float(f1_score(truth, predicted, average="macro"))
    drop = 1 - score / frozen_macro_f1
    return {
        "macro_f1": score,
        "frozen": frozen_macro_f1,
        "rows": int(len(labeled)),
        "relative_drop": float(drop),
        "warning": bool(drop > MACRO_F1_RELATIVE_DROP),
    }


def retraining_signals(
    complaints: pd.DataFrame,
    weekly: pd.DataFrame,
    mixture: pd.DataFrame,
    frozen_macro_f1: float,
) -> dict[str, Any]:
    signals: dict[str, Any] = {
        "novelty": novelty_signal(complaints),
        "coverage": coverage_signal(weekly),
        "mixture": mixture_signal(mixture),
        "macro_f1": macro_f1_signal(complaints, frozen_macro_f1),
    }
    warnings = [
        name
        for name, value in signals.items()
        if isinstance(value, dict) and value.get("warning")
    ]
    signals["retrain_recommended"] = bool(warnings)
    signals["warnings"] = warnings
    signals["default_rule"] = (
        "M9, M10 and the alert thresholds every quarter on a moving window; "
        "classifiers and patterns only when a signal warns"
    )
    return signals
