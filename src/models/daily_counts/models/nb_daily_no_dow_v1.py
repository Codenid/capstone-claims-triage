"""D-B2: the D-A negative binomial without the day-of-week effect (§28.2).

It measures what the day-of-week effect adds: same memory, same dispersion
structure, shares used as they come from the previous days.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pymc as pm

from src.models.daily_counts.models import nb_daily_hierarchical_v1 as base

MODEL_ID = "nb_daily_no_dow_v1"
SOURCE_STATUS = "native_versioned_source"
HISTORICAL_RUN_ID = None
DAY_OF_WEEK_EFFECT = False
FORMULA = (
    "count[c,d] ~ NegativeBinomial(mu[c,d], alpha[c]); "
    "mu[c,d] = daily_total[d] * r[c,d]; "
    "r[c,d] = (sum_k discount**(k-1) * count[c,d-k] + 1) / "
    "(sum_k discount**(k-1) * daily_total[d-k] + clusters); "
    "alpha[c] = exp(log_alpha_global + (zero_sum_basis @ log_alpha_contrast)[c])"
)
DIAGNOSTIC_VARIABLES = ("log_alpha_global", "log_alpha_sigma", "log_alpha_contrast")
EXPECTED_VARIABLE = base.EXPECTED_VARIABLE
OBSERVED_VARIABLE = base.OBSERVED_VARIABLE
SUMMARY_VARIABLES = DIAGNOSTIC_VARIABLES + ("alpha",)
TRACE_VARIABLES = ("log_alpha_global", "log_alpha_sigma")
MODEL_SPEC = {
    **base.MODEL_SPEC,
    "model_id": MODEL_ID,
    "formula": FORMULA,
    "mean_link": "discounted_share_input",
    "day_of_week_effect": DAY_OF_WEEK_EFFECT,
    "diagnostic_variables": DIAGNOSTIC_VARIABLES,
    "summary_variables": SUMMARY_VARIABLES,
    "trace_variables": TRACE_VARIABLES,
}


def prepare_frame(frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    return base.prepare_frame(frame, settings)


def build_model(fit: pd.DataFrame, settings: dict[str, Any]) -> pm.Model:
    return base.build_model(fit, settings, day_of_week_effect=False)
