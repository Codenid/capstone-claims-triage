"""Negative Binomial around the shares of the previous 4 complete weeks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from ..rolling_reference import row_shares

MODEL_ID = "nb_rolling_4_global_v1"
SOURCE_STATUS = "native_versioned_source"
HISTORICAL_RUN_ID = None
WINDOW_WEEKS = 4
FORMULA = (
    "complaint_count[c,t] ~ NegativeBinomial(mu[c,t], alpha); "
    "mu[c,t] = weekly_total[t] * recent_share[c,t]; "
    "recent_share[c,t] = (counts of c in the previous 4 complete weeks + 1) / "
    "(complaints in those weeks + clusters); "
    "log_alpha ~ Normal(log_alpha_mean, log_alpha_sigma); "
    "alpha = exp(log_alpha)"
)
DIAGNOSTIC_VARIABLES = ("log_alpha",)
EXPECTED_VARIABLE = "mu"
OBSERVED_VARIABLE = "observed"
SUMMARY_VARIABLES = ("log_alpha", "alpha")
TRACE_VARIABLES = ("log_alpha",)
MODEL_SPEC = {
    "model_id": MODEL_ID,
    "source_status": SOURCE_STATUS,
    "historical_run_id": HISTORICAL_RUN_ID,
    "family": "negative_binomial",
    "formula": FORMULA,
    "negative_binomial_variance": "mu + mu**2 / alpha",
    "cluster_interpretation": "operational partition, not natural categories",
    "weekly_total_usage": "observed exposure; this is a conditional forecast",
    "mean_link": "previous_weeks_share_input",
    "mean_sum_constrained": True,
    "predictive_draw_sum_constrained": False,
    "dispersion": "global",
    "share_window_weeks": WINDOW_WEEKS,
    "prior_strategy": "not_applicable_shares_are_data",
    "diagnostic_variables": DIAGNOSTIC_VARIABLES,
    "summary_variables": SUMMARY_VARIABLES,
    "expected_variable": EXPECTED_VARIABLE,
    "observed_variable": OBSERVED_VARIABLE,
    "trace_variables": TRACE_VARIABLES,
}


def prepare_frame(frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Add recent_share and drop the first 4 weeks, which have no full window."""
    shares = row_shares(frame, settings["clusters"], WINDOW_WEEKS)
    prepared = frame.assign(recent_share=shares)
    return prepared.loc[np.isfinite(shares)].reset_index(drop=True)


def build_model(fit: pd.DataFrame, settings: dict[str, Any]) -> pm.Model:
    priors = settings["priors"]
    coords = {"fit_observation": np.arange(len(fit))}

    with pm.Model(coords=coords) as model:
        weekly_total = pm.Data(
            "weekly_total",
            fit["weekly_total"].to_numpy(dtype=float),
            dims="fit_observation",
        )
        recent_share = pm.Data(
            "recent_share",
            fit["recent_share"].to_numpy(dtype=float),
            dims="fit_observation",
        )
        log_alpha = pm.Normal(
            "log_alpha",
            mu=priors["log_alpha_mean"],
            sigma=priors["log_alpha_sigma"],
        )
        alpha = pm.Deterministic("alpha", pt.exp(log_alpha))
        mu = pm.Deterministic(
            "mu",
            weekly_total * recent_share,
            dims="fit_observation",
        )
        pm.NegativeBinomial(
            "observed",
            mu=mu,
            alpha=alpha,
            observed=fit["complaint_count"].to_numpy(dtype=np.int64),
            dims="fit_observation",
        )

    return model
