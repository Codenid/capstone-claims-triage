"""Dirichlet-Multinomial around the shares of the previous 4 complete weeks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from ..data import RECENT_WEEKS, windowed_weeks

MODEL_ID = "dirichlet_multinomial_rolling_4_v1"
SOURCE_STATUS = "native_versioned_source"
HISTORICAL_RUN_ID = None
FORMULA = (
    "counts[t] ~ DirichletMultinomial(weekly_total[t], kappa * recent_share[t]); "
    "recent_share[c,t] = (counts of c in the previous 4 complete weeks + 1) / "
    "(complaints in those weeks + clusters); "
    "log_kappa ~ Normal(log_kappa_mean, log_kappa_sigma); kappa = exp(log_kappa); "
    "rho = 1 / (kappa + 1)"
)
DIAGNOSTIC_VARIABLES = ("log_kappa",)
SUMMARY_VARIABLES = ("log_kappa", "kappa", "rho")
TRACE_VARIABLES = ("log_kappa",)
MODEL_SPEC = {
    "model_id": MODEL_ID,
    "source_status": SOURCE_STATUS,
    "historical_run_id": HISTORICAL_RUN_ID,
    "family": "dirichlet_multinomial",
    "formula": FORMULA,
    "cluster_interpretation": "operational partition, not natural categories",
    "weekly_total_usage": "observed exposure; this is a conditional forecast",
    "mean_link": "previous_weeks_share_input",
    "concentration": "global",
    "share_window_weeks": RECENT_WEEKS,
    "predictive_draw_sum_constrained": True,
    "prior_strategy": "not_applicable_shares_are_data",
    "diagnostic_variables": DIAGNOSTIC_VARIABLES,
    "summary_variables": SUMMARY_VARIABLES,
    "trace_variables": TRACE_VARIABLES,
}


def prepare_panel(panel: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The first 4 weeks have no full window, so the model cannot use them."""
    return windowed_weeks(panel)


def build_model(panel: dict[str, np.ndarray], settings: dict[str, Any]) -> pm.Model:
    priors = settings["priors"]
    coords = {
        "week": np.arange(len(panel["weekly_total"])),
        "cluster": np.arange(settings["clusters"]),
    }

    with pm.Model(coords=coords) as model:
        weekly_total = pm.Data("weekly_total", panel["weekly_total"], dims="week")
        recent_share = pm.Data(
            "recent_share", panel["recent_share"], dims=("week", "cluster")
        )
        log_kappa = pm.Normal(
            "log_kappa",
            mu=priors["log_kappa_mean"],
            sigma=priors["log_kappa_sigma"],
        )
        kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
        pm.Deterministic("rho", 1 / (kappa + 1))
        concentration = pm.Deterministic(
            "concentration", kappa * recent_share, dims=("week", "cluster")
        )
        pm.DirichletMultinomial(
            "observed",
            n=weekly_total,
            a=concentration,
            observed=panel["counts"],
            dims=("week", "cluster"),
        )

    return model
