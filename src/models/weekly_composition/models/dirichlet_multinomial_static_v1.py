"""Dirichlet-Multinomial with one static share per cluster (DM-V1, §7.7)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MODEL_ID = "dirichlet_multinomial_static_v1"
SOURCE_STATUS = "native_versioned_source"
HISTORICAL_RUN_ID = None
FORMULA = (
    "share ~ Dirichlet(share_concentration); "
    "counts[t] ~ DirichletMultinomial(weekly_total[t], kappa * share); "
    "log_kappa ~ Normal(log_kappa_mean, log_kappa_sigma); kappa = exp(log_kappa); "
    "rho = 1 / (kappa + 1)"
)
DIAGNOSTIC_VARIABLES = ("share", "log_kappa")
SUMMARY_VARIABLES = ("share", "log_kappa", "kappa", "rho")
TRACE_VARIABLES = ("log_kappa",)
MODEL_SPEC = {
    "model_id": MODEL_ID,
    "source_status": SOURCE_STATUS,
    "historical_run_id": HISTORICAL_RUN_ID,
    "family": "dirichlet_multinomial",
    "formula": FORMULA,
    "cluster_interpretation": "operational partition, not natural categories",
    "weekly_total_usage": "observed exposure; this is a conditional forecast",
    "mean_link": "static_share",
    "concentration": "global",
    "predictive_draw_sum_constrained": True,
    "prior_strategy": "uniform",
    "diagnostic_variables": DIAGNOSTIC_VARIABLES,
    "summary_variables": SUMMARY_VARIABLES,
    "trace_variables": TRACE_VARIABLES,
}


def build_model(panel: dict[str, np.ndarray], settings: dict[str, Any]) -> pm.Model:
    priors = settings["priors"]
    clusters = settings["clusters"]
    coords = {
        "week": np.arange(len(panel["weekly_total"])),
        "cluster": np.arange(clusters),
    }

    with pm.Model(coords=coords) as model:
        weekly_total = pm.Data("weekly_total", panel["weekly_total"], dims="week")
        # A constant concentration: a pm.Data input would make PyMC resample
        # the share from its prior in posterior predictive.
        share = pm.Dirichlet(
            "share",
            a=np.full(clusters, priors["share_concentration"]),
            dims="cluster",
        )
        log_kappa = pm.Normal(
            "log_kappa",
            mu=priors["log_kappa_mean"],
            sigma=priors["log_kappa_sigma"],
        )
        kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
        pm.Deterministic("rho", 1 / (kappa + 1))
        concentration = pm.Deterministic(
            "concentration",
            pt.broadcast_to(kappa * share, (weekly_total.shape[0], clusters)),
            dims=("week", "cluster"),
        )
        pm.DirichletMultinomial(
            "observed",
            n=weekly_total,
            a=concentration,
            observed=panel["counts"],
            dims=("week", "cluster"),
        )

    return model
