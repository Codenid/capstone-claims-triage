"""Static Negative Binomial model with one global dispersion."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

MODEL_ID = "nb_static_global_v3"
SOURCE_STATUS = "native_versioned_source"
HISTORICAL_RUN_ID = None
FORMULA = (
    "complaint_count[c,t] ~ NegativeBinomial(mu[c,t], alpha); "
    "mu[c,t] = weekly_total[t] * share[c]; "
    "share ~ Dirichlet(share_concentration); "
    "log_alpha ~ Normal(log_alpha_mean, log_alpha_sigma); "
    "alpha = exp(log_alpha)"
)
DIAGNOSTIC_VARIABLES = ("share", "log_alpha")
EXPECTED_VARIABLE = "mu"
OBSERVED_VARIABLE = "observed"
SUMMARY_VARIABLES = ("share", "log_alpha", "alpha")
TRACE_VARIABLES = ("log_alpha", "share")
MODEL_SPEC = {
    "model_id": MODEL_ID,
    "source_status": SOURCE_STATUS,
    "historical_run_id": HISTORICAL_RUN_ID,
    "family": "negative_binomial",
    "formula": FORMULA,
    "negative_binomial_variance": "mu + mu**2 / alpha",
    "cluster_interpretation": "operational partition, not natural categories",
    "weekly_total_usage": "observed exposure; this is a conditional forecast",
    "mean_link": "static_simplex",
    "mean_sum_constrained": True,
    "predictive_draw_sum_constrained": False,
    "dispersion": "global",
    "prior_strategy": "uniform",
    "diagnostic_variables": DIAGNOSTIC_VARIABLES,
    "summary_variables": SUMMARY_VARIABLES,
    "expected_variable": EXPECTED_VARIABLE,
    "observed_variable": OBSERVED_VARIABLE,
    "trace_variables": TRACE_VARIABLES,
}


def build_model(fit: pd.DataFrame, settings: dict[str, Any]) -> pm.Model:
    clusters = settings["clusters"]
    priors = settings["priors"]
    coords = {
        "cluster": np.arange(clusters),
        "fit_observation": np.arange(len(fit)),
    }

    with pm.Model(coords=coords) as model:
        cluster_index = pm.Data(
            "cluster_index",
            fit["cluster_id"].to_numpy(dtype=np.int64),
            dims="fit_observation",
        )
        weekly_total = pm.Data(
            "weekly_total",
            fit["weekly_total"].to_numpy(dtype=float),
            dims="fit_observation",
        )
        # Keep the concentration constant: a pm.Data input would make PyMC
        # resample share from its prior during posterior prediction.
        share = pm.Dirichlet(
            "share",
            a=np.full(clusters, priors["share_concentration"]),
            dims="cluster",
        )
        log_alpha = pm.Normal(
            "log_alpha",
            mu=priors["log_alpha_mean"],
            sigma=priors["log_alpha_sigma"],
        )
        alpha = pm.Deterministic("alpha", pt.exp(log_alpha))
        mu = pm.Deterministic(
            "mu",
            weekly_total * share[cluster_index],
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
