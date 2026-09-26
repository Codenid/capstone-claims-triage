"""Static Poisson candidate for weekly cluster counts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pymc as pm

MODEL_ID = "poisson_static_pymc_v1"
SOURCE_STATUS = "native_versioned_source"
HISTORICAL_RUN_ID = None
FORMULA = (
    "complaint_count[c,t] ~ Poisson(mu[c,t]); "
    "mu[c,t] = weekly_total[t] * rate[c]; "
    "rate ~ Dirichlet(rate_concentration)"
)
DIAGNOSTIC_VARIABLES = ("rate",)
EXPECTED_VARIABLE = "mu"
OBSERVED_VARIABLE = "observed"
SUMMARY_VARIABLES = ("rate",)
TRACE_VARIABLES = ("rate",)
MODEL_SPEC = {
    "model_id": MODEL_ID,
    "source_status": SOURCE_STATUS,
    "historical_run_id": HISTORICAL_RUN_ID,
    "family": "poisson",
    "formula": FORMULA,
    "weekly_total_usage": "observed exposure; this is a conditional forecast",
    "mean_link": "static_simplex",
    "mean_sum_constrained": True,
    "predictive_draw_sum_constrained": False,
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
        rate = pm.Dirichlet(
            "rate",
            a=np.full(clusters, priors["rate_concentration"]),
            dims="cluster",
        )
        mu = pm.Deterministic(
            "mu",
            weekly_total * rate[cluster_index],
            dims="fit_observation",
        )
        pm.Poisson(
            "observed",
            mu=mu,
            observed=fit["complaint_count"].to_numpy(dtype=np.int64),
            dims="fit_observation",
        )

    return model
