"""Transparent mathematical reconstruction of the rejected M9 V1 model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

MODEL_ID = "nb_independent_linear_v1"
SOURCE_STATUS = "reconstructed_from_mlflow_spec"
HISTORICAL_RUN_ID = "3aff4cea426f435ab50b41788de998ff"
FORMULA = (
    "complaint_count[c,t] ~ NegativeBinomial(mu[c,t], alpha[c]); "
    "mu[c,t] = weekly_total[t] * exp(log_rate[c] + "
    "annual_trend[c] * time_years[t])"
)
DIAGNOSTIC_VARIABLES = (
    "log_rate_global",
    "log_rate_sigma",
    "log_rate_offset",
    "annual_trend_global",
    "annual_trend_sigma",
    "annual_trend_offset",
    "log_alpha_global",
    "log_alpha_sigma",
    "log_alpha_offset",
)
EXPECTED_VARIABLE = "mu"
OBSERVED_VARIABLE = "observed"
SUMMARY_VARIABLES = DIAGNOSTIC_VARIABLES + ("log_rate", "annual_trend", "alpha")
TRACE_VARIABLES = (
    "log_rate_global",
    "log_rate_sigma",
    "annual_trend_global",
    "annual_trend_sigma",
    "log_alpha_global",
    "log_alpha_sigma",
)
MODEL_SPEC = {
    "model_id": MODEL_ID,
    "source_status": SOURCE_STATUS,
    "historical_run_id": HISTORICAL_RUN_ID,
    "family": "negative_binomial",
    "formula": FORMULA,
    "negative_binomial_variance": "mu + mu**2 / alpha",
    "mean_link": "independent_log",
    "mean_sum_constrained": False,
    "diagnostic_variables": DIAGNOSTIC_VARIABLES,
    "summary_variables": SUMMARY_VARIABLES,
    "expected_variable": EXPECTED_VARIABLE,
    "observed_variable": OBSERVED_VARIABLE,
    "trace_variables": TRACE_VARIABLES,
}


def build_model(fit: pd.DataFrame, settings: dict[str, Any]) -> pm.Model:
    priors = settings["priors"]
    clusters = settings["clusters"]
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
        time_years = pm.Data(
            "time_years",
            fit["time_years"].to_numpy(dtype=float),
            dims="fit_observation",
        )
        weekly_total = pm.Data(
            "weekly_total",
            fit["weekly_total"].to_numpy(dtype=float),
            dims="fit_observation",
        )

        log_rate_global = pm.Normal(
            "log_rate_global",
            mu=priors["log_rate_global_mean"],
            sigma=priors["log_rate_global_sigma"],
        )
        log_rate_sigma = pm.HalfNormal(
            "log_rate_sigma",
            sigma=priors["log_rate_sigma"],
        )
        log_rate_offset = pm.Normal(
            "log_rate_offset",
            mu=0,
            sigma=1,
            dims="cluster",
        )
        log_rate = pm.Deterministic(
            "log_rate",
            log_rate_global + log_rate_sigma * log_rate_offset,
            dims="cluster",
        )

        annual_trend_global = pm.Normal(
            "annual_trend_global",
            mu=priors["annual_trend_global_mean"],
            sigma=priors["annual_trend_global_sigma"],
        )
        annual_trend_sigma = pm.HalfNormal(
            "annual_trend_sigma",
            sigma=priors["annual_trend_sigma"],
        )
        annual_trend_offset = pm.Normal(
            "annual_trend_offset",
            mu=0,
            sigma=1,
            dims="cluster",
        )
        annual_trend = pm.Deterministic(
            "annual_trend",
            annual_trend_global + annual_trend_sigma * annual_trend_offset,
            dims="cluster",
        )

        log_alpha_global = pm.Normal(
            "log_alpha_global",
            mu=priors["log_alpha_global_mean"],
            sigma=priors["log_alpha_global_sigma"],
        )
        log_alpha_sigma = pm.HalfNormal(
            "log_alpha_sigma",
            sigma=priors["log_alpha_sigma"],
        )
        log_alpha_offset = pm.Normal(
            "log_alpha_offset",
            mu=0,
            sigma=1,
            dims="cluster",
        )
        alpha = pm.Deterministic(
            "alpha",
            pt.exp(log_alpha_global + log_alpha_sigma * log_alpha_offset),
            dims="cluster",
        )

        mu = pm.Deterministic(
            "mu",
            weekly_total
            * pt.exp(
                log_rate[cluster_index]
                + annual_trend[cluster_index] * time_years
            ),
            dims="fit_observation",
        )
        pm.NegativeBinomial(
            "observed",
            mu=mu,
            alpha=alpha[cluster_index],
            observed=fit["complaint_count"].to_numpy(dtype=np.int64),
            dims="fit_observation",
        )

    return model
