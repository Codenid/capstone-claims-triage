"""Normalized-softmax Negative Binomial model with linear trends."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from ..negative_binomial import zero_sum_basis

MODEL_ID = "nb_softmax_linear_v2"
SOURCE_STATUS = "verified_equivalent_to_run"
HISTORICAL_RUN_ID = "fb71dac0a0ae40b4be9ff2628637bcfa"
FORMULA = (
    "complaint_count[c,t] ~ NegativeBinomial(mu[c,t], alpha[c]); "
    "eta[c,t] = log_rate[c] + annual_trend[c] * time_years[t]; "
    "share[c,t] = softmax_c(eta[:,t]); "
    "mu[c,t] = weekly_total[t] * share[c,t]"
)
DIAGNOSTIC_VARIABLES = (
    "log_rate_sigma",
    "log_rate_contrast",
    "annual_trend_sigma",
    "annual_trend_contrast",
    "log_alpha_global",
    "log_alpha_sigma",
    "log_alpha_contrast",
)
EXPECTED_VARIABLE = "mu"
OBSERVED_VARIABLE = "observed"
SUMMARY_VARIABLES = DIAGNOSTIC_VARIABLES + ("log_rate", "annual_trend", "alpha")
TRACE_VARIABLES = (
    "log_rate_sigma",
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
    "cluster_interpretation": "operational partition, not natural categories",
    "weekly_total_usage": "observed exposure; this is a conditional forecast",
    "mean_link": "softmax",
    "mean_sum_constrained": True,
    "predictive_draw_sum_constrained": False,
    "zero_sum_parameterization": "scaled Helmert contrasts with K-1 dimensions",
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
        "contrast": np.arange(clusters - 1),
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
        log_weekly_total = pm.Data(
            "log_weekly_total",
            np.log(fit["weekly_total"].to_numpy(dtype=float)),
            dims="fit_observation",
        )

        basis = pm.Data(
            "zero_sum_basis",
            zero_sum_basis(clusters),
            dims=("cluster", "contrast"),
        )
        log_rate_sigma = pm.HalfNormal(
            "log_rate_sigma",
            sigma=priors["log_rate_sigma"],
        )
        log_rate_contrast = pm.Normal(
            "log_rate_contrast",
            mu=0,
            sigma=1,
            dims="contrast",
        )
        log_rate = pm.Deterministic(
            "log_rate",
            log_rate_sigma * pt.dot(basis, log_rate_contrast),
            dims="cluster",
        )

        annual_trend_sigma = pm.HalfNormal(
            "annual_trend_sigma",
            sigma=priors["annual_trend_sigma"],
        )
        annual_trend_contrast = pm.Normal(
            "annual_trend_contrast",
            mu=0,
            sigma=1,
            dims="contrast",
        )
        annual_trend = pm.Deterministic(
            "annual_trend",
            annual_trend_sigma * pt.dot(basis, annual_trend_contrast),
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
        log_alpha_contrast = pm.Normal(
            "log_alpha_contrast",
            mu=0,
            sigma=1,
            dims="contrast",
        )
        log_alpha = log_alpha_global + log_alpha_sigma * pt.dot(
            basis, log_alpha_contrast
        )
        alpha = pm.Deterministic(
            "alpha",
            pt.exp(log_alpha),
            dims="cluster",
        )

        logits = log_rate + pt.outer(time_years, annual_trend)
        log_share = (
            logits[pt.arange(len(fit)), cluster_index]
            - pt.logsumexp(logits, axis=1)
        )
        mu = pm.Deterministic(
            "mu",
            pt.exp(log_weekly_total + log_share),
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
