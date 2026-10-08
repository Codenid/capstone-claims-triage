"""D-D: D-A with zero inflation per pattern (§28.2).

    count[c,d] ~ ZeroInflatedNegativeBinomial(psi[c], mu[c,d], alpha[c])
    psi[c] = 1 - pi[c], pi[c] ~ Beta(zero_alpha, zero_beta): extra zeros

The mean of the count stays psi * mu; the expected variable reported is that
mean, so the renormalized day-of-week mixture of D-A is kept and the extra
zeros only lower the expectation of the small patterns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from src.models.daily_counts.contracts import DAYS_PER_WEEK
from src.models.daily_counts.data import panel_arrays
from src.models.daily_counts.models import nb_daily_hierarchical_v1 as base
from src.models.weekly_counts.negative_binomial import zero_sum_basis

MODEL_ID = "zinb_daily_hierarchical_v1"
SOURCE_STATUS = "native_versioned_source"
HISTORICAL_RUN_ID = None
FORMULA = (
    "count[c,d] ~ ZeroInflatedNegativeBinomial(psi[c], mu_nb[c,d], alpha[c]); "
    "mu_nb as in nb_daily_hierarchical_v1; psi[c] = 1 - pi[c]; "
    "pi[c] ~ Beta(zero_alpha, zero_beta); mu[c,d] = psi[c] * mu_nb[c,d]"
)
DIAGNOSTIC_VARIABLES = base.DIAGNOSTIC_VARIABLES + ("pi",)
EXPECTED_VARIABLE = base.EXPECTED_VARIABLE
OBSERVED_VARIABLE = base.OBSERVED_VARIABLE
SUMMARY_VARIABLES = DIAGNOSTIC_VARIABLES + ("alpha",)
TRACE_VARIABLES = base.TRACE_VARIABLES
MODEL_SPEC = {
    **base.MODEL_SPEC,
    "model_id": MODEL_ID,
    "family": "zero_inflated_negative_binomial",
    "formula": FORMULA,
    "diagnostic_variables": DIAGNOSTIC_VARIABLES,
    "summary_variables": SUMMARY_VARIABLES,
}


def prepare_frame(frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    return base.prepare_frame(frame, settings)


def build_model(fit: pd.DataFrame, settings: dict[str, Any]) -> pm.Model:
    priors = settings["priors"]
    clusters = settings["clusters"]
    arrays = panel_arrays(fit, clusters)
    coords = {
        "cluster": np.arange(clusters),
        "contrast": np.arange(clusters - 1),
        "weekday": np.arange(DAYS_PER_WEEK),
        "day": np.arange(len(arrays["days"])),
    }
    basis = zero_sum_basis(clusters)

    with pm.Model(coords=coords) as model:
        totals = pm.Data("daily_total", arrays["totals"], dims="day")
        recent_share = pm.Data(
            "recent_share", arrays["recent_share"], dims=("day", "cluster")
        )
        day_of_week = pm.Data("day_of_week", arrays["day_of_week"], dims="day")
        log_alpha_global = pm.Normal(
            "log_alpha_global",
            mu=priors["log_alpha_global_mean"],
            sigma=priors["log_alpha_global_sigma"],
        )
        log_alpha_sigma = pm.HalfNormal(
            "log_alpha_sigma", sigma=priors["log_alpha_sigma"]
        )
        log_alpha_contrast = pm.Normal(
            "log_alpha_contrast", mu=0, sigma=log_alpha_sigma, dims="contrast"
        )
        alpha = pm.Deterministic(
            "alpha",
            pt.exp(log_alpha_global + pt.dot(basis, log_alpha_contrast)),
            dims="cluster",
        )
        beta_sigma = pm.HalfNormal("beta_sigma", sigma=priors["beta_sigma"])
        beta_contrast = pm.Normal(
            "beta_contrast", mu=0, sigma=beta_sigma, dims=("contrast", "weekday")
        )
        beta = pm.Deterministic(
            "beta", pt.dot(basis, beta_contrast), dims=("cluster", "weekday")
        )
        pi = pm.Beta(
            "pi", alpha=priors["zero_alpha"], beta=priors["zero_beta"], dims="cluster"
        )
        weight = recent_share * pt.exp(beta[:, day_of_week].T)
        share = weight / weight.sum(axis=1, keepdims=True)
        mu_nb = totals[:, None] * share
        pm.Deterministic("mu", (1 - pi)[None, :] * mu_nb, dims=("day", "cluster"))
        pm.ZeroInflatedNegativeBinomial(
            "observed",
            psi=(1 - pi)[None, :],
            mu=mu_nb,
            alpha=alpha[None, :],
            observed=arrays["counts"],
            dims=("day", "cluster"),
        )
    return model
