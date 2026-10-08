"""D-A: negative binomial by pattern and day with a day-of-week effect (§28.2).

    count[c,d] ~ NegativeBinomial(mu[c,d], alpha[c])
    mu[c,d] = total[d] * share[c,d]
    share[c,d] = r[c,d] * exp(beta[c,dow(d)])
                 / sum_c' r[c',d] * exp(beta[c',dow(d)])
    r[c,d]: decaying-memory share of the previous days (discount in the config)
    beta[:,w] = zero_sum_basis @ beta_contrast[:,w]
    beta_contrast ~ Normal(0, beta_sigma)
    alpha[c] = exp(log_alpha_global + zero_sum_basis @ log_alpha_contrast)

The renormalization keeps the expected counts summing to the day's total, so
the day-of-week effect moves the mixture, not the volume.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from src.models.daily_counts.contracts import DAYS_PER_WEEK
from src.models.daily_counts.data import panel_arrays
from src.models.daily_counts.references import discounted_daily_shares, warm_rows
from src.models.weekly_counts.negative_binomial import zero_sum_basis

MODEL_ID = "nb_daily_hierarchical_v1"
SOURCE_STATUS = "native_versioned_source"
HISTORICAL_RUN_ID = None
DAY_OF_WEEK_EFFECT = True
FORMULA = (
    "count[c,d] ~ NegativeBinomial(mu[c,d], alpha[c]); "
    "mu[c,d] = daily_total[d] * r[c,d] * exp(beta[c,dow(d)]) / "
    "sum_c'(r[c',d] * exp(beta[c',dow(d)])); "
    "r[c,d] = (sum_k discount**(k-1) * count[c,d-k] + 1) / "
    "(sum_k discount**(k-1) * daily_total[d-k] + clusters); "
    "beta = zero_sum_basis @ beta_contrast; beta_contrast ~ Normal(0, beta_sigma); "
    "alpha[c] = exp(log_alpha_global + (zero_sum_basis @ log_alpha_contrast)[c])"
)
DIAGNOSTIC_VARIABLES = (
    "log_alpha_global",
    "log_alpha_sigma",
    "log_alpha_contrast",
    "beta_sigma",
    "beta_contrast",
)
EXPECTED_VARIABLE = "mu"
OBSERVED_VARIABLE = "observed"
SUMMARY_VARIABLES = DIAGNOSTIC_VARIABLES + ("alpha",)
TRACE_VARIABLES = ("log_alpha_global", "log_alpha_sigma", "beta_sigma")
MODEL_SPEC = {
    "model_id": MODEL_ID,
    "source_status": SOURCE_STATUS,
    "historical_run_id": HISTORICAL_RUN_ID,
    "family": "negative_binomial",
    "granularity": "day",
    "formula": FORMULA,
    "negative_binomial_variance": "mu + mu**2 / alpha[c]",
    "cluster_interpretation": "operational partition, not natural categories",
    "daily_total_usage": "observed exposure; this is a conditional forecast",
    "mean_link": "discounted_share_with_day_of_week_renormalized",
    "mean_sum_constrained": True,
    "predictive_draw_sum_constrained": False,
    "dispersion": "hierarchical_by_cluster",
    "day_of_week_effect": DAY_OF_WEEK_EFFECT,
    "zero_sum_parameterization": "centered scaled Helmert contrasts, K-1 dimensions",
    "diagnostic_variables": DIAGNOSTIC_VARIABLES,
    "summary_variables": SUMMARY_VARIABLES,
    "expected_variable": EXPECTED_VARIABLE,
    "observed_variable": OBSERVED_VARIABLE,
    "trace_variables": TRACE_VARIABLES,
}


def prepare_frame(frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Add the decaying-memory share and drop the warm-up days of each split."""
    shares = discounted_daily_shares(
        frame, settings["clusters"], settings["share"]["discount"]
    )
    keep = warm_rows(frame, settings["share"]["warmup_days"])
    return frame.assign(recent_share=shares).loc[keep].reset_index(drop=True)


def build_model(
    fit: pd.DataFrame,
    settings: dict[str, Any],
    day_of_week_effect: bool = DAY_OF_WEEK_EFFECT,
) -> pm.Model:
    priors = settings["priors"]
    clusters = settings["clusters"]
    arrays = panel_arrays(fit, clusters)
    days = len(arrays["days"])
    coords = {
        "cluster": np.arange(clusters),
        "contrast": np.arange(clusters - 1),
        "weekday": np.arange(DAYS_PER_WEEK),
        "day": np.arange(days),
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
        if day_of_week_effect:
            beta_sigma = pm.HalfNormal("beta_sigma", sigma=priors["beta_sigma"])
            beta_contrast = pm.Normal(
                "beta_contrast", mu=0, sigma=beta_sigma, dims=("contrast", "weekday")
            )
            beta = pm.Deterministic(
                "beta", pt.dot(basis, beta_contrast), dims=("cluster", "weekday")
            )
            weight = recent_share * pt.exp(beta[:, day_of_week].T)
        else:
            weight = recent_share
        share = weight / weight.sum(axis=1, keepdims=True)
        mu = pm.Deterministic("mu", totals[:, None] * share, dims=("day", "cluster"))
        pm.NegativeBinomial(
            "observed",
            mu=mu,
            alpha=alpha[None, :],
            observed=arrays["counts"],
            dims=("day", "cluster"),
        )
    return model
