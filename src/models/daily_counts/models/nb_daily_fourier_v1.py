"""D-C: negative binomial with a Fourier weekly cycle and a local trend (§28.2).

    count[c,d] ~ NegativeBinomial(mu[c,d], alpha[c])
    mu[c,d] = total[d] * r[c,d] * exp(f[c,d]) / sum_c' r[c',d] * exp(f[c',d])
    f[c,d] = sum_k a[c,k] cos(2 pi k d / 7) + b[c,k] sin(2 pi k d / 7) + g[c] * t[d]
    t[d]: days since the start of the split, scaled to years; the trend corrects
    what the 7-day memory lags behind

The cycle is parametric (2 harmonics = 4 coefficients per pattern instead of
the 7 free day-of-week levels of D-A); every coefficient is a zero-sum
contrast across patterns with hierarchical shrinkage.
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

MODEL_ID = "nb_daily_fourier_v1"
SOURCE_STATUS = "native_versioned_source"
HISTORICAL_RUN_ID = None
HARMONICS = 2
DAYS_PER_YEAR = 365.25
FORMULA = (
    "count[c,d] ~ NegativeBinomial(mu[c,d], alpha[c]); "
    "mu[c,d] = daily_total[d] * r[c,d] * exp(f[c,d]) / sum_c'(r[c',d] * exp(f[c',d])); "
    "f[c,d] = sum_k (a[c,k] cos(2 pi k d/7) + b[c,k] sin(2 pi k d/7)) + g[c] t[d]; "
    "a, b, g = zero_sum_basis @ contrasts ~ Normal(0, sigma); "
    "alpha[c] = exp(log_alpha_global + (zero_sum_basis @ log_alpha_contrast)[c])"
)
DIAGNOSTIC_VARIABLES = (
    "log_alpha_global",
    "log_alpha_sigma",
    "log_alpha_contrast",
    "cycle_sigma",
    "cycle_contrast",
    "trend_sigma",
    "trend_contrast",
)
EXPECTED_VARIABLE = base.EXPECTED_VARIABLE
OBSERVED_VARIABLE = base.OBSERVED_VARIABLE
SUMMARY_VARIABLES = DIAGNOSTIC_VARIABLES + ("alpha",)
TRACE_VARIABLES = ("log_alpha_global", "cycle_sigma", "trend_sigma")
MODEL_SPEC = {
    **base.MODEL_SPEC,
    "model_id": MODEL_ID,
    "formula": FORMULA,
    "mean_link": "discounted_share_with_fourier_cycle_and_local_trend",
    "day_of_week_effect": "fourier",
    "harmonics": HARMONICS,
    "diagnostic_variables": DIAGNOSTIC_VARIABLES,
    "summary_variables": SUMMARY_VARIABLES,
    "trace_variables": TRACE_VARIABLES,
}


def prepare_frame(frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    return base.prepare_frame(frame, settings)


def fourier_design(days: np.ndarray) -> np.ndarray:
    """(days, 2 * HARMONICS) matrix of the weekly cycle at each calendar day."""
    ordinal = pd.DatetimeIndex(days).map(pd.Timestamp.toordinal).to_numpy()
    phase = 2 * np.pi * ordinal[:, None] * np.arange(1, HARMONICS + 1) / DAYS_PER_WEEK
    return np.hstack([np.cos(phase), np.sin(phase)])


def local_time(days: np.ndarray) -> np.ndarray:
    """Years since the first day of the frame."""
    index = pd.DatetimeIndex(days)
    return ((index - index[0]).days / DAYS_PER_YEAR).to_numpy(dtype=float)


def build_model(fit: pd.DataFrame, settings: dict[str, Any]) -> pm.Model:
    priors = settings["priors"]
    clusters = settings["clusters"]
    arrays = panel_arrays(fit, clusters)
    design = fourier_design(arrays["days"])
    coords = {
        "cluster": np.arange(clusters),
        "contrast": np.arange(clusters - 1),
        "harmonic": np.arange(design.shape[1]),
        "day": np.arange(len(arrays["days"])),
    }
    basis = zero_sum_basis(clusters)

    with pm.Model(coords=coords) as model:
        totals = pm.Data("daily_total", arrays["totals"], dims="day")
        recent_share = pm.Data(
            "recent_share", arrays["recent_share"], dims=("day", "cluster")
        )
        cycle = pm.Data("cycle", design, dims=("day", "harmonic"))
        time = pm.Data("time", local_time(arrays["days"]), dims="day")
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
        cycle_sigma = pm.HalfNormal("cycle_sigma", sigma=priors["cycle_sigma"])
        cycle_contrast = pm.Normal(
            "cycle_contrast", mu=0, sigma=cycle_sigma, dims=("contrast", "harmonic")
        )
        trend_sigma = pm.HalfNormal("trend_sigma", sigma=priors["trend_sigma"])
        trend_contrast = pm.Normal(
            "trend_contrast", mu=0, sigma=trend_sigma, dims="contrast"
        )
        coefficients = pt.dot(basis, cycle_contrast)  # (cluster, harmonic)
        trend = pt.dot(basis, trend_contrast)  # (cluster,)
        effect = pt.dot(cycle, coefficients.T) + time[:, None] * trend[None, :]
        weight = recent_share * pt.exp(effect)
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
