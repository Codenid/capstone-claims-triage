"""C-A: NB-R4-H v3 with a decaying, capped memory instead of 4 weeks (§25.5)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from ..negative_binomial import zero_sum_basis
from ..discounted_reference import row_discounted_shares

MODEL_ID = "nb_discounted_hierarchical_v1"
SOURCE_STATUS = "native_versioned_source"
HISTORICAL_RUN_ID = None
# Same rows as NB-R4-H v3: the first 4 weeks only build the memory.
WARMUP_WEEKS = 4
FORMULA = (
    "complaint_count[c,t] ~ NegativeBinomial(mu[c,t], alpha[c]); "
    "mu[c,t] = weekly_total[t] * recent_share[c,t]; "
    "recent_share[c,t] = (sum_k discount**(k-1) * capped[c,t-k] + 1) / "
    "(sum_k discount**(k-1) * capped_total[t-k] + clusters); "
    "capped[c,s] = min(count[c,s], cap * weekly_total[s] * recent_share[c,s]); "
    "log_alpha_contrast ~ Normal(0, log_alpha_sigma); "
    "log_alpha[c] = log_alpha_global + (zero_sum_basis @ log_alpha_contrast)[c]; "
    "alpha[c] = exp(log_alpha[c])"
)
DIAGNOSTIC_VARIABLES = ("log_alpha_global", "log_alpha_sigma", "log_alpha_contrast")
EXPECTED_VARIABLE = "mu"
OBSERVED_VARIABLE = "observed"
SUMMARY_VARIABLES = DIAGNOSTIC_VARIABLES + ("alpha",)
TRACE_VARIABLES = ("log_alpha_global", "log_alpha_sigma")
MODEL_SPEC = {
    "model_id": MODEL_ID,
    "source_status": SOURCE_STATUS,
    "historical_run_id": HISTORICAL_RUN_ID,
    "family": "negative_binomial",
    "formula": FORMULA,
    "negative_binomial_variance": "mu + mu**2 / alpha[c]",
    "cluster_interpretation": "operational partition, not natural categories",
    "weekly_total_usage": "observed exposure; this is a conditional forecast",
    "mean_link": "discounted_capped_share_input",
    "mean_sum_constrained": True,
    "predictive_draw_sum_constrained": False,
    "dispersion": "hierarchical_by_cluster",
    "zero_sum_parameterization": "centered scaled Helmert contrasts with K-1 dimensions",
    "share_memory": "discount and cap in the config (share)",
    "warmup_weeks": WARMUP_WEEKS,
    "prior_strategy": "not_applicable_shares_are_data",
    "diagnostic_variables": DIAGNOSTIC_VARIABLES,
    "summary_variables": SUMMARY_VARIABLES,
    "expected_variable": EXPECTED_VARIABLE,
    "observed_variable": OBSERVED_VARIABLE,
    "trace_variables": TRACE_VARIABLES,
}


def prepare_frame(frame: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    """Add recent_share and drop the warm-up weeks, as NB-R4-H v3 does."""
    shares = row_discounted_shares(
        frame,
        settings["clusters"],
        settings["share"]["discount"],
        settings["share"]["cap"],
    )
    weeks = frame["week"].drop_duplicates().sort_values()
    keep = frame["week"] >= weeks.iloc[WARMUP_WEEKS]
    return frame.assign(recent_share=shares).loc[keep].reset_index(drop=True)


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
        log_alpha_global = pm.Normal(
            "log_alpha_global",
            mu=priors["log_alpha_global_mean"],
            sigma=priors["log_alpha_global_sigma"],
        )
        log_alpha_sigma = pm.HalfNormal(
            "log_alpha_sigma",
            sigma=priors["log_alpha_sigma"],
        )
        # Centered: with 87 weeks per cluster this mixes better than v1. It
        # avoids pm.ZeroSumNormal (v2), which crashed JAX with parallel chains.
        log_alpha_contrast = pm.Normal(
            "log_alpha_contrast",
            mu=0,
            sigma=log_alpha_sigma,
            dims="contrast",
        )
        log_alpha = log_alpha_global + pt.dot(
            zero_sum_basis(clusters), log_alpha_contrast
        )
        alpha = pm.Deterministic("alpha", pt.exp(log_alpha), dims="cluster")
        mu = pm.Deterministic(
            "mu",
            weekly_total * recent_share,
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
