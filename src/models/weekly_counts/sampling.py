"""Sampling helpers for the weekly Negative Binomial model."""

from __future__ import annotations

import importlib
import platform
from typing import Any

import numpy as np
import pandas as pd

from src.models.weekly_counts.negative_binomial import (
    expected_counts,
    negative_binomial_draws,
    zero_sum_basis,
)


def simulate_prior_predictive(
    fit: pd.DataFrame,
    settings: dict[str, Any],
    seed: int,
) -> np.ndarray:
    priors = settings["priors"]
    draws = settings["sampling"]["prior_draws"]
    clusters = settings["clusters"]
    rng = np.random.default_rng(seed)

    basis = zero_sum_basis(clusters)
    log_rate_sigma = np.abs(
        rng.normal(0, priors["log_rate_sigma"], size=(draws, 1))
    )
    log_rate_contrast = rng.normal(size=(draws, clusters - 1))
    log_rate = log_rate_sigma * (log_rate_contrast @ basis.T)

    annual_trend_sigma = np.abs(
        rng.normal(0, priors["annual_trend_sigma"], size=(draws, 1))
    )
    annual_trend_contrast = rng.normal(size=(draws, clusters - 1))
    annual_trend = annual_trend_sigma * (annual_trend_contrast @ basis.T)

    log_alpha_global = rng.normal(
        priors["log_alpha_global_mean"],
        priors["log_alpha_global_sigma"],
        size=(draws, 1),
    )
    log_alpha_sigma = np.abs(
        rng.normal(0, priors["log_alpha_sigma"], size=(draws, 1))
    )
    log_alpha_contrast = rng.normal(size=(draws, clusters - 1))
    log_alpha = log_alpha_global + log_alpha_sigma * (log_alpha_contrast @ basis.T)
    alpha = np.exp(log_alpha)

    cluster_index = fit["cluster_id"].to_numpy(dtype=np.int64)
    mu = expected_counts(
        log_rate,
        annual_trend,
        cluster_index,
        fit["weekly_total"].to_numpy(dtype=float),
        fit["time_years"].to_numpy(dtype=float),
    )
    return negative_binomial_draws(mu, alpha[:, cluster_index], seed + 1)


def fit_model(
    fit: pd.DataFrame,
    settings: dict[str, Any],
    seed: int,
) -> tuple[Any, np.ndarray, dict[str, str]]:
    pm = importlib.import_module("pymc")
    pt = importlib.import_module("pytensor.tensor")
    priors = settings["priors"]
    sampling = settings["sampling"]
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
        mu = pt.exp(log_weekly_total + log_share)
        pm.NegativeBinomial(
            "observed",
            mu=mu,
            alpha=alpha[cluster_index],
            observed=fit["complaint_count"].to_numpy(dtype=np.int64),
            dims="fit_observation",
        )

        idata = pm.sample(
            draws=sampling["draws"],
            tune=sampling["tune"],
            chains=sampling["chains"],
            target_accept=sampling["target_accept"],
            random_seed=seed,
            nuts_sampler=sampling["backend"],
            nuts_sampler_kwargs={
                "chain_method": sampling["chain_method"],
                "nuts_kwargs": {"max_tree_depth": sampling["max_treedepth"]},
            },
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": False},
        )

    prior_draws = simulate_prior_predictive(fit, settings, seed)
    versions = {
        "python": platform.python_version(),
        "pymc": pm.__version__,
        "arviz": importlib.import_module("arviz").__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "jax": importlib.import_module("jax").__version__,
        "jaxlib": importlib.import_module("jaxlib").__version__,
        "numpyro": importlib.import_module("numpyro").__version__,
        "sampling_backend": sampling["backend"],
    }
    return idata, prior_draws, versions


def posterior_arrays(
    idata: Any,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def flattened(name: str) -> np.ndarray:
        values = idata.posterior[name].to_numpy()
        return values.reshape((-1,) + values.shape[2:])

    log_rate = flattened("log_rate")
    annual_trend = flattened("annual_trend")
    alpha = flattened("alpha")
    available = len(log_rate)
    selected_count = min(draws, available)
    selected = np.sort(
        np.random.default_rng(seed).choice(
            available,
            size=selected_count,
            replace=False,
        )
    )
    return log_rate[selected], annual_trend[selected], alpha[selected]
