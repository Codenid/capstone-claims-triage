"""Model-agnostic posterior sampling helpers."""

from __future__ import annotations

import importlib
import platform
from typing import Any

import numpy as np
import pandas as pd


def sample_posterior(
    model: Any,
    sampling_settings: dict[str, Any],
    seed: int,
) -> tuple[Any, dict[str, str]]:
    pm = importlib.import_module("pymc")

    with model:
        idata = pm.sample(
            draws=sampling_settings["draws"],
            tune=sampling_settings["tune"],
            chains=sampling_settings["chains"],
            target_accept=sampling_settings["target_accept"],
            random_seed=seed,
            nuts_sampler=sampling_settings["backend"],
            nuts_sampler_kwargs={
                "chain_method": sampling_settings["chain_method"],
                "nuts_kwargs": {
                    "max_tree_depth": sampling_settings["max_treedepth"]
                },
            },
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": False},
        )

    jax = importlib.import_module("jax")
    devices = jax.devices()
    versions = {
        "python": platform.python_version(),
        "pymc": pm.__version__,
        "arviz": importlib.import_module("arviz").__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "jax": jax.__version__,
        "jaxlib": importlib.import_module("jaxlib").__version__,
        "numpyro": importlib.import_module("numpyro").__version__,
        "sampling_backend": sampling_settings["backend"],
        "jax_device_count": str(len(devices)),
        "jax_devices": ",".join(str(device) for device in devices),
    }
    return idata, versions


def sample_prior_predictive(
    model: Any,
    draws: int,
    observed_variable: str,
    seed: int,
) -> np.ndarray:
    pm = importlib.import_module("pymc")

    with model:
        idata: Any = pm.sample_prior_predictive(
            draws=draws,
            var_names=[observed_variable],
            random_seed=seed,
        )

    values = idata.prior_predictive[observed_variable].to_numpy()
    return values.reshape((-1,) + values.shape[2:])


def subsample_posterior(
    idata: Any,
    draws: int,
    seed: int,
) -> Any:
    posterior = idata.posterior.stack(sample=("chain", "draw"))
    available = posterior.sizes["sample"]
    selected = np.sort(
        np.random.default_rng(seed).choice(
            available,
            size=min(draws, available),
            replace=False,
        )
    )
    return posterior.isel(sample=selected)


def sample_predictions(
    model: Any,
    posterior: Any,
    expected_variable: str,
    observed_variable: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    pm = importlib.import_module("pymc")

    with model:
        idata: Any = pm.sample_posterior_predictive(
            trace=posterior,
            sample_dims=["sample"],
            var_names=[expected_variable, observed_variable],
            random_seed=seed,
            predictions=True,
        )

    expected = idata.predictions[expected_variable].to_numpy()
    observed = idata.predictions[observed_variable].to_numpy()
    return expected.reshape((-1,) + expected.shape[1:]), observed.reshape(
        (-1,) + observed.shape[1:]
    )
