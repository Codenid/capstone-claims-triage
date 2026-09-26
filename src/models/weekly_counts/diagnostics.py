"""Diagnostics for weekly count models."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd


def posterior_diagnostics(
    idata: Any,
    diagnostic_variables: tuple[str, ...],
    summary_variables: tuple[str, ...],
    max_treedepth: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    az = importlib.import_module("arviz")
    summary = az.summary(
        idata,
        var_names=list(summary_variables),
        kind="all",
        round_to=None,
    )
    convergence = az.summary(
        idata,
        var_names=list(diagnostic_variables),
        kind="diagnostics",
        round_to=None,
    )
    if not isinstance(summary, pd.DataFrame) or not isinstance(
        convergence, pd.DataFrame
    ):
        raise TypeError("M9 posterior summaries must be pandas DataFrames.")
    diagnostics: dict[str, float | int] = {
        "rhat_max": float(convergence["r_hat"].max()),
        "ess_bulk_min": float(convergence["ess_bulk"].min()),
        "ess_tail_min": float(convergence["ess_tail"].min()),
        "divergences": int(idata.sample_stats["diverging"].sum().item()),
        "bfmi_min": float(np.min(az.bfmi(idata))),
    }
    if "tree_depth" in idata.sample_stats:
        observed_maximum = int(idata.sample_stats["tree_depth"].max().item())
        diagnostics["tree_depth_max"] = observed_maximum
        if max_treedepth is not None:
            diagnostics["tree_depth_limit_hits"] = int(
                (idata.sample_stats["tree_depth"] == max_treedepth).sum().item()
            )
    return summary, diagnostics


def prior_predictive_summary(
    prior_draws: np.ndarray,
    fit: pd.DataFrame,
) -> dict[str, float]:
    totals = fit["weekly_total"].to_numpy(dtype=float)
    return {
        "median": float(np.median(prior_draws)),
        "p99": float(np.quantile(prior_draws, 0.99)),
        "maximum": float(prior_draws.max()),
        "impossible_fraction": float((prior_draws > totals[None, :]).mean()),
    }
