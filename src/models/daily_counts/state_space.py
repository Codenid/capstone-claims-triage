"""D-B: daily state-space model with a random walk per pattern (§28.2).

    count[c,d] ~ NegativeBinomial(daily_total[d] * share[c,d], alpha[c])
    share[:,d] = softmax(basis @ z[:,d] + beta[:,dow(d)])
    z[:,d] = z[:,d-1] + e[:,d],  e ~ StudentT(nu, 0, tau)   (K-1 contrasts)
    beta: day-of-week effect per pattern, zero-sum across patterns

Fit: hyperparameters (tau, nu, alpha, beta) and the states on the fit days.
Calibration and validation: the hyperparameters stay fixed at their posterior
medians and the states are re-estimated in blocks of `block_days` (7 by
default, pre-registered): each block refits the walk on every day up to the
block start and predicts its days from the last state, with the innovation
variance growing with the horizon. Re-estimating every single day (92 refits)
is out of the compute budget; the block is the daily analogue of C-B.

    python -m src.models.daily_counts.state_space --run-mode prior|pilot|full
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
import time
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import yaml

from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    git_commit,
    load_experiment_config,
    save_run_record,
    set_seed,
)
from src.models.daily_counts import baselines
from src.models.daily_counts.contracts import DAY_COLUMN, DAYS_PER_WEEK
from src.models.daily_counts.data import as_weekly_view, load_frozen_panel, panel_arrays
from src.models.daily_counts.run import build_predictions, flatten
from src.models.semantic_space import maximum_rss_gib
from src.models.weekly_counts.comparison import compare_with_baselines
from src.models.weekly_counts.diagnostics import posterior_diagnostics
from src.models.weekly_counts.metrics import (
    candidate_status,
    evaluate_predictions,
    rejection_reason,
    run_acceptance,
)
from src.models.weekly_counts.negative_binomial import zero_sum_basis
from src.models.weekly_counts.sampling import sample_posterior

MODEL_ID = "nb_daily_state_space_v1"
STAGE = "M9D"
PLAN = "models_plan.md §28.2"
RUN_MODES = ("prior", "pilot", "full")
EVALUATED_SPLITS = ("fit", "calibration")
HYPERPARAMETERS = ("tau", "nu", "log_alpha_global", "log_alpha_sigma", "beta_sigma")
BASELINE_NAMES = ("baseline", baselines.ROLLING_NAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-mode", choices=RUN_MODES, default="full")
    return parser.parse_args()


def build_model(
    counts: np.ndarray,
    totals: np.ndarray,
    day_of_week: np.ndarray,
    priors: dict[str, Any],
    fixed: dict[str, Any] | None = None,
) -> pm.Model:
    """The walk on every day of `counts`; `fixed` holds the hyperparameters."""
    days, clusters = counts.shape
    basis = zero_sum_basis(clusters)
    coords = {
        "contrast": np.arange(clusters - 1),
        "day": np.arange(days),
        "cluster": np.arange(clusters),
        "weekday": np.arange(DAYS_PER_WEEK),
    }
    with pm.Model(coords=coords) as model:
        if fixed is None:
            tau = pm.HalfNormal("tau", sigma=priors["tau_sigma"])
            nu = pm.Gamma("nu", alpha=priors["nu_alpha"], beta=priors["nu_beta"])
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
        else:
            tau, nu = fixed["tau"], fixed["nu"]
            alpha = pt.as_tensor_variable(fixed["alpha"])
            beta = pt.as_tensor_variable(fixed["beta"])
        z = pm.RandomWalk(
            "z",
            init_dist=pm.Normal.dist(0, priors["initial_sigma"], shape=(clusters - 1,)),
            innovation_dist=pm.StudentT.dist(
                nu=nu, mu=0, sigma=tau, shape=(clusters - 1,)
            ),
            steps=days - 1,
            dims=("contrast", "day"),
        )
        pm.Deterministic("z_last", z[:, -1], dims="contrast")
        logits = pt.dot(z.T, basis.T) + beta[:, day_of_week].T
        shares = pt.special.softmax(logits, axis=1)
        pm.Deterministic("mu", totals[:, None] * shares, dims=("day", "cluster"))
        pm.NegativeBinomial(
            "observed",
            mu=totals[:, None] * shares,
            alpha=alpha[None, :],
            observed=counts,
            dims=("day", "cluster"),
        )
    return model


def hyperparameter_medians(idata: Any) -> dict[str, Any]:
    posterior = idata.posterior
    return {
        "tau": float(posterior["tau"].median()),
        "nu": float(posterior["nu"].median()),
        "alpha": posterior["alpha"].median(dim=("chain", "draw")).to_numpy(),
        "beta": posterior["beta"].median(dim=("chain", "draw")).to_numpy(),
    }


def block_draws(
    idata: Any,
    totals: np.ndarray,
    day_of_week: np.ndarray,
    fixed: dict[str, Any],
    clusters: int,
    rng: np.random.Generator,
    draws: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected and predictive draws for the days of a block, from the last state.

    Horizon h ahead of the last state adds h Student-t innovations.
    """
    basis = zero_sum_basis(clusters)
    last = np.asarray(idata.posterior["z_last"]).reshape(-1, clusters - 1)
    chosen = last[rng.choice(len(last), size=draws, replace=True)]
    horizon = len(totals)
    expected = np.empty((draws, horizon, clusters))
    predictive = np.empty((draws, horizon, clusters), dtype=np.int64)
    state = chosen.copy()
    for step in range(horizon):
        state = state + rng.standard_t(fixed["nu"], size=state.shape) * fixed["tau"]
        logits = state @ basis.T + fixed["beta"][:, day_of_week[step]][None, :]
        logits -= logits.max(axis=1, keepdims=True)
        shares = np.exp(logits)
        shares /= shares.sum(axis=1, keepdims=True)
        mu = totals[step] * shares
        expected[:, step] = mu
        probability = fixed["alpha"] / (fixed["alpha"] + mu)
        predictive[:, step] = rng.negative_binomial(fixed["alpha"], probability)
    return expected, predictive


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    config = load_experiment_config()
    settings = config["daily_state_space"]
    model_settings = yaml.safe_load(
        (PROJECT_ROOT / settings["reference_config"]).read_text(encoding="utf-8")
    )
    sampling = dict(settings["sampling"])
    if args.run_mode == "pilot":
        sampling.update(settings["pilot_sampling"])
    seed = config["experiment"]["seed"]
    set_seed(seed)
    rng = np.random.default_rng(seed)
    clusters = model_settings["clusters"]
    panel, input_sha256, source_dvc_hash = load_frozen_panel(config, model_settings)
    panel = panel.assign(recent_share=np.nan)
    arrays = {
        split: panel_arrays(rows, clusters)
        for split, rows in panel.groupby("split", observed=True)
    }
    fit = arrays["fit"]
    priors = settings["priors"]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_key = f"{timestamp}-{args.run_mode}"
    metadata = {
        "stage": STAGE,
        "plan": PLAN,
        "model_id": MODEL_ID,
        "run_key": run_key,
        "run_mode": args.run_mode,
        "git_commit": git_commit(),
        "daily_counts_sha256": input_sha256,
        "source_dvc_hash": source_dvc_hash,
        "seed": seed,
        "priors": priors,
        "block_days": settings["block_days"],
        "fit_days": int(len(fit["days"])),
        "validation_used_for_selection": False,
    }
    report_dir = PROJECT_ROOT / settings["report_dir"]

    if args.run_mode == "prior":
        with build_model(fit["counts"], fit["totals"], fit["day_of_week"], priors):
            prior = pm.sample_prior_predictive(draws=100, random_seed=seed)
        observed = prior.prior_predictive["observed"].to_numpy().reshape(-1, clusters)
        shares = observed / np.maximum(observed.sum(axis=1, keepdims=True), 1)
        summary = {
            "max_share_prior_p50": float(np.median(shares.max(axis=1))),
            "max_share_fit_p50": float(
                np.median((fit["counts"] / fit["totals"][:, None]).max(axis=1))
            ),
            "tau_p50": float(np.median(prior.prior["tau"])),
            "nu_p50": float(np.median(prior.prior["nu"])),
        }
        path = report_dir / "prior_checks" / f"{run_key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**metadata, "prior_check": summary}, indent=2))
        print(json.dumps(summary, indent=2))
        return

    run_dir = report_dir / run_key
    artifact_dir = PROJECT_ROOT / settings["artifact_dir"] / run_key
    run_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    # Pilot: a short calibration, as the weekly C-B pilot did.
    calibration = arrays["calibration"]
    if args.run_mode == "pilot":
        keep = settings["pilot_calibration_days"]
        calibration = {name: values[:keep] for name, values in calibration.items()}

    model = build_model(fit["counts"], fit["totals"], fit["day_of_week"], priors)
    idata, versions = sample_posterior(model, sampling, seed)
    idata.posterior = idata.posterior.drop_vars(["z", "mu"])
    idata.to_netcdf(artifact_dir / "posterior.nc")
    summary, diagnostics = posterior_diagnostics(
        idata, HYPERPARAMETERS, HYPERPARAMETERS + ("alpha",), sampling["max_treedepth"]
    )
    summary.reset_index().rename(columns={"index": "parameter"}).to_csv(
        run_dir / "posterior_summary.csv", index=False
    )
    fixed = hyperparameter_medians(idata)
    draws = sampling["prediction_draws"]

    # Fit days: in-sample expected and predictive draws from the fitted states.
    with model:
        fitted = pm.sample_posterior_predictive(
            idata, var_names=["mu", "observed"], random_seed=seed, predictions=True
        )
    shape = (-1, *fit["counts"].shape)
    fit_expected = flatten(fitted.predictions["mu"].to_numpy().reshape(shape))
    fit_predictive = flatten(fitted.predictions["observed"].to_numpy().reshape(shape))
    size = min(draws, len(fit_expected))
    keep = rng.choice(len(fit_expected), size=size, replace=False)
    fit_expected, fit_predictive = fit_expected[keep], fit_predictive[keep]

    # Calibration: states re-estimated block by block with fixed hyperparameters.
    block = settings["block_days"]
    counts_so_far = fit["counts"]
    totals_so_far = fit["totals"]
    dow_so_far = fit["day_of_week"]
    expected_blocks, predictive_blocks, refits = [], [], 0
    for start in range(0, len(calibration["days"]), block):
        stop = min(start + block, len(calibration["days"]))
        refit = build_model(counts_so_far, totals_so_far, dow_so_far, priors, fixed)
        with refit:
            states = pm.sample(
                draws=settings["state_sampling"]["draws"],
                tune=settings["state_sampling"]["tune"],
                chains=settings["state_sampling"]["chains"],
                target_accept=sampling["target_accept"],
                random_seed=seed + refits,
                nuts_sampler=sampling["backend"],
                nuts_sampler_kwargs={"chain_method": sampling["chain_method"]},
                progressbar=False,
                idata_kwargs={"log_likelihood": False},
            )
        refits += 1
        expected, predictive = block_draws(
            states,
            calibration["totals"][start:stop],
            calibration["day_of_week"][start:stop],
            fixed,
            clusters,
            rng,
            len(fit_expected),
        )
        expected_blocks.append(expected)
        predictive_blocks.append(predictive)
        counts_so_far = np.vstack([counts_so_far, calibration["counts"][start:stop]])
        totals_so_far = np.concatenate(
            [totals_so_far, calibration["totals"][start:stop]]
        )
        dow_so_far = np.concatenate(
            [dow_so_far, calibration["day_of_week"][start:stop]]
        )
    expected_all = np.concatenate(
        [fit_expected, flatten(np.concatenate(expected_blocks, axis=1))], axis=1
    )
    predictive_all = np.concatenate(
        [fit_predictive, flatten(np.concatenate(predictive_blocks, axis=1))], axis=1
    )

    frame = panel.loc[panel["split"].isin(EVALUATED_SPLITS)].reset_index(drop=True)
    if args.run_mode == "pilot":
        days_kept = pd.to_datetime(calibration["days"])
        keep_rows = (frame["split"] == "fit") | frame[DAY_COLUMN].isin(days_kept)
        frame = frame.loc[keep_rows].reset_index(drop=True)
    fit_frame = frame.loc[frame["split"] == "fit"]
    reference = baselines.fixed_reference_draws(
        frame, fit_frame, clusters, len(expected_all), seed + 1
    )
    predictions = build_predictions(frame, expected_all, predictive_all, reference)
    rolling = baselines.rolling_columns(
        frame, panel, clusters, len(expected_all), seed + 2
    )
    predictions = pd.concat([predictions, rolling], axis=1)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    metrics, backtest = evaluate_predictions(predictions, EVALUATED_SPLITS)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    comparison = compare_with_baselines(
        as_weekly_view(predictions), BASELINE_NAMES, seed
    )
    comparison["days"] = comparison.pop("weeks")
    acceptance = run_acceptance(
        diagnostics, backtest["calibration"], comparison, model_settings
    )
    status = candidate_status(args.run_mode, acceptance)
    report = {
        **metadata,
        "candidate_status": status,
        "rejection_reason": rejection_reason(args.run_mode, acceptance),
        "sampling": sampling,
        "state_sampling": settings["state_sampling"],
        "state_refits": refits,
        "hyperparameters": {
            "tau": fixed["tau"],
            "nu": fixed["nu"],
            "alpha_median": float(np.median(fixed["alpha"])),
        },
        "versions": versions,
        "diagnostics": diagnostics,
        "evaluated_splits": list(EVALUATED_SPLITS),
        "backtest": backtest,
        "comparison": comparison,
        "calibration_acceptance": acceptance,
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_rss_gib": maximum_rss_gib(),
            "cpu_count": os.cpu_count(),
        },
    }
    (run_dir / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    calibration_metrics = report["backtest"]["calibration"]["model"]
    record = build_run_record(
        config=config,
        run_name=f"m9d-{MODEL_ID}-{args.run_mode}",
        stage=STAGE,
        target="A1",
        split="calibration",
        view="40 patterns, daily",
        features=["daily_total", "latent random walk", "day_of_week"],
        parameters={
            "plan": PLAN,
            "run_key": run_key,
            "run_mode": args.run_mode,
            "block_days": settings["block_days"],
            "candidate_status": status,
        },
        metrics={
            **{f"calibration_{k}": float(v) for k, v in calibration_metrics.items()},
            "calibration_wis_gain": float(comparison["wis_gain"]),
            **{f"diagnostic_{k}": float(v) for k, v in diagnostics.items()},
        },
        artifacts=[str(run_dir.relative_to(PROJECT_ROOT))],
    )
    save_run_record(record, run_dir / "run.json")
    print(f"Run key: {run_key}\nDiagnostics: {json.dumps(diagnostics, sort_keys=True)}")
    print(f"Calibration: {json.dumps(comparison['metrics'], sort_keys=True)}")
    print(f"WIS gain: {comparison['wis_gain']:.4f}\nStatus: {status}")
    print(f"Report: {run_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
