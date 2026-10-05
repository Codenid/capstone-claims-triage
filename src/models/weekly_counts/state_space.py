"""C-B of models_plan.md §25.5: log-shares that follow a heavy-tailed random walk.

    z[t] = z[t-1] + tau * e[t],    e[t] ~ StudentT(nu, 0, 1), K-1 zero-sum contrasts
    p[t] = softmax(B @ z[t]),      y[c,t] ~ NegativeBinomial(N[t] * p[c,t], alpha[c])

The fit step estimates alpha, tau and nu with every fit week. Each calibration
week is then predicted from the weeks before it: the hyperparameters stay at
their fit posterior medians and only the states are re-estimated. Consecutive
complete weeks are consecutive steps, as in M9.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import yaml

from src.evaluation.experiment import (
    PROJECT_ROOT,
    git_commit,
    load_experiment_config,
    set_seed,
)
from src.models.weekly_counts.data import fit_rows
from src.models.weekly_counts.diagnostics import (
    posterior_diagnostics,
    prior_check_summary,
)
from src.models.weekly_counts.metrics import predictive_metrics
from src.models.weekly_counts.negative_binomial import zero_sum_basis
from src.models.weekly_counts.prediction import summarize_draws
from src.models.weekly_counts.reporting import file_sha256
from src.models.weekly_counts.run import load_frozen_frame
from src.models.weekly_counts.sampling import (
    sample_posterior,
    sample_prior_variables,
)

MODEL_ID = "nb_state_space_v1"
FORMULA = (
    "z[t] = z[t-1] + tau * e[t], e[t] ~ StudentT(nu, 0, 1) in K-1 zero-sum "
    "contrasts; p[t] = softmax(B z[t]); complaint_count[c,t] ~ "
    "NegativeBinomial(weekly_total[t] * p[c,t], alpha[c]); alpha, tau and nu "
    "from fit only; states re-estimated before each calibration week"
)
HYPERPARAMETERS = (
    "tau",
    "nu",
    "log_alpha_global",
    "log_alpha_sigma",
    "log_alpha_contrast",
)
STATE_VARIABLES = ("z0", "innovations")
RUN_MODES = ("prior", "pilot", "full")
PILOT_SAMPLING = {"chains": 2, "tune": 250, "draws": 250}
PILOT_CALIBRATION_WEEKS = 2
KEY_COLUMNS = ["split", "week", "cluster_id", "complaint_count", "weekly_total"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-mode", choices=RUN_MODES, default="full")
    return parser.parse_args()


def week_matrix(
    frame: pd.DataFrame,
    clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Counts as weeks x clusters, and the weekly totals, in week order."""
    counts = (
        frame.pivot(index="week", columns="cluster_id", values="complaint_count")
        .sort_index()
        .reindex(columns=range(clusters))
    )
    values = counts.to_numpy(dtype=np.int64)
    return values, values.sum(axis=1).astype(float)


def build_model(
    counts: np.ndarray,
    totals: np.ndarray,
    priors: dict[str, Any],
    fixed: dict[str, Any] | None = None,
) -> pm.Model:
    """The random walk on every week of `counts`; `fixed` holds alpha, tau and nu."""
    weeks, clusters = counts.shape
    basis = zero_sum_basis(clusters)
    coords = {
        "contrast": np.arange(clusters - 1),
        "step": np.arange(weeks - 1),
        "cluster": np.arange(clusters),
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
        else:
            tau, nu = fixed["tau"], fixed["nu"]
            alpha = pt.as_tensor_variable(fixed["alpha"])
        z0 = pm.Normal("z0", mu=0, sigma=priors["initial_sigma"], dims="contrast")
        innovations = pm.StudentT(
            "innovations", nu=nu, mu=0, sigma=1, dims=("step", "contrast")
        )
        z = pt.concatenate(
            [z0[None, :], z0[None, :] + pt.cumsum(tau * innovations, axis=0)],
            axis=0,
        )
        pm.Deterministic("z_last", z[-1], dims="contrast")
        shares = pt.special.softmax(pt.dot(z, basis.T), axis=1)
        pm.Deterministic("mu", totals[:, None] * shares)
        pm.NegativeBinomial(
            "observed",
            mu=totals[:, None] * shares,
            alpha=alpha[None, :],
            observed=counts,
        )
    return model


def hyperparameter_medians(idata: Any) -> dict[str, Any]:
    posterior = idata.posterior
    return {
        "tau": float(posterior["tau"].median()),
        "nu": float(posterior["nu"].median()),
        "alpha": posterior["alpha"].median(dim=("chain", "draw")).to_numpy(),
    }


def one_step_draws(
    idata: Any,
    total: float,
    fixed: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Predictive counts for the next week: one more random-walk step."""
    z_last = idata.posterior["z_last"].stack(sample=("chain", "draw")).to_numpy().T
    step = fixed["tau"] * rng.standard_t(fixed["nu"], size=z_last.shape)
    eta = (z_last + step) @ zero_sum_basis(z_last.shape[1] + 1).T
    shares = np.exp(eta - eta.max(axis=1, keepdims=True))
    shares /= shares.sum(axis=1, keepdims=True)
    mu = total * shares
    alpha = fixed["alpha"][None, :]
    return rng.negative_binomial(alpha, alpha / (alpha + mu)).astype(float)


def sampling_for(
    settings: dict[str, Any],
    key: str,
    run_mode: str,
) -> dict[str, Any]:
    if run_mode == "pilot":
        return {**settings[key], **PILOT_SAMPLING}
    return dict(settings[key])


def prior_check(
    fit: pd.DataFrame,
    counts: np.ndarray,
    totals: np.ndarray,
    settings: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """M9's prior summary plus the weekly change of centered log-shares."""
    model = build_model(counts, totals, settings["priors"])
    draws = sample_prior_variables(
        model, settings["sampling"]["prior_draws"], ["observed", "mu", "alpha"], seed
    )
    observed = draws["observed"].reshape(len(draws["observed"]), -1)
    expected = draws["mu"].reshape(len(draws["mu"]), -1)
    summary = prior_check_summary(observed, expected, draws["alpha"], fit)

    def weekly_change(values: np.ndarray) -> np.ndarray:
        log_share = np.log(values + 0.5)
        log_share -= log_share.mean(axis=-1, keepdims=True)
        return np.diff(log_share, axis=-2).std(axis=-2)

    prior_change = weekly_change(draws["observed"].astype(float))
    observed_change = weekly_change(counts.astype(float))
    summary["log_share_weekly_change_sd"] = {
        "prior": {
            label: float(np.quantile(prior_change, level))
            for label, level in (("p025", 0.025), ("p50", 0.5), ("p975", 0.975))
        },
        "fit_observed_median": float(np.median(observed_change)),
    }
    return summary


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_experiment_config()
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = config["experiment"]["seed"]
    set_seed(seed)
    clusters = settings["clusters"]
    panel, _, input_sha256, source_dvc_hash = load_frozen_frame(config, settings)
    fit = fit_rows(panel)
    counts, totals = week_matrix(fit, clusters)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    config_sha256 = file_sha256(config_path)
    run_key = f"{timestamp}-{args.run_mode}-{config_sha256[:8]}"
    report_root = PROJECT_ROOT / "reports/modeling/weekly_counts" / MODEL_ID
    metadata = {
        "stage": "M9 C-B",
        "plan": "models_plan.md §25.5",
        "run_key": run_key,
        "run_mode": args.run_mode,
        "model_id": MODEL_ID,
        "formula": FORMULA,
        "git_commit": git_commit(),
        "config_sha256": config_sha256,
        "weekly_counts_sha256": input_sha256,
        "source_dvc_hash": source_dvc_hash,
        "seed": seed,
        "priors": settings["priors"],
        "fit_weeks": int(counts.shape[0]),
        "validation_used_for_selection": False,
    }

    if args.run_mode == "prior":
        path = report_root / "prior_checks" / f"{run_key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = prior_check(fit, counts, totals, settings, seed)
        path.write_text(
            json.dumps({**metadata, "prior_check": summary}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Prior check: {path.relative_to(PROJECT_ROOT)}")
        return

    report_dir = report_root / run_key
    staging = report_dir.parent / f".{run_key}.inprogress"
    staging.mkdir(parents=True)
    fit_sampling = sampling_for(settings, "sampling", args.run_mode)
    idata, versions = sample_posterior(
        build_model(counts, totals, settings["priors"]), fit_sampling, seed
    )
    variables = HYPERPARAMETERS + STATE_VARIABLES
    summary, fit_diagnostics = posterior_diagnostics(
        idata, variables, HYPERPARAMETERS + ("alpha",), fit_sampling["max_treedepth"]
    )
    summary.reset_index().rename(columns={"index": "parameter"}).to_csv(
        staging / "posterior_summary.csv", index=False
    )
    fixed = hyperparameter_medians(idata)
    print(f"Fit step: {json.dumps(fit_diagnostics)}", flush=True)

    calibration = panel.loc[panel["split"] == "calibration"]
    weeks = np.sort(calibration["week"].unique())
    if args.run_mode == "pilot":
        weeks = weeks[:PILOT_CALIBRATION_WEEKS]
    filter_sampling = sampling_for(settings, "filter_sampling", args.run_mode)
    rng = np.random.default_rng(seed)
    rows = []
    filter_diagnostics = []
    for offset, week in enumerate(weeks):
        history = panel.loc[panel["week"] < week]
        history_counts, history_totals = week_matrix(history, clusters)
        state_idata, _ = sample_posterior(
            build_model(history_counts, history_totals, settings["priors"], fixed),
            filter_sampling,
            seed + 1 + offset,
        )
        _, diagnostics = posterior_diagnostics(
            state_idata, STATE_VARIABLES, ("z_last",), filter_sampling["max_treedepth"]
        )
        label = str(pd.Timestamp(week).date())
        filter_diagnostics.append({"week": label, **diagnostics})
        target = calibration.loc[calibration["week"] == week].sort_values("cluster_id")
        total = float(target["weekly_total"].iloc[0])
        draws = one_step_draws(state_idata, total, fixed, rng)
        summary_frame = summarize_draws(draws, "model")
        summary_frame["model_impossible_probability"] = (draws > total).mean(axis=0)
        keys = target[KEY_COLUMNS].reset_index(drop=True)
        rows.append(pd.concat([keys, summary_frame], axis=1))
        print(f"Calibration week {label}: {json.dumps(diagnostics)}", flush=True)

    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(staging / "predictions.csv", index=False)
    report = {
        **metadata,
        "versions": versions,
        "sampling": fit_sampling,
        "filter_sampling": filter_sampling,
        "fit_diagnostics": fit_diagnostics,
        "filter_diagnostics": filter_diagnostics,
        "hyperparameter_medians": {
            "tau": fixed["tau"],
            "nu": fixed["nu"],
            "alpha": fixed["alpha"].tolist(),
        },
        "evaluated_splits": ["calibration"],
        "backtest": {
            "calibration": {"model": predictive_metrics(predictions, "model")}
        },
        "resources": {"elapsed_seconds": time.perf_counter() - started},
    }
    (staging / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (staging / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    staging.replace(report_dir)
    artifact_dir = PROJECT_ROOT / "artifacts/models/weekly_counts" / MODEL_ID / run_key
    artifact_dir.mkdir(parents=True)
    idata.to_netcdf(artifact_dir / "posterior.nc")
    metrics = report["backtest"]["calibration"]["model"]
    print(
        f"Calibration WIS={metrics['wis']:.3f} "
        f"coverage 80%={metrics['coverage_80']:.3f} 95%={metrics['coverage_95']:.3f}"
    )
    print(f"Report: {report_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
