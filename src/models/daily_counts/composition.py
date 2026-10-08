"""D-E: Dirichlet-multinomial of the daily composition (models_plan.md §28.2).

    counts[d] ~ DirichletMultinomial(daily_total[d], kappa * r[d])
    r[d]: decaying-memory shares of the previous days (same discount as D-A)
    log_kappa ~ Normal(log_kappa_mean, log_kappa_sigma)

Scored like M10: joint log score per day against two multinomial references
(static Dirichlet-multinomial from the fit counts, and the plain 7-day share),
with the paired bootstrap of §12.3. It also yields the daily "rare mixture"
signal of §27: a posterior predictive p-value per day.

    python -m src.models.daily_counts.composition --run-mode prior|pilot|full
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
from src.models.daily_counts.contracts import DAY_COLUMN, SPLITS
from src.models.daily_counts.data import load_frozen_panel, panel_arrays
from src.models.daily_counts.references import (
    discounted_daily_shares,
    warm_rows,
    windowed_daily_shares,
)
from src.models.semantic_space import maximum_rss_gib
from src.models.weekly_composition.evaluation import compare_with_baselines
from src.models.weekly_composition.mixture_signal import week_surprise
from src.models.weekly_composition.scores import (
    dirichlet_multinomial_logpmf,
    joint_log_score,
    multinomial_logpmf,
)
from src.models.weekly_counts.diagnostics import posterior_diagnostics
from src.models.weekly_counts.sampling import sample_posterior

MODEL_ID = "dirichlet_multinomial_daily_v1"
STAGE = "M10D"
PLAN = "models_plan.md §28.2"
BASELINES = ("dm_static", "mn_rolling_7")
RUN_MODES = ("prior", "pilot", "full")
EVALUATED_SPLITS = ("fit", "calibration")
WINDOW_DAYS = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-mode", choices=RUN_MODES, default="full")
    return parser.parse_args()


def split_arrays(frame: pd.DataFrame, clusters: int) -> dict[str, dict[str, Any]]:
    """(days, clusters) arrays per split of the frame with shares attached."""
    return {
        split: panel_arrays(rows, clusters)
        for split, rows in frame.groupby("split", observed=True)
    }


def build_model(arrays: dict[str, np.ndarray], priors: dict[str, float]) -> pm.Model:
    days, clusters = arrays["counts"].shape
    coords = {"day": np.arange(days), "cluster": np.arange(clusters)}
    with pm.Model(coords=coords) as model:
        totals = pm.Data("daily_total", arrays["totals"], dims="day")
        recent = pm.Data(
            "recent_share", arrays["recent_share"], dims=("day", "cluster")
        )
        log_kappa = pm.Normal(
            "log_kappa", mu=priors["log_kappa_mean"], sigma=priors["log_kappa_sigma"]
        )
        kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
        pm.Deterministic("rho", 1 / (kappa + 1))
        pm.DirichletMultinomial(
            "observed",
            n=totals,
            a=kappa * recent,
            observed=arrays["counts"],
            dims=("day", "cluster"),
        )
    return model


def static_concentration(
    fit_counts: np.ndarray,
    share_concentration: float,
) -> np.ndarray:
    return share_concentration + fit_counts.sum(axis=0)


def windowed_shares(frame: pd.DataFrame, split: str, clusters: int) -> np.ndarray:
    """(days, clusters) plain 7-day shares of one split, in day-major order."""
    rows = frame.loc[frame["split"] == split].sort_values([DAY_COLUMN, "cluster_id"])
    return rows["window_share"].to_numpy(dtype=float).reshape(-1, clusters)


def day_scores(
    arrays: dict[str, np.ndarray],
    kappa: np.ndarray,
    static: np.ndarray,
    window: np.ndarray,
) -> dict[str, np.ndarray]:
    counts = arrays["counts"]
    concentration = kappa[:, None, None] * arrays["recent_share"][None]
    return {
        "model": joint_log_score(counts, concentration),
        "dm_static": dirichlet_multinomial_logpmf(counts, static[None, :]),
        "mn_rolling_7": multinomial_logpmf(counts, window),
    }


def surprise_table(
    arrays: dict[str, np.ndarray],
    split: str,
    kappa: np.ndarray,
    simulations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for index in range(len(arrays["days"])):
        concentration = kappa[:, None] * arrays["recent_share"][index][None, :]
        score, p_value = week_surprise(
            arrays["counts"][index],
            int(arrays["totals"][index]),
            concentration,
            simulations,
            rng,
        )
        rows.append(
            {
                "split": split,
                DAY_COLUMN: str(np.datetime64(arrays["days"][index], "D")),
                "daily_total": int(arrays["totals"][index]),
                "log_score": score,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    config = load_experiment_config()
    settings = config["daily_composition"]
    model_settings = yaml.safe_load(
        (PROJECT_ROOT / settings["reference_config"]).read_text(encoding="utf-8")
    )
    sampling = dict(settings["sampling"])
    if args.run_mode == "pilot":
        sampling.update(settings["pilot_sampling"])
    sampling.setdefault("backend", "numpyro")
    sampling.setdefault("chain_method", "parallel")
    sampling.setdefault("max_treedepth", 12)
    seed = config["experiment"]["seed"]
    set_seed(seed)
    clusters = model_settings["clusters"]
    panel, input_sha256, source_dvc_hash = load_frozen_panel(config, model_settings)
    discount = model_settings["share"]["discount"]
    # Both share references come from the full panel; the warm-up rows are
    # dropped afterwards, so the first scored day still sees its memory.
    frame = panel.assign(
        recent_share=discounted_daily_shares(panel, clusters, discount),
        window_share=windowed_daily_shares(panel, clusters, WINDOW_DAYS),
    )
    frame = frame.loc[warm_rows(frame, model_settings["share"]["warmup_days"])]
    frame = frame.reset_index(drop=True)
    arrays = split_arrays(frame, clusters)
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
        "discount": discount,
        "priors": settings["priors"],
        "fit_days": int(len(arrays["fit"]["days"])),
        "validation_used_for_selection": False,
    }
    report_dir = PROJECT_ROOT / settings["report_dir"]
    if args.run_mode == "prior":
        with build_model(arrays["fit"], settings["priors"]):
            prior = pm.sample_prior_predictive(draws=200, random_seed=seed)
        observed = prior.prior_predictive["observed"].to_numpy().reshape(-1, clusters)
        shares = observed / observed.sum(axis=1, keepdims=True)
        summary = {
            "kappa": {
                label: float(np.quantile(prior.prior["kappa"], q))
                for label, q in (("p025", 0.025), ("p50", 0.5), ("p975", 0.975))
            },
            "max_share_prior_p50": float(np.median(shares.max(axis=1))),
            "max_share_fit_p50": float(
                np.median(
                    (arrays["fit"]["counts"] / arrays["fit"]["totals"][:, None]).max(1)
                )
            ),
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
    model = build_model(arrays["fit"], settings["priors"])
    idata, versions = sample_posterior(model, sampling, seed)
    idata.to_netcdf(artifact_dir / "posterior.nc")
    summary, diagnostics = posterior_diagnostics(
        idata, ("log_kappa",), ("log_kappa", "kappa", "rho"), sampling["max_treedepth"]
    )
    summary.reset_index().rename(columns={"index": "parameter"}).to_csv(
        run_dir / "posterior_summary.csv", index=False
    )
    kappa_all = np.asarray(idata.posterior["kappa"]).reshape(-1)
    rng = np.random.default_rng(seed)
    size = min(sampling["prediction_draws"], len(kappa_all))
    kappa = kappa_all[rng.choice(len(kappa_all), size=size, replace=False)]
    static = static_concentration(
        arrays["fit"]["counts"], settings["share_concentration"]
    )

    scores_frames = []
    for split in EVALUATED_SPLITS:
        window = windowed_shares(frame, split, clusters)
        scores = day_scores(arrays[split], kappa, static, window)
        scores_frames.append(
            pd.DataFrame(
                {
                    "split": split,
                    DAY_COLUMN: [
                        str(np.datetime64(d, "D")) for d in arrays[split]["days"]
                    ],
                    **scores,
                }
            )
        )
    scores_table = pd.concat(scores_frames, ignore_index=True)
    scores_table.to_csv(run_dir / "log_scores.csv", index=False)
    calibration = scores_table.loc[scores_table["split"] == "calibration"]
    comparison = compare_with_baselines(
        {name: calibration[name].to_numpy() for name in ("model", *BASELINES)},
        BASELINES,
        seed,
    )
    comparison["days"] = comparison.pop("weeks")
    means = {
        split: {
            name: float(scores_table.loc[scores_table["split"] == split, name].mean())
            for name in ("model", *BASELINES)
        }
        for split in EVALUATED_SPLITS
    }
    signal_kappa = kappa[: settings["kappa_draws"]]
    simulations = settings["simulations_per_draw"]
    surprise = pd.concat(
        [
            surprise_table(arrays[split], split, signal_kappa, simulations, rng)
            for split in SPLITS
            if split in arrays
        ],
        ignore_index=True,
    )
    surprise.to_csv(run_dir / "daily_surprise.csv", index=False)
    flagged = surprise["p_value"] < settings["alert_p_value"]
    signal = {
        split: {
            "days": int((surprise["split"] == split).sum()),
            "flagged": int(((surprise["split"] == split) & flagged).sum()),
        }
        for split in SPLITS
        if (surprise["split"] == split).any()
    }
    accepted = (
        comparison["difference"]["p025"] > 0 and diagnostics["rhat_max"] <= 1.01
    )
    report = {
        **metadata,
        "sampling": sampling,
        "versions": versions,
        "diagnostics": diagnostics,
        "evaluated_splits": list(EVALUATED_SPLITS),
        "log_score_means": means,
        "comparison": comparison,
        "signal": {"alert_p_value": settings["alert_p_value"], **signal},
        "candidate_status": (
            "pilot_only"
            if args.run_mode == "pilot"
            else ("accepted" if accepted else "rejected")
        ),
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
    record = build_run_record(
        config=config,
        run_name=f"m10d-{MODEL_ID}-{args.run_mode}",
        stage=STAGE,
        target="A1",
        split="calibration",
        view="40 patterns, daily",
        features=["daily_total", "recent_share"],
        parameters={
            "plan": PLAN,
            "run_key": run_key,
            "run_mode": args.run_mode,
            "discount": discount,
            "candidate_status": report["candidate_status"],
        },
        metrics={
            **{
                f"calibration_log_score_{k}": v
                for k, v in means["calibration"].items()
            },
            "bootstrap_difference_estimate": float(
                comparison["difference"]["estimate"]
            ),
            "bootstrap_difference_p025": float(comparison["difference"]["p025"]),
            "bootstrap_difference_p975": float(comparison["difference"]["p975"]),
            **{f"diagnostic_{k}": float(v) for k, v in diagnostics.items()},
            **{
                f"signal_{s}_flagged": float(v["flagged"]) for s, v in signal.items()
            },
        },
        artifacts=[str(run_dir.relative_to(PROJECT_ROOT))],
    )
    save_run_record(record, run_dir / "run.json")
    print(f"Run key: {run_key}")
    print(f"Diagnostics: {json.dumps(diagnostics, sort_keys=True)}")
    print(f"Calibration log score: {json.dumps(means['calibration'], sort_keys=True)}")
    print(f"Difference vs best baseline: {json.dumps(comparison['difference'])}")
    print(f"Signal: {json.dumps(signal)}\nStatus: {report['candidate_status']}")
    print(f"Report: {run_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
