"""Reporting helpers for the M9 weekly count model."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.experiment import PROJECT_ROOT, build_run_record, save_run_record
from src.models.weekly_counts.contracts import SPLITS


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_fingerprint(config: dict[str, Any]) -> str:
    relevant = {
        "negative_binomial": config["negative_binomial"],
        "splits": config["evaluation"]["splits"],
    }
    encoded = json.dumps(relevant, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_plots(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    prior_draws: np.ndarray,
    fit: pd.DataFrame,
    idata: Any,
    paths: dict[str, Path],
    plot_clusters: int,
) -> None:
    fit_totals = (
        predictions.loc[predictions["split"] == "fit"]
        .groupby("cluster_id", observed=True)["complaint_count"]
        .sum()
        .nlargest(plot_clusters)
        .index
    )
    rows = math.ceil(plot_clusters / 2)
    figure, axes = plt.subplots(rows, 2, figsize=(15, rows * 3), sharex=True)
    for axis, cluster in zip(np.asarray(axes).ravel(), fit_totals):
        cluster_frame = predictions.loc[predictions["cluster_id"] == cluster]
        axis.fill_between(
            cluster_frame["week"],
            cluster_frame["model_p10"],
            cluster_frame["model_p90"],
            alpha=0.25,
            label="Intervalo predictivo 80%",
        )
        axis.plot(cluster_frame["week"], cluster_frame["model_p50"], label="Mediana")
        axis.scatter(
            cluster_frame["week"],
            cluster_frame["complaint_count"],
            s=10,
            color="black",
            label="Observado",
        )
        axis.axvline(pd.Timestamp("2024-10-01"), color="tab:orange", linestyle="--")
        axis.axvline(pd.Timestamp("2025-01-01"), color="tab:red", linestyle="--")
        axis.set_title(f"Patrón operativo {cluster}")
    for axis in np.asarray(axes).ravel()[len(fit_totals) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle("M9: conteo observado y distribución predictiva")
    figure.tight_layout()
    figure.savefig(paths["backtest"], dpi=160)
    plt.close(figure)

    global_metrics = metrics.loc[
        (metrics["cluster_id"] == -1)
        & (metrics["split"].isin(["calibration", "validation"]))
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for axis, level in zip(axes, (50, 80, 95)):
        pivot = global_metrics.pivot(index="split", columns="model", values=f"coverage_{level}")
        pivot.plot.bar(ax=axis)
        axis.axhline(level / 100, color="black", linestyle="--")
        axis.set_title(f"Cobertura {level}%")
        axis.set_ylim(0, 1)
        axis.set_xlabel("")
    figure.tight_layout()
    figure.savefig(paths["coverage"], dpi=160)
    plt.close(figure)

    residuals = predictions.copy()
    residuals["log_residual"] = np.log1p(residuals["complaint_count"]) - np.log1p(
        residuals["model_p50"]
    )
    pivot = residuals.pivot(index="cluster_id", columns="week", values="log_residual")
    figure, axis = plt.subplots(figsize=(15, 7))
    image = axis.imshow(pivot, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axis.set(title="Residuo logarítmico por patrón", xlabel="Semana", ylabel="Cluster")
    figure.colorbar(image, ax=axis, label="log(1 + observado) - log(1 + mediana)")
    figure.tight_layout()
    figure.savefig(paths["residuals"], dpi=160)
    plt.close(figure)

    prior_values = prior_draws.reshape(-1)
    observed_values = fit["complaint_count"].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(np.log1p(prior_values), bins=80, alpha=0.5, density=True, label="Prior")
    axis.hist(np.log1p(observed_values), bins=80, alpha=0.5, density=True, label="Fit")
    axis.set(
        title="Comprobación predictiva previa",
        xlabel="log(1 + conteo semanal)",
        ylabel="Densidad",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(paths["prior"], dpi=160)
    plt.close(figure)

    az = importlib.import_module("arviz")
    az.plot_trace(
        idata,
        var_names=[
            "log_rate_sigma",
            "annual_trend_sigma",
            "log_alpha_global",
            "log_alpha_sigma",
        ],
        compact=True,
    )
    figure = plt.gcf()
    figure.tight_layout()
    figure.savefig(paths["trace"], dpi=140)
    plt.close(figure)


def save_offline_record(config: dict[str, Any], report: dict[str, Any]) -> None:
    settings = config["negative_binomial"]
    backtest = report["backtest"]
    diagnostics = report["diagnostics"]
    acceptance = report["calibration_acceptance"]
    prior_predictive = report["prior_predictive"]
    mlflow_metrics = {
        "rhat_max": float(diagnostics["rhat_max"]),
        "ess_bulk_min": float(diagnostics["ess_bulk_min"]),
        "ess_tail_min": float(diagnostics["ess_tail_min"]),
        "divergences": float(diagnostics["divergences"]),
        "bfmi_min": float(diagnostics["bfmi_min"]),
        "tree_depth_max": float(diagnostics.get("tree_depth_max", 0)),
        "tree_depth_limit_hits": float(
            diagnostics.get("tree_depth_limit_hits", 0)
        ),
        "prior_median": prior_predictive["median"],
        "prior_p99": prior_predictive["p99"],
        "prior_maximum": prior_predictive["maximum"],
        "prior_impossible_fraction": prior_predictive["impossible_fraction"],
        "calibration_accepted": float(acceptance["accepted"]),
        "elapsed_seconds": report["resources"]["elapsed_seconds"],
        "maximum_rss_gib": report["resources"]["maximum_rss_gib"],
        "mean_normalization_max_abs_error": report["mean_normalization"][
            "maximum_abs_error"
        ],
    }
    for check, passed in acceptance["checks"].items():
        mlflow_metrics[f"acceptance_{check}"] = float(passed)
    for split in SPLITS:
        for model in ("model", "baseline"):
            for metric, value in backtest[split][model].items():
                mlflow_metrics[f"{split}_{model}_{metric}"] = float(value)

    record = build_run_record(
        config=config,
        run_name=f"m9-negative-binomial-{settings['model_version']}",
        stage="M9",
        target="weekly_cluster_volume",
        split="fit_calibration_validation",
        view="complete_weeks",
        features=["cluster_id", "week", "weekly_total"],
        parameters={
            "model_version": settings["model_version"],
            "formula": report["formula"],
            "priors": json.dumps(settings["priors"], sort_keys=True),
            "source_dvc_hash": settings["source_dvc_hash"],
            "weekly_counts_sha256": report["weekly_counts_sha256"],
            "clusters": settings["clusters"],
            "sampling_backend": settings["sampling"]["backend"],
            "chain_method": settings["sampling"]["chain_method"],
            "chains": settings["sampling"]["chains"],
            "draws": settings["sampling"]["draws"],
            "tune": settings["sampling"]["tune"],
            "target_accept": settings["sampling"]["target_accept"],
            "max_treedepth": settings["sampling"]["max_treedepth"],
            "prediction_draws": settings["sampling"]["prediction_draws"],
            "pymc_version": report["versions"]["pymc"],
            "arviz_version": report["versions"]["arviz"],
            "jax_version": report["versions"]["jax"],
            "jaxlib_version": report["versions"]["jaxlib"],
            "numpyro_version": report["versions"]["numpyro"],
            "mean_link": "softmax",
            "mean_sum_constrained": True,
            "predictive_draw_sum_constrained": False,
            "validation_used_for_selection": False,
        },
        metrics=mlflow_metrics,
        artifacts=[
            config["paths"]["negative_binomial_report"],
            config["paths"]["negative_binomial_predictions"],
            config["paths"]["negative_binomial_metrics"],
            config["paths"]["negative_binomial_summary"],
            config["paths"]["negative_binomial_backtest_plot"],
            config["paths"]["negative_binomial_coverage_plot"],
            config["paths"]["negative_binomial_residual_plot"],
            config["paths"]["negative_binomial_prior_plot"],
            config["paths"]["negative_binomial_trace_plot"],
            config["paths"]["negative_binomial_artifacts"],
        ],
    )
    record["tags"].update(
        {
            "training_split": "fit",
            "selection_split": "calibration",
            "evaluation_split": "validation",
            "validation_used_for_selection": "false",
            "calibration_accepted": str(acceptance["accepted"]).lower(),
            "candidate_status": "accepted" if acceptance["accepted"] else "rejected",
            "model_version": settings["model_version"],
            "mean_sum_constrained": "true",
            "predictive_draw_sum_constrained": "false",
        }
    )
    path = (
        PROJECT_ROOT
        / config["paths"]["offline_runs"]
        / "m9"
        / settings["model_version"]
        / "run.json"
    )
    save_run_record(record, path)
