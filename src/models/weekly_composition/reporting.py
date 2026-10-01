"""Plots and the offline MLflow record of a composition run."""

from __future__ import annotations

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


def save_plots(
    predictions: pd.DataFrame,
    idata: Any,
    paths: dict[str, Path],
    plot_clusters: int,
    trace_variables: tuple[str, ...],
    model_id: str,
) -> None:
    """Observed and predicted weekly shares of the largest clusters, and the trace."""
    largest = (
        predictions.loc[predictions["split"] == "fit"]
        .groupby("cluster_id")["complaint_count"]
        .sum()
        .nlargest(plot_clusters)
        .index
    )
    rows = math.ceil(plot_clusters / 2)
    figure, axes = plt.subplots(rows, 2, figsize=(15, rows * 3), sharex=True)
    for axis, cluster in zip(np.asarray(axes).ravel(), largest):
        part = predictions.loc[predictions["cluster_id"] == cluster]
        total = part["weekly_total"]
        axis.fill_between(
            part["week"],
            part["model_p10"] / total,
            part["model_p90"] / total,
            alpha=0.25,
            label="Intervalo predictivo 80%",
        )
        axis.plot(part["week"], part["model_p50"] / total, label="Mediana")
        axis.scatter(
            part["week"],
            part["complaint_count"] / total,
            s=10,
            color="black",
            label="Observado",
        )
        axis.axvline(pd.Timestamp("2024-10-01"), color="tab:orange", linestyle="--")
        axis.set_title(f"Patrón operativo {cluster}")
    np.asarray(axes).ravel()[0].legend(fontsize=8)
    figure.suptitle(f"M10 {model_id}: participación semanal observada y predictiva")
    figure.tight_layout()
    figure.savefig(paths["backtest"], dpi=160)
    plt.close(figure)

    az = importlib.import_module("arviz")
    az.plot_trace(idata, var_names=trace_variables, compact=True)
    figure = plt.gcf()
    figure.tight_layout()
    figure.savefig(paths["trace"], dpi=140)
    plt.close(figure)


def record_metrics(report: dict[str, Any]) -> dict[str, float]:
    diagnostics = report["diagnostics"]
    metrics = {
        "rhat_max": float(diagnostics["rhat_max"]),
        "ess_bulk_min": float(diagnostics["ess_bulk_min"]),
        "ess_tail_min": float(diagnostics["ess_tail_min"]),
        "divergences": float(diagnostics["divergences"]),
        "bfmi_min": float(diagnostics["bfmi_min"]),
        "tree_depth_limit_hits": float(diagnostics.get("tree_depth_limit_hits", 0)),
        "elapsed_seconds": float(report["resources"]["elapsed_seconds"]),
        "maximum_rss_gib": float(report["resources"]["maximum_rss_gib"]),
        "calibration_accepted": float(report["acceptance"]["accepted"]),
    }
    for split, predictors in report["metrics"].items():
        for name, values in predictors.items():
            for metric, value in values.items():
                metrics[f"{split}_{name}_{metric}"] = float(value)
    for label, value in report["comparison"]["difference"].items():
        metrics[f"calibration_log_score_difference_{label}"] = float(value)
    for check, passed in report["acceptance"]["checks"].items():
        metrics[f"acceptance_{check}"] = float(passed)
    return metrics


def save_offline_record(
    config: dict[str, Any],
    report: dict[str, Any],
    artifact_paths: list[Path],
    record_path: Path,
) -> None:
    model_spec = report["model_spec"]
    parameters = {
        "model_id": report["model_id"],
        "source_status": model_spec["source_status"],
        "family": model_spec["family"],
        "formula": model_spec["formula"],
        "config_sha256": report["config_sha256"],
        "source_manifest_sha256": report["source_manifest_sha256"],
        "weekly_counts_sha256": report["weekly_counts_sha256"],
        "source_dvc_hash": report["source_dvc_hash"],
        "weeks": json.dumps(report["weeks"], sort_keys=True),
        "priors": json.dumps(report["priors"], sort_keys=True),
        "prior_strategy": model_spec["prior_strategy"],
        "sampling": json.dumps(report["sampling"], sort_keys=True),
        "model_spec": json.dumps(model_spec, sort_keys=True),
        "versions": json.dumps(report["versions"], sort_keys=True),
    }
    artifact_names = [
        path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        for path in artifact_paths
    ]
    record = build_run_record(
        config=config,
        run_name=f"m10-{report['model_id']}-{report['run_key']}",
        stage="M10",
        target="weekly_cluster_composition",
        split="fit_calibration",
        view="complete_weeks",
        features=["cluster_id", "week", "weekly_total"],
        parameters=parameters,
        metrics=record_metrics(report),
        artifacts=artifact_names,
    )
    record["tags"].update(
        {
            "model_id": report["model_id"],
            "model_family": model_spec["family"],
            "source_status": model_spec["source_status"],
            "run_mode": report["run_mode"],
            "candidate_role": report["candidate_role"],
            "candidate_status": report["candidate_status"],
            "rejection_reason": report["rejection_reason"],
            "best_baseline": report["comparison"]["best_baseline"],
            "training_split": "fit",
            "selection_split": "calibration",
            "evaluation_split": "validation",
            "fit_only_training": "true",
            "validation_used_for_selection": "false",
            "calibration_accepted": str(report["acceptance"]["accepted"]).lower(),
        }
    )
    save_run_record(record, record_path)
