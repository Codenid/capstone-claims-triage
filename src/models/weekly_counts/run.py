"""Run a versioned weekly count model."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import io
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.evaluation.experiment import PROJECT_ROOT, load_experiment_config, set_seed
from src.models.semantic_space import load_dvc_hash, maximum_rss_gib
from src.models.weekly_counts import fixed_poisson_reference, registry
from src.models.weekly_counts.contracts import SPLITS
from src.models.weekly_counts.data import prepare_model_frame
from src.models.weekly_counts.diagnostics import (
    posterior_diagnostics,
    prior_predictive_summary,
)
from src.models.weekly_counts.metrics import calibration_acceptance, evaluate_predictions
from src.models.weekly_counts.prediction import build_predictions
from src.models.weekly_counts.reporting import (
    file_sha256,
    save_offline_record,
    save_plots,
    save_source_snapshot,
    source_manifest,
    source_manifest_sha256,
)
from src.models.weekly_counts.sampling import (
    sample_posterior,
    sample_predictions,
    sample_prior_predictive,
    subsample_posterior,
)

DEFAULT_MODEL_CONFIG = Path("configs/weekly_counts/nb_softmax_linear_v2.yaml")
MODELING_CONFIG_PATH = PROJECT_ROOT / "configs/modeling.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="Project-relative or absolute weekly-count model config path.",
    )
    return parser.parse_args()


def model_source_paths(model_module: Any, config_path: Path) -> list[Path]:
    module_dir = Path(__file__).resolve().parent
    shared_modules = (
        "run.py",
        "data.py",
        "sampling.py",
        "diagnostics.py",
        "prediction.py",
        "metrics.py",
        "reporting.py",
        "registry.py",
        "contracts.py",
        "negative_binomial.py",
        "fixed_poisson_reference.py",
    )
    return [
        Path(model_module.__file__).resolve(),
        module_dir / "models/__init__.py",
        *(module_dir / name for name in shared_modules),
        config_path,
        config_path.parent / "releases.json",
        MODELING_CONFIG_PATH,
        PROJECT_ROOT / "src/evaluation/experiment.py",
        PROJECT_ROOT / "src/models/semantic_space.py",
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "uv.lock",
        PROJECT_ROOT / "artifacts/models/weekly_patterns.dvc",
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()

    config = load_experiment_config(MODELING_CONFIG_PATH)
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_module = registry.get_model(settings["model_id"])
    model_id = model_module.MODEL_ID

    config_sha256 = file_sha256(config_path)
    source_paths = model_source_paths(model_module, config_path)
    manifest = source_manifest(source_paths)
    manifest_sha256 = source_manifest_sha256(manifest)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_key = f"{timestamp}-{config_sha256[:8]}-{manifest_sha256[:8]}"

    report_dir = PROJECT_ROOT / "reports/modeling/weekly_counts" / model_id / run_key
    artifact_dir = PROJECT_ROOT / "artifacts/models/weekly_counts" / model_id / run_key
    report_staging = report_dir.parent / f".{run_key}.inprogress"
    artifact_staging = artifact_dir.parent / f".{run_key}.inprogress"
    filenames = {
        "results": "results.json",
        "predictions": "predictions.csv",
        "metrics": "metrics.csv",
        "summary": "posterior_summary.csv",
        "backtest": "backtest.png",
        "coverage": "coverage.png",
        "residuals": "residuals.png",
        "prior": "prior.png",
        "trace": "trace.png",
        "record": "run.json",
    }
    report_paths = {name: report_staging / filename for name, filename in filenames.items()}
    final_report_paths = {name: report_dir / filename for name, filename in filenames.items()}
    report_staging.mkdir(parents=True)
    artifact_staging.mkdir(parents=True)
    snapshot_dir = artifact_staging / "source"
    save_source_snapshot(source_paths, snapshot_dir)
    snapshot_manifest = {
        path: file_sha256(snapshot_dir / path) for path in manifest
    }
    if snapshot_manifest != manifest:
        raise RuntimeError("M9 source changed while creating the snapshot.")
    write_json(artifact_staging / "source_manifest.json", manifest)

    seed = config["experiment"]["seed"]
    set_seed(seed)
    input_path = PROJECT_ROOT / config["paths"]["weekly_counts"]
    input_bytes = input_path.read_bytes()
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    if input_sha256 != settings["weekly_counts_sha256"]:
        raise ValueError("M9 weekly counts SHA-256 is not frozen.")
    source_dvc_hash = load_dvc_hash(
        PROJECT_ROOT / config["paths"]["weekly_patterns_artifacts_dvc"]
    )
    if source_dvc_hash != settings["source_dvc_hash"]:
        raise ValueError("M9 weekly patterns DVC hash is not frozen.")

    raw = pd.read_csv(io.BytesIO(input_bytes))
    frame, time_center = prepare_model_frame(
        raw,
        settings["clusters"],
        settings["expected_complete_weeks"],
        settings["time_scale_days"],
    )
    fit = frame.loc[frame["split"] == "fit"].reset_index(drop=True)

    model = model_module.build_model(fit, settings)
    idata, versions = sample_posterior(model, settings["sampling"], seed)
    prior_draws = sample_prior_predictive(
        model,
        settings["sampling"]["prior_draws"],
        model_module.OBSERVED_VARIABLE,
        seed,
    )
    idata.to_netcdf(artifact_staging / "posterior.nc")

    summary, diagnostics = posterior_diagnostics(
        idata,
        model_module.DIAGNOSTIC_VARIABLES,
        model_module.SUMMARY_VARIABLES,
        settings["sampling"]["max_treedepth"],
    )
    summary.reset_index().rename(columns={"index": "parameter"}).to_csv(
        report_paths["summary"],
        index=False,
    )
    prior_summary = prior_predictive_summary(prior_draws, fit)

    posterior = subsample_posterior(
        idata,
        settings["sampling"]["prediction_draws"],
        seed,
    )
    prediction_model = model_module.build_model(frame, settings)
    expected_draws, predictive_draws = sample_predictions(
        prediction_model,
        posterior,
        model_module.EXPECTED_VARIABLE,
        model_module.OBSERVED_VARIABLE,
        seed,
    )

    reference_rates = fixed_poisson_reference.fit_baseline_rates(
        frame,
        settings["clusters"],
    )
    reference_expected = fixed_poisson_reference.expected_counts(
        frame["cluster_id"].to_numpy(dtype=np.int64),
        frame["weekly_total"].to_numpy(dtype=float),
        reference_rates,
    )
    reference_draws = fixed_poisson_reference.predictive_draws(
        reference_expected,
        draws=len(expected_draws),
        seed=seed + 1,
    )
    predictions = build_predictions(
        frame,
        expected_draws,
        predictive_draws,
        reference_draws,
    )
    predictions.to_csv(report_paths["predictions"], index=False)
    metrics, backtest = evaluate_predictions(predictions)
    metrics.to_csv(report_paths["metrics"], index=False)

    normalization_by_split = {}
    for split in SPLITS:
        split_predictions = predictions.loc[predictions["split"] == split]
        weekly = split_predictions.groupby("week", observed=True).agg(
            expected_sum=("expected_mean", "sum"),
            weekly_total=("weekly_total", "first"),
        )
        absolute_error = np.abs(weekly["expected_sum"] / weekly["weekly_total"] - 1)
        normalization_by_split[split] = {
            "maximum_abs_error": float(absolute_error.max()),
            "mean_abs_error": float(absolute_error.mean()),
        }
    mean_normalization = {
        "maximum_abs_error": max(
            values["maximum_abs_error"] for values in normalization_by_split.values()
        ),
        "by_split": normalization_by_split,
    }
    acceptance = calibration_acceptance(
        diagnostics,
        backtest["calibration"],
        settings,
    )

    save_plots(
        predictions,
        metrics,
        prior_draws,
        fit,
        idata,
        report_paths,
        settings["plot_clusters"],
        model_module.TRACE_VARIABLES,
        model_id,
    )

    model_spec = {
        **model_module.MODEL_SPEC,
        "clusters": settings["clusters"],
        "expected_complete_weeks": settings["expected_complete_weeks"],
        "time_center": time_center.isoformat(),
        "time_scale_days": settings["time_scale_days"],
        "priors": settings["priors"],
        "sampling": settings["sampling"],
    }
    write_json(artifact_staging / "model_spec.json", model_spec)

    report = {
        "stage": "M9",
        "run_key": run_key,
        "model_id": model_id,
        "source_status": model_spec["source_status"],
        "model_spec": model_spec,
        "config_sha256": config_sha256,
        "source_manifest_sha256": manifest_sha256,
        "source_manifest": manifest,
        "weekly_counts_sha256": input_sha256,
        "source_dvc_hash": source_dvc_hash,
        "formula": model_spec["formula"],
        "fit_only_training": True,
        "validation_used_for_selection": False,
        "weeks": {
            split: int(frame.loc[frame["split"] == split, "week"].nunique())
            for split in SPLITS
        },
        "rows": {
            split: int((frame["split"] == split).sum()) for split in SPLITS
        },
        "time_center": time_center.isoformat(),
        "priors": settings["priors"],
        "sampling": settings["sampling"],
        "versions": versions,
        "diagnostics": diagnostics,
        "prior_predictive": prior_summary,
        "mean_normalization": mean_normalization,
        "backtest": backtest,
        "calibration_acceptance": acceptance,
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_rss_gib": maximum_rss_gib(),
            "cpu_count": os.cpu_count(),
        },
    }
    write_json(report_paths["results"], report)
    write_json(artifact_staging / "metadata.json", report)
    (artifact_staging / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    artifact_staging.replace(artifact_dir)

    artifact_paths = [
        path for name, path in final_report_paths.items() if name != "record"
    ]
    artifact_paths.append(artifact_dir)
    save_offline_record(
        config,
        settings,
        report,
        artifact_paths,
        report_paths["record"],
    )
    (report_staging / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    report_staging.replace(report_dir)

    print(f"Run key: {run_key}")
    print(f"Model: {model_id}")
    print(f"Diagnostics: {json.dumps(diagnostics, sort_keys=True)}")
    print(f"Status: {'accepted' if acceptance['accepted'] else 'rejected'}")
    print(f"Report: {report_dir.relative_to(PROJECT_ROOT)}")
    print(f"Artifacts: {artifact_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
