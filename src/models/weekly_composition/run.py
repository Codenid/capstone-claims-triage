"""Run a versioned weekly composition model (M10)."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import os
from pathlib import Path
import time
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.evaluation.experiment import (
    PROJECT_ROOT,
    execution_context,
    git_commit,
    load_experiment_config,
    set_seed,
)
from src.models.semantic_space import maximum_rss_gib
from src.models.weekly_composition import registry
from src.models.weekly_composition.baselines import (
    BASELINES,
    baseline_draws,
    baseline_scores,
    static_concentration,
)
from src.models.weekly_composition.data import (
    composition_panel,
    select_weeks,
    split_weeks,
    windowed_weeks,
)
from src.models.weekly_composition.diagnostics import prior_check_summary
from src.models.weekly_composition.evaluation import (
    compare_with_baselines,
    composition_acceptance,
    long_predictions,
    split_metrics,
)
from src.models.weekly_composition.reporting import save_offline_record, save_plots
from src.models.weekly_composition.scores import joint_log_score
from src.models.weekly_counts.contracts import SPLITS
from src.models.weekly_counts.diagnostics import posterior_diagnostics
from src.models.weekly_counts.metrics import candidate_status, rejection_reason
from src.models.weekly_counts.reporting import (
    file_sha256,
    save_source_snapshot,
    source_manifest,
    source_manifest_sha256,
)
from src.models.weekly_counts.run import (
    MODELING_CONFIG_PATH,
    RUN_MODES,
    load_candidate_role,
    load_frozen_frame,
    resolve_config_path,
    sampling_for_run_mode,
    write_json,
)
from src.models.weekly_counts.sampling import (
    sample_posterior,
    sample_predictions,
    sample_prior_variables,
    subsample_posterior,
)

# Validation predictions stay in predictions.csv for the single final check.
EVALUATED_SPLITS = ("fit", "calibration")
BASELINE_SEED_OFFSET = 1
REPORT_ROOT = PROJECT_ROOT / "reports/modeling/weekly_composition"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/models/weekly_composition"
M10_MODULES = (
    "__init__.py",
    "run.py",
    "data.py",
    "scores.py",
    "baselines.py",
    "diagnostics.py",
    "evaluation.py",
    "reporting.py",
    "registry.py",
    "models/__init__.py",
)
M9_MODULES = (
    "run.py",
    "data.py",
    "contracts.py",
    "rolling_reference.py",
    "fixed_poisson_reference.py",
    "prediction.py",
    "sampling.py",
    "diagnostics.py",
    "metrics.py",
    "comparison.py",
    "reporting.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Project-relative or absolute weekly-composition model config path.",
    )
    parser.add_argument(
        "--run-mode",
        choices=RUN_MODES,
        default="full",
        help="prior: prior predictive check; pilot: short technical run; "
        "full: sampling from the frozen config.",
    )
    return parser.parse_args()


def model_source_paths(module: ModuleType, config_path: Path) -> list[Path]:
    package = Path(__file__).resolve().parent
    reused = PROJECT_ROOT / "src/models/weekly_counts"
    return [
        Path(str(module.__file__)).resolve(),
        *(package / name for name in M10_MODULES),
        *(reused / name for name in M9_MODULES),
        config_path,
        config_path.parent / "releases.json",
        MODELING_CONFIG_PATH,
        PROJECT_ROOT / "src/evaluation/experiment.py",
        PROJECT_ROOT / "src/models/semantic_space.py",
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "uv.lock",
        PROJECT_ROOT / "artifacts/models/weekly_patterns.dvc",
    ]


def model_panel(
    module: ModuleType,
    panel: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Weeks a model uses; only models that need earlier weeks drop some."""
    if hasattr(module, "prepare_panel"):
        return module.prepare_panel(panel)
    return panel


def save_prior_check(
    module: ModuleType,
    fit: dict[str, np.ndarray],
    settings: dict[str, Any],
    seed: int,
    metadata: dict[str, Any],
) -> Path:
    model = module.build_model(fit, settings)
    draws = sample_prior_variables(
        model,
        settings["sampling"]["prior_draws"],
        ["kappa", "rho", "observed"],
        seed,
    )
    path = REPORT_ROOT / metadata["model_id"] / "prior_checks"
    path = path / f"{metadata['run_key']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {**metadata, "prior_check": prior_check_summary(draws, fit)})
    return path


def week_scores_frame(
    panel: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
) -> pd.DataFrame:
    return pd.DataFrame({"split": panel["split"], "week": panel["week"], **scores})


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    config_path = resolve_config_path(args.config)

    config = load_experiment_config(MODELING_CONFIG_PATH)
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings["sampling"] = sampling_for_run_mode(settings["sampling"], args.run_mode)
    module = registry.get_model(settings["model_id"])
    model_id = module.MODEL_ID
    candidate_role = load_candidate_role(config_path, model_id)

    config_sha256 = file_sha256(config_path)
    source_paths = model_source_paths(module, config_path)
    manifest = source_manifest(source_paths)
    manifest_sha256 = source_manifest_sha256(manifest)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_key = f"{timestamp}-{args.run_mode}-{config_sha256[:8]}-{manifest_sha256[:8]}"

    seed = config["experiment"]["seed"]
    set_seed(seed)
    frame, _, input_sha256, source_dvc_hash = load_frozen_frame(config, settings)
    panel = composition_panel(frame, settings["clusters"])
    fit = split_weeks(model_panel(module, panel), "fit")
    # Every model and baseline is scored on the weeks that have a full window.
    evaluation = windowed_weeks(panel)

    if args.run_mode == "prior":
        execution_host, _ = execution_context()
        prior_path = save_prior_check(
            module,
            fit,
            settings,
            seed,
            {
                "stage": "M10",
                "run_key": run_key,
                "run_mode": args.run_mode,
                "model_id": model_id,
                "candidate_role": candidate_role,
                "git_commit": git_commit(),
                "execution_host": execution_host,
                "config_sha256": config_sha256,
                "source_manifest_sha256": manifest_sha256,
                "weekly_counts_sha256": input_sha256,
                "source_dvc_hash": source_dvc_hash,
                "seed": seed,
                "priors": settings["priors"],
                "prior_draws": settings["sampling"]["prior_draws"],
                "fit_weeks": len(fit["week"]),
                "fit_only": True,
                "validation_used_for_selection": False,
            },
        )
        print(f"Run key: {run_key}")
        print(f"Model: {model_id}")
        print(f"Prior check: {prior_path.relative_to(PROJECT_ROOT)}")
        return

    report_dir = REPORT_ROOT / model_id / run_key
    artifact_dir = ARTIFACT_ROOT / model_id / run_key
    report_staging = report_dir.parent / f".{run_key}.inprogress"
    artifact_staging = artifact_dir.parent / f".{run_key}.inprogress"
    filenames = {
        "results": "results.json",
        "predictions": "predictions.csv",
        "scores": "log_scores.csv",
        "summary": "posterior_summary.csv",
        "backtest": "backtest.png",
        "trace": "trace.png",
        "record": "run.json",
    }
    report_paths = {name: report_staging / file for name, file in filenames.items()}
    final_report_paths = {name: report_dir / file for name, file in filenames.items()}
    report_staging.mkdir(parents=True)
    artifact_staging.mkdir(parents=True)
    snapshot_dir = artifact_staging / "source"
    save_source_snapshot(source_paths, snapshot_dir)
    if {path: file_sha256(snapshot_dir / path) for path in manifest} != manifest:
        raise RuntimeError("M10 source changed while creating the snapshot.")
    write_json(artifact_staging / "source_manifest.json", manifest)

    model = module.build_model(fit, settings)
    idata, versions = sample_posterior(model, settings["sampling"], seed)
    # Week-sized deterministics are recomputed for every prediction; storing
    # them would add hundreds of MB to posterior.nc.
    idata.posterior = idata.posterior.drop_vars(
        [name for name, values in idata.posterior.items() if "week" in values.dims]
    )
    idata.to_netcdf(artifact_staging / "posterior.nc")
    summary, diagnostics = posterior_diagnostics(
        idata,
        module.DIAGNOSTIC_VARIABLES,
        module.SUMMARY_VARIABLES,
        settings["sampling"]["max_treedepth"],
    )
    summary.reset_index().rename(columns={"index": "parameter"}).to_csv(
        report_paths["summary"], index=False
    )

    prediction_draws = settings["sampling"]["prediction_draws"]
    posterior = subsample_posterior(idata, prediction_draws, seed)
    concentration, model_draws = sample_predictions(
        module.build_model(evaluation, settings),
        posterior,
        "concentration",
        "observed",
        seed,
    )
    if not (model_draws.sum(axis=2) == evaluation["weekly_total"]).all():
        raise RuntimeError("M10 predictive draws must sum to weekly_total.")
    b2_concentration = static_concentration(
        split_weeks(panel, "fit")["counts"], settings["priors"]["share_concentration"]
    )
    draws = {
        "model": model_draws,
        **baseline_draws(
            evaluation, b2_concentration, len(model_draws), seed + BASELINE_SEED_OFFSET
        ),
    }
    predictions = long_predictions(evaluation, draws)
    predictions.to_csv(report_paths["predictions"], index=False)

    evaluated = np.isin(evaluation["split"], EVALUATED_SPLITS)
    scored = select_weeks(evaluation, evaluated)
    scores = {
        "model": joint_log_score(scored["counts"], concentration[:, evaluated]),
        **baseline_scores(scored, b2_concentration),
    }
    week_scores_frame(scored, scores).to_csv(report_paths["scores"], index=False)
    scored_draws = {name: values[:, evaluated] for name, values in draws.items()}
    metrics = {
        split: split_metrics(scored, scores, scored_draws, split)
        for split in EVALUATED_SPLITS
    }
    calibration = scored["split"] == "calibration"
    comparison = compare_with_baselines(
        {name: values[calibration] for name, values in scores.items()}, BASELINES, seed
    )
    acceptance = composition_acceptance(
        diagnostics, metrics["calibration"]["model"], comparison, settings
    )
    status = candidate_status(args.run_mode, acceptance)

    save_plots(
        predictions.loc[predictions["split"] != "validation"],
        idata,
        report_paths,
        settings["plot_clusters"],
        module.TRACE_VARIABLES,
        model_id,
    )
    model_spec = {
        **module.MODEL_SPEC,
        "clusters": settings["clusters"],
        "expected_complete_weeks": settings["expected_complete_weeks"],
        "priors": settings["priors"],
        "sampling": settings["sampling"],
    }
    write_json(artifact_staging / "model_spec.json", model_spec)

    report = {
        "stage": "M10",
        "run_key": run_key,
        "run_mode": args.run_mode,
        "model_id": model_id,
        "candidate_role": candidate_role,
        "candidate_status": status,
        "rejection_reason": rejection_reason(args.run_mode, acceptance),
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
            "fit_used_for_training": len(fit["week"]),
            **{split: int((evaluation["split"] == split).sum()) for split in SPLITS},
        },
        "priors": settings["priors"],
        "sampling": settings["sampling"],
        "versions": versions,
        "diagnostics": diagnostics,
        "evaluated_splits": list(EVALUATED_SPLITS),
        "metrics": metrics,
        "comparison": comparison,
        "acceptance": acceptance,
        "draw_sums_match_weekly_total": True,
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
    save_offline_record(config, report, artifact_paths, report_paths["record"])
    (report_staging / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    report_staging.replace(report_dir)

    print(f"Run key: {run_key}")
    print(f"Model: {model_id}")
    print(f"Run mode: {args.run_mode}")
    print(f"Diagnostics: {diagnostics}")
    print(f"Calibration log score: {comparison['log_score']}")
    print(f"Status: {status}")
    print(f"Report: {report_dir.relative_to(PROJECT_ROOT)}")
    print(f"Artifacts: {artifact_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
