"""Run a versioned daily count model (models_plan.md §28).

Mirrors src/models/weekly_counts/run.py at day granularity and reuses its
sampling, diagnostics, metrics and bootstrap helpers. Outputs go to
reports/modeling/daily_counts/<model>/<run_key>/ and
artifacts/models/daily_counts/<model>/<run_key>/.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    execution_context,
    git_commit,
    load_experiment_config,
    save_run_record,
    set_seed,
)
from src.models.daily_counts import baselines, registry
from src.models.daily_counts.contracts import DAY_COLUMN, SPLITS, TOTAL_COLUMN
from src.models.daily_counts.data import as_weekly_view, fit_rows, load_frozen_panel
from src.models.semantic_space import maximum_rss_gib
from src.models.weekly_counts.comparison import compare_with_baselines
from src.models.weekly_counts.diagnostics import (
    posterior_diagnostics,
    prior_check_summary,
    prior_predictive_summary,
)
from src.models.weekly_counts.metrics import (
    candidate_status,
    evaluate_predictions,
    rejection_reason,
    run_acceptance,
)
from src.models.weekly_counts.prediction import summarize_draws
from src.models.weekly_counts.reporting import (
    file_sha256,
    save_source_snapshot,
    source_manifest,
    source_manifest_sha256,
)
from src.models.weekly_counts.sampling import (
    sample_posterior,
    sample_predictions,
    sample_prior_predictive,
    sample_prior_variables,
    subsample_posterior,
)

STAGE = "M9D"
PLAN = "models_plan.md §28"
MODELING_CONFIG_PATH = PROJECT_ROOT / "configs/modeling.yaml"
REPORT_ROOT = PROJECT_ROOT / "reports/modeling/daily_counts"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/models/daily_counts"
RUN_MODES = ("prior", "pilot", "full")
PILOT_SAMPLING = {"chains": 2, "tune": 250, "draws": 250, "prediction_draws": 500}
EVALUATED_SPLITS = ("fit", "calibration")
BASELINE_NAMES = ("baseline", baselines.ROLLING_NAME)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-mode", choices=RUN_MODES, default="full")
    return parser.parse_args()


def model_source_paths(module: Any, config_path: Path) -> list[Path]:
    package = Path(__file__).resolve().parent
    shared = ("run.py", "data.py", "references.py", "baselines.py", "contracts.py")
    weekly = PROJECT_ROOT / "src/models/weekly_counts"
    return [
        Path(module.__file__).resolve(),
        package / "models/nb_daily_hierarchical_v1.py",
        *(package / name for name in shared),
        *(weekly / name for name in ("sampling.py", "diagnostics.py", "metrics.py")),
        weekly / "comparison.py",
        weekly / "prediction.py",
        weekly / "negative_binomial.py",
        weekly / "discounted_reference.py",
        config_path,
        config_path.parent / "releases.json",
        MODELING_CONFIG_PATH,
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "uv.lock",
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def flatten(values: np.ndarray) -> np.ndarray:
    """(draws, days, clusters) draws to (draws, rows) in day-major row order."""
    return values.reshape(values.shape[0], -1)


def build_predictions(
    frame: pd.DataFrame,
    expected: np.ndarray,
    predictive: np.ndarray,
    reference: np.ndarray,
) -> pd.DataFrame:
    observed = frame["complaint_count"].to_numpy(dtype=np.int64)
    totals = frame[TOTAL_COLUMN].to_numpy(dtype=float)
    columns = ["split", DAY_COLUMN, "cluster_id", "complaint_count", TOTAL_COLUMN]
    result = pd.concat(
        [
            frame[columns].reset_index(drop=True),
            summarize_draws(expected, "expected"),
            summarize_draws(predictive, "model"),
            summarize_draws(reference, "baseline"),
        ],
        axis=1,
    )
    for name, draws in (("model", predictive), ("baseline", reference)):
        result[f"{name}_upper_tail_probability"] = (draws >= observed[None, :]).mean(
            axis=0
        )
        result[f"{name}_impossible_probability"] = (draws > totals[None, :]).mean(
            axis=0
        )
    return result


def save_plots(
    predictions: pd.DataFrame,
    report_paths: dict[str, Path],
    clusters: int,
    idata: Any,
    trace_variables: tuple[str, ...],
) -> None:
    import arviz as az

    rows = predictions.loc[predictions["split"] != "validation"]
    sizes = rows.groupby("cluster_id")["complaint_count"].sum()
    largest = sizes.nlargest(clusters).index
    figure, axes = plt.subplots(len(largest), 1, figsize=(12, 2.2 * len(largest)))
    for axis, cluster in zip(np.atleast_1d(axes), largest):
        part = rows.loc[rows["cluster_id"] == cluster].tail(120)
        days = pd.to_datetime(part[DAY_COLUMN])
        axis.fill_between(
            days, part["model_p025"], part["model_p975"], color="#4C72B0", alpha=0.15
        )
        axis.fill_between(
            days, part["model_p10"], part["model_p90"], color="#4C72B0", alpha=0.25
        )
        axis.plot(days, part["model_p50"], color="#4C72B0", linewidth=1)
        axis.plot(days, part["complaint_count"], color="#222", linewidth=1)
        axis.set_title(f"patrón {cluster}, últimos 120 días", fontsize=9)
    figure.tight_layout()
    figure.savefig(report_paths["backtest"], dpi=100)
    plt.close(figure)
    axes = az.plot_trace(idata, var_names=list(trace_variables))
    figure = np.atleast_1d(axes).ravel()[0].get_figure()
    figure.savefig(report_paths["trace"], dpi=80)
    plt.close(figure)


def save_offline_record(
    config: dict[str, Any],
    settings: dict[str, Any],
    report: dict[str, Any],
    artifacts: list[Path],
    path: Path,
) -> None:
    calibration = report["backtest"]["calibration"]["model"]
    metrics = {f"calibration_{k}": float(v) for k, v in calibration.items()}
    comparison = report["comparison"]
    metrics["calibration_wis_gain"] = float(comparison["wis_gain"])
    for name, values in comparison["difference"].items():
        for label, value in values.items():
            metrics[f"bootstrap_{name}_{label}"] = float(value)
    diagnostics = report["diagnostics"]
    metrics.update({f"diagnostic_{k}": float(v) for k, v in diagnostics.items()})
    record = build_run_record(
        config=config,
        run_name=f"m9d-{report['model_id']}-{report['run_mode']}",
        stage=STAGE,
        target="A1",
        split="calibration",
        view="40 patterns, daily",
        features=["daily_total", "recent_share", "day_of_week"],
        parameters={
            "plan": PLAN,
            "model_id": report["model_id"],
            "run_key": report["run_key"],
            "run_mode": report["run_mode"],
            "discount": settings["share"]["discount"],
            "priors": json.dumps(settings["priors"]),
            "candidate_status": report["candidate_status"],
        },
        metrics=metrics,
        artifacts=[str(item.relative_to(PROJECT_ROOT)) for item in artifacts],
    )
    save_run_record(record, path)


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    config = load_experiment_config(MODELING_CONFIG_PATH)
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.run_mode == "pilot":
        settings["sampling"] = {**settings["sampling"], **PILOT_SAMPLING}
    module = registry.get_model(settings["model_id"])
    model_id = module.MODEL_ID
    releases = json.loads((config_path.parent / "releases.json").read_text("utf-8"))
    candidate_role = releases[model_id].get("candidate_role", "candidate")

    config_sha256 = file_sha256(config_path)
    source_paths = model_source_paths(module, config_path)
    manifest = source_manifest(source_paths)
    manifest_sha256 = source_manifest_sha256(manifest)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_key = f"{timestamp}-{args.run_mode}-{config_sha256[:8]}-{manifest_sha256[:8]}"

    seed = config["experiment"]["seed"]
    set_seed(seed)
    panel, input_sha256, source_dvc_hash = load_frozen_panel(config, settings)
    frame = module.prepare_frame(panel, settings)
    fit = fit_rows(frame)
    metadata = {
        "stage": STAGE,
        "plan": PLAN,
        "run_key": run_key,
        "run_mode": args.run_mode,
        "model_id": model_id,
        "candidate_role": candidate_role,
        "git_commit": git_commit(),
        "config_sha256": config_sha256,
        "source_manifest_sha256": manifest_sha256,
        "daily_counts_sha256": input_sha256,
        "source_dvc_hash": source_dvc_hash,
        "seed": seed,
        "priors": settings["priors"],
        "share": settings["share"],
        "fit_days": int(fit[DAY_COLUMN].nunique()),
        "fit_only": True,
        "validation_used_for_selection": False,
    }

    if args.run_mode == "prior":
        model = module.build_model(fit, settings)
        names = [module.EXPECTED_VARIABLE, module.OBSERVED_VARIABLE, "alpha"]
        draws = sample_prior_variables(
            model, settings["sampling"]["prior_draws"], names, seed
        )
        summary = prior_check_summary(
            flatten(draws[module.OBSERVED_VARIABLE]),
            flatten(draws[module.EXPECTED_VARIABLE]),
            draws["alpha"],
            as_weekly_view(fit),
        )
        path = REPORT_ROOT / model_id / "prior_checks" / f"{run_key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        execution_host, _ = execution_context()
        write_json(
            path, {**metadata, "execution_host": execution_host, "prior_check": summary}
        )
        print(f"Run key: {run_key}\nPrior check: {path.relative_to(PROJECT_ROOT)}")
        return

    report_dir = REPORT_ROOT / model_id / run_key
    artifact_dir = ARTIFACT_ROOT / model_id / run_key
    report_staging = report_dir.parent / f".{run_key}.inprogress"
    artifact_staging = artifact_dir.parent / f".{run_key}.inprogress"
    names = {
        "results": "results.json",
        "predictions": "predictions.csv",
        "metrics": "metrics.csv",
        "bootstrap": "bootstrap.json",
        "summary": "posterior_summary.csv",
        "backtest": "backtest.png",
        "trace": "trace.png",
        "record": "run.json",
    }
    report_paths = {key: report_staging / name for key, name in names.items()}
    report_staging.mkdir(parents=True)
    artifact_staging.mkdir(parents=True)
    save_source_snapshot(source_paths, artifact_staging / "source")
    write_json(artifact_staging / "source_manifest.json", manifest)

    model = module.build_model(fit, settings)
    idata, versions = sample_posterior(model, settings["sampling"], seed)
    prior_draws = flatten(
        sample_prior_predictive(
            model, settings["sampling"]["prior_draws"], module.OBSERVED_VARIABLE, seed
        )
    )
    idata.posterior = idata.posterior.drop_vars(
        [name for name, values in idata.posterior.items() if "day" in values.dims]
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
    expected, predictive = sample_predictions(
        module.build_model(frame, settings),
        posterior,
        module.EXPECTED_VARIABLE,
        module.OBSERVED_VARIABLE,
        seed,
    )
    expected, predictive = flatten(expected), flatten(predictive)
    draws = len(expected)
    reference = baselines.fixed_reference_draws(
        frame, fit, settings["clusters"], draws, seed + baselines.BASELINE_SEED_OFFSET
    )
    predictions = build_predictions(frame, expected, predictive, reference)
    rolling = baselines.rolling_columns(
        frame, panel, settings["clusters"], draws, seed + baselines.ROLLING_SEED_OFFSET
    )
    predictions = pd.concat([predictions, rolling], axis=1)
    predictions.to_csv(report_paths["predictions"], index=False)

    metrics, backtest = evaluate_predictions(predictions, EVALUATED_SPLITS)
    metrics.to_csv(report_paths["metrics"], index=False)
    comparison = compare_with_baselines(
        as_weekly_view(predictions), BASELINE_NAMES, seed
    )
    comparison["days"] = comparison.pop("weeks")
    write_json(report_paths["bootstrap"], comparison)
    acceptance = run_acceptance(
        diagnostics, backtest["calibration"], comparison, settings
    )
    status = candidate_status(args.run_mode, acceptance)
    save_plots(
        predictions,
        report_paths,
        settings["plot_clusters"],
        idata,
        module.TRACE_VARIABLES,
    )

    model_spec = {
        **module.MODEL_SPEC,
        "clusters": settings["clusters"],
        "expected_days": settings["expected_days"],
        "share": settings["share"],
        "priors": settings["priors"],
        "sampling": settings["sampling"],
    }
    write_json(artifact_staging / "model_spec.json", model_spec)
    report = {
        **metadata,
        "candidate_status": status,
        "rejection_reason": rejection_reason(args.run_mode, acceptance),
        "model_spec": model_spec,
        "formula": module.FORMULA,
        "source_manifest": manifest,
        "days": {
            split: int(frame.loc[frame["split"] == split, DAY_COLUMN].nunique())
            for split in SPLITS
        },
        "sampling": settings["sampling"],
        "versions": versions,
        "diagnostics": diagnostics,
        "prior_predictive": prior_predictive_summary(prior_draws, as_weekly_view(fit)),
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
    write_json(report_paths["results"], report)
    write_json(artifact_staging / "metadata.json", report)
    (artifact_staging / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    artifact_staging.replace(artifact_dir)
    final_paths = [report_dir / name for key, name in names.items() if key != "record"]
    save_offline_record(
        config, settings, report, [*final_paths, artifact_dir], report_paths["record"]
    )
    (report_staging / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    report_staging.replace(report_dir)

    print(f"Run key: {run_key}\nModel: {model_id}\nRun mode: {args.run_mode}")
    print(f"Diagnostics: {json.dumps(diagnostics, sort_keys=True)}")
    print(f"Calibration: {json.dumps(comparison['metrics'], sort_keys=True)}")
    print(f"WIS gain: {comparison['wis_gain']:.4f}\nStatus: {status}")
    print(f"Report: {report_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
