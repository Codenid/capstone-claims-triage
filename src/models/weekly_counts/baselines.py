"""Register the deterministic rolling-share baselines B1-R4 and B1-R13."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import time
from typing import Any

import pandas as pd
import yaml

from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    load_experiment_config,
    save_run_record,
)
from src.models.weekly_counts import rolling_reference
from src.models.weekly_counts.metrics import predictive_metrics
from src.models.weekly_counts.reporting import (
    file_sha256,
    source_manifest,
    source_manifest_sha256,
)
from src.models.weekly_counts.rolling_reference import (
    ROLLING_SEED_OFFSET,
    ROLLING_VERSION,
    ROLLING_WINDOWS,
    rolling_columns,
)
from src.models.weekly_counts.run import (
    MODELING_CONFIG_PATH,
    load_frozen_frame,
    model_source_paths,
    write_json,
)

# B1-R4 and B1-R13 are rolling versions of B1 and share its frozen data contract.
CONTRACT_CONFIG = PROJECT_ROOT / "configs/weekly_counts/poisson_static_pymc_v1.yaml"
FRAME_COLUMNS = ["split", "week", "cluster_id", "complaint_count", "weekly_total"]


def baseline_predictions(
    frame: pd.DataFrame,
    columns: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """Calibration and validation rows; validation is kept but not evaluated."""
    later = frame["split"] != "fit"
    own_columns = [name for name in columns if name.startswith(f"{prefix}_")]
    return pd.concat(
        [frame.loc[later, FRAME_COLUMNS], columns.loc[later, own_columns]],
        axis=1,
    ).reset_index(drop=True)


def save_baseline_run(
    config: dict[str, Any],
    report: dict[str, Any],
    predictions: pd.DataFrame,
) -> Path:
    model_id = report["model_id"]
    report_dir = (
        PROJECT_ROOT / "reports/modeling/weekly_counts" / model_id / report["run_key"]
    )
    staging = report_dir.parent / f".{report['run_key']}.inprogress"
    staging.mkdir(parents=True)
    predictions.to_csv(staging / "predictions.csv", index=False)
    write_json(staging / "results.json", report)

    record = build_run_record(
        config=config,
        run_name=f"m9-{model_id}-{report['run_key']}",
        stage="M9",
        target="weekly_cluster_volume",
        split="calibration",
        view="complete_weeks",
        features=["cluster_id", "week", "weekly_total"],
        parameters={
            "model_id": model_id,
            "definition": report["definition"],
            "config_sha256": report["config_sha256"],
            "source_manifest_sha256": report["source_manifest_sha256"],
            "weekly_counts_sha256": report["weekly_counts_sha256"],
            "source_dvc_hash": report["source_dvc_hash"],
        },
        metrics={
            f"calibration_model_{name}": value
            for name, value in report["calibration"].items()
        },
        artifacts=[
            (report_dir / name).relative_to(PROJECT_ROOT).as_posix()
            for name in ("predictions.csv", "results.json")
        ],
    )
    record["tags"].update(
        {
            "model_id": model_id,
            "family": "deterministic_baseline",
            "model_family": "deterministic_baseline",
            "run_mode": "full",
            "candidate_role": "deterministic_baseline",
            "source_status": "native_versioned_source",
            "selection_split": "calibration",
            "evaluation_split": "validation",
            "validation_used_for_selection": "false",
        }
    )
    save_run_record(record, staging / "run.json")
    (staging / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    staging.replace(report_dir)
    return report_dir


def main() -> None:
    started = time.perf_counter()
    config = load_experiment_config(MODELING_CONFIG_PATH)
    settings = yaml.safe_load(CONTRACT_CONFIG.read_text(encoding="utf-8"))
    frame, _, input_sha256, source_dvc_hash = load_frozen_frame(config, settings)
    seed = config["experiment"]["seed"]
    columns = rolling_columns(
        frame,
        settings["clusters"],
        settings["sampling"]["prediction_draws"],
        seed + ROLLING_SEED_OFFSET,
    )

    source_paths = [
        *model_source_paths(rolling_reference, CONTRACT_CONFIG),
        Path(__file__).resolve(),
    ]
    manifest = source_manifest(source_paths)
    manifest_sha256 = source_manifest_sha256(manifest)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")

    for prefix, window in ROLLING_WINDOWS.items():
        predictions = baseline_predictions(frame, columns, prefix)
        calibration = predictions.loc[predictions["split"] == "calibration"]
        report = {
            "stage": "M9",
            "run_key": f"{timestamp}-full-{manifest_sha256[:8]}",
            "model_id": f"{prefix}_{ROLLING_VERSION}",
            "definition": (
                f"one-step-ahead shares of the previous {window} complete weeks, "
                "(count + 1) / (total + clusters), Poisson(weekly_total * share)"
            ),
            "window_weeks": window,
            "config_sha256": file_sha256(CONTRACT_CONFIG),
            "source_manifest_sha256": manifest_sha256,
            "source_manifest": manifest,
            "weekly_counts_sha256": input_sha256,
            "source_dvc_hash": source_dvc_hash,
            "seed": seed + ROLLING_SEED_OFFSET,
            "prediction_draws": settings["sampling"]["prediction_draws"],
            "evaluated_splits": ["calibration"],
            "validation_used_for_selection": False,
            "calibration_weeks": int(calibration["week"].nunique()),
            "calibration": predictive_metrics(calibration, prefix),
            "elapsed_seconds": time.perf_counter() - started,
        }
        report_dir = save_baseline_run(config, report, predictions)
        print(f"Model: {report['model_id']}")
        print(f"Report: {report_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
