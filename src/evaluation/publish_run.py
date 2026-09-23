"""Publish an offline experiment record to MLflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import mlflow

REQUIRED_KEYS = {
    "run_name",
    "experiment_name",
    "tags",
    "parameters",
    "metrics",
    "artifacts",
}


def load_run_record(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_KEYS - record.keys()
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Run record is missing: {names}")

    missing_artifacts = [item for item in record["artifacts"] if not Path(item).exists()]
    if missing_artifacts:
        names = ", ".join(missing_artifacts)
        raise FileNotFoundError(f"Run artifacts do not exist: {names}")
    return record


def publish(record_path: Path) -> str:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("Set MLFLOW_TRACKING_URI before publishing a run.")

    record = load_run_record(record_path)
    experiment_name = os.environ.get(
        "MLFLOW_EXPERIMENT_NAME",
        record["experiment_name"],
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=record["run_name"]) as run:
        mlflow.set_tags(record["tags"])
        mlflow.log_params(record["parameters"])
        mlflow.log_metrics(record["metrics"])
        mlflow.log_artifact(str(record_path), artifact_path="run")
        for item in record["artifacts"]:
            path = Path(item)
            if path.is_dir():
                mlflow.log_artifacts(str(path), artifact_path="outputs")
            else:
                mlflow.log_artifact(str(path), artifact_path="outputs")

    return run.info.run_id


def find_records(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        records = sorted(path.rglob("run.json"))
        if records:
            return records
    raise FileNotFoundError(f"No run records found at: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    args = parser.parse_args()

    for record_path in find_records(args.record):
        run_id = publish(record_path)
        print(f"Run ID: {run_id}")
        print(f"Record: {record_path}")


if __name__ == "__main__":
    main()
