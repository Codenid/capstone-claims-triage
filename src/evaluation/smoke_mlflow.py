"""Log the approved input contract as the first MLflow run."""

import json
import os
from pathlib import Path
import subprocess

import mlflow
import yaml

CONTRACT_PATH = Path("reports/modeling/input_contract.json")
LOCK_PATH = Path("dvc.lock")


def prepared_data_hash(lock_path: Path) -> str:
    """Read the prepared Parquet hash recorded by DVC."""
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    output = lock["stages"]["finalize_prepared"]["outs"][0]
    return str(output["md5"])


def main() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("Set MLFLOW_TRACKING_URI before running this smoke test.")

    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()
    experiment_name = os.environ.get(
        "MLFLOW_EXPERIMENT_NAME",
        "claims-triage-modeling",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="m0-input-contract") as run:
        mlflow.set_tags(
            {
                "git_commit": git_commit,
                "dvc_data_hash": prepared_data_hash(LOCK_PATH),
                "stage": "M0",
                "target": "input_contract",
            }
        )
        mlflow.log_params(
            {
                "input": contract["input"],
                "rows": contract["rows"],
                "columns": contract["columns"],
                "text_normalizer": contract["versions"]["text_normalizer"],
                "taxonomy_version": contract["versions"]["taxonomy_version"],
                "targets_version": contract["versions"]["targets_periods_version"],
                "holdout_final_evaluation_allowed": contract["versions"][
                    "holdout_final_evaluation_allowed"
                ],
            }
        )
        mlflow.log_metric("input_size_gib", contract["size_bytes"] / 1024**3)
        mlflow.log_artifact(str(CONTRACT_PATH), artifact_path="contracts")

    print(f"Experiment: {experiment_name}")
    print(f"Run ID: {run.info.run_id}")
    print(f"Tracking URI: {tracking_uri}")


if __name__ == "__main__":
    main()
