"""Add the course provenance tags to the MLflow runs published before them.

The course repository (utec-dsia/pi1-262-g1) expects `version_datos`,
`datos_md5` and `git_commit` on every run, plus `tipo` to filter by equality.
Runs published before 2026-10-07 carry `dvc_data_hash` and `git_commit` only;
this script copies `dvc_data_hash` into `datos_md5`, writes the data version
from the config and sets `tipo=experimento` where the tag is missing. It never
changes a tag that already exists.

Run it from the login node with the MLflow variables of `.env` loaded.
"""

from __future__ import annotations

import argparse
import os

import mlflow

from src.evaluation.experiment import load_experiment_config

NEW_TAGS = ("version_datos", "datos_md5", "tipo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the tags; without it the script only reports what it would do.",
    )
    return parser.parse_args()


def missing_tags(tags: dict[str, str], data_version: str) -> dict[str, str]:
    """The course tags a run lacks, derived from the tags it already has."""
    wanted = {
        "version_datos": data_version,
        "datos_md5": tags.get("dvc_data_hash", ""),
        "tipo": "experimento",
    }
    return {
        name: value
        for name, value in wanted.items()
        if name not in tags and value
    }


def main() -> None:
    args = parse_args()
    config = load_experiment_config()
    data_version = config["experiment"]["data_version"]
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    name = os.environ.get("MLFLOW_EXPERIMENT_NAME", config["experiment"]["name"])
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        raise SystemExit(f"No MLflow experiment named {name}.")
    client = mlflow.MlflowClient()
    runs = client.search_runs([experiment.experiment_id], max_results=1000)
    changed = 0
    for run in runs:
        tags = missing_tags(run.data.tags, data_version)
        if not tags:
            continue
        changed += 1
        print(f"{run.info.run_name}: {sorted(tags)}")
        if args.apply:
            for key, value in tags.items():
                client.set_tag(run.info.run_id, key, value)
    verb = "updated" if args.apply else "would update"
    print(f"{len(runs)} runs in {name}; {verb} {changed}.")


if __name__ == "__main__":
    main()
