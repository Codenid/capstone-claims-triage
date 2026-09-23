"""Create a small offline run to verify the M1 experiment setup."""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import random

import mlflow
import numpy as np

from src.evaluation.experiment import (
    build_run_record,
    load_experiment_config,
    save_run_record,
    set_seed,
)

DEFAULT_OUTPUT = Path("reports/modeling/runs/m1-experiment-setup/run.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = load_experiment_config()
    seed = config["experiment"]["seed"]
    set_seed(seed)

    record = build_run_record(
        config=config,
        run_name="m1-experiment-setup",
        stage="M1",
        target="experiment_setup",
        split="not_applicable",
        view="not_applicable",
        features=[],
        parameters={
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "mlflow_version": mlflow.__version__,
        },
        metrics={
            "setup_ok": 1.0,
            "python_seed_probe": random.random(),
            "numpy_seed_probe": float(np.random.random()),
        },
    )
    save_run_record(record, args.output)

    print(f"Seed: {seed}")
    print(f"Execution: {record['tags']['execution_host']}/{record['tags']['execution_mode']}")
    print(f"Offline record: {args.output}")


if __name__ == "__main__":
    main()
