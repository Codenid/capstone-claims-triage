"""Shared metadata and seeds for modeling experiments."""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/modeling.yaml"
VALID_HOSTS = {"local", "khipu"}


def load_experiment_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "experiment" not in config or "paths" not in config:
        raise ValueError(f"Invalid modeling config: {path}")
    return config


def set_seed(seed: int) -> None:
    """Reset Python and NumPy random generators."""
    random.seed(seed)
    np.random.seed(seed)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


def execution_context() -> tuple[str, str]:
    default_host = "khipu" if os.environ.get("SLURM_JOB_ID") else "local"
    host = os.environ.get("CLAIMS_EXECUTION_HOST", default_host).lower()
    if host not in VALID_HOSTS:
        raise ValueError("CLAIMS_EXECUTION_HOST must be 'local' or 'khipu'.")

    mode = "slurm" if os.environ.get("SLURM_JOB_ID") else "interactive"
    return host, mode


def build_run_record(
    *,
    config: dict[str, Any],
    run_name: str,
    stage: str,
    target: str,
    split: str,
    view: str,
    features: list[str],
    parameters: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    artifacts: list[str] | None = None,
) -> dict[str, Any]:
    """Build the information that every MLflow run must contain."""
    experiment = config["experiment"]
    paths = config["paths"]
    contract_path = Path(paths["input_contract"])
    if not contract_path.is_absolute():
        contract_path = PROJECT_ROOT / contract_path
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    host, mode = execution_context()

    tags = {
        "run_contract_version": experiment["run_contract_version"],
        "git_commit": git_commit(),
        "dvc_data_hash": contract["dvc_md5"],
        "stage": stage,
        "target": target,
        "split": split,
        "view": view,
        "execution_host": host,
        "execution_mode": mode,
    }
    if os.environ.get("SLURM_JOB_ID"):
        tags["slurm_job_id"] = os.environ["SLURM_JOB_ID"]

    base_parameters = {
        "seed": experiment["seed"],
        "input_data": paths["input_data"],
        "features": json.dumps(features, ensure_ascii=False),
    }
    parameters = parameters or {}
    repeated = base_parameters.keys() & parameters.keys()
    if repeated:
        names = ", ".join(sorted(repeated))
        raise ValueError(f"Parameters already defined by the run contract: {names}")
    base_parameters.update(parameters)

    return {
        "run_name": run_name,
        "experiment_name": experiment["name"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "tags": tags,
        "parameters": base_parameters,
        "metrics": metrics or {},
        "artifacts": artifacts or [],
    }


def save_run_record(record: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
