"""Export every run of the MLflow experiment to a small JSON for the reports.

The Quarto page `reports/experimentos/` reads `reports/modeling/mlflow_runs.json`
instead of querying DagsHub, so anyone can render it without credentials.
Re-run this script when runs are added; it needs the usual variables
(`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`),
read from the environment or from `.env`.

    uv run --no-sync python scripts/export_mlflow_runs.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "reports" / "modeling" / "mlflow_runs.json"
EXPERIMENT = "claims-triage-modeling"
TAGS = (
    "stage",
    "target",
    "split",
    "view",
    "version_datos",
    "datos_md5",
    "git_commit",
    "execution_host",
    "execution_mode",
    "slurm_job_id",
    "tipo",
)
PARAMS = ("model_id", "run_key", "run_mode", "candidate_status", "discount", "seed")


def load_env() -> None:
    """Read `.env` when the tracking variables are not already set."""
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - dotenv is a project dependency
        return
    load_dotenv(PROJECT_ROOT / ".env")


def run_rows(experiment_id: str) -> list[dict]:
    import mlflow

    rows = []
    for run in mlflow.search_runs(
        [experiment_id], max_results=5000, output_format="list"
    ):
        info, data = run.info, run.data
        rows.append(
            {
                "run_id": info.run_id,
                "run_name": info.run_name,
                "status": info.status,
                "start_time_utc": datetime.fromtimestamp(
                    info.start_time / 1000, tz=timezone.utc
                ).isoformat(timespec="seconds"),
                "duration_seconds": (
                    (info.end_time - info.start_time) / 1000 if info.end_time else None
                ),
                "tags": {tag: data.tags.get(tag) for tag in TAGS},
                "params": {param: data.params.get(param) for param in PARAMS},
                "metrics": {key: value for key, value in sorted(data.metrics.items())},
            }
        )
    rows.sort(key=lambda row: row["start_time_utc"])
    return rows


def main() -> int:
    load_env()
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("Set MLFLOW_TRACKING_URI (or put it in .env) before exporting.")
        return 1
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        print(f"Experiment {EXPERIMENT!r} not found at the tracking server.")
        return 1
    rows = run_rows(experiment.experiment_id)
    payload = {
        "experiment": EXPERIMENT,
        "tracking_host": tracking_uri.split("//")[-1].split("/")[0],
        "exported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": rows,
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"{len(rows)} runs -> {OUTPUT.relative_to(PROJECT_ROOT)} ({size_kb:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
