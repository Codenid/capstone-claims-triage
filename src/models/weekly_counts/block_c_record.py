"""One MLflow record for block C: every candidate against the frozen M9.

models_plan.md §14.5 keeps MLflow to one run per decision, so the comparisons
that `challenge.py` saved for block C (§25.5) become the metrics of a single run.
"""

from __future__ import annotations

import json
from typing import Any

from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    load_experiment_config,
    save_run_record,
)


def candidate_metrics(report: dict[str, Any]) -> dict[str, float]:
    name = report["candidate"]
    comparison = report["comparison"]
    difference = comparison["difference"]["wis"]
    return {
        f"{name}_wis": comparison["metrics"]["model"]["wis"],
        f"{name}_m9_wis": comparison["metrics"]["m9"]["wis"],
        f"{name}_wis_gain": comparison["wis_gain"],
        f"{name}_wis_difference_p025": difference["p025"],
        f"{name}_wis_difference_p975": difference["p975"],
        f"{name}_wins": float(report["wins"]),
    }


def main() -> None:
    config = load_experiment_config()
    settings = config["m9_challenge"]
    paths = sorted((PROJECT_ROOT / settings["report_dir"]).glob("*.json"))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    metrics: dict[str, float] = {}
    for report in reports:
        metrics.update(candidate_metrics(report))
    winners = [report["candidate"] for report in reports if report["wins"]]
    record = build_run_record(
        config=config,
        run_name="m9-block-c-challenge",
        stage="M9",
        target="weekly_cluster_counts",
        split="calibration",
        view="complete_weeks",
        features=["cluster_id", "week", "weekly_total"],
        parameters={
            "plan": "models_plan.md §25.5",
            "reference_run": settings["reference_run"],
            "candidates": json.dumps([report["candidate"] for report in reports]),
            "winners": json.dumps(winners),
        },
        metrics=metrics,
        artifacts=[path.relative_to(PROJECT_ROOT).as_posix() for path in paths],
    )
    record["tags"]["run_mode"] = "full"
    path = PROJECT_ROOT / config["paths"]["offline_runs"] / "m9" / "block_c" / "run.json"
    save_run_record(record, path)
    print(f"Candidates: {[report['candidate'] for report in reports]}")
    print(f"Winners: {winners}")
    print(f"Record: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
