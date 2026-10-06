"""Compare a block C candidate with the frozen M9, NB-R4-H v3 (models_plan.md §25.2).

The candidate wins if its calibration WIS is at least 5% lower, the 95%
interval of the weekly paired bootstrap stays below 0, WAPE and MAE do not
clearly degrade, and its 80% and 95% coverage stay in the §12.1 ranges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.experiment import PROJECT_ROOT, git_commit, load_experiment_config
from src.models.weekly_counts.comparison import compare_with_baselines
from src.models.weekly_counts.contracts import INTERVALS

KEYS = ["split", "week", "cluster_id"]
REFERENCE = "m9"
WIDTHS = {50: 0.50, 80: 0.20, 95: 0.05}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "candidate_run",
        type=Path,
        help="Project-relative report directory of the candidate's full run.",
    )
    return parser.parse_args()


def attach_reference(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Candidate predictions plus the frozen M9 quantiles as `m9_*` columns."""
    columns = [column for column in reference if column.startswith("model_")]
    renamed = reference[KEYS + ["complaint_count", *columns]].rename(
        columns={
            "complaint_count": "m9_complaint_count",
            **{column: f"{REFERENCE}_{column[6:]}" for column in columns},
        }
    )
    merged = candidate.merge(renamed, on=KEYS, how="left", validate="one_to_one")
    calibration = merged["split"] == "calibration"
    if merged.loc[calibration, "m9_complaint_count"].isna().any():
        raise ValueError("M9 has no prediction for some candidate calibration rows.")
    if not merged.loc[calibration, "m9_complaint_count"].eq(
        merged.loc[calibration, "complaint_count"]
    ).all():
        raise ValueError("The candidate and M9 disagree on the observed counts.")
    return merged.drop(columns="m9_complaint_count")


def decide(
    comparison: dict[str, Any],
    coverage: dict[str, float],
    acceptance: dict[str, Any],
    minimum_gain: float,
) -> dict[str, Any]:
    """§25.2 and §12.1/§12.4: every check must pass for the candidate to win.

    A candidate without a 95% interval (TimesFM 3.0, §25.5) skips that coverage.
    """
    difference = comparison["difference"]
    checks = {
        "wis_gain": comparison["wis_gain"] >= minimum_gain,
        "wis_interval_below_zero": difference["wis"]["p975"] < 0,
        "wape_not_worse": difference["wape"]["p025"] <= 0,
        "mae_not_worse": difference["mae"]["p025"] <= 0,
    }
    for width in (80, 95):
        value = coverage.get(f"coverage_{width}")
        if value is not None:
            checks[f"coverage_{width}"] = (
                acceptance[f"coverage_{width}_minimum"]
                <= value
                <= acceptance[f"coverage_{width}_maximum"]
            )
    return {"checks": checks, "wins": all(checks.values())}


def candidate_intervals(results: dict[str, Any]) -> tuple[tuple[float, str, str], ...]:
    """The WIS intervals the candidate provides; M9 is scored on the same ones."""
    widths = results.get("interval_widths", list(WIDTHS))
    alphas = {WIDTHS[width] for width in widths}
    return tuple(interval for interval in INTERVALS if interval[0] in alphas)


def main() -> None:
    args = parse_args()
    config = load_experiment_config()
    settings = config["m9_challenge"]
    run_dir = PROJECT_ROOT / args.candidate_run
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    if results["run_mode"] != "full":
        raise ValueError("Only full runs are compared with M9.")
    report_dir = PROJECT_ROOT / settings["report_dir"]
    report_path = report_dir / f"{results['model_id']}.json"
    if report_path.exists():
        raise FileExistsError(f"The M9 challenge never overwrites: {report_path}")

    predictions = attach_reference(
        pd.read_csv(run_dir / "predictions.csv"),
        pd.read_csv(PROJECT_ROOT / settings["reference_run"] / "predictions.csv"),
    )
    intervals = candidate_intervals(results)
    comparison = compare_with_baselines(
        predictions, (REFERENCE,), config["experiment"]["seed"], intervals=intervals
    )
    coverage = results["backtest"]["calibration"]["model"]
    decision = decide(
        comparison,
        coverage,
        settings["acceptance"],
        settings["minimum_wis_gain"],
    )
    report = {
        "plan": "models_plan.md §25.2",
        "git_commit": git_commit(),
        "candidate": results["model_id"],
        "candidate_run": str(args.candidate_run).replace("\\", "/"),
        "reference_run": settings["reference_run"],
        "split": "calibration",
        "validation_used": False,
        "formula": results["formula"],
        "comparison": comparison,
        "interval_widths": [round((1 - alpha) * 100) for alpha, *_ in intervals],
        "coverage": {key: coverage.get(key) for key in ("coverage_80", "coverage_95")},
        **decision,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metrics = comparison["metrics"]
    print(
        f"WIS {results['model_id']}={metrics['model']['wis']:.3f} "
        f"M9={metrics[REFERENCE]['wis']:.3f} gain={comparison['wis_gain']:+.1%}"
    )
    print(f"Checks: {decision['checks']}")
    print(f"Wins: {decision['wins']}")
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
