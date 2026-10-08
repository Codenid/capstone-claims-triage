"""Choose the daily count winner among the full runs (models_plan.md §28.2).

Each full run already accepted or rejected itself: convergence, coverage and
the paired bootstrap against the best Poisson baseline under the §25.2 rule.
The pre-registered tie-break is the lowest calibration WIS among the accepted
candidates. The winner is also bootstrapped, pairwise, against every other
accepted candidate on the same calibration days; that comparison is reported,
it does not decide.

    python -m src.models.daily_counts.challenge <run_dir> [<run_dir> ...]
    python -m src.models.daily_counts.challenge --validation <run_dir> ...

With --validation the chosen winner is scored once on 2025-H1 against the
baselines and the other accepted candidates; nothing is decided there.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    git_commit,
    load_experiment_config,
    save_run_record,
)
from src.models.daily_counts.baselines import ROLLING_NAME
from src.models.daily_counts.data import as_weekly_view
from src.models.weekly_counts.comparison import compare_with_baselines
from src.models.weekly_counts.metrics import predictive_metrics

STAGE = "M9D"
PLAN = "models_plan.md §28.2"
RULE = "lowest calibration WIS among the accepted full runs"
KEYS = ["split", "day", "cluster_id"]
BASELINE_NAMES = ("baseline", ROLLING_NAME)
CANDIDATE_METRICS = ("wis", "wape", "mae", "coverage_80", "coverage_95")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Project-relative report directories of the full runs.",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Report the winner once on 2025-H1 (§28.2), with no decision.",
    )
    return parser.parse_args()


def load_run(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    run_dir = PROJECT_ROOT / path
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    if results["run_mode"] != "full":
        raise ValueError(f"Only full runs compete: {path}")
    predictions = pd.read_csv(run_dir / "predictions.csv")
    return results, predictions


def candidate_row(results: dict[str, Any], path: Path) -> dict[str, Any]:
    """What the decision needs from one run, as its results.json reports it."""
    calibration = results["backtest"]["calibration"]["model"]
    comparison = results["comparison"]
    return {
        "model_id": results["model_id"],
        "run_key": results["run_key"],
        "run_dir": str(path).replace("\\", "/"),
        "status": results["candidate_status"],
        "rejection_reason": results.get("rejection_reason", ""),
        **{name: float(calibration[name]) for name in CANDIDATE_METRICS},
        "best_baseline": comparison["best_baseline"],
        "best_baseline_wis": float(
            comparison["metrics"][comparison["best_baseline"]]["wis"]
        ),
        "wis_gain": float(comparison["wis_gain"]),
        "wis_difference_p025": float(comparison["difference"]["wis"]["p025"]),
        "wis_difference_p975": float(comparison["difference"]["wis"]["p975"]),
        "rhat_max": float(results["diagnostics"].get("rhat_max", float("nan"))),
        "divergences": int(results["diagnostics"].get("divergences", 0)),
    }


def choose(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row["status"] == "accepted"]
    if not accepted:
        raise ValueError("No accepted full run: nothing to choose (§28.2).")
    return min(accepted, key=lambda row: row["wis"])


def attach_candidate(
    winner: pd.DataFrame, other: pd.DataFrame, name: str
) -> pd.DataFrame:
    """Winner predictions plus the other candidate's quantiles as `<name>_*`."""
    columns = [column for column in other if column.startswith("model_")]
    names: dict[str, str] = {column: f"{name}_{column[6:]}" for column in columns}
    names["complaint_count"] = f"{name}_complaint_count"
    renamed = other[KEYS + list(names)].rename(columns=names)
    merged = winner.merge(renamed, on=KEYS, how="left", validate="one_to_one")
    observed = merged[f"{name}_complaint_count"]
    missing = bool(observed.isna().any())
    if missing or not bool(observed.eq(merged["complaint_count"]).all()):
        raise ValueError(f"{name} does not cover the winner's days and counts.")
    return merged.drop(columns=f"{name}_complaint_count")


def pairwise(
    winner: pd.DataFrame,
    others: dict[str, pd.DataFrame],
    seed: int,
    split: str,
) -> dict[str, Any]:
    """Paired bootstrap winner - other over the same days, per other candidate."""
    rows = winner.loc[winner["split"] == split]
    result = {}
    for name, frame in others.items():
        merged = attach_candidate(rows, frame.loc[frame["split"] == split], name)
        comparison = compare_with_baselines(
            as_weekly_view(merged), (name,), seed, split=split
        )
        result[name] = {
            "wis_gain": comparison["wis_gain"],
            "difference": comparison["difference"],
        }
    return result


def validation_report(
    winner: pd.DataFrame, others: dict[str, pd.DataFrame], seed: int
) -> dict[str, Any]:
    """The single 2025-H1 report of the winner; nothing is decided here."""
    rows = winner.loc[winner["split"] == "validation"]
    if rows.empty:
        raise ValueError("The winner has no validation predictions.")
    return {
        "split": "validation",
        "already_consulted": True,
        "note": "2025-H1 was opened in §22 and §25; this is not clean evidence.",
        "comparison": compare_with_baselines(
            as_weekly_view(rows), BASELINE_NAMES, seed, split="validation"
        ),
        "metrics": {
            name: predictive_metrics(rows, name)
            for name in ("model", *BASELINE_NAMES)
        },
        "pairwise": pairwise(winner, others, seed, "validation"),
    }


def main() -> None:
    args = parse_args()
    config = load_experiment_config()
    settings = config["daily_challenge"]
    seed = config["experiment"]["seed"]
    runs = {path: load_run(path) for path in args.runs}
    rows = [candidate_row(results, path) for path, (results, _) in runs.items()]
    winner = choose(rows)
    predictions = {row["model_id"]: runs[Path(row["run_dir"])][1] for row in rows}
    others = {
        row["model_id"]: predictions[row["model_id"]]
        for row in rows
        if row["status"] == "accepted" and row["model_id"] != winner["model_id"]
    }
    split = "validation" if args.validation else "calibration"
    report: dict[str, Any] = {
        "plan": PLAN,
        "rule": RULE,
        "git_commit": git_commit(),
        "split": split,
        "validation_used_for_selection": False,
        "seed": seed,
        "candidates": rows,
        "winner": {key: winner[key] for key in ("model_id", "run_key", "run_dir")},
    }
    if args.validation:
        report["validation"] = validation_report(
            predictions[winner["model_id"]], others, seed
        )
    else:
        report["pairwise"] = pairwise(
            predictions[winner["model_id"]], others, seed, "calibration"
        )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    report_dir = PROJECT_ROOT / settings["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{timestamp}-{split}.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metrics = {f"{row['model_id']}_calibration_wis": row["wis"] for row in rows}
    metrics["winner_calibration_wis"] = winner["wis"]
    metrics["winner_calibration_wis_gain"] = winner["wis_gain"]
    for name, values in report.get("pairwise", {}).items():
        metrics[f"winner_minus_{name}_wis_p975"] = values["difference"]["wis"]["p975"]
    if args.validation:
        validation = report["validation"]
        metrics["winner_validation_wis"] = validation["metrics"]["model"]["wis"]
        metrics["winner_validation_wis_gain"] = validation["comparison"]["wis_gain"]
    record = build_run_record(
        config=config,
        run_name="m9d-daily-challenge" + ("-validation" if args.validation else ""),
        stage=STAGE,
        target="A1",
        split=split,
        view="40 patterns, daily",
        features=["daily_total", "recent_share", "day_of_week"],
        parameters={
            "plan": PLAN,
            "rule": RULE,
            "candidates": ",".join(row["model_id"] for row in rows),
            "winner": winner["model_id"],
            "winner_run_key": winner["run_key"],
        },
        metrics=metrics,
        artifacts=[str(report_path.relative_to(PROJECT_ROOT)).replace("\\", "/")],
    )
    save_run_record(record, report_dir / f"{timestamp}-{split}.run.json")

    for row in sorted(rows, key=lambda row: row["wis"]):
        print(
            f"{row['model_id']:<32} WIS {row['wis']:.3f} "
            f"gain {row['wis_gain']:+.1%} {row['status']}"
        )
    print(f"Winner ({RULE}): {winner['model_id']} {winner['run_key']}")
    for name, values in report.get("pairwise", {}).items():
        difference = values["difference"]["wis"]
        print(
            f"winner - {name}: WIS {difference['estimate']:+.3f} "
            f"[{difference['p025']:+.3f}, {difference['p975']:+.3f}]"
        )
    if args.validation:
        validation = report["validation"]["metrics"]
        print(
            f"2025-H1 WIS winner={validation['model']['wis']:.3f} "
            f"{ROLLING_NAME}={validation[ROLLING_NAME]['wis']:.3f}"
        )
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
