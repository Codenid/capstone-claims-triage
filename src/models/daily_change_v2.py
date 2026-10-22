"""D-11 v2: the daily alert with a 3-day scan and empirical thresholds (§32.5).

Same excesses as D-11 (normal score of the mid-PIT under the frozen D-D
predictive). Three rules share the budget of 4 false alarms a month over 40
patterns: a single day, a 3-day scan S = (z_d + z_{d-1} + z_{d-2}) / sqrt(3),
and the daily CUSUM. The thresholds are empirical, fixed on fit: the quantile
1 - 1/304 of z and S, and the CUSUM h that gives the budget on the real fit
excesses. Injected bursts in calibration decide with the rule of §28.3 plus
a check that the real calibration alarms stay within 1.5 times the budget.

    python -m src.models.daily_change_v2               # design in fit + calibration
    python -m src.models.daily_change_v2 --validation  # single 2025-H1 report
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    git_commit,
    load_experiment_config,
    save_run_record,
    set_seed,
)
from src.models.daily_change import (
    DAYS_PER_MONTH,
    DESIGN_SPLITS,
    DailyExpectation,
    burst_factors,
    check_run,
    detection_summary,
    false_alarm_rate,
    first_alarm,
    injected_daily_scores,
    load_model,
    load_weekly,
    warm_rows,
    weekly_detection,
)
from src.models.daily_counts.contracts import DAY_COLUMN, SPLITS
from src.models.daily_counts.data import load_frozen_panel, panel_arrays
from src.models.daily_counts.references import discounted_daily_shares
from src.models.persistent_change import BISECTION_STEPS, cusum

STAGE = "M11D-V2"
PLAN = "models_plan.md §32.5"
RULES = ("daily", "scan", "cusum")


# ------------------------------------------------------------ pure rules
def scan_statistic(z: np.ndarray, window: int) -> np.ndarray:
    """Sum of the last `window` scores along axis 0 over sqrt(window); NaN before."""
    if window < 1:
        raise ValueError("The scan window must be at least one day.")
    frame = pd.DataFrame(np.asarray(z, dtype=float))
    return np.asarray(frame.rolling(window).sum(), dtype=float) / np.sqrt(window)


def empirical_cusum_threshold(z: np.ndarray, k: float, rate: float) -> float:
    """Smallest h whose CUSUM alarm rate on the real scores is at most `rate`."""
    low, high = 0.0, 50.0
    for _ in range(BISECTION_STEPS):
        middle = (low + high) / 2
        if cusum(z, k, middle)[1].mean() > rate:
            low = middle
        else:
            high = middle
    return high


def empirical_thresholds(
    z_fit: np.ndarray,
    rate: float,
    k: float,
    window: int,
) -> dict[str, float]:
    """Thresholds that give the budget on the fit decisions of each rule."""
    scan = scan_statistic(z_fit, window)
    return {
        "daily": float(np.quantile(z_fit, 1 - rate)),
        "scan": float(np.nanquantile(scan, 1 - rate)),
        "cusum": empirical_cusum_threshold(z_fit, k, rate),
    }


def add_alarms(
    rows: pd.DataFrame,
    thresholds: dict[str, float],
    k: float,
    window: int,
) -> pd.DataFrame:
    """The three rules on day-major rows; the CUSUM runs without restarting."""
    clusters = int(rows["cluster_id"].nunique())
    z = rows["z"].to_numpy().reshape(-1, clusters)
    statistic, alarms = cusum(z, k, thresholds["cusum"])
    scan = scan_statistic(z, window)
    return rows.assign(
        cusum=statistic.ravel(),
        cusum_alarm=alarms.ravel(),
        daily_alarm=rows["z"] > thresholds["daily"],
        scan=scan.ravel(),
        scan_alarm=np.nan_to_num(scan.ravel(), nan=-np.inf) > thresholds["scan"],
    )


def alarm_summary(rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for split, group in rows.groupby("split", sort=False):
        days = int(group[DAY_COLUMN].nunique())
        months = days / DAYS_PER_MONTH
        summary[str(split)] = {"days": days}
        for rule in RULES:
            alarms = int(group[f"{rule}_alarm"].sum())
            summary[str(split)][rule] = alarms
            summary[str(split)][f"{rule}_per_month"] = alarms / months
    return summary


def injected_window_scores(
    counts: np.ndarray,
    day_of_week: np.ndarray,
    alpha: np.ndarray,
    model: DailyExpectation,
    cluster: int,
    start: int,
    factors: np.ndarray,
    window: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Scores over the horizon for each rule, with the scan primed on prior days.

    The excess is computed from `window - 1` days before the burst (real
    counts) so that the scan is defined from the first burst day; the daily
    and CUSUM rules see the horizon only, as D-11 did.
    """
    lead = window - 1
    if start < lead:
        raise ValueError("The burst must start after the scan window.")
    padded = np.concatenate([np.ones(lead), factors])
    z_extended, increased = injected_daily_scores(
        counts, day_of_week, alpha, model, cluster, start - lead, padded
    )
    z = z_extended[lead:]
    scan = scan_statistic(z_extended[:, None], window)[lead:, 0]
    return {"daily": z, "scan": scan, "cusum": z}, increased


def detection_runs(
    arrays: dict[str, np.ndarray],
    alpha: np.ndarray,
    model: DailyExpectation,
    thresholds: dict[str, float],
    settings: dict[str, Any],
    weekly: dict[str, Any] | None,
    seed: int,
) -> pd.DataFrame:
    """Each rule's delay for every scenario, pattern and sampled start day."""
    counts = arrays["counts"]
    horizon = settings["horizon_days"]
    window = settings["scan_days"]
    rng = np.random.default_rng(seed)
    first = max(settings["warmup_days"], window - 1)
    last = len(counts) - horizon
    starts = rng.choice(np.arange(first, last), size=settings["starts_per_pattern"])
    rows = []
    for name, scenario in settings["scenarios"].items():
        factors = burst_factors(scenario, horizon)
        for cluster in range(counts.shape[1]):
            for start in starts:
                scores, increased = injected_window_scores(
                    counts,
                    arrays["day_of_week"],
                    alpha,
                    model,
                    cluster,
                    int(start),
                    factors,
                    window,
                )
                alarms = cusum(
                    scores["cusum"][:, None], settings["cusum_k"], thresholds["cusum"]
                )[1]
                row = {
                    "scenario": name,
                    "cluster_id": cluster,
                    "start_day": str(pd.Timestamp(arrays["days"][start]).date()),
                    "daily_delay": first_alarm(scores["daily"] > thresholds["daily"]),
                    "scan_delay": first_alarm(scores["scan"] > thresholds["scan"]),
                    "cusum_delay": first_alarm(alarms.ravel()),
                }
                if weekly is not None and scenario["kind"] == "burst":
                    row["weekly_cusum_delay"] = weekly_detection(
                        increased, arrays["days"], int(start), weekly
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def decide(
    summary: dict[str, Any],
    calibration_alarms: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """§32.5: detection and delay of §28.3, plus the calibration budget check."""
    deciding = settings["deciding_scenarios"]
    limit = settings["false_alarms_per_month"] * settings["calibration_budget_factor"]
    checks: dict[str, bool] = {}
    for rule in RULES:
        for scenario in deciding:
            result = summary[scenario][rule]
            checks[f"{scenario}:{rule}"] = bool(
                result["detected"] >= settings["minimum_detection"]
                and result["median_delay"] is not None
                and result["median_delay"] <= settings["maximum_median_delay"]
            )
        checks[f"budget:{rule}"] = bool(
            calibration_alarms[f"{rule}_per_month"] <= limit
        )
    passing = [
        rule
        for rule in RULES
        if all(checks[f"{s}:{rule}"] for s in deciding) and checks[f"budget:{rule}"]
    ]
    chosen = None
    if passing:
        score = {r: sum(summary[s][r]["detected"] for s in deciding) for r in passing}
        best = max(score.values())
        chosen = next(r for r in RULES if r in passing and score[r] == best)
    return {
        "adopted": chosen is not None,
        "rule": chosen,
        "checks": checks,
        "calibration_budget_limit": limit,
    }


# -------------------------------------------------------------------- run
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=STAGE)
    parser.add_argument("--validation", action="store_true")
    return parser.parse_args()


def merged_settings(config: dict[str, Any]) -> dict[str, Any]:
    """D-11 v2 keys over the frozen D-11 ones (run, budget, horizon, rule)."""
    return {**config["daily_change"], **config["daily_change_v2"]}


def scored_rows(
    panel: pd.DataFrame,
    splits: tuple[str, ...],
    clusters: int,
    alpha: np.ndarray,
    model: DailyExpectation,
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray]:
    frame = panel.loc[panel["split"].isin(splits)].reset_index(drop=True)
    frame = frame.assign(recent_share=discounted_daily_shares(frame, clusters, 0.5))
    arrays = panel_arrays(frame, clusters)
    split_of_day = frame.drop_duplicates(DAY_COLUMN)["split"].to_numpy()
    rows = warm_rows(arrays, split_of_day, alpha, model, settings["warmup_days"])
    return rows, arrays, split_of_day


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    config = load_experiment_config()
    settings = merged_settings(config)
    seed = config["experiment"]["seed"]
    set_seed(seed)
    model_config = yaml.safe_load(
        (
            PROJECT_ROOT / "configs/daily_counts" / f"{settings['m9d_model']}.yaml"
        ).read_text(encoding="utf-8")
    )
    panel, input_sha256, source_dvc_hash = load_frozen_panel(config, model_config)
    clusters = model_config["clusters"]
    alpha, model = load_model(settings, seed)
    report_dir = PROJECT_ROOT / settings["report_dir"]
    window = settings["scan_days"]

    if args.validation:
        design = json.loads((report_dir / "results.json").read_text("utf-8"))
        if not design["decision"]["adopted"]:
            raise ValueError("D-11 v2 was not adopted: no 2025-H1 report.")
        path = report_dir / "validation_results.json"
        if path.exists():
            raise FileExistsError(f"D-11 v2 reports 2025-H1 only once: {path}")
        rows, _, _ = scored_rows(panel, SPLITS, clusters, alpha, model, settings)
        rows = add_alarms(rows, design["thresholds"], settings["cusum_k"], window)
        validation = rows.loc[rows["split"] == "validation"]
        report = {
            "stage": STAGE,
            "plan": PLAN,
            "split": "validation",
            "already_consulted": True,
            "git_commit": git_commit(),
            "rule": design["decision"],
            "thresholds": design["thresholds"],
            "real_alarms": alarm_summary(validation),
        }
        path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8"
        )
        validation.to_csv(report_dir / "validation_alerts.csv", index=False)
        print(json.dumps(report["real_alarms"], indent=2))
        return

    if report_dir.exists():
        raise FileExistsError(f"D-11 v2 never overwrites its design: {report_dir}")
    rows, arrays, split_of_day = scored_rows(
        panel, DESIGN_SPLITS, clusters, alpha, model, settings
    )
    check = check_run(rows, settings)
    rate = false_alarm_rate(settings["false_alarms_per_month"], clusters)
    z_fit = rows.loc[rows["split"] == "fit", "z"].to_numpy().reshape(-1, clusters)
    thresholds = empirical_thresholds(z_fit, rate, settings["cusum_k"], window)
    rows = add_alarms(rows, thresholds, settings["cusum_k"], window)
    real_alarms = alarm_summary(rows)
    calibration_days = np.flatnonzero(split_of_day == "calibration")
    weekly = load_weekly(config, seed) if settings.get("compare_weekly", True) else None
    runs = detection_runs(
        {name: values[calibration_days] for name, values in arrays.items()},
        alpha,
        model,
        thresholds,
        settings,
        weekly,
        seed,
    )
    summary = detection_summary(runs)
    decision = decide(summary, real_alarms["calibration"], settings)
    report_dir.mkdir(parents=True)
    rows.to_csv(report_dir / "alerts.csv", index=False)
    runs.to_csv(report_dir / "detection.csv", index=False)
    report = {
        "stage": STAGE,
        "plan": PLAN,
        "git_commit": git_commit(),
        "seed": seed,
        "m9d_model": settings["m9d_model"],
        "m9d_run_key": settings["m9d_run_key"],
        "daily_counts_sha256": input_sha256,
        "source_dvc_hash": source_dvc_hash,
        "settings": {k: v for k, v in settings.items() if k != "scenarios"},
        "scenarios": settings["scenarios"],
        "false_alarm_rate_per_decision": rate,
        "thresholds": thresholds,
        "run_check": check,
        "real_alarms": real_alarms,
        "detection": summary,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (report_dir / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metrics = {
        **{f"threshold_{rule}": thresholds[rule] for rule in RULES},
        **{
            f"{split}_{rule}_per_month": values[f"{rule}_per_month"]
            for split, values in real_alarms.items()
            for rule in RULES
        },
        **{
            f"detect_{scenario}_{rule}": result[rule]["detected"]
            for scenario, result in summary.items()
            for rule in result
            if rule != "runs"
        },
    }
    record = build_run_record(
        config=config,
        run_name=settings["run_name"],
        stage=STAGE,
        target="A1",
        split="calibration",
        view="40 patterns, daily",
        features=[
            "mid-PIT excess of D-D",
            f"{window}-day scan",
            "empirical thresholds",
        ],
        parameters={
            "plan": PLAN,
            "m9d_run_key": settings["m9d_run_key"],
            "false_alarms_per_month": settings["false_alarms_per_month"],
            "scan_days": window,
            "adopted": decision["adopted"],
            "rule": decision["rule"],
        },
        metrics=metrics,
        artifacts=[f"{settings['report_dir']}/results.json"],
    )
    run_root = PROJECT_ROOT / config["paths"]["offline_runs"] / "m11d_v2"
    save_run_record(record, run_root / "run.json")
    print(json.dumps({"thresholds": thresholds}, indent=2))
    print(json.dumps(real_alarms, indent=2))
    print(json.dumps(summary, indent=2))
    print(json.dumps(decision, indent=2))
    print(f"Report: {report_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
