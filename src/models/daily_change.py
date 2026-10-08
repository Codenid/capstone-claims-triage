"""D-11: the daily alert rule on the excesses of the winning daily model (§28.3).

Each day and pattern gets an excess score z: the normal score of the mid-PIT
of the observed count in the model's negative binomial predictive, with the
expected count read from the frozen run and the dispersion from its posterior.
Two rules share one false-alarm budget (4 a month over 40 patterns, 1/300 per
decision, pre-registered): a single day with z > z* and a daily CUSUM.
Thresholds come from simulated N(0, 1) scores. Injected bursts in calibration
measure what each rule detects, and the same bursts aggregated to weeks are
run through the weekly CUSUM of M11, so the table shows what each scale sees.

    python -m src.models.daily_change            # design in fit + calibration
    python -m src.models.daily_change --validation   # single 2025-H1 report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    git_commit,
    load_experiment_config,
    save_run_record,
    set_seed,
)
from src.models.daily_counts.contracts import DAY_COLUMN, SPLITS
from src.models.daily_counts.data import load_frozen_panel, panel_arrays
from src.models.daily_counts.references import discounted_daily_shares
from src.models.persistent_change import (
    cusum,
    cusum_threshold,
    excess_scores,
    first_alarm,
    m9_shares,
)
from src.models.weekly_counts.discounted_reference import shares_from_counts
from src.models.weekly_counts.sampling import subsample_posterior

STAGE = "M11D"
PLAN = "models_plan.md §28.3"
RULES = ("cusum", "daily")
DAYS_PER_MONTH = 365.25 / 12
WEEKS_PER_MONTH = 365.25 / 7 / 12
DESIGN_SPLITS = ("fit", "calibration")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", action="store_true")
    return parser.parse_args()


def false_alarm_rate(per_month: float, clusters: int) -> float:
    """Allowed false alarms per decision when every pattern is tested daily."""
    return per_month / (clusters * DAYS_PER_MONTH)


def run_dir(settings: dict[str, Any]) -> Path:
    return (
        PROJECT_ROOT
        / "reports/modeling/daily_counts"
        / settings["m9d_model"]
        / settings["m9d_run_key"]
    )


def artifact_dir(settings: dict[str, Any]) -> Path:
    return (
        PROJECT_ROOT
        / "artifacts/models/daily_counts"
        / settings["m9d_model"]
        / settings["m9d_run_key"]
    )


class DailyExpectation:
    """Expected counts of the frozen daily winner (D-A family, §28.2).

    Decaying shares and the day-of-week effect give the Negative Binomial
    mean; a zero-inflated winner (D-D) also carries `zero`, the posterior
    median probability of an extra zero per pattern, which deflates the
    expected count and enters the predictive CDF of the excess score.
    """

    def __init__(
        self,
        discount: float,
        beta: np.ndarray,
        zero: np.ndarray | None = None,
    ) -> None:
        self.discount = discount
        self.beta = beta  # (clusters, 7), posterior median
        self.zero = zero  # (clusters,), posterior median, or None

    def shares(self, counts: np.ndarray, day_of_week: np.ndarray) -> np.ndarray:
        memory = shares_from_counts(counts.astype(float), self.discount, None)
        weight = memory * np.exp(self.beta[:, day_of_week].T)
        return weight / np.nansum(weight, axis=1, keepdims=True)

    def nb_mean(self, counts: np.ndarray, day_of_week: np.ndarray) -> np.ndarray:
        """Mean of the Negative Binomial part: total times share."""
        return counts.sum(axis=1, keepdims=True) * self.shares(counts, day_of_week)

    def expected(self, counts: np.ndarray, day_of_week: np.ndarray) -> np.ndarray:
        """Expected count, as the run's `expected_mean` reports it."""
        mean = self.nb_mean(counts, day_of_week)
        return mean if self.zero is None else mean * (1.0 - self.zero)[None, :]

    def excess(
        self,
        observed: np.ndarray,
        nb_mean: np.ndarray,
        alpha: np.ndarray,
        clusters: np.ndarray,
    ) -> np.ndarray:
        """Excess scores of rows whose pattern ids are `clusters`."""
        zero = None if self.zero is None else self.zero[clusters]
        return excess_scores(observed, nb_mean, alpha[:, clusters], zero)


def load_model(
    settings: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, DailyExpectation]:
    """Dispersion draws and the share function of the frozen daily run."""
    spec = json.loads((artifact_dir(settings) / "model_spec.json").read_text("utf-8"))
    idata = az.from_netcdf(artifact_dir(settings) / "posterior.nc")
    posterior = subsample_posterior(idata, settings["prediction_draws"], seed)
    alpha = posterior["alpha"].transpose("sample", "cluster").to_numpy()
    beta = idata.posterior["beta"].median(dim=("chain", "draw")).to_numpy()
    zero = None
    if "pi" in idata.posterior:
        zero = idata.posterior["pi"].median(dim=("chain", "draw")).to_numpy()
    return alpha, DailyExpectation(spec["share"]["discount"], beta, zero)


def warm_rows(
    arrays: dict[str, np.ndarray],
    splits: np.ndarray,
    alpha: np.ndarray,
    model: DailyExpectation,
    warmup: int,
) -> pd.DataFrame:
    """Observed, expected and excess score of every day after the warm-up."""
    counts = arrays["counts"]
    nb_mean = model.nb_mean(counts, arrays["day_of_week"])
    expected = model.expected(counts, arrays["day_of_week"])
    days, clusters = np.meshgrid(
        np.arange(warmup, len(counts)), np.arange(counts.shape[1]), indexing="ij"
    )
    rows = pd.DataFrame(
        {
            "split": splits[days.ravel()],
            DAY_COLUMN: arrays["days"][days.ravel()],
            "cluster_id": clusters.ravel(),
            "observed": counts[days, clusters].ravel(),
            "expected": expected[days, clusters].ravel(),
        }
    )
    rows["z"] = model.excess(
        rows["observed"].to_numpy(),
        nb_mean[days, clusters].ravel(),
        alpha,
        rows["cluster_id"].to_numpy(),
    )
    return rows


def check_run(rows: pd.DataFrame, settings: dict[str, Any]) -> dict[str, float]:
    """The recomputed expected counts must match the frozen run's expected_mean."""
    predictions = pd.read_csv(run_dir(settings) / "predictions.csv")
    predictions[DAY_COLUMN] = pd.to_datetime(predictions[DAY_COLUMN])
    keys = [DAY_COLUMN, "cluster_id"]
    merged = rows.merge(predictions, on=keys, validate="one_to_one")
    gap = np.abs(merged["expected"] - merged["expected_mean"])
    check = {
        "rows": len(merged),
        "expected_max_difference": float(gap.max()),
        "relative": float((gap / merged["expected_mean"]).max()),
    }
    if len(merged) != len(rows) or check["relative"] > settings["expected_tolerance"]:
        raise ValueError(f"D-11 does not reproduce the daily run's expected: {check}")
    return check


def add_alarms(
    rows: pd.DataFrame,
    thresholds: dict[str, float],
    k: float,
) -> pd.DataFrame:
    z = rows["z"].to_numpy().reshape(-1, int(rows["cluster_id"].nunique()))
    statistic, alarms = cusum(z, k, thresholds["cusum"])
    return rows.assign(
        cusum=statistic.ravel(),
        cusum_alarm=alarms.ravel(),
        daily_alarm=rows["z"] > thresholds["daily"],
    )


def alarm_summary(rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary = {}
    for split, group in rows.groupby("split", sort=False):
        days = group[DAY_COLUMN].nunique()
        months = days / DAYS_PER_MONTH
        summary[str(split)] = {
            "days": int(days),
            "cusum": int(group["cusum_alarm"].sum()),
            "cusum_per_month": float(group["cusum_alarm"].sum() / months),
            "daily": int(group["daily_alarm"].sum()),
            "daily_per_month": float(group["daily_alarm"].sum() / months),
        }
    return summary


def burst_factors(scenario: dict[str, Any], horizon: int) -> np.ndarray:
    """Multiplicative factors over the horizon: `size` for `days`, then 1."""
    factors = np.ones(horizon)
    if scenario["kind"] == "burst":
        factors[: scenario["days"]] = scenario["size"]
    return factors


def injected_daily_scores(
    counts: np.ndarray,
    day_of_week: np.ndarray,
    alpha: np.ndarray,
    model: DailyExpectation,
    cluster: int,
    start: int,
    factors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Excess scores of one pattern during an injected burst, and the new counts."""
    window = slice(start, start + len(factors))
    increased = counts.copy()
    increased[window, cluster] = np.rint(counts[window, cluster] * factors)
    nb_mean = model.nb_mean(increased, day_of_week)[window, cluster]
    ids = np.full(len(nb_mean), cluster)
    z = model.excess(increased[window, cluster], nb_mean, alpha, ids)
    return z, increased


def weekly_view(
    counts: np.ndarray,
    days: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate daily counts to Monday-to-Sunday weeks; keep complete weeks only."""
    index = pd.DatetimeIndex(days)
    weeks = index.to_period("W-SUN").start_time
    frame = pd.DataFrame(counts)
    frame["week"] = weeks.to_numpy()
    grouped = frame.groupby("week")
    sizes = grouped.size()
    complete = sizes.index[sizes == 7]
    summed = grouped.sum().loc[complete]
    return summed.to_numpy(), summed.index.to_numpy()


def weekly_detection(
    increased: np.ndarray,
    days: np.ndarray,
    burst_day: int,
    weekly: dict[str, Any],
) -> float:
    """Delay in weeks of the weekly CUSUM on the same injected counts; NaN if none."""
    counts, weeks = weekly_view(increased, days)
    shares = m9_shares(counts, weekly["m9_settings"])
    expected = counts.sum(axis=1, keepdims=True) * shares
    burst_week = np.datetime64(days[burst_day], "D")
    start = int(np.searchsorted(weeks, burst_week, side="right") - 1)
    valid = np.arange(len(counts)) >= weekly["warmup_weeks"]
    z = np.full(counts.shape, np.nan)
    rows = np.flatnonzero(valid)
    for cluster in range(counts.shape[1]):
        z[rows, cluster] = excess_scores(
            counts[rows, cluster],
            expected[rows, cluster],
            weekly["alpha"][:, [cluster]],
        )
    window = z[start : start + weekly["horizon_weeks"]]
    alarms = cusum(np.nan_to_num(window), weekly["k"], weekly["h"])[1]
    return first_alarm(np.asarray(alarms.any(axis=1)))


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
    rng = np.random.default_rng(seed)
    first = settings["warmup_days"]
    last = len(counts) - horizon
    starts = rng.choice(np.arange(first, last), size=settings["starts_per_pattern"])
    rows = []
    for name, scenario in settings["scenarios"].items():
        factors = burst_factors(scenario, horizon)
        for cluster in range(counts.shape[1]):
            for start in starts:
                z, increased = injected_daily_scores(
                    counts,
                    arrays["day_of_week"],
                    alpha,
                    model,
                    cluster,
                    int(start),
                    factors,
                )
                k, h = settings["cusum_k"], thresholds["cusum"]
                alarms = cusum(z[:, None], k, h)[1]
                row = {
                    "scenario": name,
                    "cluster_id": cluster,
                    "start_day": str(np.datetime64(arrays["days"][start], "D")),
                    "cusum_delay": first_alarm(alarms.ravel()),
                    "daily_delay": first_alarm(z > thresholds["daily"]),
                }
                if weekly is not None and scenario["kind"] == "burst":
                    row["weekly_cusum_delay"] = weekly_detection(
                        increased, arrays["days"], int(start), weekly
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def detection_summary(runs: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    columns = [c for c in runs.columns if c.endswith("_delay")]
    for scenario, group in runs.groupby("scenario", sort=False):
        summary[str(scenario)] = {"runs": len(group)}
        for column in columns:
            delays = group[column].dropna()
            summary[str(scenario)][column.replace("_delay", "")] = {
                "detected": len(delays) / len(group),
                "median_delay": float(delays.median()) if len(delays) else None,
            }
    return summary


def decide(summary: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Adopt the daily rule if it sees the pre-registered bursts fast enough."""
    checks = {}
    for scenario in settings["deciding_scenarios"]:
        for rule in RULES:
            result = summary[scenario][rule]
            checks[f"{scenario}:{rule}"] = (
                result["detected"] >= settings["minimum_detection"]
                and result["median_delay"] is not None
                and result["median_delay"] <= settings["maximum_median_delay"]
            )
    deciding = settings["deciding_scenarios"]
    passing = {rule: all(checks[f"{s}:{rule}"] for s in deciding) for rule in RULES}
    chosen = None
    if passing["cusum"] and passing["daily"]:
        # Both qualify: the one with more detections on the deciding scenarios.
        score = {
            rule: sum(summary[s][rule]["detected"] for s in deciding) for rule in RULES
        }
        chosen = max(RULES, key=lambda rule: score[rule])
    elif passing["cusum"] or passing["daily"]:
        chosen = "cusum" if passing["cusum"] else "daily"
    return {"adopted": chosen is not None, "rule": chosen, "checks": checks}


def load_weekly(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """The weekly M11 pieces: C-A shares, its dispersion draws and thresholds."""
    change = config["persistent_change"]
    results = json.loads(
        (PROJECT_ROOT / change["report_dir"] / "results.json").read_text("utf-8")
    )
    m9_config = yaml.safe_load(
        (PROJECT_ROOT / "configs/weekly_counts" / f"{change['m9_model']}.yaml")
        .read_text(encoding="utf-8")
    )
    artifact = (
        PROJECT_ROOT
        / "artifacts/models/weekly_counts"
        / change["m9_model"]
        / change["m9_run_key"]
    )
    idata = az.from_netcdf(artifact / "posterior.nc")
    draws = m9_config["sampling"]["prediction_draws"]
    posterior = subsample_posterior(idata, draws, seed)
    return {
        "m9_settings": m9_config,
        "alpha": posterior["alpha"].transpose("sample", "cluster").to_numpy(),
        "k": change["cusum_k"],
        "h": results["thresholds"]["cusum"],
        "warmup_weeks": 4,
        "horizon_weeks": 4,
    }


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    config = load_experiment_config()
    settings = config["daily_change"]
    seed = config["experiment"]["seed"]
    set_seed(seed)
    model_config = yaml.safe_load(
        (PROJECT_ROOT / "configs/daily_counts" / f"{settings['m9d_model']}.yaml")
        .read_text(encoding="utf-8")
    )
    panel, input_sha256, source_dvc_hash = load_frozen_panel(config, model_config)
    clusters = model_config["clusters"]
    alpha, model = load_model(settings, seed)
    report_dir = PROJECT_ROOT / settings["report_dir"]

    splits = DESIGN_SPLITS if not args.validation else SPLITS
    frame = panel.loc[panel["split"].isin(splits)].reset_index(drop=True)
    frame = frame.assign(recent_share=discounted_daily_shares(frame, clusters, 0.5))
    arrays = panel_arrays(frame, clusters)
    split_of_day = frame.drop_duplicates(DAY_COLUMN)["split"].to_numpy()
    rows = warm_rows(arrays, split_of_day, alpha, model, settings["warmup_days"])
    check = check_run(rows.loc[rows["split"].isin(DESIGN_SPLITS)], settings)

    if args.validation:
        design = json.loads((report_dir / "results.json").read_text("utf-8"))
        thresholds = design["thresholds"]
        rows = add_alarms(rows, thresholds, settings["cusum_k"])
        validation = rows.loc[rows["split"] == "validation"]
        path = report_dir / "validation_results.json"
        if path.exists():
            raise FileExistsError(f"D-11 reports 2025-H1 only once: {path}")
        report = {
            "stage": STAGE,
            "plan": PLAN,
            "split": "validation",
            "already_consulted": True,
            "git_commit": git_commit(),
            "rule": design["decision"],
            "thresholds": thresholds,
            "real_alarms": alarm_summary(validation),
        }
        path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        validation.to_csv(report_dir / "validation_alerts.csv", index=False)
        print(json.dumps(report["real_alarms"], indent=2))
        return

    if report_dir.exists():
        raise FileExistsError(f"D-11 never overwrites its design: {report_dir}")
    rate = false_alarm_rate(settings["false_alarms_per_month"], clusters)
    thresholds = {
        "cusum": cusum_threshold(
            rate,
            settings["cusum_k"],
            settings["simulation_series"],
            settings["simulation_days"],
            seed,
        ),
        "daily": float(norm.ppf(1 - rate)),
    }
    rows = add_alarms(rows, thresholds, settings["cusum_k"])
    calibration = {name: values for name, values in arrays.items()}
    calibration_days = np.flatnonzero(split_of_day == "calibration")
    weekly = load_weekly(config, seed) if settings.get("compare_weekly", True) else None
    runs = detection_runs(
        {name: values[calibration_days] for name, values in calibration.items()},
        alpha,
        model,
        thresholds,
        settings,
        weekly,
        seed,
    )
    summary = detection_summary(runs)
    decision = decide(summary, settings)
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
        "zero_inflated": model.zero is not None,
        "daily_counts_sha256": input_sha256,
        "source_dvc_hash": source_dvc_hash,
        "settings": {k: v for k, v in settings.items() if k != "scenarios"},
        "scenarios": settings["scenarios"],
        "false_alarm_rate_per_decision": rate,
        "thresholds": thresholds,
        "run_check": check,
        "real_alarms": alarm_summary(rows),
        "detection": summary,
        "decision": decision,
        "weekly_reference": (
            None
            if weekly is None
            else {name: weekly[name] for name in ("h", "k", "horizon_weeks")}
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (report_dir / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metrics = {
        "threshold_cusum": thresholds["cusum"],
        "threshold_daily": thresholds["daily"],
        **{
            f"{split}_{rule}_per_month": values[f"{rule}_per_month"]
            for split, values in report["real_alarms"].items()
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
        features=["mid-PIT excess of the daily model"],
        parameters={
            "plan": PLAN,
            "m9d_run_key": settings["m9d_run_key"],
            "false_alarms_per_month": settings["false_alarms_per_month"],
            "cusum_k": settings["cusum_k"],
            "decision": json.dumps(decision["rule"]),
        },
        metrics=metrics,
        artifacts=[f"{settings['report_dir']}/results.json"],
    )
    run_root = PROJECT_ROOT / config["paths"]["offline_runs"] / "m11d"
    save_run_record(record, run_root / "run.json")
    print(json.dumps({"thresholds": thresholds}, indent=2))
    print(json.dumps(report["real_alarms"], indent=2))
    print(json.dumps(summary, indent=2))
    print(json.dumps(decision, indent=2))
    print(f"Report: {report_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
