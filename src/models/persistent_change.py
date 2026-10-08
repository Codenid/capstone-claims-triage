"""M11: persistent increases in the M8 patterns (models_plan.md §24).

Each week's excess is the normal score of the mid-PIT under the frozen M9
predictive. An upper CUSUM per pattern is compared with a weekly excess rule at
the same false-alarm budget, with increases injected on fit + calibration.
--validation then reports the 2025-H1 alerts once, with everything frozen.

The M9 run comes from `persistent_change` in configs/modeling.yaml: NB-R4-H v3
for the first M11, C-A after block C (§25.5). Its expected counts follow its
own shares: the 4 previous weeks, or C-A's decaying memory.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial
import json
from pathlib import Path
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm

from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    load_experiment_config,
    save_run_record,
    set_seed,
)
from src.evaluation.final_confirmation import load_settings
from src.models.representation_comparison import (
    check_new_outputs,
    read_json,
    write_json,
)
from src.models.weekly_composition.data import (
    RECENT_WEEKS,
    composition_panel,
    select_weeks,
)
from src.models.weekly_counts.discounted_reference import shares_from_counts
from src.models.weekly_counts.run import load_frozen_frame
from src.models.weekly_counts.sampling import subsample_posterior

Shares = Callable[[np.ndarray], np.ndarray]
M9_ROOTS = {
    "report": "reports/modeling/weekly_counts",
    "artifact": "artifacts/models/weekly_counts",
}
RULES = ("cusum", "weekly")
RULE_LABELS = {"cusum": "CUSUM", "weekly": "Regla semanal"}
DESIGN_SPLITS = ("fit", "calibration")
WEEKS_PER_MONTH = 365.25 / 7 / 12
# Keeps z finite; |z| <= 4.75 still alarms in a single week.
PIT_CLIP = 1e-6
BISECTION_STEPS = 30
DESIGN_FILES = {
    "results": "results.json",
    "alerts": "alerts.csv",
    "detection": "detection.csv",
    "plot": "detection.png",
}
VALIDATION_FILES = {
    "results": "validation_results.json",
    "alerts": "validation_alerts.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Report the 2025-H1 alerts once, with the frozen thresholds.",
    )
    return parser.parse_args()


def false_alarm_rate(per_month: float, clusters: int) -> float:
    """Alarms per pattern and week that add up to `per_month` in total."""
    return per_month / (WEEKS_PER_MONTH * clusters)


def recent_shares(counts: np.ndarray) -> np.ndarray:
    """R4 shares as in M9's rolling_shares, for a (weeks, clusters) array."""
    rolling = pd.DataFrame(counts).shift(1).rolling(RECENT_WEEKS)
    previous = np.asarray(rolling.sum(), dtype=float)
    return (previous + 1) / (previous.sum(axis=1, keepdims=True) + counts.shape[1])


def m9_shares(counts: np.ndarray, m9_settings: dict[str, Any]) -> np.ndarray:
    """The shares behind M9's expected counts: C-A's memory, or the 4 weeks."""
    share = m9_settings.get("share")
    if share is None:
        return recent_shares(counts)
    return shares_from_counts(counts.astype(float), share["discount"], share["cap"])


def m9_run(settings: dict[str, Any], kind: str) -> Path:
    """Report or artifact directory of the M9 run that M11 uses."""
    return (
        PROJECT_ROOT / M9_ROOTS[kind] / settings["m9_model"] / settings["m9_run_key"]
    )


def predictive_cdf(
    values: np.ndarray,
    mu: np.ndarray,
    alpha: np.ndarray,
    zero: np.ndarray | None = None,
) -> np.ndarray:
    """P(Y <= value) under the Negative Binomial, averaged over posterior draws.

    `values` and `mu` have one entry per row; `alpha` has one row per draw.
    With `zero` (extra-zero probability per row, D-D of §28.2) the count is
    zero-inflated: P(Y <= y) = zero + (1 - zero) P_NB(Y <= y) for y >= 0.
    """
    cdf = nbinom.cdf(values, alpha, alpha / (alpha + mu))
    if zero is not None:
        cdf = np.where(values < 0, 0.0, zero + (1.0 - zero) * cdf)
    return cdf.mean(axis=0)


def excess_scores(
    observed: np.ndarray,
    mu: np.ndarray,
    alpha: np.ndarray,
    zero: np.ndarray | None = None,
) -> np.ndarray:
    """Normal score of the mid-PIT: about N(0, 1) when the model is calibrated."""
    mid_pit = (
        predictive_cdf(observed - 1, mu, alpha, zero)
        + predictive_cdf(observed, mu, alpha, zero)
    ) / 2
    return norm.ppf(np.clip(mid_pit, PIT_CLIP, 1 - PIT_CLIP))


def cusum(z: np.ndarray, k: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Upper CUSUM along the first axis; it restarts at 0 after each alarm."""
    statistic = np.zeros(z.shape)
    alarms = np.zeros(z.shape, dtype=bool)
    current = np.zeros(z.shape[1:])
    for week, value in enumerate(z):
        current = np.maximum(0.0, current + value - k)
        statistic[week] = current
        alarms[week] = current > h
        current = np.where(alarms[week], 0.0, current)
    return statistic, alarms


def cusum_threshold(rate: float, k: float, series: int, weeks: int, seed: int) -> float:
    """Smallest h whose alarm rate on independent N(0, 1) scores is at most `rate`."""
    z = np.random.default_rng(seed).standard_normal((weeks, series))
    low, high = 0.0, 20.0
    for _ in range(BISECTION_STEPS):
        middle = (low + high) / 2
        if cusum(z, k, middle)[1].mean() > rate:
            low = middle
        else:
            high = middle
    return high


def load_alpha(settings: dict[str, Any], artifact: Path, seed: int) -> np.ndarray:
    """M9's dispersion draws used by its predictions, shape (draws, clusters)."""
    idata = az.from_netcdf(artifact / "posterior.nc")
    draws = settings["sampling"]["prediction_draws"]
    posterior = subsample_posterior(idata, draws, seed)
    return posterior["alpha"].transpose("sample", "cluster").to_numpy()


def windowed_rows(
    panel: dict[str, np.ndarray],
    alpha: np.ndarray,
    shares: Shares = recent_shares,
) -> pd.DataFrame:
    """Observed and expected count and excess score of every week with a full window."""
    counts = panel["counts"]
    expected = counts.sum(axis=1, keepdims=True) * shares(counts)
    weeks, clusters = np.meshgrid(
        np.arange(RECENT_WEEKS, len(counts)),
        np.arange(counts.shape[1]),
        indexing="ij",
    )
    rows = pd.DataFrame(
        {
            "split": panel["split"][weeks.ravel()],
            "week": panel["week"][weeks.ravel()],
            "cluster_id": clusters.ravel(),
            "observed": counts[weeks, clusters].ravel(),
            "expected": expected[weeks, clusters].ravel(),
        }
    )
    rows["z"] = excess_scores(
        rows["observed"].to_numpy(),
        rows["expected"].to_numpy(),
        alpha[:, rows["cluster_id"].to_numpy()],
    )
    return rows


def check_m9(
    rows: pd.DataFrame,
    alpha: np.ndarray,
    tolerance: float,
    report: Path,
) -> dict[str, float]:
    """M11 must reproduce M9's expected counts and its saved upper tail P(Y >= y)."""
    predictions = pd.read_csv(report / "predictions.csv", parse_dates=["week"])
    merged = rows.merge(predictions, on=["week", "cluster_id"], validate="one_to_one")
    upper_tail = 1 - predictive_cdf(
        merged["observed"].to_numpy() - 1,
        merged["expected"].to_numpy(),
        alpha[:, merged["cluster_id"].to_numpy()],
    )
    check = {
        "rows": len(merged),
        "expected_max_difference": float(
            np.abs(merged["expected"] - merged["expected_mean"]).max()
        ),
        "upper_tail_max_difference": float(
            np.abs(upper_tail - merged["model_upper_tail_probability"]).max()
        ),
    }
    same_rows = len(merged) == len(rows) and (
        merged["observed"] == merged["complaint_count"]
    ).all()
    if not same_rows or check["expected_max_difference"] > 1e-6:
        raise ValueError(f"M11 does not reproduce M9's expected counts: {check}")
    if check["upper_tail_max_difference"] > tolerance:
        raise ValueError(f"M11 does not reproduce M9's upper tail: {check}")
    return check


def add_alarms(
    rows: pd.DataFrame,
    thresholds: dict[str, float],
    k: float,
) -> pd.DataFrame:
    """Both rules on the real series; the CUSUM runs without restarting."""
    z = rows["z"].to_numpy().reshape(-1, int(rows["cluster_id"].nunique()))
    statistic, alarms = cusum(z, k, thresholds["cusum"])
    return rows.assign(
        cusum=statistic.ravel(),
        cusum_alarm=alarms.ravel(),
        weekly_alarm=rows["z"] > thresholds["weekly"],
    )


def alarm_summary(rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Alarms of each rule per split, also as alarms per month in total."""
    summary: dict[str, dict[str, Any]] = {}
    for split, group in rows.groupby("split", sort=False):
        weeks = int(group["week"].nunique())
        summary[str(split)] = {"weeks": weeks}
        for rule in RULES:
            alarms = int(group[f"{rule}_alarm"].sum())
            summary[str(split)][rule] = alarms
            summary[str(split)][f"{rule}_per_month"] = alarms / weeks * WEEKS_PER_MONTH
    return summary


def alerts_table(rows: pd.DataFrame, cluster_summary: pd.DataFrame) -> pd.DataFrame:
    """Weeks where any rule alarms, with the pattern's words for human review."""
    alerts = rows.loc[rows["cusum_alarm"] | rows["weekly_alarm"]]
    terms = cluster_summary[["cluster_id", "main_product", "representative_terms"]]
    return alerts.merge(terms, on="cluster_id", how="left")


def scenario_factors(kind: str, size: float, horizon: int) -> np.ndarray:
    """Multiplier of the real count in each week of an artificial increase."""
    if kind == "growth":
        return (1 + size) ** np.arange(1, horizon + 1)
    if kind == "step":
        return np.full(horizon, 1 + size)
    if kind == "none":
        return np.ones(horizon)
    raise ValueError(f"Unknown scenario kind: {kind}")


def increased_counts(
    counts: np.ndarray,
    cluster: int,
    start: int,
    factors: np.ndarray,
) -> np.ndarray:
    """Copy of the counts with one pattern multiplied by `factors` from `start`."""
    window = slice(start, start + len(factors))
    increased = counts.copy()
    increased[window, cluster] = np.rint(counts[window, cluster] * factors)
    return increased


def injected_scores(
    counts: np.ndarray,
    alpha: np.ndarray,
    cluster: int,
    start: int,
    factors: np.ndarray,
    shares: Shares = recent_shares,
) -> np.ndarray:
    """Excess scores of one pattern during an artificial increase.

    The added complaints raise the weekly total, and the following weeks'
    shares use the increased counts, as M9 would.
    """
    window = slice(start, start + len(factors))
    increased = increased_counts(counts, cluster, start, factors)
    expected = increased[window].sum(axis=1) * shares(increased)[window, cluster]
    return excess_scores(increased[window, cluster], expected, alpha[:, [cluster]])


def first_alarm(alarms: np.ndarray) -> float:
    """Week of the first alarm, counting the first week as 1; NaN without one."""
    hits = np.flatnonzero(alarms)
    return float(hits[0] + 1) if hits.size else np.nan


def detection_runs(
    panel: dict[str, np.ndarray],
    alpha: np.ndarray,
    thresholds: dict[str, float],
    settings: dict[str, Any],
    shares: Shares = recent_shares,
) -> pd.DataFrame:
    """Each rule's delay for every scenario, pattern and start week that fits."""
    counts = panel["counts"]
    horizon = settings["horizon_weeks"]
    rows = []
    for name, scenario in settings["scenarios"].items():
        factors = scenario_factors(scenario["kind"], scenario["size"], horizon)
        for cluster in range(counts.shape[1]):
            for start in range(RECENT_WEEKS, len(counts) - horizon + 1):
                z = injected_scores(counts, alpha, cluster, start, factors, shares)
                alarms = cusum(z, settings["cusum_k"], thresholds["cusum"])[1]
                rows.append(
                    {
                        "scenario": name,
                        "cluster_id": cluster,
                        "start_week": panel["week"][start],
                        "cusum_delay": first_alarm(alarms),
                        "weekly_delay": first_alarm(z > thresholds["weekly"]),
                    }
                )
    return pd.DataFrame(rows)


def detection_summary(runs: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Share of runs detected within the horizon and median delay when detected."""
    summary: dict[str, dict[str, Any]] = {}
    for scenario, group in runs.groupby("scenario", sort=False):
        summary[str(scenario)] = {"runs": len(group)}
        for rule in RULES:
            delays = group[f"{rule}_delay"].dropna()
            summary[str(scenario)][rule] = {
                "detected": len(delays) / len(group),
                "median_delay": float(delays.median()) if len(delays) else None,
            }
    return summary


def choose_rule(summary: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """CUSUM only if it detects clearly more in every deciding scenario (§24)."""
    gains = {
        scenario: summary[scenario]["cusum"]["detected"]
        - summary[scenario]["weekly"]["detected"]
        for scenario in settings["deciding_scenarios"]
    }
    minimum = settings["minimum_detection_gain"]
    clear = all(gain >= minimum for gain in gains.values())
    return {"rule": "cusum" if clear else "weekly", "gains": gains, "minimum": minimum}


def scenario_label(scenario: dict[str, Any]) -> str:
    if scenario["kind"] == "growth":
        return f"Crecimiento {scenario['size']:.0%}"
    if scenario["kind"] == "step":
        return f"Salto {scenario['size']:.0%}"
    return "Sin aumento"


def detection_figure(summary: dict[str, Any], settings: dict[str, Any]) -> Any:
    """Share of increases each rule detects within the horizon, by scenario."""
    names = list(settings["scenarios"])
    positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(8, 4))
    for offset, rule in zip((-0.2, 0.2), RULES):
        rates = [summary[name][rule]["detected"] for name in names]
        axis.bar(positions + offset, rates, width=0.4, label=RULE_LABELS[rule])
    labels = [scenario_label(settings["scenarios"][name]) for name in names]
    axis.set_xticks(positions, labels)
    axis.set_ylabel(f"Avisa dentro de {settings['horizon_weeks']} semanas")
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.set_ylim(0, 1)
    axis.legend()
    figure.tight_layout()
    return figure


def flat_metrics(prefix: str, value: Any) -> dict[str, float]:
    """Nested numbers as metric names; text and missing values are skipped."""
    if isinstance(value, dict):
        return {
            name: number
            for key, item in value.items()
            for name, number in flat_metrics(f"{prefix}_{key}", item).items()
        }
    if isinstance(value, (bool, int, float)):
        return {prefix: float(value)}
    return {}


def record_path(config: dict[str, Any], split: str) -> Path:
    directory = PROJECT_ROOT / config["paths"]["offline_runs"]
    name = config["persistent_change"]["run_name"].replace("-", "_")
    return directory / f"{name}_{split}" / "run.json"


def offline_record(
    config: dict[str, Any],
    report: dict[str, Any],
    artifacts: list[Path],
) -> dict[str, Any]:
    """MLflow record of one M11 run, published later from the login node."""
    metrics = {
        **flat_metrics("thresholds", report["thresholds"]),
        **flat_metrics("alarms", report["real_alarms"]),
        **flat_metrics("detection", report.get("detection", {})),
    }
    record = build_run_record(
        config=config,
        run_name=f"{config['persistent_change']['run_name']}-{report['split']}",
        stage="M11",
        target="weekly_cluster_counts",
        split=report["split"],
        view="complete_weeks",
        features=["cluster_id", "week", "weekly_total"],
        parameters={
            "rule": "models_plan.md §24",
            "m9_model": report["m9_model"],
            "m9_run_key": report["m9_run_key"],
            "settings": json.dumps(report["settings"], sort_keys=True),
            "chosen_rule": report["decision"]["rule"],
        },
        metrics=metrics,
        artifacts=[path.relative_to(PROJECT_ROOT).as_posix() for path in artifacts],
    )
    record["tags"].update(
        {"run_mode": "full", "validation_used_for_selection": "false"}
    )
    return record


def print_summary(report: dict[str, Any]) -> None:
    keys = ("thresholds", "real_alarms", "decision")
    print(json.dumps({key: report[key] for key in keys}, indent=2))


def run_design(
    config: dict[str, Any],
    panel: dict[str, np.ndarray],
    alpha: np.ndarray,
    report_dir: Path,
    shares: Shares,
) -> None:
    """Thresholds, real alerts and injected increases on fit + calibration."""
    settings = config["persistent_change"]
    paths = {name: report_dir / file for name, file in DESIGN_FILES.items()}
    record_file = record_path(config, "fit_calibration")
    check_new_outputs([*paths.values(), record_file])
    clusters = panel["counts"].shape[1]
    seed = config["experiment"]["seed"]
    rate = false_alarm_rate(settings["false_alarms_per_month"], clusters)
    thresholds = {
        "cusum": cusum_threshold(
            rate,
            settings["cusum_k"],
            settings["simulation_series"],
            settings["simulation_weeks"],
            seed,
        ),
        "weekly": float(norm.ppf(1 - rate)),
    }
    rows = windowed_rows(panel, alpha, shares)
    m9_check = check_m9(
        rows, alpha, settings["pit_tolerance"], m9_run(settings, "report")
    )
    rows = add_alarms(rows, thresholds, settings["cusum_k"])
    runs = detection_runs(panel, alpha, thresholds, settings, shares)
    detection = detection_summary(runs)
    report = {
        "stage": "M11",
        "split": "fit_calibration",
        "rule": "models_plan.md §24",
        "m9_model": settings["m9_model"],
        "m9_run_key": settings["m9_run_key"],
        "seed": seed,
        "settings": settings,
        "false_alarm_rate": rate,
        "thresholds": thresholds,
        "m9_check": m9_check,
        "real_alarms": alarm_summary(rows),
        "detection": detection,
        "decision": choose_rule(detection, settings),
    }
    cluster_summary = pd.read_csv(PROJECT_ROOT / config["paths"]["cluster_summary"])
    alerts = alerts_table(rows, cluster_summary)
    record = offline_record(config, report, list(paths.values()))
    figure = detection_figure(detection, settings)
    # Serialize before writing anything, so a failure leaves no partial results.
    json.dumps([report, record])

    report_dir.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(paths["alerts"], index=False)
    runs.to_csv(paths["detection"], index=False)
    figure.savefig(paths["plot"], dpi=150)
    plt.close(figure)
    write_json(paths["results"], report)
    save_run_record(record, record_file)
    print_summary(report)


def run_validation(
    config: dict[str, Any],
    panel: dict[str, np.ndarray],
    alpha: np.ndarray,
    report_dir: Path,
    shares: Shares,
) -> None:
    """The single 2025-H1 report, with the CUSUM continued from calibration."""
    design = read_json(report_dir / DESIGN_FILES["results"])
    paths = {name: report_dir / file for name, file in VALIDATION_FILES.items()}
    record_file = record_path(config, "validation")
    check_new_outputs([*paths.values(), record_file])
    settings = design["settings"]
    rows = windowed_rows(panel, alpha, shares)
    m9_check = check_m9(
        rows, alpha, settings["pit_tolerance"], m9_run(settings, "report")
    )
    rows = add_alarms(rows, design["thresholds"], settings["cusum_k"])
    cluster_summary = pd.read_csv(PROJECT_ROOT / config["paths"]["cluster_summary"])
    alerts = alerts_table(rows, cluster_summary)

    # The design weeks must give exactly the alerts the design run saved.
    saved = pd.read_csv(report_dir / DESIGN_FILES["alerts"], parse_dates=["week"])
    keys = ["week", "cluster_id", "cusum_alarm", "weekly_alarm"]
    design_alerts = alerts.loc[alerts["split"].isin(DESIGN_SPLITS), keys]
    if not design_alerts.reset_index(drop=True).equals(saved[keys]):
        raise ValueError("The fit + calibration alerts differ from the design run.")

    report = {
        "stage": "M11",
        "split": "validation",
        "rule": "models_plan.md §24",
        "m9_model": settings["m9_model"],
        "m9_run_key": settings["m9_run_key"],
        "settings": settings,
        "thresholds": design["thresholds"],
        "decision": design["decision"],
        "m9_check": m9_check,
        "real_alarms": alarm_summary(rows),
    }
    record = offline_record(config, report, list(paths.values()))
    json.dumps([report, record])

    alerts.loc[alerts["split"] == "validation"].to_csv(paths["alerts"], index=False)
    write_json(paths["results"], report)
    save_run_record(record, record_file)
    print_summary(report)


def main() -> None:
    args = parse_args()
    config = load_experiment_config()
    seed = config["experiment"]["seed"]
    set_seed(seed)
    settings = config["persistent_change"]
    m9_settings = load_settings("weekly_counts", settings["m9_model"])
    frame, _, _, _ = load_frozen_frame(config, m9_settings)
    panel = composition_panel(frame, m9_settings["clusters"])
    alpha = load_alpha(m9_settings, m9_run(settings, "artifact"), seed)
    shares = partial(m9_shares, m9_settings=m9_settings)
    report_dir = PROJECT_ROOT / settings["report_dir"]
    if args.validation:
        run_validation(config, panel, alpha, report_dir, shares)
        return
    design = select_weeks(panel, np.isin(panel["split"], DESIGN_SPLITS))
    run_design(config, design, alpha, report_dir, shares)


if __name__ == "__main__":
    main()
