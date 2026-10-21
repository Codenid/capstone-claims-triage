"""C-M11-FARR v1: M11 with the memory reweighted by the excess (§31.2).

C-A's decaying memory m_t = d * m_{t-1} + w_t * y_t gives each pattern's
expected share. In M11 every week enters with weight 1, so a burst raises the
pattern's expectation for weeks (the echo). Here a week enters with weight
(z0 / z)^2 when its upward excess z (M11's normal score of the mid-PIT,
computed with the memory before that week) exceeds z0, as Farrington (1996)
and Noufaily (2013) do with the baseline of outbreak detection. Nothing is
refitted: alpha, the discount, the budget and the thresholds are M11's. The
candidate is judged with M11's own simulation against the frozen M11 report.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    git_commit,
    load_experiment_config,
    save_run_record,
    set_seed,
)
from src.evaluation.final_confirmation import load_settings
from src.models.persistent_change import (
    DESIGN_SPLITS,
    RULES,
    add_alarms,
    alarm_summary,
    alerts_table,
    check_m9,
    cusum,
    detection_summary,
    excess_scores,
    first_alarm,
    increased_counts,
    load_alpha,
    m9_run,
    m9_shares,
    scenario_factors,
    windowed_rows,
)
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
from src.models.weekly_counts.metrics import weighted_interval_score
from src.models.weekly_counts.run import load_frozen_frame

STAGE = "C-M11-FARR"
PLAN = "models_plan.md §31.2"
WIS_INTERVALS = ((0.5, 0.25, 0.75), (0.2, 0.10, 0.90), (0.05, 0.025, 0.975))
DESIGN_FILES = {
    "results": "results.json",
    "alerts": "alerts.csv",
    "detection": "detection.csv",
}
VALIDATION_FILES = {
    "results": "validation_results.json",
    "alerts": "validation_alerts.csv",
}


# ------------------------------------------------------------ pure rules
def excess_weights(z: np.ndarray, threshold: float) -> np.ndarray:
    """1 up to the threshold, then (threshold / z)^2: upward excesses only."""
    weights = np.ones(np.shape(z))
    above = z > threshold
    weights[above] = (threshold / z[above]) ** 2
    return weights


class ReweightedShares:
    """C-A's memory where each week enters with the weight of its excess.

    `alpha` has shape (draws, clusters). The first call fixes the baseline
    panel and keeps the memory before every week, so a panel that differs
    from the baseline only from some week on (an injected increase) resumes
    from that week; `partial(counts, stop)` stops after week `stop - 1`.
    """

    def __init__(self, alpha: np.ndarray, discount: float, threshold: float) -> None:
        if not 0 < discount <= 1:
            raise ValueError("The discount must be in (0, 1].")
        self.alpha = alpha
        self.discount = discount
        self.threshold = threshold
        self._baseline: np.ndarray | None = None
        self._states: dict[int, np.ndarray] = {}
        self._shares: np.ndarray | None = None

    def __call__(self, counts: np.ndarray) -> np.ndarray:
        return self.partial(counts, np.shape(counts)[0])

    def partial(self, counts: np.ndarray, stop: int) -> np.ndarray:
        """Shares of the weeks before `stop`; NaN in the first week and after."""
        values = np.asarray(counts, dtype=float)
        if (
            self._baseline is None
            or self._shares is None
            or self._baseline.shape != values.shape
        ):
            shares, states = self._run(
                values, 0, np.zeros(values.shape[1]), len(values)
            )
            self._baseline, self._states, self._shares = values.copy(), states, shares
        changed = np.flatnonzero((values != self._baseline).any(axis=1))
        if not changed.size:
            return self._shares.copy()
        first = int(changed[0])
        shares = self._shares.copy()
        shares[first:] = np.nan
        resumed, _ = self._run(values, first, self._states[first], max(stop, first))
        shares[first:stop] = resumed[first:stop]
        return shares

    def _run(
        self,
        counts: np.ndarray,
        start: int,
        memory: np.ndarray,
        stop: int,
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Shares of weeks start..stop-1 from the memory before `start`."""
        clusters = counts.shape[1]
        shares = np.full(counts.shape, np.nan)
        memory = memory.copy()
        states = {start: memory.copy()}
        for week in range(start, stop):
            observed = counts[week]
            weights = np.ones(clusters)
            if week > 0:
                shares[week] = (memory + 1) / (memory.sum() + clusters)
                expected = observed.sum() * shares[week]
                z = excess_scores(observed, expected, self.alpha)
                weights = excess_weights(z, self.threshold)
            memory = self.discount * memory + weights * observed
            states[week + 1] = memory.copy()
        return shares, states


def detection_runs(
    panel: dict[str, np.ndarray],
    alpha: np.ndarray,
    thresholds: dict[str, float],
    settings: dict[str, Any],
    shares: ReweightedShares,
) -> pd.DataFrame:
    """M11's injected increases, computing the memory only through the window."""
    counts = panel["counts"]
    horizon = settings["horizon_weeks"]
    rows = []
    for name, scenario in settings["scenarios"].items():
        factors = scenario_factors(scenario["kind"], scenario["size"], horizon)
        for cluster in range(counts.shape[1]):
            for start in range(RECENT_WEEKS, len(counts) - horizon + 1):
                window = slice(start, start + horizon)
                increased = increased_counts(counts, cluster, start, factors)
                share = shares.partial(increased, start + horizon)[window, cluster]
                expected = increased[window].sum(axis=1) * share
                z = excess_scores(
                    increased[window, cluster], expected, alpha[:, [cluster]]
                )
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


def plugin_wis(
    counts: np.ndarray,
    shares: np.ndarray,
    alpha: np.ndarray,
    weeks: np.ndarray,
    samples_per_draw: int,
    seed: int,
) -> float:
    """Mean WIS over the given weeks of NB(total * share, alpha) sampled per draw."""
    rng = np.random.default_rng(seed)
    draws = alpha.shape[0]
    scores = []
    for week in weeks:
        mu = counts[week].sum() * shares[week]
        n = np.broadcast_to(
            alpha[:, None, :], (draws, samples_per_draw, alpha.shape[1])
        )
        samples = rng.negative_binomial(n, n / (n + mu)).reshape(-1, alpha.shape[1])
        median = np.quantile(samples, 0.5, axis=0)
        intervals = [
            (
                level,
                np.quantile(samples, low, axis=0),
                np.quantile(samples, high, axis=0),
            )
            for level, low, high in WIS_INTERVALS
        ]
        scores.append(weighted_interval_score(counts[week], median, intervals))
    return float(np.mean(np.concatenate(scores)))


def decide(
    detection: dict[str, Any],
    alarms: dict[str, Any],
    reference: dict[str, Any],
    deciding_scenarios: list[str],
    maximum_drop: float,
) -> dict[str, Any]:
    """§31.2: detection does not drop more than `maximum_drop`, fewer real alarms."""
    drops = {
        scenario: reference["detection"][scenario]["cusum"]["detected"]
        - detection[scenario]["cusum"]["detected"]
        for scenario in deciding_scenarios
    }
    fewer = {
        split: alarms[split]["cusum_per_month"]
        < reference["real_alarms"][split]["cusum_per_month"]
        for split in DESIGN_SPLITS
    }
    adopted = all(drop <= maximum_drop for drop in drops.values()) and all(
        fewer.values()
    )
    return {
        "rule": (
            "adopt if CUSUM detection in the deciding scenarios drops at most "
            f"{maximum_drop:.2f} against M11 and real CUSUM alarms per month fall "
            "in fit and calibration"
        ),
        "detection_drops": drops,
        "fewer_alarms": fewer,
        "adopted": bool(adopted),
    }


# ------------------------------------------------------------------- runs
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=STAGE)
    parser.add_argument(
        "--validation",
        action="store_true",
        help="report the 2025-H1 alerts once, after the design decision",
    )
    return parser.parse_args()


def weight_draws(alpha: np.ndarray, draws: int, seed: int) -> np.ndarray:
    """A fixed subsample of the alpha draws for the weights only."""
    chosen = np.random.default_rng(seed).choice(len(alpha), draws, replace=False)
    return alpha[np.sort(chosen)]


def split_weeks(panel: dict[str, np.ndarray], split: str) -> np.ndarray:
    weeks = np.flatnonzero(panel["split"] == split)
    return weeks[weeks >= RECENT_WEEKS]


def evaluate_threshold(
    threshold: float,
    panel: dict[str, np.ndarray],
    alpha: np.ndarray,
    weights_alpha: np.ndarray,
    discount: float,
    thresholds: dict[str, float],
    m11: dict[str, Any],
    settings: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    shares = ReweightedShares(weights_alpha, discount, threshold)
    rows = add_alarms(windowed_rows(panel, alpha, shares), thresholds, m11["cusum_k"])
    runs = detection_runs(panel, alpha, thresholds, m11, shares)
    share_values = shares(panel["counts"])
    summary = {
        "threshold": threshold,
        "reweighted_rows": int((rows["z"] > threshold).sum()),
        "real_alarms": alarm_summary(rows),
        "detection": detection_summary(runs),
        "wis": {
            split: plugin_wis(
                panel["counts"],
                share_values,
                alpha,
                split_weeks(panel, split),
                settings["wis_samples_per_draw"],
                seed,
            )
            for split in DESIGN_SPLITS
        },
    }
    return summary, rows, runs


def run_design(
    config: dict[str, Any],
    panel: dict[str, np.ndarray],
    alpha: np.ndarray,
    m9_settings: dict[str, Any],
    report_dir: Path,
    started: float,
) -> None:
    settings = config["persistent_change_farr"]
    m11 = config["persistent_change"]
    seed = config["experiment"]["seed"]
    paths = {name: report_dir / file for name, file in DESIGN_FILES.items()}
    record_file = (
        PROJECT_ROOT / config["paths"]["offline_runs"] / "m11_farr" / "run.json"
    )
    check_new_outputs([*paths.values(), record_file])
    reference = read_json(PROJECT_ROOT / settings["reference_report"])
    thresholds = reference["thresholds"]
    discount = m9_settings["share"]["discount"]

    baseline_shares = partial(m9_shares, m9_settings=m9_settings)
    baseline_rows = windowed_rows(panel, alpha, baseline_shares)
    m9_check = check_m9(
        baseline_rows, alpha, m11["pit_tolerance"], m9_run(m11, "report")
    )
    baseline_values = baseline_shares(panel["counts"])
    baseline_wis = {
        split: plugin_wis(
            panel["counts"],
            baseline_values,
            alpha,
            split_weeks(panel, split),
            settings["wis_samples_per_draw"],
            seed,
        )
        for split in DESIGN_SPLITS
    }

    weights_alpha = weight_draws(alpha, settings["weight_draws"], seed)
    candidates: dict[str, Any] = {}
    primary_rows = primary_runs = None
    for threshold in (settings["excess_threshold"], *settings["secondary_thresholds"]):
        summary, rows, runs = evaluate_threshold(
            threshold,
            panel,
            alpha,
            weights_alpha,
            discount,
            thresholds,
            m11,
            settings,
            seed,
        )
        candidates[str(threshold)] = summary
        if threshold == settings["excess_threshold"]:
            primary_rows, primary_runs = rows, runs
        print(
            json.dumps({"threshold": threshold, "real_alarms": summary["real_alarms"]})
        )
    assert primary_rows is not None and primary_runs is not None
    primary = candidates[str(settings["excess_threshold"])]
    decision = decide(
        primary["detection"],
        primary["real_alarms"],
        reference,
        m11["deciding_scenarios"],
        settings["maximum_detection_drop"],
    )
    report = {
        "stage": STAGE,
        "plan": PLAN,
        "split": "fit_calibration",
        "git_commit": git_commit(),
        "seed": seed,
        "settings": settings,
        "m9_model": m11["m9_model"],
        "m9_run_key": m11["m9_run_key"],
        "discount": discount,
        "thresholds": thresholds,
        "m9_check": m9_check,
        "reference": {
            "real_alarms": reference["real_alarms"],
            "detection": reference["detection"],
            "wis": baseline_wis,
        },
        "candidates": candidates,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    cluster_summary = pd.read_csv(PROJECT_ROOT / config["paths"]["cluster_summary"])
    alerts = alerts_table(primary_rows, cluster_summary)
    metrics = {
        "wis_calibration": primary["wis"]["calibration"],
        "wis_calibration_reference": baseline_wis["calibration"],
        **{
            f"alarms_{split}_cusum_per_month": values["cusum_per_month"]
            for split, values in primary["real_alarms"].items()
        },
        **{
            f"detection_{scenario}_{rule}": result[rule]["detected"]
            for scenario, result in primary["detection"].items()
            for rule in RULES
        },
    }
    record = build_run_record(
        config=config,
        run_name=settings["run_name"],
        stage=STAGE,
        target="weekly_cluster_counts",
        split="calibration",
        view="complete_weeks",
        features=["cluster_id", "week", "weekly_total", "reweighted memory"],
        parameters={
            "plan": PLAN,
            "m9_run_key": m11["m9_run_key"],
            "excess_threshold": settings["excess_threshold"],
            "weight_draws": settings["weight_draws"],
            "decision": decision["rule"],
            "adopted": decision["adopted"],
        },
        metrics=metrics,
        artifacts=[
            path.relative_to(PROJECT_ROOT).as_posix() for path in paths.values()
        ],
    )
    json.dumps([report, record])
    report_dir.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(paths["alerts"], index=False)
    primary_runs.to_csv(paths["detection"], index=False)
    write_json(paths["results"], report)
    save_run_record(record, record_file)
    print(
        json.dumps({"reference": report["reference"], "wis": primary["wis"]}, indent=2)
    )
    print(json.dumps(decision, indent=2))
    print(f"Report: {report_dir.relative_to(PROJECT_ROOT)}")


def run_validation(
    config: dict[str, Any],
    panel: dict[str, np.ndarray],
    alpha: np.ndarray,
    m9_settings: dict[str, Any],
    report_dir: Path,
) -> None:
    """The single 2025-H1 report of the primary candidate, already consulted."""
    design = read_json(report_dir / DESIGN_FILES["results"])
    settings = design["settings"]
    m11 = config["persistent_change"]
    seed = config["experiment"]["seed"]
    paths = {name: report_dir / file for name, file in VALIDATION_FILES.items()}
    check_new_outputs(list(paths.values()))
    weights_alpha = weight_draws(alpha, settings["weight_draws"], seed)
    shares = ReweightedShares(
        weights_alpha, m9_settings["share"]["discount"], settings["excess_threshold"]
    )
    rows = add_alarms(
        windowed_rows(panel, alpha, shares), design["thresholds"], m11["cusum_k"]
    )
    cluster_summary = pd.read_csv(PROJECT_ROOT / config["paths"]["cluster_summary"])
    alerts = alerts_table(rows, cluster_summary)
    saved = pd.read_csv(report_dir / DESIGN_FILES["alerts"], parse_dates=["week"])
    keys = ["week", "cluster_id", "cusum_alarm", "weekly_alarm"]
    design_alerts = alerts.loc[alerts["split"].isin(DESIGN_SPLITS), keys]
    if not design_alerts.reset_index(drop=True).equals(saved[keys]):
        raise ValueError("The fit + calibration alerts differ from the design run.")
    report = {
        "stage": STAGE,
        "plan": PLAN,
        "split": "validation",
        "already_consulted": True,
        "git_commit": git_commit(),
        "thresholds": design["thresholds"],
        "decision": design["decision"],
        "real_alarms": alarm_summary(rows),
    }
    alerts.loc[alerts["split"] == "validation"].to_csv(paths["alerts"], index=False)
    write_json(paths["results"], report)
    print(json.dumps(report["real_alarms"], indent=2))


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    config = load_experiment_config()
    seed = config["experiment"]["seed"]
    set_seed(seed)
    m11 = config["persistent_change"]
    m9_settings = load_settings("weekly_counts", m11["m9_model"])
    frame, _, _, _ = load_frozen_frame(config, m9_settings)
    panel = composition_panel(frame, m9_settings["clusters"])
    alpha = load_alpha(m9_settings, m9_run(m11, "artifact"), seed)
    report_dir = PROJECT_ROOT / config["persistent_change_farr"]["report_dir"]
    if args.validation:
        run_validation(config, panel, alpha, m9_settings, report_dir)
        return
    design = select_weeks(panel, np.isin(panel["split"], DESIGN_SPLITS))
    run_design(config, design, alpha, m9_settings, report_dir, started)


if __name__ == "__main__":
    main()
