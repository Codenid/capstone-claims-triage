"""Weekly "rare mixture" signal from M10, models_plan.md §27.

For every complete week, the joint log score of the observed composition
under the frozen DM-R4 posterior is compared with the scores of compositions
simulated from the same predictive. The share of simulated scores at or below
the observed one is a posterior predictive p-value: it does not depend on the
weekly total nor on the 4-week memory, and a week is flagged when it falls
under the threshold fixed a priori (1 %). Nothing is refitted.
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
from scipy.special import logsumexp

from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    git_commit,
    load_experiment_config,
    save_run_record,
    set_seed,
)
from src.models.weekly_composition.data import composition_panel, windowed_weeks
from src.models.weekly_composition.scores import (
    dirichlet_multinomial_draws,
    dirichlet_multinomial_logpmf,
)
from src.models.weekly_counts.run import load_frozen_frame

STAGE = "M10S"
PLAN = "models_plan.md §27"
REPORTED_SPLITS = ("fit", "calibration", "validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Score a few weeks with few simulations and save nothing.",
    )
    return parser.parse_args()


def load_kappa(artifact_dir: Path, draws: int, seed: int) -> np.ndarray:
    """A fixed subsample of the posterior concentration draws."""
    idata = az.from_netcdf(artifact_dir / "posterior.nc")
    kappa = np.asarray(idata.posterior["kappa"]).reshape(-1)
    rng = np.random.default_rng(seed)
    return kappa[rng.choice(len(kappa), size=draws, replace=False)]


def mixture_log_score(counts: np.ndarray, concentration: np.ndarray) -> np.ndarray:
    """log p(counts | data) over the posterior draws, for one or many compositions.

    counts is (..., clusters) and concentration is (draws, clusters).
    """
    per_draw = dirichlet_multinomial_logpmf(counts[..., None, :], concentration)
    return logsumexp(per_draw, axis=-1) - np.log(len(concentration))


def week_surprise(
    counts: np.ndarray,
    total: int,
    concentration: np.ndarray,
    simulations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Observed log score and its posterior predictive p-value for one week."""
    observed = float(mixture_log_score(counts, concentration))
    totals = np.full((len(concentration), simulations), total)
    simulated = dirichlet_multinomial_draws(
        totals, np.repeat(concentration[:, None, :], simulations, axis=1), rng
    )
    scores = mixture_log_score(simulated.reshape(-1, counts.shape[-1]), concentration)
    p_value = (1 + int((scores <= observed).sum())) / (1 + len(scores))
    return observed, p_value


def largest_excess(
    counts: np.ndarray,
    total: int,
    recent_share: np.ndarray,
) -> tuple[int, float, float]:
    """The pattern whose observed share moved most against its expected share."""
    observed = counts / total
    difference = observed - recent_share
    cluster = int(np.argmax(np.abs(difference)))
    return cluster, float(observed[cluster]), float(recent_share[cluster])


def score_weeks(
    panel: dict[str, np.ndarray],
    kappa: np.ndarray,
    simulations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(len(panel["week"])):
        counts = panel["counts"][index]
        total = int(panel["weekly_total"][index])
        recent = panel["recent_share"][index]
        concentration = kappa[:, None] * recent[None, :]
        log_score, p_value = week_surprise(
            counts, total, concentration, simulations, rng
        )
        cluster, share, expected = largest_excess(counts, total, recent)
        rows.append(
            {
                "week": str(np.datetime64(panel["week"][index], "D")),
                "split": str(panel["split"][index]),
                "weekly_total": total,
                "log_score": log_score,
                "p_value": p_value,
                "largest_excess_cluster": cluster,
                "largest_excess_share": share,
                "largest_excess_expected_share": expected,
            }
        )
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame, alert_p_value: float) -> dict[str, Any]:
    flagged = table["p_value"] < alert_p_value
    summary: dict[str, Any] = {"splits": {}, "flagged_weeks": []}
    for split in REPORTED_SPLITS:
        rows = table["split"] == split
        if not rows.any():
            continue
        summary["splits"][split] = {
            "weeks": int(rows.sum()),
            "flagged": int((rows & flagged).sum()),
            "expected_flagged_if_calibrated": float(rows.sum() * alert_p_value),
            "median_p_value": float(table.loc[rows, "p_value"].median()),
        }
    for _, row in table.loc[flagged].iterrows():
        summary["flagged_weeks"].append(
            {
                "week": row["week"],
                "split": row["split"],
                "p_value": float(row["p_value"]),
                "weekly_total": int(row["weekly_total"]),
                "cluster": int(row["largest_excess_cluster"]),
                "share": float(row["largest_excess_share"]),
                "expected_share": float(row["largest_excess_expected_share"]),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = load_experiment_config()
    settings = config["mixture_signal"]
    seed = config["experiment"]["seed"]
    set_seed(seed)
    model_config = yaml.safe_load(
        (PROJECT_ROOT / "configs/weekly_composition" / f"{settings['m10_model']}.yaml")
        .read_text(encoding="utf-8")
    )
    frame, _, input_sha256, source_dvc_hash = load_frozen_frame(config, model_config)
    panel = windowed_weeks(composition_panel(frame, model_config["clusters"]))
    artifact_dir = (
        PROJECT_ROOT
        / "artifacts/models/weekly_composition"
        / settings["m10_model"]
        / settings["m10_run_key"]
    )
    draws = settings["kappa_draws"]
    simulations = settings["simulations_per_draw"]
    if args.smoke:
        keep = np.isin(panel["split"], ["calibration"])
        panel = {name: values[keep][:3] for name, values in panel.items()}
        draws, simulations = 10, 5
    kappa = load_kappa(artifact_dir, draws, seed)
    table = score_weeks(panel, kappa, simulations, seed)
    summary = summarize(table, settings["alert_p_value"])
    for split, values in summary["splits"].items():
        print(f"{split}: {values['flagged']} of {values['weeks']} weeks flagged")
    for week in summary["flagged_weeks"]:
        print(
            f"  {week['week']} ({week['split']}): p={week['p_value']:.4f}, "
            f"cluster {week['cluster']} at {week['share']:.1%} "
            f"vs {week['expected_share']:.1%} expected"
        )
    if args.smoke:
        print("Mixture signal smoke finished; nothing was saved.")
        return

    report_dir = PROJECT_ROOT / settings["report_dir"]
    if report_dir.exists():
        raise FileExistsError(f"The mixture signal never overwrites {report_dir}.")
    report_dir.mkdir(parents=True)
    table.to_csv(report_dir / "weekly.csv", index=False)
    report = {
        "stage": STAGE,
        "plan": PLAN,
        "git_commit": git_commit(),
        "seed": seed,
        "m10_model": settings["m10_model"],
        "m10_run_key": settings["m10_run_key"],
        "weekly_counts_sha256": input_sha256,
        "source_dvc_hash": source_dvc_hash,
        "kappa_draws": draws,
        "simulations_per_draw": simulations,
        "alert_p_value": settings["alert_p_value"],
        "validation_note": "2025-H1 was opened in §22; reported as already consulted.",
        **summary,
        "weeks": {
            row["week"]: {"split": row["split"], "p_value": float(row["p_value"])}
            for _, row in table.iterrows()
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (report_dir / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    save_offline_record(config, settings, report)
    print(f"Report: {report_dir.relative_to(PROJECT_ROOT)}")


def save_offline_record(
    config: dict[str, Any],
    settings: dict[str, Any],
    report: dict[str, Any],
) -> None:
    metrics = {
        f"{split}_{name}": float(value)
        for split, values in report["splits"].items()
        for name, value in values.items()
    }
    record = build_run_record(
        config=config,
        run_name=settings["run_name"],
        stage=STAGE,
        target="A1",
        split="calibration",
        view="40 patterns",
        features=["M10 joint log score", "posterior predictive p-value"],
        parameters={
            "plan": PLAN,
            "m10_run_key": settings["m10_run_key"],
            "kappa_draws": report["kappa_draws"],
            "simulations_per_draw": report["simulations_per_draw"],
            "alert_p_value": report["alert_p_value"],
        },
        metrics=metrics,
        artifacts=[f"{settings['report_dir']}/results.json"],
    )
    run_root = PROJECT_ROOT / config["paths"]["offline_runs"] / "m10"
    save_run_record(record, run_root / "mixture_signal" / "run.json")


if __name__ == "__main__":
    main()
