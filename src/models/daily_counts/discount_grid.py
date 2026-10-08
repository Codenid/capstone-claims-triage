"""Choose the daily discount in fit (models_plan.md §28.2).

Each discount of the pre-registered grid is scored by its WIS on the fit days
after the warm-up, with a plug-in negative binomial whose per-pattern
dispersion is the posterior mean of a reference D-A run (the pilot). The
lowest WIS fixes `share.discount` in every daily config; PyMC then
re-estimates the dispersion with that share. Nothing from calibration or
validation is read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import yaml

from src.evaluation.experiment import PROJECT_ROOT, git_commit, load_experiment_config
from src.models.daily_counts.contracts import INTERVALS, TOTAL_COLUMN
from src.models.daily_counts.data import load_frozen_panel
from src.models.daily_counts.references import (
    discounted_daily_shares,
    warm_rows,
    windowed_daily_shares,
)
from src.models.weekly_counts.metrics import weighted_interval_score


def quantile_level(label: str) -> float:
    return float(f"0.{label[1:]}")


def reference_alpha(run_dir: Path, clusters: int) -> np.ndarray:
    summary = pd.read_csv(run_dir / "posterior_summary.csv")
    means = summary.set_index("parameter")["mean"]
    return np.array([means[f"alpha[{cluster}]"] for cluster in range(clusters)])


def plug_in_wis(rows: pd.DataFrame, shares: np.ndarray, alpha: np.ndarray) -> float:
    """Mean WIS of NegativeBinomial(daily_total * share, alpha[pattern])."""
    mu = rows[TOTAL_COLUMN].to_numpy(dtype=float) * shares
    dispersion = alpha[rows["cluster_id"].to_numpy()]
    probability = dispersion / (dispersion + mu)
    labels = {"p50"} | {label for _, *bounds in INTERVALS for label in bounds}
    quantiles = {
        label: stats.nbinom.ppf(quantile_level(label), dispersion, probability)
        for label in labels
    }
    intervals = [
        (level, quantiles[lower], quantiles[upper]) for level, lower, upper in INTERVALS
    ]
    observed = rows["complaint_count"].to_numpy(dtype=float)
    return float(weighted_interval_score(observed, quantiles["p50"], intervals).mean())


def grid_scores(
    panel: pd.DataFrame,
    clusters: int,
    alpha: np.ndarray,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    mask = ((panel["split"] == "fit") & warm_rows(panel, settings["warmup_days"]))
    mask = mask.to_numpy()
    rows = panel.loc[mask]
    scores = [
        {
            "discount": discount,
            "memory_days": 1 / (1 - discount),
            "wis": plug_in_wis(
                rows, discounted_daily_shares(panel, clusters, discount)[mask], alpha
            ),
        }
        for discount in settings["discounts"]
    ]
    window = windowed_daily_shares(panel, clusters, settings["window_days"])[mask]
    return scores, plug_in_wis(rows, window, alpha)


def main() -> None:
    config = load_experiment_config()
    settings = config["daily_discount_grid"]
    report_path = PROJECT_ROOT / settings["report"]
    if report_path.exists():
        raise FileExistsError(f"The daily grid never overwrites: {report_path}")
    model_settings = yaml.safe_load(
        (PROJECT_ROOT / settings["reference_config"]).read_text(encoding="utf-8")
    )
    panel, input_sha256, source_dvc_hash = load_frozen_panel(config, model_settings)
    clusters = model_settings["clusters"]
    alpha = reference_alpha(PROJECT_ROOT / settings["reference_run"], clusters)
    scores, window_wis = grid_scores(panel, clusters, alpha, settings)
    chosen = min(scores, key=lambda score: score["wis"])
    report = {
        "stage": "M9D",
        "plan": "models_plan.md §28.2",
        "git_commit": git_commit(),
        "daily_counts_sha256": input_sha256,
        "source_dvc_hash": source_dvc_hash,
        "reference_run": settings["reference_run"],
        "dispersion": "posterior mean of alpha[c] in the reference run",
        "scored_split": "fit",
        "validation_used": False,
        "window_wis": {"window_days": settings["window_days"], "wis": window_wis},
        "scores": scores,
        "chosen": chosen,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for score in sorted(scores, key=lambda score: score["wis"]):
        print(f"discount={score['discount']}: WIS={score['wis']:.3f}")
    print(f"{settings['window_days']}-day window: WIS={window_wis:.3f}")
    print(f"Chosen: discount={chosen['discount']}")
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
