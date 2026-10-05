"""Choose the C-A discount and cap in fit (models_plan.md §25.5).

Each (discount, cap) pair is scored by its WIS on the fit weeks that NB-R4-H v3
also scores, with the per-cluster dispersion of the frozen v3 run as a plug-in
value. The pair with the lowest WIS fixes the C-A config; PyMC then
re-estimates the dispersion with that share.
"""

from __future__ import annotations

from itertools import product
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import yaml

from src.evaluation.experiment import PROJECT_ROOT, git_commit, load_experiment_config
from src.models.weekly_counts.contracts import INTERVALS
from src.models.weekly_counts.discounted_reference import row_discounted_shares
from src.models.weekly_counts.metrics import weighted_interval_score
from src.models.weekly_counts.rolling_reference import row_shares
from src.models.weekly_counts.run import load_frozen_frame

V3_WINDOW_WEEKS = 4


def quantile_level(label: str) -> float:
    """Level of a quantile column label: p025 -> 0.025, p90 -> 0.90."""
    return float(f"0.{label[1:]}")


def reference_alpha(run_dir: Path, clusters: int) -> np.ndarray:
    """Posterior mean of each cluster's dispersion in the frozen v3 run."""
    summary = pd.read_csv(run_dir / "posterior_summary.csv")
    means = summary.set_index("parameter")["mean"]
    return np.array([means[f"alpha[{cluster}]"] for cluster in range(clusters)])


def plug_in_wis(rows: pd.DataFrame, shares: np.ndarray, alpha: np.ndarray) -> float:
    """Mean WIS of NegativeBinomial(weekly_total * share, alpha[cluster])."""
    mu = rows["weekly_total"].to_numpy(dtype=float) * shares
    dispersion = alpha[rows["cluster_id"].to_numpy()]
    probability = dispersion / (dispersion + mu)
    labels = {"p50"} | {label for _, *bounds in INTERVALS for label in bounds}
    quantiles = {
        label: stats.nbinom.ppf(quantile_level(label), dispersion, probability)
        for label in labels
    }
    intervals = [
        (alpha_level, quantiles[lower], quantiles[upper])
        for alpha_level, lower, upper in INTERVALS
    ]
    observed = rows["complaint_count"].to_numpy(dtype=float)
    return float(weighted_interval_score(observed, quantiles["p50"], intervals).mean())


def scored_rows(panel: pd.DataFrame, warmup_weeks: int) -> np.ndarray:
    """Fit rows after the warm-up weeks: the rows NB-R4-H v3 was fitted on."""
    weeks = panel["week"].drop_duplicates().sort_values()
    after_warmup = panel["week"] >= weeks.iloc[warmup_weeks]
    return ((panel["split"] == "fit") & after_warmup).to_numpy()


def grid_scores(
    panel: pd.DataFrame,
    clusters: int,
    alpha: np.ndarray,
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    """WIS of every (discount, cap) pair, and of the v3 4-week window."""
    mask = scored_rows(panel, settings["warmup_weeks"])
    rows = panel.loc[mask]
    scores = []
    for discount, cap in product(settings["discounts"], settings["caps"]):
        shares = row_discounted_shares(panel, clusters, discount, cap)[mask]
        scores.append(
            {
                "discount": discount,
                "cap": cap,
                "memory_weeks": 1 / (1 - discount) if discount < 1 else None,
                "wis": plug_in_wis(rows, shares, alpha),
            }
        )
    v3_shares = row_shares(panel, clusters, V3_WINDOW_WEEKS)[mask]
    return scores, plug_in_wis(rows, v3_shares, alpha)


def main() -> None:
    config = load_experiment_config()
    settings = config["discount_grid"]
    report_path = PROJECT_ROOT / settings["report"]
    if report_path.exists():
        raise FileExistsError(f"C-A never overwrites its selection: {report_path}")
    model_settings = yaml.safe_load(
        (PROJECT_ROOT / settings["reference_config"]).read_text(encoding="utf-8")
    )
    panel, _, input_sha256, source_dvc_hash = load_frozen_frame(config, model_settings)
    clusters = model_settings["clusters"]
    alpha = reference_alpha(PROJECT_ROOT / settings["reference_run"], clusters)

    scores, v3_wis = grid_scores(panel, clusters, alpha, settings)
    scored = panel.loc[scored_rows(panel, settings["warmup_weeks"])]
    chosen = min(scores, key=lambda score: score["wis"])
    report = {
        "stage": "M9 C-A",
        "plan": "models_plan.md §25.5",
        "git_commit": git_commit(),
        "weekly_counts_sha256": input_sha256,
        "source_dvc_hash": source_dvc_hash,
        "reference_run": settings["reference_run"],
        "scored_split": "fit",
        "scored_weeks": int(scored["week"].nunique()),
        "validation_used": False,
        "dispersion": "posterior mean of alpha[c] in the v3 run",
        "v3_window_wis": v3_wis,
        "scores": scores,
        "chosen": chosen,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for score in sorted(scores, key=lambda score: score["wis"]):
        print(
            f"discount={score['discount']} cap={score['cap']}: WIS={score['wis']:.3f}"
        )
    print(f"v3 4-week window: WIS={v3_wis:.3f}")
    print(f"Chosen: discount={chosen['discount']} cap={chosen['cap']}")
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
