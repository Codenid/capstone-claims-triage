"""PB-24 check 2: the WIS of C-A's memory as a function of the discount (§32.6).

Theory 7 of the fourth Theorizer question predicts a U: too little memory
chases the noise, too much memory lags real drift. The §25.5 grid stopped at
0.5, the lowest value tried and the chosen one, so the left arm was never
measured. This script scores the same plug-in WIS on a wider grid, on the fit
weeks that chose the discount and, as information only, on calibration.
Nothing is chosen here: C-A keeps 0.5.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.evaluation.experiment import PROJECT_ROOT, git_commit, load_experiment_config
from src.models.weekly_counts.discount_grid import (
    plug_in_wis,
    reference_alpha,
    scored_rows,
)
from src.models.weekly_counts.discounted_reference import row_discounted_shares
from src.models.weekly_counts.run import load_frozen_frame

PLAN = "models_plan.md §32.6"
REPORT = (
    "reports/modeling/weekly_counts/nb_discounted_hierarchical_v1/discount_curve.json"
)
DISCOUNTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95)


def curve(
    panel: pd.DataFrame,
    clusters: int,
    alpha: np.ndarray,
    discounts: tuple[float, ...],
    mask: np.ndarray,
) -> list[dict[str, Any]]:
    """Plug-in WIS of the uncapped memory for each discount on the masked rows."""
    rows = panel.loc[mask]
    scores = []
    for discount in discounts:
        shares = row_discounted_shares(panel, clusters, discount, None)[mask]
        scores.append(
            {
                "discount": discount,
                "memory_weeks": 1 / (1 - discount),
                "wis": plug_in_wis(rows, shares, alpha),
            }
        )
    return scores


def shape(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Where the minimum sits and whether both arms of the U were observed."""
    best = min(scores, key=lambda s: s["wis"])
    ordered = sorted(scores, key=lambda s: s["discount"])
    position = ordered.index(best)
    return {
        "argmin_discount": best["discount"],
        "minimum_wis": best["wis"],
        "left_arm_observed": position > 0,
        "right_arm_observed": position < len(ordered) - 1,
        "u_shaped": 0 < position < len(ordered) - 1,
    }


def main() -> None:
    config = load_experiment_config()
    settings = config["discount_grid"]
    report_path = PROJECT_ROOT / REPORT
    if report_path.exists():
        raise FileExistsError(f"The discount curve is written once: {report_path}")
    model_settings = yaml.safe_load(
        (PROJECT_ROOT / settings["reference_config"]).read_text(encoding="utf-8")
    )
    panel, _, input_sha256, source_dvc_hash = load_frozen_frame(config, model_settings)
    clusters = model_settings["clusters"]
    alpha = reference_alpha(PROJECT_ROOT / settings["reference_run"], clusters)

    fit_mask = scored_rows(panel, settings["warmup_weeks"])
    calibration_mask = (panel["split"] == "calibration").to_numpy()
    splits = {
        "fit": curve(panel, clusters, alpha, DISCOUNTS, fit_mask),
        "calibration": curve(panel, clusters, alpha, DISCOUNTS, calibration_mask),
    }
    report = {
        "stage": "PB-24 check",
        "plan": PLAN,
        "git_commit": git_commit(),
        "weekly_counts_sha256": input_sha256,
        "source_dvc_hash": source_dvc_hash,
        "reference_run": settings["reference_run"],
        "dispersion": "posterior mean of alpha[c] in the v3 run, as in §25.5",
        "chosen_discount": 0.5,
        "decision": "none: C-A keeps its discount; this is information",
        "splits": splits,
        "shape": {split: shape(scores) for split, scores in splits.items()},
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for split, scores in splits.items():
        print(split)
        for score in scores:
            print(f"  discount={score['discount']:.2f}  WIS={score['wis']:.3f}")
        print(" ", json.dumps(report["shape"][split]))
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
