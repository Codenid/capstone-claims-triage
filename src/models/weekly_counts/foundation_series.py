"""Zero-shot time-series foundation models for block C (models_plan.md §25.5).

Chronos-2 and TimesFM 3.0 forecast each cluster's count one week ahead from
the weeks before it, with the weekly total as a covariate known for the target
week, as M9 conditions on it. Nothing is trained. Consecutive complete weeks
are consecutive steps, as in M9, so the series get a regular weekly index.

TimesFM 3.0 only predicts deciles: its 50% interval interpolates the 20%-30%
and 70%-80% deciles, and it has no 95% interval, so M9 is compared with it on
the 50% and 80% intervals only (decision of 2026-10-06).
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.evaluation.experiment import PROJECT_ROOT, git_commit, load_experiment_config
from src.models.weekly_counts.challenge import WIDTHS
from src.models.weekly_counts.contracts import INTERVALS
from src.models.weekly_counts.metrics import interval_bounds, weighted_interval_score
from src.models.weekly_counts.reporting import file_sha256
from src.models.weekly_counts.run import load_frozen_frame
from src.models.weekly_counts.state_space import week_matrix

LABELS = {"p025": 0.025, "p10": 0.10, "p25": 0.25, "p50": 0.50, "p75": 0.75}
LABELS |= {"p90": 0.90, "p975": 0.975}
MODELS = {
    "chronos2": {
        "model_id": "chronos2_zero_shot_v1",
        "interval_widths": [50, 80, 95],
        "formula": "Chronos-2 zero-shot, weekly total as a known future covariate",
    },
    "timesfm3": {
        "model_id": "timesfm3_zero_shot_v1",
        "interval_widths": [50, 80],
        "formula": "TimesFM 3.0 zero-shot, weekly total as a past-future covariate; "
        "deciles only, 50% interval interpolated",
    },
}
KEY_COLUMNS = ["split", "week", "cluster_id", "complaint_count", "weekly_total"]
SMOKE_WEEKS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=tuple(MODELS))
    parser.add_argument("--smoke", action="store_true", help="Two weeks, no files.")
    return parser.parse_args()


def regular_weeks(count: int) -> pd.DatetimeIndex:
    """A gap-free weekly index: complete weeks are consecutive steps."""
    return pd.date_range("2000-01-03", periods=count, freq="W-MON")


def chronos_quantiles(
    pipeline: Any,
    counts: np.ndarray,
    totals: np.ndarray,
    target_total: float,
) -> dict[str, np.ndarray]:
    weeks, clusters = counts.shape
    index = regular_weeks(weeks + 1)
    context = pd.DataFrame(
        {
            "item_id": np.repeat(np.arange(clusters), weeks),
            "timestamp": np.tile(index[:-1], clusters),
            "target": counts.T.reshape(-1).astype(float),
            "weekly_total": np.tile(totals, clusters),
        }
    )
    future = pd.DataFrame(
        {
            "item_id": np.arange(clusters),
            "timestamp": index[-1],
            "weekly_total": target_total,
        }
    )
    output = pipeline.predict_df(
        context,
        future_df=future,
        prediction_length=1,
        quantile_levels=list(LABELS.values()),
    ).sort_values("item_id")
    quantiles = {
        label: output[str(level)].to_numpy(dtype=float)
        for label, level in LABELS.items()
    }
    quantiles["mean"] = output["predictions"].to_numpy(dtype=float)
    return quantiles


def timesfm_quantiles(
    model: Any,
    counts: np.ndarray,
    totals: np.ndarray,
    target_total: float,
) -> dict[str, np.ndarray]:
    covariate = np.append(totals, target_total)
    outputs = list(
        model.predict_batch(
            [counts[:, cluster].astype(float) for cluster in range(counts.shape[1])],
            horizon=1,
            past_future_covariates=[covariate] * counts.shape[1],
            return_quantiles=True,
        )
    )
    deciles = np.vstack([output.quantiles[0] for output in outputs])
    return decile_quantiles(deciles) | {
        "mean": np.array([output.forecast[0] for output in outputs], dtype=float)
    }


def decile_quantiles(deciles: np.ndarray) -> dict[str, np.ndarray]:
    """M9's quantile columns from the 10%..90% deciles; no 2.5% or 97.5%."""
    levels = np.linspace(0.1, 0.9, 9)

    def at(level: float) -> np.ndarray:
        return np.array([np.interp(level, levels, row) for row in deciles])

    missing = np.full(len(deciles), np.nan)
    return {
        "p025": missing,
        "p10": deciles[:, 0],
        "p25": at(0.25),
        "p50": deciles[:, 4],
        "p75": at(0.75),
        "p90": deciles[:, 8],
        "p975": missing,
    }


def prediction_rows(
    target: pd.DataFrame,
    quantiles: dict[str, np.ndarray],
) -> pd.DataFrame:
    """M9's prediction columns; counts cannot be negative, so quantiles stop at 0."""
    rows = target[KEY_COLUMNS].reset_index(drop=True)
    for label in ("mean", *LABELS):
        rows[f"model_{label}"] = np.maximum(quantiles[label], 0.0)
    total = rows["weekly_total"].to_numpy(dtype=float)
    available = [
        f"model_{label}" for label in LABELS if np.isfinite(quantiles[label]).all()
    ]
    # Coarse: the share of predicted quantiles above the weekly total.
    above = rows[available].to_numpy() > total[:, None]
    rows["model_impossible_probability"] = above.mean(axis=1)
    return rows


def calibration_metrics(
    predictions: pd.DataFrame,
    widths: list[int],
) -> dict[str, Any]:
    """WIS, errors and coverage on the intervals the model provides."""
    alphas = {WIDTHS[width] for width in widths}
    intervals = tuple(item for item in INTERVALS if item[0] in alphas)
    observed = predictions["complaint_count"].to_numpy(dtype=float)
    median = predictions["model_p50"].to_numpy(dtype=float)
    bounds = interval_bounds(predictions, "model", intervals)
    errors = median - observed
    metrics: dict[str, Any] = {
        "wis": float(weighted_interval_score(observed, median, bounds).mean()),
        "mae": float(np.abs(errors).mean()),
        "wape": float(np.abs(errors).sum() / observed.sum()),
        "relative_bias": float(errors.sum() / observed.sum()),
        "coverage_95": None,
    }
    for alpha, lower, upper in bounds:
        inside = (observed >= lower) & (observed <= upper)
        metrics[f"coverage_{round((1 - alpha) * 100)}"] = float(inside.mean())
    return metrics


def load_model(name: str, settings: dict[str, Any]) -> Any:
    path = str(Path(settings[f"{name}_path"]).expanduser())
    if name == "chronos2":
        from chronos import Chronos2Pipeline

        return Chronos2Pipeline.from_pretrained(path, device_map="cpu")
    from timesfm3.torch.timesfm3_forecaster import TimesFM3Forecaster

    return TimesFM3Forecaster.from_pretrained(path, device="cpu")


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = load_experiment_config()
    settings = config["foundation_series"]
    spec = MODELS[args.model]
    model_settings = yaml.safe_load(
        (PROJECT_ROOT / settings["reference_config"]).read_text(encoding="utf-8")
    )
    panel, _, input_sha256, source_dvc_hash = load_frozen_frame(config, model_settings)
    clusters = model_settings["clusters"]
    calibration = panel.loc[panel["split"] == "calibration"]
    weeks = np.sort(calibration["week"].unique())
    if args.smoke:
        weeks = weeks[:SMOKE_WEEKS]

    model = load_model(args.model, settings)
    forecast = chronos_quantiles if args.model == "chronos2" else timesfm_quantiles
    parts = []
    for week in weeks:
        counts, totals = week_matrix(panel.loc[panel["week"] < week], clusters)
        target = calibration.loc[calibration["week"] == week].sort_values("cluster_id")
        total = float(target["weekly_total"].iloc[0])
        parts.append(prediction_rows(target, forecast(model, counts, totals, total)))
        label = pd.Timestamp(week).date()
        print(f"{label}: {len(counts)} weeks of context", flush=True)
    predictions = pd.concat(parts, ignore_index=True)
    metrics = calibration_metrics(predictions, spec["interval_widths"])
    print(json.dumps(metrics))
    if args.smoke:
        print("Smoke run finished; nothing was saved.")
        return

    run_key = (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S.%fZ')}-full-"
        f"{file_sha256(PROJECT_ROOT / 'configs/modeling.yaml')[:8]}"
    )
    report_root = PROJECT_ROOT / "reports/modeling/weekly_counts" / spec["model_id"]
    report_dir = report_root / run_key
    report_dir.mkdir(parents=True)
    predictions.to_csv(report_dir / "predictions.csv", index=False)
    report = {
        "stage": "M9 block C",
        "plan": "models_plan.md §25.5",
        "run_key": run_key,
        "run_mode": "full",
        "model_id": spec["model_id"],
        "formula": spec["formula"],
        "interval_widths": spec["interval_widths"],
        "git_commit": git_commit(),
        "weekly_counts_sha256": input_sha256,
        "source_dvc_hash": source_dvc_hash,
        "weights": Path(settings[f"{args.model}_path"]).name,
        "trained": False,
        "validation_used_for_selection": False,
        "evaluated_splits": ["calibration"],
        "backtest": {"calibration": {"model": metrics}},
        "resources": {"elapsed_seconds": time.perf_counter() - started},
    }
    (report_dir / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Report: {report_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
