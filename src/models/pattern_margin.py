"""PB-24 check 1: how much of each pattern sits on a boundary (§32.6).

Theory 4 of the second Theorizer question makes the boundary mass of a
cluster the control variable of its alarm fragility. This script measures it
on the frozen partition without changing anything: a Gaussian mixture with
one component per pattern, fitted on a fixed sample of the fit UMAP
coordinates and started at the k-means centres, gives each complaint a
probability per pattern; the margin between the first and second probability
says how firmly it belongs. A geometric margin from the k-means distances is
reported next to it. The output is one table per split and pattern.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from src.evaluation.experiment import PROJECT_ROOT, git_commit, load_experiment_config
from src.models.semantic_space import fixed_sample_positions

PLAN = "models_plan.md §32.6"
REPORT = "reports/modeling/patterns/boundary_mass.json"
SPLITS = ("fit", "calibration", "validation")
SAMPLE_ROWS = 200_000
BATCH_ROWS = 100_000
PROBABILITY_MARGIN = 0.2
PROBABILITY_FLOOR = 0.6
DISTANCE_MARGIN = 0.1


# ------------------------------------------------------------ pure rules
def probability_margins(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top probability and the gap to the second, per row."""
    if probabilities.shape[1] < 2:
        raise ValueError("Margins need at least two patterns.")
    top_two = -np.partition(-probabilities, 1, axis=1)[:, :2]
    return top_two[:, 0], top_two[:, 0] - top_two[:, 1]


def distance_margins(values: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """(d2 - d1) / d1 with d1, d2 the two nearest centre distances, per row."""
    squared = (
        (values**2).sum(axis=1, keepdims=True)
        - 2 * values @ centres.T
        + (centres**2).sum(axis=1)[None, :]
    )
    nearest = np.sqrt(np.maximum(np.partition(squared, 1, axis=1)[:, :2], 0.0))
    return (nearest[:, 1] - nearest[:, 0]) / np.maximum(nearest[:, 0], 1e-12)


def boundary_table(
    labels: np.ndarray,
    top: np.ndarray,
    gap: np.ndarray,
    geometric: np.ndarray,
    clusters: int,
) -> list[dict[str, Any]]:
    """Per pattern: size, mean margins and the fractions below the thresholds."""
    rows = []
    for cluster in range(clusters):
        member = labels == cluster
        size = int(member.sum())
        if not size:
            rows.append({"cluster_id": cluster, "rows": 0})
            continue
        rows.append(
            {
                "cluster_id": cluster,
                "rows": size,
                "mean_probability_gap": float(gap[member].mean()),
                "boundary_fraction_probability": float(
                    (gap[member] < PROBABILITY_MARGIN).mean()
                ),
                "uncertain_fraction": float((top[member] < PROBABILITY_FLOOR).mean()),
                "mean_distance_margin": float(geometric[member].mean()),
                "boundary_fraction_distance": float(
                    (geometric[member] < DISTANCE_MARGIN).mean()
                ),
            }
        )
    return rows


# -------------------------------------------------------------------- run
def patterns_dir(config: dict[str, Any]) -> Path:
    return PROJECT_ROOT / config["paths"]["weekly_patterns_artifacts"]


def fit_mixture(values: np.ndarray, centres: np.ndarray, seed: int) -> GaussianMixture:
    sample = values[fixed_sample_positions(len(values), SAMPLE_ROWS, seed)]
    mixture = GaussianMixture(
        n_components=len(centres),
        covariance_type="full",
        means_init=centres,
        random_state=seed,
        max_iter=200,
    )
    return mixture.fit(sample)


def batched_probabilities(mixture: GaussianMixture, values: np.ndarray) -> np.ndarray:
    parts = [
        mixture.predict_proba(values[start : start + BATCH_ROWS])
        for start in range(0, len(values), BATCH_ROWS)
    ]
    return np.concatenate(parts)


def evaluate_split(
    split: str,
    directory: Path,
    mixture: GaussianMixture,
    centres: np.ndarray,
) -> dict[str, Any]:
    values = np.load(directory / f"{split}_umap_15d.npy", mmap_mode="r")
    values = np.asarray(values, dtype=np.float64)
    labels = np.load(directory / f"{split}_labels.npy")
    probabilities = batched_probabilities(mixture, values)
    top, gap = probability_margins(probabilities)
    geometric = distance_margins(values, centres)
    mixture_labels = probabilities.argmax(axis=1)
    return {
        "rows": int(len(values)),
        "mixture_agreement_with_kmeans": float((mixture_labels == labels).mean()),
        "boundary_fraction_probability": float((gap < PROBABILITY_MARGIN).mean()),
        "uncertain_fraction": float((top < PROBABILITY_FLOOR).mean()),
        "boundary_fraction_distance": float((geometric < DISTANCE_MARGIN).mean()),
        "patterns": boundary_table(labels, top, gap, geometric, len(centres)),
    }


def main() -> None:
    started = time.perf_counter()
    config = load_experiment_config()
    seed = config["experiment"]["seed"]
    report_path = PROJECT_ROOT / REPORT
    if report_path.exists():
        raise FileExistsError(f"The boundary mass is written once: {report_path}")
    directory = patterns_dir(config)
    kmeans = joblib.load(directory / "kmeans.joblib")
    centres = np.asarray(kmeans.cluster_centers_, dtype=np.float64)
    fit_values = np.asarray(
        np.load(directory / "fit_umap_15d.npy", mmap_mode="r"), dtype=np.float64
    )
    mixture = fit_mixture(fit_values, centres, seed)
    del fit_values
    splits = {
        split: evaluate_split(split, directory, mixture, centres) for split in SPLITS
    }
    report = {
        "stage": "PB-24 check",
        "plan": PLAN,
        "git_commit": git_commit(),
        "mixture": {
            "components": int(len(centres)),
            "covariance": "full",
            "sample_rows": SAMPLE_ROWS,
            "converged": bool(mixture.converged_),
            "iterations": int(mixture.n_iter_),
        },
        "thresholds": {
            "probability_margin": PROBABILITY_MARGIN,
            "probability_floor": PROBABILITY_FLOOR,
            "distance_margin": DISTANCE_MARGIN,
        },
        "decision": "none: the partition is unchanged; this is information",
        "splits": splits,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for split, result in splits.items():
        summary = {k: v for k, v in result.items() if k != "patterns"}
        print(split, json.dumps(summary))
    table = pd.DataFrame(splits["calibration"]["patterns"])
    print(table.sort_values("boundary_fraction_probability", ascending=False).head(8))
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
