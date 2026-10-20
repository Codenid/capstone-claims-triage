"""C-PAT-INC v1: the pattern of a new complaint by its fit neighbors (§31.1).

The frozen M8B projection (PCA -> UMAP -> k-means) gives a different pattern
to the same complaint depending on the batch it travels in. This candidate
replaces the projection, for new complaints only, by a similarity-weighted
vote of the k nearest fit complaints in the frozen M6 FAISS index, each with
its frozen M8B label. Nothing is fitted: search, vote, and compare with the
frozen labels of calibration. The adoption rule is written in §31.1.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import time
from typing import Any

import joblib
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
from src.models.bge_full import load_embedding_array
from src.models.semantic_space import fixed_sample_positions
from src.models.weekly_patterns import assign_clusters
from src.pipeline.frozen import FrozenArtifacts, frozen_artifacts

STAGE = "C-PAT-INC"
PLAN = "models_plan.md §31.1"
ID_COLUMN = "Complaint ID"
DECIDING_SPLIT = "calibration"
CONSULTED_SPLIT = "validation"


# ------------------------------------------------------------ pure rules
def vote(
    similarities: np.ndarray,
    neighbor_labels: np.ndarray,
    k: int,
    clusters: int,
) -> np.ndarray:
    """Similarity-weighted vote over the first k neighbors; ties go to the nearest.

    `similarities` and `neighbor_labels` are (rows, >= k) arrays sorted from
    the most to the least similar neighbor.
    """
    if k < 1 or k > similarities.shape[1]:
        raise ValueError("k must be between 1 and the number of neighbors.")
    rows = np.arange(len(similarities))[:, None]
    scores = np.zeros((len(similarities), clusters), dtype=np.float64)
    np.add.at(scores, (rows, neighbor_labels[:, :k]), similarities[:, :k])
    nearest = neighbor_labels[:, 0]
    scores[np.arange(len(scores)), nearest] += 1e-9
    return scores.argmax(axis=1).astype(np.int32)


def novelty_threshold(top1_similarity: np.ndarray, quantile: float) -> float:
    """The similarity below which a complaint is novel."""
    return float(np.quantile(top1_similarity, quantile))


def agreement(labels: np.ndarray, reference: np.ndarray) -> float:
    if len(labels) != len(reference):
        raise ValueError("Label arrays must have the same length.")
    return float(np.mean(labels == reference))


def core_and_border(
    candidate: np.ndarray,
    frozen: np.ndarray,
    reprojected: np.ndarray,
) -> dict[str, Any]:
    """Agreement of the candidate where the frozen projection agrees with itself."""
    stable = reprojected == frozen
    return {
        "stable_fraction": float(stable.mean()),
        "agreement_stable": agreement(candidate[stable], frozen[stable]),
        "agreement_border": agreement(candidate[~stable], frozen[~stable])
        if (~stable).any()
        else None,
    }


# ------------------------------------------------------------ frozen data
def index_labels(frozen: FrozenArtifacts) -> np.ndarray:
    """Frozen M8B label of every row of the FAISS fit index, in index order."""
    ids = pd.read_parquet(frozen.semantic_dir / "faiss_fit_ids.parquet")[ID_COLUMN]
    fit = pd.read_parquet(
        frozen.patterns_dir / "fit_assignments.parquet",
        columns=[ID_COLUMN, "cluster_id"],
    )
    lookup = fit.assign(**{ID_COLUMN: fit[ID_COLUMN].astype(str)}).set_index(ID_COLUMN)
    labels = lookup.loc[ids.astype(str).to_numpy(), "cluster_id"].to_numpy()
    return labels.astype(np.int32)


def load_index(frozen: FrozenArtifacts, threads: int) -> Any:
    faiss = importlib.import_module("faiss")
    faiss.omp_set_num_threads(threads)
    return faiss.read_index(str(frozen.semantic_dir / "faiss_fit.index"))


def search(index: Any, values: np.ndarray, k: int, batch_rows: int) -> tuple:
    similarities = np.empty((len(values), k), dtype=np.float32)
    positions = np.empty((len(values), k), dtype=np.int64)
    for start in range(0, len(values), batch_rows):
        stop = min(start + batch_rows, len(values))
        batch = np.ascontiguousarray(values[start:stop], dtype=np.float32)
        similarities[start:stop], positions[start:stop] = index.search(batch, k)
    return similarities, positions


def split_embeddings(
    frozen: FrozenArtifacts, split: str
) -> tuple[np.ndarray, pd.DataFrame]:
    """Embeddings of a split aligned with its frozen assignments."""
    assignments = pd.read_parquet(
        frozen.patterns_dir / f"{split}_assignments.parquet",
        columns=[ID_COLUMN, "embedding_row", "cluster_id"],
    )
    values = materialize(load_embedding_array(frozen.bge_full_dir, split))
    rows = assignments["embedding_row"].to_numpy()
    return values[rows], assignments


def materialize(array: Any) -> np.ndarray:
    """Copy an EmbeddingArray into memory before closing its memory maps.

    A slice of a single-file array is a view of the map; reading it after
    close() returns garbage (job 57788 of 2026-10-20).
    """
    values = np.array(array[0 : len(array)], dtype=np.float32, copy=True)
    array.close()
    return values


def fit_rows_outside_index(frozen: FrozenArtifacts, rows: int, seed: int) -> np.ndarray:
    """Embeddings of a fixed sample of fit complaints that are not in the index."""
    ids = pd.read_parquet(frozen.semantic_dir / "faiss_fit_ids.parquet")[ID_COLUMN]
    fit = pd.read_parquet(
        frozen.patterns_dir / "fit_assignments.parquet",
        columns=[ID_COLUMN, "embedding_row"],
    )
    outside = fit.loc[~fit[ID_COLUMN].astype(str).isin(ids.astype(str).to_numpy())]
    chosen = outside.iloc[fixed_sample_positions(len(outside), rows, seed)]
    positions = np.sort(chosen["embedding_row"].to_numpy())
    array = load_embedding_array(frozen.bge_full_dir, "fit")
    values = np.stack([np.array(array[p : p + 1])[0] for p in positions])
    array.close()
    return values.astype(np.float32)


def reprojected_labels(
    values: np.ndarray,
    frozen: FrozenArtifacts,
    batch_rows: int,
    seed: int,
) -> np.ndarray:
    """Frozen PCA -> UMAP -> k-means on the same rows in a permuted order."""
    pca = joblib.load(frozen.semantic_dir / "pca.joblib")
    reducer = joblib.load(frozen.clustering_dir / "umap_15d.joblib")
    model = joblib.load(frozen.patterns_dir / "kmeans.joblib")
    order = np.random.default_rng(seed).permutation(len(values))
    reduced = np.empty((len(values), reducer.n_components), dtype=np.float32)
    for start in range(0, len(values), batch_rows):
        rows = order[start : start + batch_rows]
        reduced[rows] = reducer.transform(pca.transform(values[rows]))
    labels, _ = assign_clusters(model, reduced, batch_rows)
    return labels.astype(np.int32)


# -------------------------------------------------------------- evaluate
def evaluate_split(
    split: str,
    frozen: FrozenArtifacts,
    index: Any,
    labels_of_index: np.ndarray,
    threshold: float,
    settings: dict[str, Any],
    clusters: int,
    seed: int,
) -> dict[str, Any]:
    values, assignments = split_embeddings(frozen, split)
    reference = assignments["cluster_id"].to_numpy().astype(np.int32)
    ks = sorted({settings["neighbors"], *settings["secondary_neighbors"]})
    similarities, positions = search(
        index, values, max(ks), settings["search_batch_rows"]
    )
    neighbor_labels = labels_of_index[positions]
    candidate = vote(similarities, neighbor_labels, settings["neighbors"], clusters)

    order = np.random.default_rng(seed).permutation(len(values))
    permuted = vote(
        similarities[order], neighbor_labels[order], settings["neighbors"], clusters
    )
    deterministic = bool(np.array_equal(permuted[np.argsort(order)], candidate))

    reprojected = reprojected_labels(values, frozen, settings["umap_batch_rows"], seed)
    return {
        "rows": int(len(values)),
        "agreement_knn": agreement(candidate, reference),
        "agreement_reprojected": agreement(reprojected, reference),
        "agreement_by_k": {
            str(k): agreement(
                vote(similarities, neighbor_labels, k, clusters), reference
            )
            for k in ks
        },
        "novel_fraction": float(np.mean(similarities[:, 0] < threshold)),
        "deterministic_under_permutation": deterministic,
        **core_and_border(candidate, reference, reprojected),
    }


def decide(calibration: dict[str, Any]) -> dict[str, Any]:
    adopted = calibration["agreement_knn"] >= calibration["agreement_reprojected"]
    return {
        "rule": (
            "adopt if the k-NN agreement with the frozen calibration labels is at "
            "least the agreement of the permuted frozen projection"
        ),
        "agreement_knn": calibration["agreement_knn"],
        "agreement_reprojected": calibration["agreement_reprojected"],
        "adopted": bool(adopted),
    }


def main() -> None:
    started = time.perf_counter()
    config = load_experiment_config()
    settings = config["pattern_knn"]
    seed = config["experiment"]["seed"]
    set_seed(seed)
    frozen = frozen_artifacts(config)
    report_dir = PROJECT_ROOT / settings["report_dir"]
    if report_dir.exists():
        raise FileExistsError(f"C-PAT-INC never overwrites its report: {report_dir}")
    clusters = int(len(np.load(frozen.patterns_dir / "novelty_thresholds.npy")))
    threads = int(config["experiment"].get("threads", 8))

    index = load_index(frozen, threads)
    labels_of_index = index_labels(frozen)
    outside = fit_rows_outside_index(frozen, settings["novelty_sample_rows"], seed)
    top1, _ = search(index, outside, 1, settings["search_batch_rows"])
    threshold = novelty_threshold(top1[:, 0], settings["novelty_quantile"])

    results = {
        split: evaluate_split(
            split, frozen, index, labels_of_index, threshold, settings, clusters, seed
        )
        for split in (DECIDING_SPLIT, CONSULTED_SPLIT)
    }
    results[CONSULTED_SPLIT]["already_consulted"] = True
    decision = decide(results[DECIDING_SPLIT])
    report = {
        "stage": STAGE,
        "plan": PLAN,
        "git_commit": git_commit(),
        "seed": seed,
        "settings": settings,
        "index_rows": int(index.ntotal),
        "clusters": clusters,
        "novelty_threshold": threshold,
        "splits": results,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_dir.mkdir(parents=True)
    (report_dir / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    record = build_run_record(
        config=config,
        run_name=settings["run_name"],
        stage=STAGE,
        target="A1",
        split=DECIDING_SPLIT,
        view="complete",
        features=["BGE-large, frozen M6 FAISS fit index, frozen M8B labels"],
        parameters={
            "plan": PLAN,
            "neighbors": settings["neighbors"],
            "novelty_quantile": settings["novelty_quantile"],
            "decision": json.dumps(decision["rule"]),
        },
        metrics={
            "agreement_knn": results[DECIDING_SPLIT]["agreement_knn"],
            "agreement_reprojected": results[DECIDING_SPLIT]["agreement_reprojected"],
            "novel_fraction": results[DECIDING_SPLIT]["novel_fraction"],
            "novelty_threshold": threshold,
        },
        artifacts=[f"{settings['report_dir']}/results.json"],
    )
    run_root = PROJECT_ROOT / config["paths"]["offline_runs"] / "pattern_knn"
    save_run_record(record, run_root / "run.json")
    print(json.dumps({"novelty_threshold": threshold}, indent=2))
    print(json.dumps(results, indent=2))
    print(json.dumps(decision, indent=2))
    print(f"Report: {Path(settings['report_dir'])}")


if __name__ == "__main__":
    main()
