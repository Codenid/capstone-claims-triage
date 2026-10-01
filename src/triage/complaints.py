"""Signals of each complaint from the frozen models (models_plan.md §25)."""

from __future__ import annotations

import importlib
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.apply_taxonomy import CANONICAL_PRODUCT_COLUMN
from src.data.normalize_text import NORMALIZED_COLUMN
from src.evaluation.experiment import PROJECT_ROOT
from src.evaluation.final_confirmation import (
    CLASSIFIERS,
    frozen_selection,
    model_path,
    split_features,
)
from src.models.semantic_space import ID_COLUMN
from src.models.tfidf_models import DATE_COLUMN

# T4 left the triage (§23).
BINARY_TARGETS = ("T2", "T3")
TOP_ISSUES = 3


def pattern_assignments(
    rows: pd.DataFrame,
    config: dict[str, Any],
    split: str,
) -> pd.DataFrame:
    """M8B pattern, distance to its centroid and novelty of each complaint."""
    directory = PROJECT_ROOT / config["paths"]["weekly_patterns_artifacts"]
    assigned = pd.read_parquet(
        directory / f"{split}_assignments.parquet",
        columns=[ID_COLUMN, "cluster_id", "distance", "is_novel"],
    )
    assigned[ID_COLUMN] = assigned[ID_COLUMN].astype(str)
    keys = pd.DataFrame({ID_COLUMN: rows[ID_COLUMN].astype(str).to_numpy()})
    merged = keys.merge(assigned, on=ID_COLUMN, how="left", validate="one_to_one")
    if merged["cluster_id"].isna().to_numpy().any():
        raise ValueError("M12 found complaints without an M8B pattern.")
    thresholds = np.load(directory / "novelty_thresholds.npy")
    return merged.assign(
        cluster_id=merged["cluster_id"].astype("int64"),
        novelty_threshold=thresholds[merged["cluster_id"].to_numpy(dtype=np.int64)],
    )


def top_issues(scores: np.ndarray, classes: np.ndarray) -> list[list[dict[str, Any]]]:
    """The T1 classes with the highest scores, best first."""
    order = np.argsort(-scores, axis=1)[:, :TOP_ISSUES]
    return [
        [{"issue": str(classes[i]), "score": float(row[i])} for i in positions]
        for positions, row in zip(order, scores)
    ]


def classifier_signals(
    rows: pd.DataFrame,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """T1's top issues and the T2 and T3 probabilities against their thresholds."""
    selection = frozen_selection(config)
    features = split_features(rows, "validation", config)
    t1 = joblib.load(model_path(CLASSIFIERS["T1"], "T1", config))
    issues = top_issues(
        t1.decision_function(features[CLASSIFIERS["T1"]]), np.asarray(t1.classes_)
    )
    signals: list[dict[str, Any]] = [{"t1": row} for row in issues]
    for target in BINARY_TARGETS:
        representation = CLASSIFIERS[target]
        model = joblib.load(model_path(representation, target, config))
        probabilities = model.predict_proba(features[representation])[:, 1]
        threshold = float(selection[target]["metrics"][representation]["threshold"])
        for signal, probability in zip(signals, probabilities):
            signal[target.lower()] = {
                "probability": float(probability),
                "threshold": threshold,
                # As in M4, a probability equal to the threshold counts as likely.
                "likely": bool(probability >= threshold),
            }
    return signals


def neighbor_index(config: dict[str, Any]) -> tuple[Any, np.ndarray]:
    """M6's FAISS index of 120,000 fit embeddings and their complaint IDs."""
    directory = PROJECT_ROOT / config["paths"]["semantic_artifacts"]
    faiss = importlib.import_module("faiss")
    index = faiss.read_index(str(directory / "faiss_fit.index"))
    ids = pd.read_parquet(directory / "faiss_fit_ids.parquet")[ID_COLUMN]
    return index, ids.astype(str).to_numpy()


def optional_bool(value: Any) -> bool | None:
    return None if pd.isna(value) else bool(value)


def iso_date(value: Any) -> str:
    return str(pd.Timestamp(value).date())


def similar_complaints(
    index: Any,
    ids: np.ndarray,
    embeddings: np.ndarray,
    fit: pd.DataFrame,
    neighbors: int,
) -> list[list[dict[str, Any]]]:
    """Nearest fit complaints by inner product of the normalized embeddings."""
    scores, positions = index.search(
        np.ascontiguousarray(embeddings, dtype=np.float32), neighbors
    )
    info = fit.set_index(fit[ID_COLUMN].astype(str))
    similar = []
    for row_positions, row_scores in zip(positions, scores):
        row = []
        for position, score in zip(row_positions, row_scores):
            match = info.loc[ids[position]]
            row.append(
                {
                    "complaint_id": ids[position],
                    "received": iso_date(match[DATE_COLUMN]),
                    "product": str(match[CANONICAL_PRODUCT_COLUMN]),
                    "issue": str(match["T1"]),
                    "relief": optional_bool(match["T2"]),
                    "monetary": optional_bool(match["T3"]),
                    "similarity": float(score),
                }
            )
        similar.append(row)
    return similar


def complaint_card(
    row: pd.Series,
    signals: dict[str, Any],
    similar: list[dict[str, Any]],
    status: dict[str, Any],
    deadline: dict[str, Any],
    action: dict[str, str],
    text_characters: int,
) -> dict[str, Any]:
    """Everything the agent sees for one complaint, as plain JSON data."""
    return {
        "complaint_id": str(row[ID_COLUMN]),
        "received": iso_date(row[DATE_COLUMN]),
        "product": str(row[CANONICAL_PRODUCT_COLUMN]),
        "text": str(row[NORMALIZED_COLUMN])[:text_characters],
        **signals,
        "pattern": {
            "cluster_id": int(row["cluster_id"]),
            "distance": float(row["distance"]),
            "novelty_threshold": float(row["novelty_threshold"]),
            "is_novel": bool(row["is_novel"]),
        },
        "similar": similar,
        "pattern_status": status,
        "deadline": deadline,
        "action": action,
    }
