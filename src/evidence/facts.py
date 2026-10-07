"""Facts of the evidence card: five frozen signals for a few real complaints.

Decision of 2026-10-06 (docs/registro/models_plan.md §25): no automatic action.
Each complaint gets its likely issues from TabPFN-3.5 and from the linear
BGE + product classifier, its T2 and T3 probabilities, its nearest fit
complaints, its pattern with the week's alert status, and a deadline
simulated as if it were registered on `--as-of`. The complaints come from
calibration without shared text, chosen by a fixed rule and seed. Nothing is
retrained; the output is one JSON for reports/evidence_card/ficha.qmd.
"""

from __future__ import annotations

import argparse
from datetime import date
import importlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.apply_taxonomy import CANONICAL_PRODUCT_COLUMN
from src.evaluation.experiment import (
    PROJECT_ROOT,
    git_commit,
    load_experiment_config,
    set_seed,
)
from src.evaluation.final_confirmation import (
    frozen_selection,
    load_split,
    model_path,
    split_features,
)
from src.evidence.deadline import deadlines, load_holidays
from src.models.bge_sample import ID_COLUMN
from src.models.foundation_t1 import features, run_tabpfn
from src.models.representation_comparison import (
    attach_embedding_rows,
    load_embeddings,
    write_json,
)
from src.models.semantic_space import fixed_sample_positions
from src.models.tfidf_models import DATE_COLUMN, eligible_mask

NARRATIVE_COLUMN = "Consumer complaint narrative"
TOP_ISSUES = 3
BINARY_TARGETS = ("T2", "T3")
NEIGHBOR_SNIPPET = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=date.today(),
        help="Registration date assumed for the deadline (ISO).",
    )
    return parser.parse_args()


def choose_complaints(
    rows: pd.DataFrame,
    alerts: pd.DataFrame,
    settings: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    """One complaint of an alerted pattern, one novel, and one per top product."""
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    reasons: dict[int, str] = {}

    def pick(mask: pd.Series, reason: str) -> None:
        candidates = rows.index[mask.to_numpy() & ~rows.index.isin(chosen)]
        if len(candidates) == 0:
            raise ValueError(f"No complaint available for: {reason}")
        index = int(rng.choice(candidates.to_numpy()))
        chosen.append(index)
        reasons[index] = reason

    alerted_keys = set(zip(alerts["week"], alerts["cluster_id"]))
    keys = pd.Series(list(zip(rows["week"], rows["cluster_id"])), index=rows.index)
    pick(keys.isin(alerted_keys), "patrón con alerta de M11 esa semana")
    pick(rows["is_novel"].astype(bool), "marcado como novedoso por M8")
    counts = rows[CANONICAL_PRODUCT_COLUMN].value_counts()
    for product in counts.index[: settings["products"]]:
        pick(rows[CANONICAL_PRODUCT_COLUMN] == product, f"producto: {product}")
    selected = rows.loc[chosen].copy()
    selected["reason"] = [reasons[index] for index in chosen]
    return selected.reset_index(drop=True)


def narratives(ids: list[str], config: dict[str, Any]) -> pd.DataFrame:
    """Raw narrative, company and state of the given complaints."""
    table = pq.read_table(
        PROJECT_ROOT / config["paths"]["input_data"],
        columns=[ID_COLUMN, DATE_COLUMN, NARRATIVE_COLUMN, "Company", "State",
                 CANONICAL_PRODUCT_COLUMN, "T1", "T2", "T3"],
        filters=[(ID_COLUMN, "in", ids)],
    )
    frame = table.to_pandas().drop_duplicates(ID_COLUMN)
    return frame.set_index(frame[ID_COLUMN].astype(str))


def top_issues(
    probability: np.ndarray,
    classes: np.ndarray,
) -> list[list[dict[str, Any]]]:
    order = np.argsort(-probability, axis=1)[:, :TOP_ISSUES]
    return [
        [{"issue": str(classes[i]), "probability": float(row[i])} for i in positions]
        for positions, row in zip(order, probability)
    ]


def linear_issues(
    rows: pd.DataFrame,
    config: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    """Top issues of the frozen BGE + product classifier (§22)."""
    inputs = split_features(rows, "calibration", config)["bge_product"]
    model = joblib.load(model_path("bge_product", "T1", config))
    return top_issues(model.predict_proba(inputs), np.asarray(model.classes_))


def tabpfn_issues(
    rows: pd.DataFrame,
    fit: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
) -> list[list[dict[str, Any]]]:
    """Top issues of TabPFN-3.5 with the block B context (§25.4)."""
    settings = config["foundation_t1"]
    eligible = fit.loc[eligible_mask(fit, "T1", "complete")]
    positions = fixed_sample_positions(len(eligible), settings["context_rows"], seed)
    context = eligible.iloc[positions].reset_index(drop=True)
    manifest = PROJECT_ROOT / config["paths"]["bge_full_artifacts"] / "manifest.parquet"
    attach_embedding_rows({"fit": context}, manifest)
    numerical, codes, _ = features(
        {"fit": context, "calibration": rows}, config, settings["pca_components"]
    )
    probability, classes = run_tabpfn(
        numerical,
        codes,
        context["T1"].astype(str).to_numpy(),
        settings,
        settings["estimators"],
        seed,
        "calibration",
    )
    return top_issues(probability, classes)


def binary_signals(rows: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    """T2 and T3 probabilities of the frozen TF-IDF classifiers and their thresholds."""
    inputs = split_features(rows, "calibration", config)["tfidf_text"]
    selection = frozen_selection(config)
    signals: list[dict[str, Any]] = [{} for _ in range(len(rows))]
    for target in BINARY_TARGETS:
        model = joblib.load(model_path("tfidf_text", target, config))
        probability = model.predict_proba(inputs)[:, 1]
        threshold = float(selection[target]["metrics"]["tfidf_text"]["threshold"])
        for signal, value in zip(signals, probability):
            signal[target] = {
                "probability": float(value),
                "threshold": threshold,
                "likely": bool(value >= threshold),
            }
    return signals


def similar_complaints(
    rows: pd.DataFrame,
    config: dict[str, Any],
    neighbors: int,
) -> list[list[dict[str, Any]]]:
    """Nearest fit complaints in M6's exact FAISS index, with a narrative snippet."""
    directory = PROJECT_ROOT / config["paths"]["semantic_artifacts"]
    faiss = importlib.import_module("faiss")
    index = faiss.read_index(str(directory / "faiss_fit.index"))
    ids = pd.read_parquet(directory / "faiss_fit_ids.parquet")[ID_COLUMN]
    ids = ids.astype(str).to_numpy()
    embeddings = load_embeddings(
        PROJECT_ROOT / config["paths"]["bge_full_artifacts"], {"calibration": rows}
    )["calibration"]
    scores, positions = index.search(np.ascontiguousarray(embeddings), neighbors)
    found = narratives(sorted({ids[p] for p in positions.ravel()}), config)
    result = []
    for row_positions, row_scores in zip(positions, scores):
        matches = []
        for position, score in zip(row_positions, row_scores):
            match = found.loc[ids[position]]
            matches.append(
                {
                    "complaint_id": ids[position],
                    "received": str(pd.Timestamp(match[DATE_COLUMN]).date()),
                    "product": str(match[CANONICAL_PRODUCT_COLUMN]),
                    "issue": str(match["T1"]),
                    "relief": None if pd.isna(match["T2"]) else bool(match["T2"]),
                    "monetary": None if pd.isna(match["T3"]) else bool(match["T3"]),
                    "similarity": float(score),
                    "snippet": str(match[NARRATIVE_COLUMN])[:NEIGHBOR_SNIPPET],
                }
            )
        result.append(matches)
    return result


def pattern_info(
    rows: pd.DataFrame,
    alerts: pd.DataFrame,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """The complaint's M8 pattern, its words, and M11's status for that week."""
    summary = pd.read_csv(PROJECT_ROOT / config["paths"]["cluster_summary"])
    summary = summary.set_index("cluster_id")
    keyed = alerts.set_index(["week", "cluster_id"])
    result = []
    for _, row in rows.iterrows():
        cluster = int(row["cluster_id"])
        info = summary.loc[cluster]
        key = (row["week"], cluster)
        alert = keyed.loc[key] if key in keyed.index else None
        result.append(
            {
                "cluster_id": cluster,
                "main_product": str(info["main_product"]),
                "terms": str(info["representative_terms"]),
                "distance": float(row["distance"]),
                "p95_distance_fit": float(info["p95_distance"]),
                "is_novel": bool(row["is_novel"]),
                "week": str(pd.Timestamp(row["week"]).date()),
                "alert": None
                if alert is None
                else {
                    "cusum": bool(alert["cusum_alarm"]),
                    "weekly": bool(alert["weekly_alarm"]),
                    "observed": int(alert["observed"]),
                    "expected": float(alert["expected"]),
                },
            }
        )
    return result


def main() -> None:
    args = parse_args()
    config = load_experiment_config()
    settings = config["evidence_card"]
    seed = config["experiment"]["seed"]
    set_seed(seed)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    output = PROJECT_ROOT / settings["facts"]
    if output.exists():
        raise FileExistsError(f"The evidence card never overwrites: {output}")

    frames = load_split(config, "calibration")
    calibration = frames["calibration"]
    calibration = calibration.loc[calibration["no_shared_text"]]
    assignments = pd.read_parquet(
        PROJECT_ROOT / config["paths"]["weekly_patterns_artifacts"]
        / "calibration_assignments.parquet",
        columns=[ID_COLUMN, "cluster_id", "distance", "is_novel"],
    )
    assignments[ID_COLUMN] = assignments[ID_COLUMN].astype(str)
    calibration = calibration.assign(**{ID_COLUMN: calibration[ID_COLUMN].astype(str)})
    calibration = calibration.merge(assignments, on=ID_COLUMN, validate="one_to_one")
    alerts = pd.read_csv(
        PROJECT_ROOT / config["persistent_change"]["report_dir"] / "alerts.csv",
        parse_dates=["week"],
    )
    alerts = alerts.loc[(alerts["split"] == "calibration") & alerts["cusum_alarm"]]
    rows = choose_complaints(calibration, alerts, settings, seed)
    print(f"Chosen: {rows[ID_COLUMN].tolist()}", flush=True)

    linear = linear_issues(rows, config)
    tabpfn = tabpfn_issues(rows, frames["fit"], config, seed)
    binary = binary_signals(rows, config)
    similar = similar_complaints(rows, config, settings["neighbors"])
    patterns = pattern_info(rows, alerts, config)
    texts = narratives(rows[ID_COLUMN].tolist(), config)
    holidays = load_holidays()
    deadline = deadlines(args.as_of, holidays)

    cards = []
    for position, (_, row) in enumerate(rows.iterrows()):
        text = texts.loc[row[ID_COLUMN]]
        cards.append(
            {
                "complaint_id": row[ID_COLUMN],
                "reason": row["reason"],
                "received": str(pd.Timestamp(text[DATE_COLUMN]).date()),
                "product": str(row[CANONICAL_PRODUCT_COLUMN]),
                "company": str(text["Company"]),
                "state": str(text["State"]),
                "narrative": str(text[NARRATIVE_COLUMN]),
                "actual": {
                    "issue": str(row["T1"]),
                    "relief": None if pd.isna(row["T2"]) else bool(row["T2"]),
                    "monetary": None if pd.isna(row["T3"]) else bool(row["T3"]),
                },
                "issues": {"tabpfn": tabpfn[position], "linear": linear[position]},
                **binary[position],
                "similar": similar[position],
                "pattern": patterns[position],
                "deadline": deadline,
            }
        )
    report = {
        "plan": "docs/registro/models_plan.md §25",
        "git_commit": git_commit(),
        "seed": seed,
        "split": "calibration",
        "view": "no_shared_text",
        "selection_rule": {
            "alerted_pattern": 1,
            "novel": 1,
            "top_products": settings["products"],
        },
        "models": {
            "T1": ["tabpfn-v3.5 (block B context)", "bge_product (M5B)"],
            "T2": "tfidf_text (M5B)",
            "T3": "tfidf_text (M5B)",
            "neighbors": "M6 faiss_fit.index, 120,000 fit rows",
            "patterns": "M8 kmeans k=40, novelty p99",
            "alerts": config["persistent_change"]["report_dir"],
        },
        "deadline_assumption": (
            f"registro supuesto el {args.as_of.isoformat()}; feriados de "
            "configs/peru_holidays.yaml"
        ),
        "cards": cards,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(f"Facts: {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
