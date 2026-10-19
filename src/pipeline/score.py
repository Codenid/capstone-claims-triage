"""Score a new table with the frozen components (models_plan.md §29).

Per complaint: the three most likely issues (TabPFN-3.5 with the block B
context, or the linear BGE + product classifier when the TabPFN weights are
not available), the T2 and T3 probabilities of the frozen TF-IDF models, the
pattern and its novelty flag, and the nearest fit complaints. Per week and
per day: the counts per pattern, the expectation and intervals of the frozen
M9 and D-D runs, the M11 excess, CUSUM and alarms with the frozen thresholds,
and the M10 rare-mixture p-value. Nothing is refitted.
"""

from __future__ import annotations

from datetime import UTC, datetime
import importlib
import json
import os
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
    save_run_record,
    set_seed,
)
from src.evaluation.final_confirmation import frozen_selection, load_split, model_path
from src.models.bge_sample import bge_feature_matrices
from src.models.daily_change import load_model as load_daily_model
from src.models.foundation_t1 import run_tabpfn
from src.models.persistent_change import load_alpha
from src.models.representation_comparison import attach_embedding_rows, load_embeddings
from src.models.semantic_space import fixed_sample_positions
from src.models.tfidf_models import eligible_mask
from src.models.weekly_composition.mixture_signal import load_kappa
from src.models.weekly_patterns import apply_novelty, assign_clusters
from src.pipeline import counts as count_rules
from src.pipeline.contract import (
    CANONICAL_PRODUCT_COLUMN,
    DATE_COLUMN,
    ID_COLUMN,
    NORMALIZED_COLUMN,
    prepare_table,
)
from src.pipeline.frozen import (
    FrozenArtifacts,
    frozen_artifacts,
    load_yaml,
    manifest,
    missing_artifacts,
)
from src.pipeline.signals import retraining_signals

STAGE = "PIPELINE"
PLAN = "models_plan.md §29"
TOP_ISSUES = 3
NEIGHBORS = 5
BATCH_ROWS = 20_000


# ------------------------------------------------------------- embeddings
def load_encoder(config: dict[str, Any], device: str):
    """The frozen BGE model from the local Hugging Face cache (no download)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sentence_transformers = importlib.import_module("sentence_transformers")
    settings = config["bge"]
    model = sentence_transformers.SentenceTransformer(
        settings["model"],
        revision=settings["revision"],
        device=device,
        local_files_only=True,
    )
    model.max_seq_length = settings["max_sequence_length"]
    return model


def embed(texts: list[str], model: Any, batch_size: int) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


# --------------------------------------------------------------- patterns
def assign_patterns(embeddings: np.ndarray, frozen: FrozenArtifacts) -> pd.DataFrame:
    """PCA -> UMAP -> nearest k-means center, with the frozen novelty threshold."""
    pca = joblib.load(frozen.semantic_dir / "pca.joblib")
    reducer = joblib.load(frozen.clustering_dir / "umap_15d.joblib")
    model = joblib.load(frozen.patterns_dir / "kmeans.joblib")
    thresholds = np.load(frozen.patterns_dir / "novelty_thresholds.npy")
    reduced = reducer.transform(pca.transform(embeddings))
    labels, distances = assign_clusters(
        model, np.asarray(reduced, dtype=np.float32), BATCH_ROWS
    )
    return pd.DataFrame(
        {
            "cluster_id": labels.astype(int),
            "distance": distances,
            "is_novel": apply_novelty(labels, distances, thresholds),
        }
    )


def nearest_fit_complaints(
    embeddings: np.ndarray, frozen: FrozenArtifacts, neighbors: int
) -> tuple[list[list[str]], np.ndarray]:
    faiss = importlib.import_module("faiss")
    index = faiss.read_index(str(frozen.semantic_dir / "faiss_fit.index"))
    ids = pd.read_parquet(frozen.semantic_dir / "faiss_fit_ids.parquet")[ID_COLUMN]
    ids = ids.astype(str).to_numpy()
    scores, positions = index.search(np.ascontiguousarray(embeddings), neighbors)
    return [[str(ids[p]) for p in row] for row in positions], scores


# ------------------------------------------------------------ classifiers
def top_issue_columns(probability: np.ndarray, classes: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-probability, axis=1)[:, :TOP_ISSUES]
    columns: dict[str, Any] = {}
    for rank in range(TOP_ISSUES):
        columns[f"issue_top{rank + 1}"] = classes[order[:, rank]]
        columns[f"issue_top{rank + 1}_probability"] = probability[
            np.arange(len(probability)), order[:, rank]
        ]
    return columns


def binary_columns(
    prepared: pd.DataFrame, frozen: FrozenArtifacts, config: dict[str, Any]
) -> dict[str, Any]:
    """T2 and T3 of the frozen TF-IDF classifiers with their calibration thresholds."""
    vectorizer = joblib.load(frozen.tfidf_dir / "vectorizer.joblib")
    text = vectorizer.transform(prepared[NORMALIZED_COLUMN])
    selection = frozen_selection(config)
    columns: dict[str, Any] = {}
    for target in ("T2", "T3"):
        model = joblib.load(model_path("tfidf_text", target, config))
        probability = model.predict_proba(text)[:, 1]
        threshold = float(selection[target]["metrics"]["tfidf_text"]["threshold"])
        columns[f"{target.lower()}_probability"] = probability
        columns[f"{target.lower()}_likely"] = probability >= threshold
    return columns


def linear_issue_columns(
    prepared: pd.DataFrame,
    embeddings: np.ndarray,
    frozen: FrozenArtifacts,
    config: dict[str, Any],
) -> dict[str, Any]:
    encoder = joblib.load(frozen.tfidf_dir / "product_encoder.joblib")
    features = bge_feature_matrices({"new": embeddings}, {"new": prepared}, encoder)[
        "new"
    ]
    model = joblib.load(model_path("bge_product", "T1", config))
    return top_issue_columns(model.predict_proba(features), np.asarray(model.classes_))


def tabpfn_issue_columns(
    prepared: pd.DataFrame,
    embeddings: np.ndarray,
    frozen: FrozenArtifacts,
    config: dict[str, Any],
    seed: int,
    device: str,
) -> dict[str, Any]:
    """TabPFN-3.5 with the same fixed block B context, on fresh embeddings."""
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    settings = config["foundation_t1"]
    fit = load_split(config, "calibration")["fit"]
    eligible = fit.loc[eligible_mask(fit, "T1", "complete")]
    positions = fixed_sample_positions(len(eligible), settings["context_rows"], seed)
    context = eligible.iloc[positions].reset_index(drop=True)
    attach_embedding_rows({"fit": context}, frozen.bge_full_dir / "manifest.parquet")
    context_embeddings = load_embeddings(frozen.bge_full_dir, {"fit": context})["fit"]
    pca = joblib.load(frozen.semantic_dir / "pca.joblib")
    encoder = joblib.load(frozen.tfidf_dir / "product_encoder.joblib")
    products = [str(product) for product in encoder.categories_[0]]
    components = settings["pca_components"]
    numerical = {
        "fit": pca.transform(context_embeddings)[:, :components].astype(np.float32),
        "query": pca.transform(embeddings)[:, :components].astype(np.float32),
    }
    codes = {}
    for split, frame in (("fit", context), ("query", prepared)):
        product = pd.Categorical(
            frame[CANONICAL_PRODUCT_COLUMN].astype(str), categories=products
        )
        # A product M4 never saw (or __UNKNOWN__) takes the first code; it is flagged.
        codes[split] = np.where(product.codes < 0, 0, product.codes).astype(np.int64)
    probability, classes = run_tabpfn(
        numerical,
        codes,
        context["T1"].astype(str).to_numpy(),
        settings,
        settings["estimators"],
        seed,
        "query",
        device,
    )
    return top_issue_columns(probability, classes)


# ------------------------------------------------------------------ counts
def frozen_weekly_history(frozen: FrozenArtifacts) -> pd.DataFrame:
    history = pd.read_csv(
        frozen.weekly_counts, usecols=("week", "cluster_id", "complaint_count")
    )
    return history.rename(columns={"week": "period"})


def frozen_daily_history(frozen: FrozenArtifacts) -> pd.DataFrame:
    history = pd.read_csv(
        frozen.daily_counts, usecols=("day", "cluster_id", "complaint_count")
    )
    return history.rename(columns={"day": "period"})


def weekly_outputs(
    complaints: pd.DataFrame,
    frozen: FrozenArtifacts,
    config: dict[str, Any],
    clusters: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    m11 = config["persistent_change"]
    m9_config = load_yaml(
        PROJECT_ROOT / "configs/weekly_counts" / f"{m11['m9_model']}.yaml"
    )
    thresholds = json.loads(frozen.m11_report.read_text(encoding="utf-8"))["thresholds"]
    alpha = load_alpha(m9_config, frozen.m9_artifact, seed)
    new = count_rules.period_counts(complaints, clusters, "week")
    panel = count_rules.extend_history(frozen_weekly_history(frozen), new, clusters)
    weekly = count_rules.weekly_predictive(
        panel,
        alpha,
        m9_config["share"],
        thresholds,
        m11["cusum_k"],
        np.random.default_rng(seed),
    )
    complete = new.set_index(["period", "cluster_id"])["is_complete"]
    weekly["is_complete"] = [
        bool(complete.get((period, cluster), True))
        for period, cluster in zip(weekly["period"], weekly["cluster_id"])
    ]
    signal = config["mixture_signal"]
    kappa = load_kappa(frozen.m10_artifact, signal["kappa_draws"], seed)
    mixture = count_rules.weekly_mixture(
        panel, kappa, signal["simulations_per_draw"], signal["alert_p_value"], seed
    )
    mixture["source"] = mixture["period"].map(
        panel.drop_duplicates("period").set_index("period")["source"]
    )
    return weekly, mixture


def daily_outputs(
    complaints: pd.DataFrame,
    frozen: FrozenArtifacts,
    config: dict[str, Any],
    clusters: int,
    seed: int,
) -> pd.DataFrame:
    alpha, model = load_daily_model(config["daily_change"], seed)
    new = count_rules.period_counts(complaints, clusters, "day")
    panel = count_rules.extend_history(frozen_daily_history(frozen), new, clusters)
    return count_rules.daily_predictive(
        panel, alpha, model, np.random.default_rng(seed)
    )


# -------------------------------------------------------------------- run
def score_table(
    frame: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    device: str = "cpu",
    neighbors: int = NEIGHBORS,
    skip_tabpfn: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    seed = config["experiment"]["seed"]
    set_seed(seed)
    frozen = frozen_artifacts(config)
    missing = [m for m in missing_artifacts(frozen) if "TabPFN weights" not in m]
    if missing:
        raise FileNotFoundError(f"Frozen artifacts missing (dvc pull): {missing}")
    clusters = int(
        load_yaml(
            PROJECT_ROOT / "configs/weekly_counts/nb_discounted_hierarchical_v1.yaml"
        )["clusters"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = prepare_table(frame)
    encoder = load_encoder(config, device)
    embeddings = embed(
        prepared[NORMALIZED_COLUMN].tolist(), encoder, config["bge"]["batch_size"]
    )

    complaints = prepared[
        [ID_COLUMN, DATE_COLUMN, "Product", CANONICAL_PRODUCT_COLUMN, "known_product"]
    ].copy()
    if "Issue canonical" in prepared:
        complaints["Issue canonical"] = prepared["Issue canonical"]
    patterns = assign_patterns(embeddings, frozen)
    for column in patterns:
        complaints[column] = patterns[column].to_numpy()
    for column, values in binary_columns(prepared, frozen, config).items():
        complaints[column] = values
    use_tabpfn = frozen.tabpfn_weights.exists() and not skip_tabpfn
    issues = (
        tabpfn_issue_columns(prepared, embeddings, frozen, config, seed, device)
        if use_tabpfn
        else linear_issue_columns(prepared, embeddings, frozen, config)
    )
    for column, values in issues.items():
        complaints[column] = values
    complaints["t1_model"] = "tabpfn-3.5" if use_tabpfn else "bge_product (linear)"
    neighbor_ids, similarity = nearest_fit_complaints(embeddings, frozen, neighbors)
    complaints["neighbors"] = [json.dumps(ids) for ids in neighbor_ids]
    complaints["neighbor_similarity_top1"] = similarity[:, 0]

    weekly, mixture = weekly_outputs(complaints, frozen, config, clusters, seed)
    daily = daily_outputs(complaints, frozen, config, clusters, seed)
    weekly_new = weekly.loc[weekly["source"] == "new"].reset_index(drop=True)
    daily_new = daily.loc[daily["source"] == "new"].reset_index(drop=True)
    mixture_new = mixture.loc[mixture["source"] == "new"].reset_index(drop=True)

    frozen_f1 = json.loads(frozen.foundation_report.read_text(encoding="utf-8"))[
        "metrics"
    ]["tabpfn"]["macro_f1"]
    signals = retraining_signals(complaints, weekly_new, mixture_new, frozen_f1)

    complaints.to_parquet(output_dir / "complaints.parquet", index=False)
    complaints.drop(columns=["neighbors"]).to_csv(
        output_dir / "complaints.csv", index=False
    )
    weekly_new.to_csv(output_dir / "weekly.csv", index=False)
    daily_new.to_csv(output_dir / "daily.csv", index=False)
    mixture_new.to_csv(output_dir / "mixture.csv", index=False)
    report = {
        "stage": STAGE,
        "plan": PLAN,
        "git_commit": git_commit(),
        "scored_at_utc": datetime.now(UTC).isoformat(),
        "rows": int(len(complaints)),
        "dates": {
            "first": str(complaints[DATE_COLUMN].min().date()),
            "last": str(complaints[DATE_COLUMN].max().date()),
        },
        "weeks": int(weekly_new["period"].nunique()),
        "days": int(daily_new["period"].nunique()),
        "t1_model": complaints["t1_model"].iloc[0],
        "device": device,
        "frozen": manifest(frozen, config),
        "signals": signals,
        "alerts": {
            "cusum_weeks": int(
                weekly_new.loc[weekly_new["cusum_alarm"], "period"].nunique()
            ),
            "cusum_rows": int(weekly_new["cusum_alarm"].sum()),
            "weekly_rule_rows": int(weekly_new["weekly_alarm"].sum()),
            "rare_mixture_weeks": int(
                mixture_new["rare_mixture"].fillna(False).astype(bool).sum()
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "score.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    record = build_run_record(
        config=config,
        run_name="pipeline-score",
        stage=STAGE,
        target="A1",
        split="new",
        view="score",
        features=["frozen components, no refit"],
        parameters={
            "plan": PLAN,
            "rows": report["rows"],
            "first_date": report["dates"]["first"],
            "last_date": report["dates"]["last"],
            "t1_model": report["t1_model"],
            "m9_run": report["frozen"]["m9_run"],
        },
        metrics={
            "novel_fraction": float(signals["novelty"]["novel_fraction"]),
            "cusum_alarm_rows": float(report["alerts"]["cusum_rows"]),
            "rare_mixture_weeks": float(report["alerts"]["rare_mixture_weeks"]),
            "elapsed_seconds": float(report["elapsed_seconds"]),
        },
        artifacts=[
            str(output_dir.relative_to(PROJECT_ROOT))
            if output_dir.is_relative_to(PROJECT_ROOT)
            else str(output_dir)
        ],
    )
    save_run_record(record, output_dir / "run.json")
    return report
