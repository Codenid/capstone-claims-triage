"""Fit the final semantic partition and build complete weekly counts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, cast

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer

from src.data.apply_taxonomy import CANONICAL_PRODUCT_COLUMN
from src.data.normalize_text import HASH_COLUMN, NORMALIZED_COLUMN
from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    load_experiment_config,
    save_run_record,
    set_seed,
)
from src.models.bge_full import load_embedding_array
from src.models.semantic_space import ID_COLUMN, load_dvc_hash, maximum_rss_gib
from src.models.tfidf_models import DATE_COLUMN

SPLITS = ("fit", "calibration", "validation")


def config_fingerprint(config: dict[str, Any]) -> str:
    relevant = {
        "weekly_patterns": config["weekly_patterns"],
        "splits": config["evaluation"]["splits"],
    }
    encoded = json.dumps(relevant, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_frozen_hash(name: str, actual: str, expected: str | None) -> None:
    if not expected:
        raise ValueError(f"Set weekly_patterns.{name} before running M8B.")
    if actual != expected:
        raise ValueError(f"M8B {name} does not match the frozen configuration.")


def save_state(path: Path, state: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_state(
    path: Path,
    source_hash: str,
    fingerprint: str,
) -> dict[str, Any]:
    expected = {
        "source_dvc_hash": source_hash,
        "config_sha256": fingerprint,
    }
    if not path.exists():
        state = {
            **expected,
            "transform_progress": {split: 0 for split in SPLITS},
            "status": "running",
            "resume_count": 0,
        }
        save_state(path, state)
        return state
    state = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"M8B checkpoint does not match current {key}.")
    state["resume_count"] += 1
    save_state(path, state)
    return state


def open_memmap(path: Path, rows: int, dimensions: int) -> np.memmap:
    if path.exists():
        values = cast(np.memmap, np.load(path, mmap_mode="r+"))
        if values.shape != (rows, dimensions) or values.dtype != np.float32:
            raise ValueError(f"M8B staging array is invalid: {path}")
        return values
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(rows, dimensions),
    )


def transform_split(
    split: str,
    source: Any,
    pca: Any,
    reducer: Any,
    output_path: Path,
    state: dict[str, Any],
    state_path: Path,
    batch_rows: int,
    output_dimensions: int,
) -> np.ndarray:
    progress = state["transform_progress"][split]
    if progress < 0 or progress > len(source):
        raise ValueError(f"M8B checkpoint progress is invalid for {split}.")
    if progress and not output_path.exists():
        raise ValueError(f"M8B checkpoint references a missing {split} array.")
    output = open_memmap(output_path, len(source), output_dimensions)
    for start in range(progress, len(source), batch_rows):
        stop = min(start + batch_rows, len(source))
        pca_values = pca.transform(source[start:stop]).astype(np.float32, copy=False)
        output[start:stop] = reducer.transform(pca_values).astype(np.float32, copy=False)
        output.flush()
        state["transform_progress"][split] = stop
        save_state(state_path, state)
    return np.load(output_path, mmap_mode="r")


def fit_final_model(
    fit_values: np.ndarray,
    clusters: int,
    n_init: int,
    batch_size: int,
    max_iter: int,
    seed: int,
) -> MiniBatchKMeans:
    model = MiniBatchKMeans(
        n_clusters=clusters,
        init="k-means++",
        n_init=n_init,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=seed,
    ).fit(fit_values)
    if len(np.unique(model.labels_)) != clusters:
        raise ValueError("M8B final model contains an empty fit cluster.")
    return model


def assign_clusters(
    model: MiniBatchKMeans,
    values: np.ndarray,
    batch_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.empty(len(values), dtype=np.int32)
    distances = np.empty(len(values), dtype=np.float32)
    for start in range(0, len(values), batch_rows):
        stop = min(start + batch_rows, len(values))
        batch = np.asarray(values[start:stop])
        batch_labels = model.predict(batch).astype(np.int32)
        labels[start:stop] = batch_labels
        distances[start:stop] = np.linalg.norm(
            batch - model.cluster_centers_[batch_labels],
            axis=1,
        ).astype(np.float32)
    return labels, distances


def novelty_thresholds(
    labels: np.ndarray,
    distances: np.ndarray,
    clusters: int,
    quantile: float,
) -> np.ndarray:
    thresholds = np.empty(clusters, dtype=np.float32)
    for cluster in range(clusters):
        cluster_distances = distances[labels == cluster]
        if not len(cluster_distances):
            raise ValueError(f"M8B final model has an empty cluster: {cluster}")
        thresholds[cluster] = np.quantile(cluster_distances, quantile)
    return thresholds


def apply_novelty(
    labels: np.ndarray,
    distances: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    return distances > thresholds[labels]


def split_week_index(period: dict[str, str]) -> pd.DatetimeIndex:
    start = cast(pd.Timestamp, pd.Timestamp(period["start"]))
    end = cast(pd.Timestamp, pd.Timestamp(period["end"]))
    if end <= start:
        raise ValueError("M8B split end must be after its start.")
    first_week = start.to_period("W-SUN").start_time
    last_date = cast(pd.Timestamp, end - pd.Timedelta(days=1))
    last_week = last_date.to_period("W-SUN").start_time
    return pd.date_range(first_week, last_week, freq="7D")


def complete_weekly_counts(
    assignments: pd.DataFrame,
    split_periods: dict[str, dict[str, str]],
    clusters: int,
) -> pd.DataFrame:
    frame = assignments.copy()
    actual_splits = set(frame["split"].unique())
    unexpected_splits = actual_splits - set(SPLITS)
    if unexpected_splits:
        raise ValueError("M8B assignments contain an unknown split.")

    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN])
    for split in SPLITS:
        period = split_periods[split]
        start = pd.Timestamp(period["start"])
        end = pd.Timestamp(period["end"])
        dates = frame.loc[frame["split"] == split, DATE_COLUMN]
        if ((dates < start) | (dates >= end)).any():
            raise ValueError(f"M8B found a date outside the {split} period.")

    frame["week"] = frame[DATE_COLUMN].dt.to_period("W-SUN").dt.start_time
    grouped = (
        frame.groupby(["split", "week", "cluster_id"], observed=True)
        .agg(
            complaint_count=(ID_COLUMN, "size"),
            unique_text_count=(HASH_COLUMN, "nunique"),
            novel_count=("is_novel", "sum"),
        )
        .reset_index()
    )

    grids = []
    for split in SPLITS:
        period = split_periods[split]
        start = pd.Timestamp(period["start"])
        end = pd.Timestamp(period["end"])
        index = pd.MultiIndex.from_product(
            [[split], split_week_index(period), range(clusters)],
            names=["split", "week", "cluster_id"],
        )
        complete = (
            grouped.loc[grouped["split"] == split]
            .set_index(["split", "week", "cluster_id"])
            .reindex(index, fill_value=0)
            .reset_index()
        )
        complete["is_complete_week"] = (
            (complete["week"] >= start)
            & (complete["week"] + pd.Timedelta(days=7) <= end)
        )
        grids.append(complete)

    result = pd.concat(grids, ignore_index=True)
    count_columns = ("complaint_count", "unique_text_count", "novel_count")
    result[list(count_columns)] = result[list(count_columns)].astype(np.int64)
    result["weekly_total"] = result.groupby(
        ["split", "week"],
        observed=True,
    )["complaint_count"].transform("sum")
    result["proportion"] = np.divide(
        result["complaint_count"],
        result["weekly_total"],
        out=np.zeros(len(result), dtype=float),
        where=result["weekly_total"].to_numpy() != 0,
    )
    return result.sort_values(["split", "week", "cluster_id"]).reset_index(drop=True)


def class_tfidf_terms(
    texts: list[str],
    labels: np.ndarray,
    top_terms: int,
    max_features: int,
) -> dict[int, str]:
    if len(texts) != len(labels):
        raise ValueError("M8B texts and cluster labels do not align.")
    frame = pd.DataFrame({"text": texts, "cluster_id": labels})
    clusters = sorted(frame["cluster_id"].unique())
    documents = (
        frame.groupby("cluster_id", observed=True)["text"]
        .agg(" ".join)
        .reindex(clusters)
        .tolist()
    )
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
    )
    counts = vectorizer.fit_transform(documents).astype(np.float64).tocsr()
    class_lengths = np.asarray(counts.sum(axis=1)).ravel()
    if (class_lengths == 0).any():
        raise ValueError("M8B found a cluster without usable representative terms.")
    term_totals = np.asarray(counts.sum(axis=0)).ravel()
    average_class_length = float(class_lengths.mean())
    inverse_document_frequency = np.log1p(average_class_length / term_totals)
    scores = (
        counts.multiply(1.0 / class_lengths[:, None])
        .multiply(inverse_document_frequency)
        .tocsr()
    )
    vocabulary = vectorizer.get_feature_names_out()

    result = {}
    for row_index, cluster in enumerate(clusters):
        cluster_id = int(cast(Any, cluster))
        row = scores.getrow(row_index)
        order = np.argsort(row.data)[-top_terms:][::-1]
        result[cluster_id] = ", ".join(vocabulary[row.indices[order]])
    return result


def representative_terms(
    input_path: Path,
    fit_assignments: pd.DataFrame,
    rows_per_cluster: int,
    seed: int,
    top_terms: int,
    max_features: int,
) -> dict[int, str]:
    samples = []
    for cluster, group in fit_assignments.groupby("cluster_id", observed=True):
        cluster_id = int(cast(Any, cluster))
        count = min(rows_per_cluster, len(group))
        samples.append(group.sample(n=count, random_state=seed + cluster_id))
    selected = pd.concat(samples, ignore_index=True)
    ids = selected[ID_COLUMN].astype(str).tolist()
    table = pq.read_table(
        input_path,
        columns=[ID_COLUMN, NORMALIZED_COLUMN],
        filters=[(ID_COLUMN, "in", ids)],
    ).to_pandas()
    text_by_id = table.drop_duplicates(ID_COLUMN).set_index(ID_COLUMN)[NORMALIZED_COLUMN]
    selected["text"] = text_by_id.reindex(selected[ID_COLUMN]).to_numpy()
    if selected["text"].isna().any():
        raise ValueError("M8B could not recover representative narratives.")

    return class_tfidf_terms(
        selected["text"].astype(str).tolist(),
        selected["cluster_id"].to_numpy(dtype=np.int32),
        top_terms,
        max_features,
    )


def cluster_summary(
    fit_assignments: pd.DataFrame,
    terms: dict[int, str],
) -> pd.DataFrame:
    rows = []
    for cluster, group in fit_assignments.groupby("cluster_id", observed=True):
        cluster_id = int(cast(Any, cluster))
        hash_counts = group[HASH_COLUMN].value_counts()
        product_counts = group[CANONICAL_PRODUCT_COLUMN].value_counts()
        rows.append(
            {
                "cluster_id": cluster_id,
                "fit_rows": len(group),
                "unique_text_fraction": group[HASH_COLUMN].nunique() / len(group),
                "dominant_template_fraction": hash_counts.iloc[0] / len(group),
                "main_product": str(product_counts.index[0]),
                "main_product_fraction": product_counts.iloc[0] / len(group),
                "median_distance": group["distance"].median(),
                "p95_distance": group["distance"].quantile(0.95),
                "representative_terms": terms[cluster_id],
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def save_plots(
    summary: pd.DataFrame,
    weekly: pd.DataFrame,
    size_path: Path,
    weekly_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    ordered = summary.sort_values("fit_rows", ascending=False)
    axis.bar(ordered["cluster_id"].astype(str), ordered["fit_rows"])
    axis.set(
        title=f"Tamaño final de los {len(summary)} patrones",
        xlabel="Cluster",
        ylabel="Reclamos de ajuste",
    )
    axis.tick_params(axis="x", labelrotation=90)
    figure.tight_layout()
    figure.savefig(size_path, dpi=160)
    plt.close(figure)

    complete = weekly.loc[weekly["is_complete_week"]].copy()
    pivot = complete.pivot_table(
        index="cluster_id",
        columns="week",
        values="proportion",
        fill_value=0,
    )
    figure, axis = plt.subplots(figsize=(15, 7))
    image = axis.imshow(pivot, aspect="auto", cmap="viridis")
    axis.set(
        title="Proporción semanal por patrón",
        xlabel="Semana",
        ylabel="Cluster",
    )
    figure.colorbar(image, ax=axis, label="Proporción")
    figure.tight_layout()
    figure.savefig(weekly_path, dpi=160)
    plt.close(figure)


def save_offline_record(config: dict[str, Any], report: dict[str, Any]) -> None:
    record = build_run_record(
        config=config,
        run_name="m8b-weekly-patterns",
        stage="M8B",
        target="A1",
        split="fit_calibration_validation",
        view="complete_eligible_narratives",
        features=["BGE", "PCA 256D", "UMAP 15D"],
        parameters={
            "clusters": config["weekly_patterns"]["clusters"],
            "novelty_quantile": config["weekly_patterns"]["novelty_quantile"],
            "bge_full_dvc_hash": config["weekly_patterns"]["source_dvc_hash"],
            "semantic_dvc_hash": config["weekly_patterns"]["semantic_dvc_hash"],
            "clustering_dvc_hash": config["weekly_patterns"][
                "clustering_dvc_hash"
            ],
            "model_artifact": config["paths"]["weekly_patterns_artifacts"],
        },
        metrics={
            "fit_novel_fraction": report["assignment"]["fit"]["novel_fraction"],
            "calibration_novel_fraction": report["assignment"]["calibration"][
                "novel_fraction"
            ],
            "validation_novel_fraction": report["assignment"]["validation"][
                "novel_fraction"
            ],
            "largest_cluster_fraction": report["fit_clusters"][
                "largest_cluster_fraction"
            ],
            "weekly_rows": float(report["weekly_rows"]),
            "elapsed_seconds": report["resources"]["elapsed_seconds"],
        },
        artifacts=[
            config["paths"]["weekly_patterns_report"],
            config["paths"]["weekly_counts"],
            config["paths"]["cluster_summary"],
            config["paths"]["weekly_cluster_size_plot"],
            config["paths"]["weekly_patterns_plot"],
        ],
    )
    path = (
        PROJECT_ROOT
        / config["paths"]["offline_runs"]
        / "m8b"
        / "weekly_patterns"
        / "run.json"
    )
    save_run_record(record, path)


def main() -> None:
    started = time.perf_counter()
    config = load_experiment_config()
    settings = config["weekly_patterns"]
    seed = config["experiment"]["seed"]
    set_seed(seed)

    if not settings.get("source_dvc_hash"):
        raise ValueError(
            "Set weekly_patterns.source_dvc_hash after M8A is added to DVC."
        )
    source_hash = load_dvc_hash(
        PROJECT_ROOT / config["paths"]["bge_full_artifacts_dvc"]
    )
    semantic_hash = load_dvc_hash(
        PROJECT_ROOT / config["paths"]["semantic_artifacts_dvc"]
    )
    clustering_hash = load_dvc_hash(
        PROJECT_ROOT / config["paths"]["clustering_artifacts_dvc"]
    )
    require_frozen_hash("source_dvc_hash", source_hash, settings["source_dvc_hash"])
    require_frozen_hash(
        "semantic_dvc_hash",
        semantic_hash,
        settings["semantic_dvc_hash"],
    )
    require_frozen_hash(
        "clustering_dvc_hash",
        clustering_hash,
        settings["clustering_dvc_hash"],
    )

    source_dir = PROJECT_ROOT / config["paths"]["bge_full_artifacts"]
    semantic_dir = PROJECT_ROOT / config["paths"]["semantic_artifacts"]
    clustering_dir = PROJECT_ROOT / config["paths"]["clustering_artifacts"]
    output_dir = PROJECT_ROOT / config["paths"]["weekly_patterns_artifacts"]
    staging_dir = PROJECT_ROOT / config["paths"]["weekly_patterns_staging"]
    report_path = PROJECT_ROOT / config["paths"]["weekly_patterns_report"]
    weekly_path = PROJECT_ROOT / config["paths"]["weekly_counts"]
    summary_path = PROJECT_ROOT / config["paths"]["cluster_summary"]
    size_plot_path = PROJECT_ROOT / config["paths"]["weekly_cluster_size_plot"]
    weekly_plot_path = PROJECT_ROOT / config["paths"]["weekly_patterns_plot"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if (output_dir / "_SUCCESS").exists():
        print(f"M8B is already complete: {output_dir.relative_to(PROJECT_ROOT)}")
        return
    if output_dir.exists():
        raise ValueError("M8B final output exists without _SUCCESS.")
    staging_dir.mkdir(parents=True, exist_ok=True)

    state_path = staging_dir / "checkpoint.json"
    state = load_state(state_path, source_hash, config_fingerprint(config))
    pca = joblib.load(semantic_dir / "pca.joblib")
    reducer = joblib.load(clustering_dir / "umap_15d.joblib")
    transformed = {}
    for split in SPLITS:
        source = load_embedding_array(source_dir, split)
        transformed[split] = transform_split(
            split,
            source,
            pca,
            reducer,
            staging_dir / f"{split}_umap_15d.npy",
            state,
            state_path,
            settings["batch_rows"],
            settings["umap_dimensions"],
        )
        source.close()
    del pca, reducer

    model = fit_final_model(
        transformed["fit"],
        settings["clusters"],
        settings["n_init"],
        settings["kmeans_batch_size"],
        settings["max_iter"],
        seed,
    )
    joblib.dump(model, staging_dir / "kmeans.joblib", compress=3)

    labels = {}
    distances = {}
    for split in SPLITS:
        labels[split], distances[split] = assign_clusters(
            model,
            transformed[split],
            settings["batch_rows"],
        )
        np.save(staging_dir / f"{split}_labels.npy", labels[split])
        np.save(staging_dir / f"{split}_distances.npy", distances[split])
    del transformed
    thresholds = novelty_thresholds(
        labels["fit"],
        distances["fit"],
        settings["clusters"],
        settings["novelty_quantile"],
    )
    np.save(staging_dir / "novelty_thresholds.npy", thresholds)

    assignment_frames = []
    assignment_report = {}
    for split in SPLITS:
        frame = pd.read_parquet(
            source_dir / "manifest.parquet",
            filters=[("split", "=", split)],
        ).sort_values("embedding_row")
        if len(frame) != len(labels[split]):
            raise ValueError(f"M8B manifest does not align for {split}.")
        frame["cluster_id"] = labels[split]
        frame["distance"] = distances[split]
        frame["is_novel"] = apply_novelty(
            labels[split],
            distances[split],
            thresholds,
        )
        frame.to_parquet(staging_dir / f"{split}_assignments.parquet", index=False)
        assignment_frames.append(frame)
        assignment_report[split] = {
            "rows": len(frame),
            "novel_rows": int(frame["is_novel"].sum()),
            "novel_fraction": float(frame["is_novel"].mean()),
        }

    assignments = pd.concat(assignment_frames, ignore_index=True)
    weekly = complete_weekly_counts(
        assignments,
        config["evaluation"]["splits"],
        settings["clusters"],
    )
    weekly.to_csv(weekly_path, index=False)

    fit_assignments = assignment_frames[0]
    terms = representative_terms(
        PROJECT_ROOT / config["paths"]["input_data"],
        fit_assignments,
        settings["term_rows_per_cluster"],
        seed,
        settings["top_terms"],
        settings["term_max_features"],
    )
    summary = cluster_summary(fit_assignments, terms)
    summary.to_csv(summary_path, index=False)
    save_plots(summary, weekly, size_plot_path, weekly_plot_path)

    fit_sizes = summary["fit_rows"].to_numpy()
    report = {
        "stage": "M8B",
        "seed": seed,
        "source_dvc_hash": source_hash,
        "semantic_dvc_hash": semantic_hash,
        "clustering_dvc_hash": clustering_hash,
        "clusters": settings["clusters"],
        "fit_only_training": True,
        "novelty_quantile": settings["novelty_quantile"],
        "assignment": assignment_report,
        "fit_clusters": {
            "minimum_rows": int(fit_sizes.min()),
            "median_rows": float(np.median(fit_sizes)),
            "maximum_rows": int(fit_sizes.max()),
            "largest_cluster_fraction": float(fit_sizes.max() / fit_sizes.sum()),
        },
        "weekly_rows": len(weekly),
        "complete_weekly_rows": int(weekly["is_complete_week"].sum()),
        "partial_weeks": int(
            (~weekly["is_complete_week"])
            .groupby([weekly["split"], weekly["week"]])
            .any()
            .sum()
        ),
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_rss_gib": maximum_rss_gib(),
            "resume_count": state["resume_count"],
        },
    }
    (staging_dir / "metadata.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    save_offline_record(config, report)
    state["status"] = "complete"
    save_state(state_path, state)
    (staging_dir / "_SUCCESS").write_text("complete\n", encoding="utf-8")
    staging_dir.replace(output_dir)

    print(f"Clusters: {settings['clusters']}")
    print(f"Weekly rows: {len(weekly):,}")
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")
    print(f"Artifacts: {output_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
