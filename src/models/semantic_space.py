"""Prepare the M6 semantic space from the frozen M5 BGE sample."""

from __future__ import annotations

import importlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA

from src.data.apply_taxonomy import CANONICAL_PRODUCT_COLUMN
from src.data.build_targets import COMPLETE_COLUMNS
from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    load_experiment_config,
    save_run_record,
    set_seed,
)

SPLITS = ("fit", "calibration", "validation")
ID_COLUMN = "Complaint ID"
NO_SHARED_COLUMN = "no_shared_text"


def load_dvc_hash(path: Path) -> str:
    pointer = yaml.safe_load(path.read_text(encoding="utf-8"))
    return str(pointer["outs"][0]["md5"])


def load_inputs(
    input_dir: Path,
    expected_rows: dict[str, int],
) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], dict[str, Any]]:
    manifest = pd.read_parquet(input_dir / "sample_manifest.parquet")
    if manifest[ID_COLUMN].duplicated().any():
        raise ValueError("M6 manifest contains duplicate complaint IDs.")

    frames = {}
    embeddings = {}
    for split in SPLITS:
        frame = manifest.loc[manifest["sample_split"] == split].reset_index(drop=True)
        if len(frame) != expected_rows[split]:
            raise ValueError(f"M6 manifest row count does not match {split}.")
        frames[split] = frame
        embeddings[split] = np.load(
            input_dir / f"{split}_embeddings.npy",
            mmap_mode="r",
        )

    metadata = json.loads(
        (input_dir / "metadata.json").read_text(encoding="utf-8")
    )
    return frames, embeddings, metadata


def validate_embeddings(
    embeddings: dict[str, np.ndarray],
    expected_rows: dict[str, int],
    dimensions: int,
    tolerance: float,
    batch_rows: int,
) -> dict[str, dict[str, Any]]:
    validation = {}
    for split in SPLITS:
        values = embeddings[split]
        if values.shape != (expected_rows[split], dimensions):
            raise ValueError(f"Unexpected embedding shape for {split}: {values.shape}")
        if values.dtype != np.float32:
            raise ValueError(f"M6 embeddings for {split} must use float32.")

        maximum_norm_error = 0.0
        for start in range(0, len(values), batch_rows):
            batch = np.asarray(values[start : start + batch_rows])
            if not np.isfinite(batch).all():
                raise ValueError(f"M6 embeddings for {split} contain non-finite values.")
            errors = np.abs(np.linalg.norm(batch, axis=1) - 1.0)
            maximum_norm_error = max(maximum_norm_error, float(errors.max()))
        if maximum_norm_error > tolerance:
            raise ValueError(f"M6 embeddings for {split} are not normalized.")

        validation[split] = {
            "rows": len(values),
            "dimensions": values.shape[1],
            "dtype": str(values.dtype),
            "maximum_norm_error": maximum_norm_error,
        }
    return validation


def fit_pca(values: np.ndarray, components: int, seed: int) -> PCA:
    model = PCA(
        n_components=components,
        svd_solver="randomized",
        random_state=seed,
    )
    model.fit(values)
    return model


def transform_to_npy(
    model: PCA,
    values: np.ndarray,
    output_path: Path,
    batch_rows: int,
) -> np.ndarray:
    transformed = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(values), model.n_components_),
    )
    for start in range(0, len(values), batch_rows):
        stop = min(start + batch_rows, len(values))
        transformed[start:stop] = model.transform(values[start:stop]).astype(
            np.float32,
            copy=False,
        )
    transformed.flush()
    del transformed
    return np.load(output_path, mmap_mode="r")


def fit_umap(
    fit_values: np.ndarray,
    settings: dict[str, Any],
    seed: int,
) -> tuple[Any, np.ndarray]:
    umap = importlib.import_module("umap")
    model = umap.UMAP(
        n_components=settings["n_components"],
        n_neighbors=settings["n_neighbors"],
        min_dist=settings["min_dist"],
        metric=settings["metric"],
        init=settings["init"],
        random_state=seed,
        transform_seed=seed,
        low_memory=True,
        n_jobs=1,
    )
    coordinates = model.fit_transform(fit_values).astype(np.float32)
    return model, coordinates


def fixed_sample_positions(rows: int, requested: int, seed: int) -> np.ndarray:
    if requested >= rows:
        return np.arange(rows)
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(rows, size=requested, replace=False))


def save_pca_plot(pca: PCA, checkpoints: list[int], output_path: Path) -> None:
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(np.arange(1, len(cumulative) + 1), cumulative)
    for checkpoint in checkpoints:
        if checkpoint <= len(cumulative):
            axis.scatter(checkpoint, cumulative[checkpoint - 1], s=25)
    axis.set(
        title="Varianza acumulada por componentes PCA",
        xlabel="Número de componentes",
        ylabel="Proporción de varianza acumulada",
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_umap_plot(
    frames: dict[str, pd.DataFrame],
    coordinates: dict[str, np.ndarray],
    rows_per_split: int,
    seed: int,
    sample_path: Path,
    plot_path: Path,
) -> int:
    samples = []
    for offset, split in enumerate(SPLITS):
        positions = fixed_sample_positions(
            len(frames[split]),
            rows_per_split,
            seed + offset,
        )
        samples.append(
            pd.DataFrame(
                {
                    ID_COLUMN: frames[split].iloc[positions][ID_COLUMN].astype(str),
                    "split": split,
                    "product": frames[split]
                    .iloc[positions][CANONICAL_PRODUCT_COLUMN]
                    .astype(str),
                    "umap_x": coordinates[split][positions, 0],
                    "umap_y": coordinates[split][positions, 1],
                }
            )
        )
    sample = pd.concat(samples, ignore_index=True)
    sample.to_parquet(sample_path, index=False)

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    for split, group in sample.groupby("split", sort=False):
        axes[0].scatter(
            group["umap_x"],
            group["umap_y"],
            s=2,
            alpha=0.35,
            label=split,
            rasterized=True,
        )
    axes[0].set_title("UMAP por periodo")
    axes[0].legend(markerscale=4)

    top_products = sample["product"].value_counts().head(10).index
    sample["product_plot"] = sample["product"].where(
        sample["product"].isin(top_products),
        "Other",
    )
    for product, group in sample.groupby("product_plot", sort=False):
        axes[1].scatter(
            group["umap_x"],
            group["umap_y"],
            s=2,
            alpha=0.3,
            label=product,
            rasterized=True,
        )
    axes[1].set_title("UMAP por producto")
    axes[1].legend(markerscale=4, fontsize=7, loc="best")
    for axis in axes:
        axis.set(xlabel="UMAP 1", ylabel="UMAP 2")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    return len(sample)


def build_faiss_index(values: np.ndarray, batch_rows: int, threads: int) -> Any:
    faiss = importlib.import_module("faiss")
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexFlatIP(values.shape[1])
    for start in range(0, len(values), batch_rows):
        batch = np.ascontiguousarray(values[start : start + batch_rows])
        index.add(batch)
    return index


def query_positions(frame: pd.DataFrame, requested: int, seed: int) -> np.ndarray:
    if NO_SHARED_COLUMN in frame:
        eligible = np.flatnonzero(frame[NO_SHARED_COLUMN].to_numpy(dtype=bool))
    else:
        eligible = np.arange(len(frame))
    selected = fixed_sample_positions(len(eligible), requested, seed)
    return eligible[selected]


def search_neighbor_sample(
    index: Any,
    fit_frame: pd.DataFrame,
    fit_embeddings: np.ndarray,
    query_frames: dict[str, pd.DataFrame],
    query_embeddings: dict[str, np.ndarray],
    settings: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    fit_ids = fit_frame[ID_COLUMN].astype(str).to_numpy()
    fit_products = fit_frame[CANONICAL_PRODUCT_COLUMN].astype(str).to_numpy()
    fit_t1 = fit_frame["T1"].astype(str).to_numpy()
    fit_t1_complete = fit_frame[COMPLETE_COLUMNS["T1"]].to_numpy(dtype=bool)
    neighbors = []
    metrics = {}

    for offset, split in enumerate(("calibration", "validation")):
        frame = query_frames[split]
        positions = query_positions(
            frame,
            settings["query_rows_per_split"],
            seed + offset,
        )
        queries = np.ascontiguousarray(query_embeddings[split][positions])
        scores, indices = index.search(queries, settings["neighbors"])

        exact_rows = min(settings["exact_check_rows_per_split"], len(queries))
        exact_scores = queries[:exact_rows] @ np.asarray(fit_embeddings).T
        expected = np.sort(exact_scores, axis=1)[:, -settings["neighbors"] :][:, ::-1]
        maximum_exact_error = float(
            np.max(np.abs(expected - scores[:exact_rows]))
        )

        query_ids = frame.iloc[positions][ID_COLUMN].astype(str).to_numpy()
        query_products = (
            frame.iloc[positions][CANONICAL_PRODUCT_COLUMN].astype(str).to_numpy()
        )
        query_t1 = frame.iloc[positions]["T1"].astype(str).to_numpy()
        query_t1_complete = (
            frame.iloc[positions][COMPLETE_COLUMNS["T1"]].to_numpy(dtype=bool)
        )
        top_one = indices[:, 0]
        same_t1_mask = query_t1_complete & fit_t1_complete[top_one]

        metrics[split] = {
            "queries": float(len(queries)),
            "top1_similarity_mean": float(scores[:, 0].mean()),
            "top1_similarity_median": float(np.median(scores[:, 0])),
            "top1_similarity_p05": float(np.quantile(scores[:, 0], 0.05)),
            "top1_similarity_p95": float(np.quantile(scores[:, 0], 0.95)),
            "top1_same_product_rate": float(
                np.mean(query_products == fit_products[top_one])
            ),
            "top1_same_t1_rate": float(
                np.mean(query_t1[same_t1_mask] == fit_t1[top_one][same_t1_mask])
            ),
            "exact_score_maximum_error": maximum_exact_error,
        }

        repeated_queries = np.repeat(np.arange(len(queries)), settings["neighbors"])
        flat_indices = indices.reshape(-1)
        neighbors.append(
            pd.DataFrame(
                {
                    "query_split": split,
                    "query_complaint_id": query_ids[repeated_queries],
                    "rank": np.tile(
                        np.arange(1, settings["neighbors"] + 1),
                        len(queries),
                    ),
                    "fit_complaint_id": fit_ids[flat_indices],
                    "cosine_similarity": scores.reshape(-1),
                    "query_product": query_products[repeated_queries],
                    "fit_product": fit_products[flat_indices],
                    "query_t1": query_t1[repeated_queries],
                    "fit_t1": fit_t1[flat_indices],
                }
            )
        )

    return pd.concat(neighbors, ignore_index=True), metrics


def maximum_rss_gib() -> float:
    resource: Any = importlib.import_module("resource")
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_maxrss / 1024**2)


def save_offline_record(config: dict[str, Any], report: dict[str, Any]) -> None:
    settings = config["semantic_space"]
    metrics = {
        "pca_explained_variance": report["pca"]["explained_variance"],
        "maximum_norm_error": max(
            split["maximum_norm_error"] for split in report["input"].values()
        ),
    }
    for split, values in report["faiss"]["metrics"].items():
        for name, value in values.items():
            metrics[f"{split}_{name}"] = value

    record = build_run_record(
        config=config,
        run_name="m6-semantic-space",
        stage="M6",
        target="A1",
        split="fit_calibration_validation_sample",
        view="no_shared_text_neighbors",
        features=["BGE normalized embeddings"],
        parameters={
            "source_artifact": config["paths"]["bge_artifacts"],
            "source_dvc_hash": settings["source_dvc_hash"],
            "pca_components": settings["pca"]["n_components"],
            "umap_neighbors": settings["umap"]["n_neighbors"],
            "umap_min_dist": settings["umap"]["min_dist"],
            "faiss_index_type": settings["faiss"]["index_type"],
            "faiss_metric": settings["faiss"]["metric"],
            "faiss_neighbors": settings["faiss"]["neighbors"],
            "model_artifact": config["paths"]["semantic_artifacts"],
        },
        metrics=metrics,
        artifacts=[
            config["paths"]["semantic_report"],
            config["paths"]["semantic_pca_plot"],
            config["paths"]["semantic_umap_plot"],
        ],
    )
    path = (
        PROJECT_ROOT
        / config["paths"]["offline_runs"]
        / "m6"
        / "semantic_space"
        / "run.json"
    )
    save_run_record(record, path)


def main() -> None:
    started = time.perf_counter()
    config = load_experiment_config()
    settings = config["semantic_space"]
    seed = config["experiment"]["seed"]
    set_seed(seed)

    input_dir = PROJECT_ROOT / config["paths"]["bge_artifacts"]
    dvc_path = PROJECT_ROOT / config["paths"]["bge_artifacts_dvc"]
    output_dir = PROJECT_ROOT / config["paths"]["semantic_artifacts"]
    report_path = PROJECT_ROOT / config["paths"]["semantic_report"]
    pca_plot_path = PROJECT_ROOT / config["paths"]["semantic_pca_plot"]
    umap_plot_path = PROJECT_ROOT / config["paths"]["semantic_umap_plot"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    source_hash = load_dvc_hash(dvc_path)
    if source_hash != settings["source_dvc_hash"]:
        raise ValueError("M6 source DVC hash does not match the frozen configuration.")

    frames, embeddings, source_metadata = load_inputs(
        input_dir,
        config["bge"]["sample_rows"],
    )
    input_validation = validate_embeddings(
        embeddings,
        config["bge"]["sample_rows"],
        settings["embedding_dimensions"],
        settings["normalization_tolerance"],
        settings["batch_rows"],
    )

    pca_started = time.perf_counter()
    pca = fit_pca(
        embeddings["fit"],
        settings["pca"]["n_components"],
        seed,
    )
    pca_values = {
        split: transform_to_npy(
            pca,
            embeddings[split],
            output_dir / f"{split}_pca.npy",
            settings["batch_rows"],
        )
        for split in SPLITS
    }
    joblib.dump(pca, output_dir / "pca.joblib", compress=3)
    save_pca_plot(
        pca,
        settings["pca"]["variance_checkpoints"],
        pca_plot_path,
    )
    pca_seconds = time.perf_counter() - pca_started

    umap_started = time.perf_counter()
    reducer, fit_coordinates = fit_umap(pca_values["fit"], settings["umap"], seed)
    coordinates = {"fit": fit_coordinates}
    for split in ("calibration", "validation"):
        coordinates[split] = reducer.transform(pca_values[split]).astype(np.float32)
    del reducer
    for split, values in coordinates.items():
        np.save(output_dir / f"{split}_umap_2d.npy", values)
    plotted_rows = save_umap_plot(
        frames,
        coordinates,
        settings["umap"]["plot_rows_per_split"],
        seed,
        output_dir / "umap_plot_sample.parquet",
        umap_plot_path,
    )
    umap_seconds = time.perf_counter() - umap_started

    faiss_started = time.perf_counter()
    index = build_faiss_index(
        embeddings["fit"],
        settings["batch_rows"],
        settings["faiss"]["threads"],
    )
    faiss = importlib.import_module("faiss")
    faiss.write_index(index, str(output_dir / "faiss_fit.index"))
    frames["fit"][[ID_COLUMN]].to_parquet(
        output_dir / "faiss_fit_ids.parquet",
        index=False,
    )
    neighbor_sample, neighbor_metrics = search_neighbor_sample(
        index,
        frames["fit"],
        embeddings["fit"],
        frames,
        embeddings,
        settings["faiss"],
        seed,
    )
    neighbor_sample.to_parquet(output_dir / "neighbor_sample.parquet", index=False)
    faiss_seconds = time.perf_counter() - faiss_started

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    checkpoints = {
        str(checkpoint): float(cumulative_variance[checkpoint - 1])
        for checkpoint in settings["pca"]["variance_checkpoints"]
        if checkpoint <= len(cumulative_variance)
    }
    report = {
        "stage": "M6",
        "seed": seed,
        "split_version": config["evaluation"]["split_version"],
        "source": {
            "path": config["paths"]["bge_artifacts"],
            "dvc_hash": source_hash,
            "bge_model": source_metadata["bge_model"],
            "bge_revision": source_metadata["bge_revision"],
        },
        "input": input_validation,
        "pca": {
            "fitted_on": "fit",
            "components": settings["pca"]["n_components"],
            "explained_variance": float(cumulative_variance[-1]),
            "variance_checkpoints": checkpoints,
            "elapsed_seconds": pca_seconds,
        },
        "umap": {
            "fitted_on": "fit",
            **settings["umap"],
            "plotted_rows": plotted_rows,
            "elapsed_seconds": umap_seconds,
        },
        "faiss": {
            "reference_split": "fit",
            **settings["faiss"],
            "index_rows": int(index.ntotal),
            "metrics": neighbor_metrics,
            "elapsed_seconds": faiss_seconds,
        },
        "versions": {
            "faiss_cpu": version("faiss-cpu"),
            "umap_learn": version("umap-learn"),
            "scikit_learn": version("scikit-learn"),
        },
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_rss_gib": maximum_rss_gib(),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    save_offline_record(config, report)

    print(f"PCA explained variance: {report['pca']['explained_variance']:.4f}")
    for split, values in neighbor_metrics.items():
        print(
            f"{split} top-1 similarity median: "
            f"{values['top1_similarity_median']:.4f}"
        )
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")
    print(f"Artifacts: {output_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
