"""Compare clustering methods on the frozen M6 semantic space."""

from __future__ import annotations

import hashlib
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time
from typing import Any
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    pairwise_distances,
    pairwise_distances_argmin_min,
    silhouette_samples,
)

from src.data.apply_taxonomy import CANONICAL_PRODUCT_COLUMN
from src.data.build_targets import COMPLETE_COLUMNS
from src.data.normalize_text import HASH_COLUMN
from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    load_experiment_config,
    save_run_record,
    set_seed,
)
from src.models.semantic_space import (
    ID_COLUMN,
    fixed_sample_positions,
    load_dvc_hash,
    maximum_rss_gib,
)

NOISE_LABEL = -1
MISSING_LABEL = -999


def load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    semantic_dir = PROJECT_ROOT / config["paths"]["semantic_artifacts"]
    bge_dir = PROJECT_ROOT / config["paths"]["bge_artifacts"]
    manifest = pd.read_parquet(bge_dir / "sample_manifest.parquet")
    frames = {
        split: manifest.loc[manifest["sample_split"] == split].reset_index(drop=True)
        for split in ("fit", "calibration")
    }
    inputs = {
        "fit_pca": np.load(semantic_dir / "fit_pca.npy", mmap_mode="r"),
        "calibration_pca": np.load(
            semantic_dir / "calibration_pca.npy",
            mmap_mode="r",
        ),
        "fit_umap_2d": np.load(semantic_dir / "fit_umap_2d.npy", mmap_mode="r"),
        "fit_bge": np.load(bge_dir / "fit_embeddings.npy", mmap_mode="r"),
        "frames": frames,
    }
    for split, key in (("fit", "fit_pca"), ("calibration", "calibration_pca")):
        if len(inputs[key]) != len(frames[split]):
            raise ValueError(f"M7 {split} rows do not align with the M5 manifest.")
    if len(inputs["fit_bge"]) != len(frames["fit"]):
        raise ValueError("M7 BGE rows do not align with the M5 manifest.")
    return inputs


def fit_umap_space(
    fit_values: np.ndarray,
    calibration_values: np.ndarray,
    settings: dict[str, Any],
    seed: int,
) -> tuple[Any, np.ndarray, np.ndarray]:
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
    fit_transformed = model.fit_transform(fit_values).astype(np.float32)
    calibration_transformed = model.transform(calibration_values).astype(np.float32)
    return model, fit_transformed, calibration_transformed


def candidate_definitions(settings: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for clusters in settings["kmeans"]["clusters"]:
        candidates.append(
            {
                "id": f"kmeans_k{clusters}",
                "algorithm": "kmeans",
                "parameters": {"clusters": clusters},
            }
        )
    for minimum_size in settings["hdbscan"]["min_cluster_size"]:
        for minimum_samples in settings["hdbscan"]["min_samples"]:
            candidates.append(
                {
                    "id": f"hdbscan_mcs{minimum_size}_ms{minimum_samples}",
                    "algorithm": "hdbscan",
                    "parameters": {
                        "min_cluster_size": minimum_size,
                        "min_samples": minimum_samples,
                    },
                }
            )
    for parameters in settings["cure"]["candidates"]:
        candidates.append(
            {
                "id": (
                    f"cure_k{parameters['clusters']}_r{parameters['representatives']}"
                    f"_c{parameters['compression']}"
                ),
                "algorithm": "cure",
                "parameters": parameters,
            }
        )
    return candidates


def labels_from_cure_clusters(clusters: list[list[int]], rows: int) -> np.ndarray:
    labels = np.full(rows, MISSING_LABEL, dtype=np.int32)
    for cluster_id, positions in enumerate(clusters):
        indices = np.asarray(positions, dtype=np.int64)
        if np.any(labels[indices] != MISSING_LABEL):
            raise ValueError("CURE assigned an input row more than once.")
        labels[indices] = cluster_id
    if np.any(labels == MISSING_LABEL):
        raise ValueError("CURE did not assign every input row.")
    return labels


def fit_algorithm(
    algorithm: str,
    parameters: dict[str, Any],
    values: np.ndarray,
    settings: dict[str, Any],
    seed: int,
    threads: int,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    if algorithm == "kmeans":
        model = MiniBatchKMeans(
            n_clusters=parameters["clusters"],
            init="k-means++",
            n_init=settings["kmeans"]["n_init"],
            batch_size=settings["kmeans"]["batch_size"],
            max_iter=settings["kmeans"]["max_iter"],
            random_state=seed,
        ).fit(values)
        return model, model.labels_.astype(np.int32), {}

    if algorithm == "hdbscan":
        model = HDBSCAN(
            min_cluster_size=parameters["min_cluster_size"],
            min_samples=parameters["min_samples"],
            metric="euclidean",
            cluster_selection_method="eom",
            allow_single_cluster=False,
            copy=True,
            n_jobs=threads,
        ).fit(values)
        return model, model.labels_.astype(np.int32), {
            "mean_probability": float(model.probabilities_.mean()),
        }

    if algorithm == "cure":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            cure_module = importlib.import_module("pyclustering.cluster.cure")
            wrapper = importlib.import_module("pyclustering.core.wrapper")
        if not wrapper.ccore_library.workable():
            raise RuntimeError("M7 requires the native pyclustering CCORE library.")
        model = cure_module.cure(
            values.tolist(),
            parameters["clusters"],
            parameters["representatives"],
            parameters["compression"],
            ccore=True,
        ).process()
        labels = labels_from_cure_clusters(model.get_clusters(), len(values))
        return model, labels, {
            "representatives": model.get_representors(),
        }

    raise ValueError(f"Unknown clustering algorithm: {algorithm}")


def silhouette_summary(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    selected_labels = labels[positions]
    assigned = selected_labels >= 0
    selected_labels = selected_labels[assigned]
    assigned_positions = np.flatnonzero(assigned)
    selected_distances = distance_matrix[
        np.ix_(assigned_positions, assigned_positions)
    ]
    groups = np.unique(selected_labels)
    if len(groups) < 2 or len(groups) >= len(selected_labels):
        return {
            "silhouette_rows": float(len(selected_labels)),
            "silhouette_mean": -1.0,
            "silhouette_median": -1.0,
            "negative_silhouette_fraction": 1.0,
        }, np.array([], dtype=float)

    values = silhouette_samples(
        selected_distances,
        selected_labels,
        metric="precomputed",
    )
    return {
        "silhouette_rows": float(len(values)),
        "silhouette_mean": float(values.mean()),
        "silhouette_median": float(np.median(values)),
        "negative_silhouette_fraction": float(np.mean(values < 0)),
    }, values


def structure_metrics(
    labels: np.ndarray,
    small_cluster_rows: int,
) -> dict[str, float]:
    assigned = labels[labels >= 0]
    groups, sizes = np.unique(assigned, return_counts=True)
    if len(groups) == 0:
        return {
            "clusters": 0.0,
            "noise_fraction": 1.0,
            "small_cluster_fraction": 0.0,
            "largest_cluster_fraction": 0.0,
            "minimum_cluster_rows": 0.0,
            "median_cluster_rows": 0.0,
            "maximum_cluster_rows": 0.0,
        }
    return {
        "clusters": float(len(groups)),
        "noise_fraction": float(np.mean(labels < 0)),
        "small_cluster_fraction": float(
            sizes[sizes < small_cluster_rows].sum() / len(labels)
        ),
        "largest_cluster_fraction": float(sizes.max() / len(labels)),
        "minimum_cluster_rows": float(sizes.min()),
        "median_cluster_rows": float(np.median(sizes)),
        "maximum_cluster_rows": float(sizes.max()),
    }


def semantic_neighbor_indices(values: np.ndarray, neighbors: int) -> np.ndarray:
    faiss = importlib.import_module("faiss")
    index = faiss.IndexFlatIP(values.shape[1])
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    index.add(contiguous)
    _, raw_indices = index.search(contiguous, neighbors + 2)
    result = np.empty((len(values), neighbors), dtype=np.int32)
    for row, indices in enumerate(raw_indices):
        filtered = indices[indices != row][:neighbors]
        if len(filtered) != neighbors:
            raise ValueError("M7 could not construct the semantic neighbor sample.")
        result[row] = filtered
    return result


def semantic_metrics(labels: np.ndarray, neighbors: np.ndarray) -> dict[str, float]:
    query_labels = labels[:, None]
    neighbor_labels = labels[neighbors]
    valid = (query_labels >= 0) & (neighbor_labels >= 0)
    _, sizes = np.unique(labels[labels >= 0], return_counts=True)
    if not valid.any() or len(sizes) == 0:
        return {
            "neighbor_same_cluster_rate": 0.0,
            "neighbor_chance_rate": 0.0,
            "neighbor_lift": 0.0,
        }
    observed = float(np.mean((query_labels == neighbor_labels)[valid]))
    proportions = sizes / sizes.sum()
    chance = float(np.sum(proportions**2))
    return {
        "neighbor_same_cluster_rate": observed,
        "neighbor_chance_rate": chance,
        "neighbor_lift": observed - chance,
    }


def template_metrics(labels: np.ndarray, hashes: np.ndarray) -> dict[str, float]:
    assigned = labels >= 0
    frame = pd.DataFrame(
        {
            "cluster": labels[assigned],
            "hash": hashes[assigned],
        }
    )
    if frame.empty:
        return {
            "unique_hash_fraction": 0.0,
            "template_dominated_rows_fraction": 0.0,
        }
    cluster_sizes = frame.groupby("cluster", observed=True).size()
    dominant = (
        frame.groupby(["cluster", "hash"], observed=True)
        .size()
        .groupby(level=0)
        .max()
    )
    dominant_fraction = dominant / cluster_sizes
    dominated_clusters = dominant_fraction[dominant_fraction > 0.5].index
    dominated_rows = int(cluster_sizes.loc[dominated_clusters].sum())
    return {
        "unique_hash_fraction": float(frame["hash"].nunique() / len(frame)),
        "template_dominated_rows_fraction": float(dominated_rows / len(labels)),
    }


def audit_metrics(
    labels: np.ndarray,
    frame: pd.DataFrame,
) -> dict[str, float]:
    assigned = labels >= 0
    product_nmi = (
        normalized_mutual_info_score(
            frame.loc[assigned, CANONICAL_PRODUCT_COLUMN].astype(str),
            labels[assigned],
        )
        if assigned.any()
        else 0.0
    )
    t1_complete = frame[COMPLETE_COLUMNS["T1"]].to_numpy(dtype=bool) & assigned
    t1_nmi = (
        normalized_mutual_info_score(
            frame.loc[t1_complete, "T1"].astype(str),
            labels[t1_complete],
        )
        if t1_complete.any()
        else 0.0
    )
    return {
        "product_nmi": float(product_nmi),
        "t1_nmi": float(t1_nmi),
    }


def acceptance_failures(
    metrics: dict[str, Any],
    acceptance: dict[str, Any],
) -> list[str]:
    checks = [
        (
            metrics["clusters"] < acceptance["minimum_clusters"]
            or metrics["clusters"] > acceptance["maximum_clusters"],
            "cluster_count",
        ),
        (
            metrics["largest_cluster_fraction"]
            > acceptance["maximum_largest_cluster_fraction"],
            "largest_cluster",
        ),
        (
            metrics["small_cluster_fraction"]
            > acceptance["maximum_small_cluster_fraction"],
            "small_clusters",
        ),
        (
            metrics["noise_fraction"] > acceptance["maximum_noise_fraction"],
            "noise",
        ),
        (
            metrics["silhouette_mean"] <= acceptance["minimum_silhouette"],
            "silhouette",
        ),
        (
            metrics["negative_silhouette_fraction"]
            > acceptance["maximum_negative_silhouette_fraction"],
            "negative_silhouette",
        ),
        (
            metrics["neighbor_lift"] < acceptance["minimum_neighbor_lift"],
            "neighbor_lift",
        ),
        (
            metrics["template_dominated_rows_fraction"]
            > acceptance["maximum_template_dominated_fraction"],
            "templates",
        ),
    ]
    return [name for failed, name in checks if failed]


def select_candidate(records: list[dict[str, Any]], algorithm: str) -> dict[str, Any]:
    available = [record for record in records if record["algorithm"] == algorithm]
    passing = [record for record in available if not record["initial_failures"]]
    pool = passing or available
    return max(pool, key=lambda record: record["silhouette_mean"])


def stability_metrics(
    algorithm: str,
    parameters: dict[str, Any],
    values: np.ndarray,
    settings: dict[str, Any],
    seed: int,
    threads: int,
) -> dict[str, float]:
    runs = []
    rows = int(round(len(values) * settings["stability_fraction"]))
    for run_seed in settings["stability_seeds"]:
        positions = fixed_sample_positions(len(values), rows, run_seed)
        _, labels, _ = fit_algorithm(
            algorithm,
            parameters,
            values[positions],
            settings,
            seed + run_seed,
            threads,
        )
        mapped = np.full(len(values), MISSING_LABEL, dtype=np.int32)
        mapped[positions] = labels
        runs.append(mapped)

    all_scores = []
    assigned_scores = []
    for left in range(len(runs)):
        for right in range(left + 1, len(runs)):
            common = (runs[left] != MISSING_LABEL) & (runs[right] != MISSING_LABEL)
            all_scores.append(
                adjusted_rand_score(runs[left][common], runs[right][common])
            )
            assigned = common & (runs[left] >= 0) & (runs[right] >= 0)
            if assigned.sum() > 1:
                assigned_scores.append(
                    adjusted_rand_score(
                        runs[left][assigned],
                        runs[right][assigned],
                    )
                )

    primary = assigned_scores if algorithm == "hdbscan" and assigned_scores else all_scores
    return {
        "stability_ari": float(np.median(primary)),
        "stability_ari_all": float(np.median(all_scores)),
    }


def future_assignment(
    algorithm: str,
    model: Any,
    extras: dict[str, Any],
    train_values: np.ndarray,
    train_labels: np.ndarray,
    future_values: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[int, float]]:
    clusters = sorted(np.unique(train_labels[train_labels >= 0]).tolist())
    if not clusters:
        return (
            {
                "assignment_type": "unavailable",
                "future_coverage": 0.0,
                "future_rejected_fraction": 1.0,
            },
            np.empty((0, train_values.shape[1]), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            {},
        )
    if algorithm == "kmeans":
        references = np.asarray(model.cluster_centers_, dtype=np.float32)
        reference_labels = np.arange(len(references), dtype=np.int32)
        assignment_type = "native_centers"
    elif algorithm == "cure":
        points = []
        point_labels = []
        for cluster_id, representatives in enumerate(extras["representatives"]):
            points.extend(representatives)
            point_labels.extend([cluster_id] * len(representatives))
        references = np.asarray(points, dtype=np.float32)
        reference_labels = np.asarray(point_labels, dtype=np.int32)
        assignment_type = "representative_adapter"
    else:
        references = np.vstack(
            [train_values[train_labels == cluster].mean(axis=0) for cluster in clusters]
        ).astype(np.float32)
        reference_labels = np.asarray(clusters, dtype=np.int32)
        assignment_type = "centroid_adapter"

    thresholds = {}
    for cluster in clusters:
        rows = train_values[train_labels == cluster]
        own_references = references[reference_labels == cluster]
        _, distances = pairwise_distances_argmin_min(rows, own_references)
        thresholds[cluster] = float(np.quantile(distances, 0.99))

    reference_positions, future_distances = pairwise_distances_argmin_min(
        future_values,
        references,
    )
    future_labels = reference_labels[reference_positions]
    accepted = np.array(
        [distance <= thresholds[int(label)] for label, distance in zip(future_labels, future_distances)]
    )
    metrics = {
        "assignment_type": assignment_type,
        "future_coverage": float(accepted.mean()),
        "future_rejected_fraction": float(1.0 - accepted.mean()),
    }
    return metrics, references, reference_labels, thresholds


def load_template_hashes(input_path: Path, complaint_ids: np.ndarray) -> np.ndarray:
    table = pq.read_table(
        input_path,
        columns=[ID_COLUMN, HASH_COLUMN],
        filters=[(ID_COLUMN, "in", complaint_ids.tolist())],
    )
    frame = table.to_pandas().drop_duplicates(ID_COLUMN)
    hashes = frame.set_index(ID_COLUMN)[HASH_COLUMN].reindex(complaint_ids)
    if hashes.isna().any():
        raise ValueError("M7 could not recover every narrative hash.")
    return hashes.astype(str).to_numpy()


def save_silhouette_plot(
    values: np.ndarray,
    labels: dict[str, np.ndarray],
    positions: np.ndarray,
    selected: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 6))
    for axis, algorithm in zip(axes, ("kmeans", "hdbscan", "cure")):
        current_labels = labels[selected[algorithm]["id"]][positions]
        assigned = current_labels >= 0
        current_labels = current_labels[assigned]
        distances = pairwise_distances(values[positions][assigned], n_jobs=-1)
        if len(np.unique(current_labels)) < 2:
            axis.text(0.5, 0.5, "Sin silhouette válido", ha="center")
            axis.set_title(algorithm)
            continue
        current_values = silhouette_samples(
            distances,
            current_labels,
            metric="precomputed",
        )
        y_lower = 10
        for cluster in np.unique(current_labels):
            cluster_values = np.sort(current_values[current_labels == cluster])
            y_upper = y_lower + len(cluster_values)
            axis.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_values,
                alpha=0.7,
            )
            y_lower = y_upper + 4
        axis.axvline(current_values.mean(), color="red", linestyle="--")
        axis.set(
            title=f"{algorithm}: {current_values.mean():.3f}",
            xlabel="Silhouette",
            ylabel="Casos ordenados por grupo",
            yticks=[],
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_size_plot(
    labels: dict[str, np.ndarray],
    selected: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    for algorithm in ("kmeans", "hdbscan", "cure"):
        current = labels[selected[algorithm]["id"]]
        _, sizes = np.unique(current[current >= 0], return_counts=True)
        axis.plot(
            np.arange(1, len(sizes) + 1),
            np.sort(sizes)[::-1],
            marker="o",
            markersize=3,
            label=algorithm,
        )
    axis.set(
        title="Tamaño de los grupos seleccionados",
        xlabel="Grupo ordenado por tamaño",
        ylabel="Número de reclamos",
        yscale="log",
    )
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_umap_plot(
    umap_2d: np.ndarray,
    sample_positions: np.ndarray,
    labels: dict[str, np.ndarray],
    selected: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    coordinates = umap_2d[sample_positions]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, algorithm in zip(axes, ("kmeans", "hdbscan", "cure")):
        current = labels[selected[algorithm]["id"]]
        colors = current.astype(float)
        colors[current < 0] = np.nan
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=colors,
            cmap="nipy_spectral",
            s=3,
            alpha=0.45,
            rasterized=True,
        )
        noise = current < 0
        if noise.any():
            axis.scatter(
                coordinates[noise, 0],
                coordinates[noise, 1],
                c="lightgray",
                s=2,
                alpha=0.3,
                rasterized=True,
            )
        axis.set(title=algorithm, xlabel="UMAP 1", ylabel="UMAP 2")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_offline_record(config: dict[str, Any], report: dict[str, Any]) -> None:
    metrics = {}
    for algorithm, values in report["selected"].items():
        for name in (
            "silhouette_mean",
            "noise_fraction",
            "largest_cluster_fraction",
            "neighbor_lift",
            "stability_ari",
            "future_coverage",
        ):
            metrics[f"{algorithm}_{name}"] = float(values[name])

    settings = config["clustering"]
    record = build_run_record(
        config=config,
        run_name="m7-cluster-comparison",
        stage="M7",
        target="A1",
        split="fit_sample_calibration_assignment",
        view="semantic_patterns",
        features=["BGE", "PCA 256D", "UMAP 15D"],
        parameters={
            "source_artifact": config["paths"]["semantic_artifacts"],
            "source_dvc_hash": settings["source_dvc_hash"],
            "comparison_rows": settings["comparison_rows"],
            "silhouette_rows": settings["silhouette_rows"],
            "recommended_algorithm": report["decision"]["recommended_algorithm"],
            "model_artifact": config["paths"]["clustering_artifacts"],
        },
        metrics=metrics,
        artifacts=[
            config["paths"]["clustering_report"],
            config["paths"]["clustering_candidates"],
            config["paths"]["clustering_silhouette_plot"],
            config["paths"]["clustering_umap_plot"],
            config["paths"]["clustering_sizes_plot"],
        ],
    )
    path = (
        PROJECT_ROOT
        / config["paths"]["offline_runs"]
        / "m7"
        / "cluster_comparison"
        / "run.json"
    )
    save_run_record(record, path)


def main() -> None:
    started = time.perf_counter()
    config = load_experiment_config()
    settings = config["clustering"]
    seed = config["experiment"]["seed"]
    threads = settings.get("threads", 16)
    set_seed(seed)

    source_hash = load_dvc_hash(
        PROJECT_ROOT / config["paths"]["semantic_artifacts_dvc"]
    )
    if source_hash != settings["source_dvc_hash"]:
        raise ValueError("M7 source DVC hash does not match the frozen configuration.")

    output_dir = PROJECT_ROOT / config["paths"]["clustering_artifacts"]
    report_path = PROJECT_ROOT / config["paths"]["clustering_report"]
    candidates_path = PROJECT_ROOT / config["paths"]["clustering_candidates"]
    silhouette_plot_path = PROJECT_ROOT / config["paths"]["clustering_silhouette_plot"]
    umap_plot_path = PROJECT_ROOT / config["paths"]["clustering_umap_plot"]
    sizes_plot_path = PROJECT_ROOT / config["paths"]["clustering_sizes_plot"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "labels").mkdir(parents=True)
    (output_dir / "selected").mkdir(parents=True)

    inputs = load_inputs(config)
    umap_started = time.perf_counter()
    umap_model, fit_umap, calibration_umap = fit_umap_space(
        inputs["fit_pca"],
        inputs["calibration_pca"],
        settings["umap"],
        seed,
    )
    np.save(output_dir / "fit_umap_15d.npy", fit_umap)
    np.save(output_dir / "calibration_umap_15d.npy", calibration_umap)
    joblib.dump(umap_model, output_dir / "umap_15d.joblib", compress=3)
    del umap_model
    umap_seconds = time.perf_counter() - umap_started

    sample_positions = fixed_sample_positions(
        len(fit_umap),
        settings["comparison_rows"],
        seed,
    )
    sample_values = fit_umap[sample_positions]
    sample_frame = inputs["frames"]["fit"].iloc[sample_positions].reset_index(drop=True)
    sample_bge = np.ascontiguousarray(inputs["fit_bge"][sample_positions])
    hashes = load_template_hashes(
        PROJECT_ROOT / config["paths"]["input_data"],
        sample_frame[ID_COLUMN].astype(str).to_numpy(),
    )
    sample_id_hash = hashlib.sha256(
        "\n".join(sample_frame[ID_COLUMN].astype(str)).encode("utf-8")
    ).hexdigest()
    private_manifest = sample_frame[[ID_COLUMN]].copy()
    private_manifest["source_position"] = sample_positions
    private_manifest.to_parquet(output_dir / "comparison_sample.parquet", index=False)

    silhouette_positions = fixed_sample_positions(
        len(sample_values),
        settings["silhouette_rows"],
        seed + 100,
    )
    silhouette_distances = pairwise_distances(
        sample_values[silhouette_positions],
        metric="euclidean",
        n_jobs=threads,
    ).astype(np.float32)
    neighbor_indices = semantic_neighbor_indices(
        sample_bge,
        settings["semantic_neighbors"],
    )

    records = []
    candidate_labels = {}
    for candidate in candidate_definitions(settings):
        candidate_started = time.perf_counter()
        _, labels, extras = fit_algorithm(
            candidate["algorithm"],
            candidate["parameters"],
            sample_values,
            settings,
            seed,
            threads,
        )
        metrics = structure_metrics(
            labels,
            settings["acceptance"]["small_cluster_rows"],
        )
        silhouette, _ = silhouette_summary(
            silhouette_distances,
            labels,
            silhouette_positions,
        )
        metrics.update(silhouette)
        metrics.update(semantic_metrics(labels, neighbor_indices))
        metrics.update(template_metrics(labels, hashes))
        metrics.update(audit_metrics(labels, sample_frame))
        metrics.update(
            {
                name: value
                for name, value in extras.items()
                if isinstance(value, (int, float))
            }
        )
        record = {
            **candidate,
            **metrics,
            "elapsed_seconds": time.perf_counter() - candidate_started,
        }
        record["initial_failures"] = acceptance_failures(
            record,
            settings["acceptance"],
        )
        records.append(record)
        candidate_labels[candidate["id"]] = labels
        np.save(output_dir / "labels" / f"{candidate['id']}.npy", labels)

    selected = {
        algorithm: select_candidate(records, algorithm)
        for algorithm in ("kmeans", "hdbscan", "cure")
    }
    for algorithm, record in selected.items():
        model, labels, extras = fit_algorithm(
            algorithm,
            record["parameters"],
            sample_values,
            settings,
            seed,
            threads,
        )
        stability = stability_metrics(
            algorithm,
            record["parameters"],
            sample_values,
            settings,
            seed,
            threads,
        )
        future, references, reference_labels, thresholds = future_assignment(
            algorithm,
            model,
            extras,
            sample_values,
            labels,
            calibration_umap,
        )
        record.update(stability)
        record.update(future)
        final_failures = list(record["initial_failures"])
        if record["stability_ari"] < settings["acceptance"]["minimum_stability_ari"]:
            final_failures.append("stability")
        if record["future_coverage"] < settings["acceptance"]["minimum_future_coverage"]:
            final_failures.append("future_coverage")
        record["final_failures"] = final_failures
        record["accepted"] = not final_failures
        record["selected"] = True
        np.save(output_dir / "selected" / f"{algorithm}_references.npy", references)
        np.save(
            output_dir / "selected" / f"{algorithm}_reference_labels.npy",
            reference_labels,
        )
        (output_dir / "selected" / f"{algorithm}_thresholds.json").write_text(
            json.dumps(thresholds, indent=2) + "\n",
            encoding="utf-8",
        )
        if algorithm in {"kmeans", "hdbscan"}:
            joblib.dump(model, output_dir / "selected" / f"{algorithm}.joblib", compress=3)

    passing = [
        record for record in selected.values() if record["accepted"]
    ]
    native = [record for record in passing if record["algorithm"] == "kmeans"]
    decision_pool = native or passing
    recommended = (
        max(
            decision_pool,
            key=lambda record: (record["stability_ari"], record["silhouette_mean"]),
        )["algorithm"]
        if decision_pool
        else "none"
    )
    geometry_pool = passing or list(selected.values())
    best_geometry = max(
        geometry_pool,
        key=lambda record: record["silhouette_mean"],
    )["algorithm"]

    for record in records:
        record.setdefault("selected", False)
        record.setdefault("stability_ari", np.nan)
        record.setdefault("stability_ari_all", np.nan)
        record.setdefault("future_coverage", np.nan)
        record.setdefault("future_rejected_fraction", np.nan)
        record.setdefault("assignment_type", "not_evaluated")
        record.setdefault("final_failures", record["initial_failures"])
        record.setdefault("accepted", False)

    candidate_table = pd.DataFrame(records)
    candidate_table["parameters"] = candidate_table["parameters"].map(json.dumps)
    candidate_table["initial_failures"] = candidate_table["initial_failures"].map(
        lambda values: ",".join(values)
    )
    candidate_table["final_failures"] = candidate_table["final_failures"].map(
        lambda values: ",".join(values)
    )
    candidate_table.to_csv(candidates_path, index=False)

    save_silhouette_plot(
        sample_values,
        candidate_labels,
        silhouette_positions,
        selected,
        silhouette_plot_path,
    )
    save_size_plot(candidate_labels, selected, sizes_plot_path)
    save_umap_plot(
        inputs["fit_umap_2d"],
        sample_positions,
        candidate_labels,
        selected,
        umap_plot_path,
    )

    public_selected = {
        algorithm: {
            key: value
            for key, value in record.items()
            if key not in {"parameters"}
        }
        | {"parameters": record["parameters"]}
        for algorithm, record in selected.items()
    }
    report = {
        "stage": "M7",
        "seed": seed,
        "split_version": config["evaluation"]["split_version"],
        "source": {
            "path": config["paths"]["semantic_artifacts"],
            "dvc_hash": source_hash,
        },
        "sample": {
            "fit_rows": settings["comparison_rows"],
            "silhouette_rows": settings["silhouette_rows"],
            "id_sha256": sample_id_hash,
            "validation_used": False,
        },
        "umap": {
            **settings["umap"],
            "fitted_on_rows": len(fit_umap),
            "elapsed_seconds": umap_seconds,
        },
        "candidate_count": len(records),
        "selected": public_selected,
        "decision": {
            "recommended_algorithm": recommended,
            "best_geometry": best_geometry,
            "accepted_algorithms": [
                record["algorithm"] for record in passing
            ],
        },
        "acceptance": settings["acceptance"],
        "versions": {
            "scikit_learn": version("scikit-learn"),
            "umap_learn": version("umap-learn"),
            "pyclustering": version("pyclustering"),
        },
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_rss_gib": maximum_rss_gib(),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_offline_record(config, report)

    print(f"Recommended algorithm: {recommended}")
    for algorithm, values in public_selected.items():
        print(
            f"{algorithm}: silhouette={values['silhouette_mean']:.4f}; "
            f"stability={values['stability_ari']:.4f}; "
            f"failures={values['final_failures']}"
        )
    print(f"Report: {report_path.relative_to(PROJECT_ROOT)}")
    print(f"Artifacts: {output_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
