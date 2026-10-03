"""Block A of models_plan.md §25.3: does the PCA/UMAP space change M7's clusters?

Each step is a SLURM job (scripts/hpc/README.md):

- ``prepare``: builds one space (PCA or UMAP), fitted only on the M5 fit
  sample, and keeps the M7 comparison rows and the calibration rows.
- ``cluster``: runs one algorithm of the M7 grid, plus GMM, in one space, with
  the M7 rules for choosing and accepting a candidate.
- ``summarize``: compares every accepted candidate with the M7 clusters by the
  BGE neighbor lift (§25.2) and writes the report.

``smoke`` runs every step on a few rows, in memory, and saves nothing.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import adjusted_rand_score, pairwise_distances

from src.evaluation.experiment import (
    PROJECT_ROOT,
    build_run_record,
    load_experiment_config,
    save_run_record,
    set_seed,
)
from src.models.cluster_comparison import (
    acceptance_failures,
    audit_metrics,
    candidate_definitions,
    fit_algorithm,
    fit_umap_space,
    future_assignment,
    load_template_hashes,
    select_candidate,
    semantic_metrics,
    semantic_neighbor_indices,
    silhouette_summary,
    stability_metrics,
    structure_metrics,
    template_metrics,
)
from src.models.representation_comparison import week_resamples
from src.models.semantic_space import (
    ID_COLUMN,
    fit_pca,
    fixed_sample_positions,
    load_dvc_hash,
    maximum_rss_gib,
)

ALGORITHMS = ("kmeans", "hdbscan", "cure", "gmm")
DATE_COLUMN = "Date received"
DIFFERENCE_QUANTILES = {"p025": 0.025, "p50": 0.50, "p975": 0.975}
SMOKE_ROWS = {"fit": 3_000, "calibration": 1_000, "sample": 1_500, "silhouette": 500}
SMOKE_BOOTSTRAP_DRAWS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("prepare", "cluster", "summarize", "smoke"))
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Task of prepare or cluster; defaults to SLURM_ARRAY_TASK_ID.",
    )
    return parser.parse_args()


def task_index(argument: int | None) -> int:
    if argument is not None:
        return argument
    if "SLURM_ARRAY_TASK_ID" not in os.environ:
        raise ValueError("prepare and cluster need --index or a SLURM array task.")
    return int(os.environ["SLURM_ARRAY_TASK_ID"])


def space_names(settings: dict[str, Any]) -> list[str]:
    return list(settings["spaces"])


def cluster_tasks(settings: dict[str, Any]) -> list[tuple[str, str]]:
    """Every (space, algorithm) pair, in the order of the SLURM array."""
    return [
        (space, algorithm)
        for space in space_names(settings)
        for algorithm in ALGORITHMS
    ]


def task_key(space: str, algorithm: str) -> str:
    return f"{space}__{algorithm}"


def clustering_settings(
    config: dict[str, Any],
    space: dict[str, Any],
) -> dict[str, Any]:
    """M7 settings plus GMM, with the covariance that §25.3 fixes for the space."""
    gmm = dict(config["space_sensitivity"]["gmm"])
    gmm["covariance"] = gmm["covariance"][space["kind"]]
    return {**config["clustering"], "gmm": gmm}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def staging_path(final: Path) -> Path:
    return final.with_name(f".{final.name}.inprogress")


def start_output(final: Path) -> Path | None:
    """Staging directory of a new output, or None when it is already complete."""
    staging = staging_path(final)
    if final.exists():
        return None
    if staging.exists():
        raise FileExistsError(f"Review the unfinished output by hand: {staging}")
    staging.mkdir(parents=True)
    return staging


def check_sources(config: dict[str, Any]) -> None:
    paths = config["paths"]
    settings = config["space_sensitivity"]
    expected = {
        "bge_artifacts_dvc": settings["bge_dvc_hash"],
        "semantic_artifacts_dvc": settings["semantic_dvc_hash"],
        "clustering_artifacts_dvc": settings["clustering_dvc_hash"],
    }
    for key, frozen in expected.items():
        if load_dvc_hash(PROJECT_ROOT / paths[key]) != frozen:
            raise ValueError(f"M7S source {paths[key]} does not match its frozen hash.")


def load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """M5 manifest, BGE embeddings and M6 PCA of the fit and calibration samples."""
    bge_dir = PROJECT_ROOT / config["paths"]["bge_artifacts"]
    semantic_dir = PROJECT_ROOT / config["paths"]["semantic_artifacts"]
    manifest = pd.read_parquet(bge_dir / "sample_manifest.parquet")
    inputs: dict[str, Any] = {
        "frames": {},
        "pca_model": joblib.load(semantic_dir / "pca.joblib"),
    }
    for split in ("fit", "calibration"):
        rows = manifest.loc[manifest["sample_split"] == split]
        inputs["frames"][split] = rows.reset_index(drop=True)
        inputs[f"{split}_bge"] = np.load(
            bge_dir / f"{split}_embeddings.npy", mmap_mode="r"
        )
        inputs[f"{split}_pca"] = np.load(
            semantic_dir / f"{split}_pca.npy", mmap_mode="r"
        )
        for key in (f"{split}_bge", f"{split}_pca"):
            if len(inputs[key]) != len(inputs["frames"][split]):
                raise ValueError(f"M7S {key} rows do not align with the M5 manifest.")
    return inputs


def subset_inputs(
    inputs: dict[str, Any],
    rows: dict[str, int],
    seed: int,
) -> dict[str, Any]:
    """A small fixed subset of every array, for the smoke run."""
    subset = {"frames": {}, "pca_model": inputs["pca_model"]}
    for split in ("fit", "calibration"):
        positions = fixed_sample_positions(
            len(inputs["frames"][split]), rows[split], seed
        )
        subset["frames"][split] = (
            inputs["frames"][split].iloc[positions].reset_index(drop=True)
        )
        for kind in ("bge", "pca"):
            subset[f"{split}_{kind}"] = np.asarray(inputs[f"{split}_{kind}"][positions])
    return subset


def build_space(
    space: dict[str, Any],
    inputs: dict[str, Any],
    umap_settings: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit and calibration coordinates of one space, learned only from fit."""
    if space["kind"] == "pca":
        components = space["components"]
        m6 = inputs["pca_model"]
        if components <= m6.n_components_:
            # PCA components are nested: the first k of M6 span its k-component PCA.
            ratios = m6.explained_variance_ratio_[:components]
            fit_values = inputs["fit_pca"][:, :components]
            calibration_values = inputs["calibration_pca"][:, :components]
        else:
            model = fit_pca(np.asarray(inputs["fit_bge"]), components, seed)
            ratios = model.explained_variance_ratio_
            fit_values = model.transform(inputs["fit_bge"])
            calibration_values = model.transform(inputs["calibration_bge"])
        return (
            np.asarray(fit_values, dtype=np.float32),
            np.asarray(calibration_values, dtype=np.float32),
            {"explained_variance": float(np.sum(ratios))},
        )

    if space["kind"] == "umap":
        _, fit_values, calibration_values = fit_umap_space(
            inputs["fit_pca"],
            inputs["calibration_pca"],
            {**umap_settings, "n_components": space["components"]},
            space["seed"],
        )
        return fit_values, calibration_values, {}

    raise ValueError(f"Unknown space kind: {space['kind']}")


def load_m7_space(
    clustering_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """The UMAP 15D coordinates that M7 used, so the reference space is exact."""
    return (
        np.load(clustering_dir / "fit_umap_15d.npy"),
        np.load(clustering_dir / "calibration_umap_15d.npy"),
        {"reused": "M7"},
    )


def build_context(
    fit_frame: pd.DataFrame,
    fit_bge: np.ndarray,
    sample_positions: np.ndarray,
    silhouette_rows: int,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Everything about the comparison rows that does not depend on the space."""
    frame = fit_frame.iloc[sample_positions].reset_index(drop=True)
    ids = frame[ID_COLUMN].astype(str).to_numpy()
    input_path = PROJECT_ROOT / config["paths"]["input_data"]
    return {
        "frame": frame,
        "hashes": load_template_hashes(input_path, ids),
        "neighbors": semantic_neighbor_indices(
            np.ascontiguousarray(fit_bge[sample_positions]),
            config["clustering"]["semantic_neighbors"],
        ),
        "silhouette_positions": fixed_sample_positions(
            len(sample_positions),
            silhouette_rows,
            seed + 100,
        ),
        "id_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
    }


def run_algorithm(
    algorithm: str,
    values: np.ndarray,
    calibration_values: np.ndarray,
    context: dict[str, Any],
    settings: dict[str, Any],
    seed: int,
    threads: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """The M7 procedure for one algorithm: grid, silhouette choice, final gates."""
    acceptance = settings["acceptance"]
    positions = context["silhouette_positions"]
    distances = pairwise_distances(
        values[positions], metric="euclidean", n_jobs=threads
    )
    records = []
    labels = {}
    for candidate in candidate_definitions(settings):
        if candidate["algorithm"] != algorithm:
            continue
        started = time.perf_counter()
        _, candidate_labels, extras = fit_algorithm(
            algorithm, candidate["parameters"], values, settings, seed, threads
        )
        record = {
            **candidate,
            **structure_metrics(candidate_labels, acceptance["small_cluster_rows"]),
            **silhouette_summary(distances, candidate_labels, positions)[0],
            **semantic_metrics(candidate_labels, context["neighbors"]),
            **template_metrics(candidate_labels, context["hashes"]),
            **audit_metrics(candidate_labels, context["frame"]),
            **{k: v for k, v in extras.items() if isinstance(v, (int, float))},
            "elapsed_seconds": time.perf_counter() - started,
            "selected": False,
            "stability_ari": None,
            "future_coverage": None,
            "accepted": False,
        }
        record["initial_failures"] = acceptance_failures(record, acceptance)
        record["final_failures"] = list(record["initial_failures"])
        records.append(record)
        labels[candidate["id"]] = candidate_labels

    chosen = select_candidate(records, algorithm)
    model, chosen_labels, extras = fit_algorithm(
        algorithm, chosen["parameters"], values, settings, seed, threads
    )
    chosen.update(
        stability_metrics(
            algorithm, chosen["parameters"], values, settings, seed, threads
        )
    )
    future, _, _, _ = future_assignment(
        algorithm, model, extras, values, chosen_labels, calibration_values
    )
    chosen.update(future)
    if chosen["stability_ari"] < acceptance["minimum_stability_ari"]:
        chosen["final_failures"].append("stability")
    if chosen["future_coverage"] < acceptance["minimum_future_coverage"]:
        chosen["final_failures"].append("future_coverage")
    chosen["selected"] = True
    chosen["accepted"] = not chosen["final_failures"]
    return records, labels


def neighbor_pairs(
    labels: np.ndarray,
    neighbors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per row: BGE neighbors in the same cluster, and neighbors that count."""
    query = labels[:, None]
    neighbor_labels = labels[neighbors]
    valid = (query >= 0) & (neighbor_labels >= 0)
    same = valid & (query == neighbor_labels)
    return same.sum(axis=1), valid.sum(axis=1)


def weighted_neighbor_lift(
    labels: np.ndarray,
    same: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray,
) -> float:
    """M7's neighbor lift with row weights; unit weights give the M7 value."""
    pairs = float(np.sum(weights * valid))
    assigned = labels >= 0
    totals = np.bincount(labels[assigned], weights=weights[assigned])
    if pairs == 0 or totals.sum() == 0:
        return 0.0
    observed = float(np.sum(weights * same)) / pairs
    proportions = totals / totals.sum()
    return observed - float(np.sum(proportions**2))


def lift_draws(
    labels: np.ndarray,
    neighbors: np.ndarray,
    counts: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    same, valid = neighbor_pairs(labels, neighbors)
    return np.array(
        [
            weighted_neighbor_lift(labels, same, valid, draw[positions].astype(float))
            for draw in counts
        ]
    )


def compare_with_reference(
    challenger: dict[str, Any],
    reference: dict[str, Any],
    draws: dict[str, np.ndarray],
    minimum_gain: float,
) -> dict[str, Any]:
    """§25.2: at least 5% more lift and a 95% interval entirely above 0."""
    difference = draws[challenger["key"]] - draws[reference["key"]]
    quantiles = np.quantile(difference, list(DIFFERENCE_QUANTILES.values()))
    interval = {
        label: float(value) for label, value in zip(DIFFERENCE_QUANTILES, quantiles)
    }
    gain = challenger["neighbor_lift"] / reference["neighbor_lift"] - 1
    return {
        "challenger": challenger["key"],
        "relative_gain": float(gain),
        "difference": {
            "estimate": float(challenger["neighbor_lift"] - reference["neighbor_lift"]),
            **interval,
        },
        "wins": bool(gain >= minimum_gain and interval["p025"] > 0),
    }


def decide(
    results: dict[str, dict[str, Any]],
    neighbors: np.ndarray,
    weeks: np.ndarray,
    settings: dict[str, Any],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Compare every accepted candidate with the M7 reference (§25.2 and §25.3)."""
    reference_space = settings["reference_space"]
    reference_id = settings["reference_candidate"]
    reference_result = results[task_key(reference_space, "kmeans")]
    reference_labels = reference_result["labels"][reference_id]

    candidates = []
    for result in results.values():
        for record in result["records"]:
            candidate_labels = result["labels"][record["id"]]
            candidates.append(
                {
                    **record,
                    "space": result["space"],
                    "key": f"{result['space']}/{record['id']}",
                    "ari_vs_reference": float(
                        adjusted_rand_score(reference_labels, candidate_labels)
                    ),
                }
            )
    by_key = {candidate["key"]: candidate for candidate in candidates}
    reference = by_key[f"{reference_space}/{reference_id}"]

    pool = [
        candidate
        for candidate in candidates
        if candidate["selected"]
        and candidate["accepted"]
        and candidate is not reference
    ]
    counts, positions = week_resamples(weeks, draws, seed)
    lifts = {}
    for candidate in [reference, *pool]:
        result = results[task_key(candidate["space"], candidate["algorithm"])]
        lifts[candidate["key"]] = lift_draws(
            result["labels"][candidate["id"]], neighbors, counts, positions
        )
    comparisons = [
        compare_with_reference(
            candidate, reference, lifts, settings["minimum_relative_gain"]
        )
        for candidate in pool
    ]
    winners = [
        by_key[comparison["challenger"]]
        for comparison in comparisons
        if comparison["wins"]
    ]
    winner = max(
        winners, key=lambda candidate: candidate["neighbor_lift"], default=None
    )

    space_change = {}
    for space in space_names(settings):
        same_k = by_key[f"{space}/{reference_id}"]["ari_vs_reference"]
        space_change[space] = {
            "ari_same_candidate": same_k,
            "changes": bool(same_k < settings["change_ari"]),
        }
    return {
        "reference": {
            "key": reference["key"],
            "accepted": reference["accepted"],
            "neighbor_lift": reference["neighbor_lift"],
        },
        "comparisons": comparisons,
        "winner": winner["key"] if winner else None,
        "replaces_reference": winner is not None,
        "space_change": space_change,
        "candidates": candidates,
    }


def reproduction_check(
    reference_result: dict[str, Any],
    clustering_dir: Path,
    minimum_ari: float,
    reference_id: str,
) -> dict[str, Any]:
    """The reference space must reproduce the M7 labels before anything is compared."""
    scores = {
        candidate_id: float(
            adjusted_rand_score(
                np.load(clustering_dir / "labels" / f"{candidate_id}.npy"),
                labels,
            )
        )
        for candidate_id, labels in reference_result["labels"].items()
        if (clustering_dir / "labels" / f"{candidate_id}.npy").exists()
    }
    passed = scores.get(reference_id, 0.0) >= minimum_ari
    if not passed:
        raise ValueError(
            f"M7S does not reproduce M7 {reference_id}: ARI {scores.get(reference_id)}."
        )
    return {"ari_vs_m7": scores, "minimum_ari": minimum_ari, "passed": passed}


def load_weeks(input_path: Path, complaint_ids: np.ndarray) -> np.ndarray:
    """Week of each comparison row, with the weeks of M8 (Monday to Sunday)."""
    table = pq.read_table(
        input_path,
        columns=[ID_COLUMN, DATE_COLUMN],
        filters=[(ID_COLUMN, "in", complaint_ids.tolist())],
    )
    frame = table.to_pandas().drop_duplicates(ID_COLUMN).set_index(ID_COLUMN)
    dates = pd.to_datetime(frame[DATE_COLUMN].reindex(complaint_ids))
    if dates.isna().any():
        raise ValueError("M7S could not recover every complaint date.")
    return dates.dt.to_period("W-SUN").dt.start_time.to_numpy()


def candidate_table(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    table = pd.DataFrame(candidates).drop(columns=["key"])
    table["parameters"] = table["parameters"].map(json.dumps)
    for column in ("initial_failures", "final_failures"):
        table[column] = table[column].map(",".join)
    first = ["space", "id", "algorithm", "selected", "accepted", "neighbor_lift"]
    rest = [column for column in table.columns if column not in first]
    return table.reindex(columns=first + rest)


def public_report(decision: dict[str, Any]) -> dict[str, Any]:
    selected = [
        {
            key: candidate[key]
            for key in (
                "key",
                "algorithm",
                "accepted",
                "final_failures",
                "neighbor_lift",
                "silhouette_mean",
                "stability_ari",
                "future_coverage",
                "noise_fraction",
                "largest_cluster_fraction",
                "ari_vs_reference",
            )
        }
        for candidate in decision["candidates"]
        if candidate["selected"]
    ]
    return {key: value for key, value in decision.items() if key != "candidates"} | {
        "selected": selected
    }


def prepare(config: dict[str, Any], index: int) -> None:
    settings = config["space_sensitivity"]
    seed = config["experiment"]["seed"]
    name = space_names(settings)[index]
    space = settings["spaces"][name]
    root = PROJECT_ROOT / config["paths"]["space_sensitivity_artifacts"] / "spaces"
    final = root / name
    staging = start_output(final)
    if staging is None:
        print(f"M7S space {name} is already complete.")
        return

    started = time.perf_counter()
    clustering_dir = PROJECT_ROOT / config["paths"]["clustering_artifacts"]
    inputs = load_inputs(config)
    if space.get("reuse_m7"):
        fit_values, calibration_values, details = load_m7_space(clustering_dir)
    else:
        fit_values, calibration_values, details = build_space(
            space, inputs, config["clustering"]["umap"], seed
        )
    positions = fixed_sample_positions(
        len(fit_values), config["clustering"]["comparison_rows"], seed
    )
    m7_positions = pd.read_parquet(clustering_dir / "comparison_sample.parquet")
    if not np.array_equal(positions, m7_positions["source_position"].to_numpy()):
        raise ValueError("M7S comparison rows differ from the M7 sample.")
    np.save(staging / "sample.npy", fit_values[positions])
    np.save(staging / "calibration.npy", calibration_values)
    write_json(
        staging / "metadata.json",
        {
            "space": name,
            **space,
            **details,
            "fitted_on_rows": len(fit_values),
            "sample_rows": len(positions),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    staging.rename(final)
    print(f"M7S space {name} ready: {final.relative_to(PROJECT_ROOT)}")


def cluster(config: dict[str, Any], index: int) -> None:
    settings = config["space_sensitivity"]
    seed = config["experiment"]["seed"]
    space_name, algorithm = cluster_tasks(settings)[index]
    root = PROJECT_ROOT / config["paths"]["space_sensitivity_artifacts"]
    final = root / "clusters" / task_key(space_name, algorithm)
    staging = start_output(final)
    if staging is None:
        print(f"M7S task {final.name} is already complete.")
        return

    space_dir = root / "spaces" / space_name
    if not space_dir.exists():
        raise FileNotFoundError(f"Run M7S prepare first: {space_dir}")
    inputs = load_inputs(config)
    positions = fixed_sample_positions(
        len(inputs["frames"]["fit"]), config["clustering"]["comparison_rows"], seed
    )
    context = build_context(
        inputs["frames"]["fit"],
        inputs["fit_bge"],
        positions,
        config["clustering"]["silhouette_rows"],
        config,
        seed,
    )
    records, labels = run_algorithm(
        algorithm,
        np.load(space_dir / "sample.npy"),
        np.load(space_dir / "calibration.npy"),
        context,
        clustering_settings(config, settings["spaces"][space_name]),
        seed,
        settings["threads"],
    )
    write_json(staging / "records.json", records)
    # One row per candidate, in the order of records.json.
    np.save(staging / "labels.npy", np.stack([labels[r["id"]] for r in records]))
    staging.rename(final)
    print(f"M7S task {final.name} ready.")


def load_results(root: Path, settings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results = {}
    missing = []
    for space, algorithm in cluster_tasks(settings):
        directory = root / "clusters" / task_key(space, algorithm)
        if not directory.exists():
            missing.append(directory.name)
            continue
        records = read_json(directory / "records.json")
        stored = np.load(directory / "labels.npy")
        results[task_key(space, algorithm)] = {
            "space": space,
            "records": records,
            "labels": {
                record["id"]: row for record, row in zip(records, stored, strict=True)
            },
        }
    if missing:
        raise FileNotFoundError(f"M7S tasks without results: {', '.join(missing)}")
    return results


def summarize(config: dict[str, Any]) -> None:
    started = time.perf_counter()
    settings = config["space_sensitivity"]
    seed = config["experiment"]["seed"]
    paths = config["paths"]
    report_path = PROJECT_ROOT / paths["space_sensitivity_report"]
    candidates_path = PROJECT_ROOT / paths["space_sensitivity_candidates"]
    for path in (report_path, candidates_path):
        if path.exists():
            raise FileExistsError(f"M7S never overwrites results: {path}")

    root = PROJECT_ROOT / paths["space_sensitivity_artifacts"]
    clustering_dir = PROJECT_ROOT / paths["clustering_artifacts"]
    results = load_results(root, settings)
    reproduction = reproduction_check(
        results[task_key(settings["reference_space"], "kmeans")],
        clustering_dir,
        settings["reproduction_ari"],
        settings["reference_candidate"],
    )
    inputs = load_inputs(config)
    positions = fixed_sample_positions(
        len(inputs["frames"]["fit"]), config["clustering"]["comparison_rows"], seed
    )
    frame = inputs["frames"]["fit"].iloc[positions]
    ids = frame[ID_COLUMN].astype(str).to_numpy()
    neighbors = semantic_neighbor_indices(
        np.ascontiguousarray(inputs["fit_bge"][positions]),
        config["clustering"]["semantic_neighbors"],
    )
    weeks = load_weeks(PROJECT_ROOT / paths["input_data"], ids)
    draws = settings["bootstrap_draws"]
    decision = decide(results, neighbors, weeks, settings, draws, seed)

    spaces = {
        name: read_json(root / "spaces" / name / "metadata.json")
        for name in space_names(settings)
    }
    report = {
        "stage": "M7S",
        "plan": "models_plan.md §25.3",
        "seed": seed,
        "split_version": config["evaluation"]["split_version"],
        "validation_used": False,
        "sources": {
            "bge_dvc_hash": settings["bge_dvc_hash"],
            "semantic_dvc_hash": settings["semantic_dvc_hash"],
            "clustering_dvc_hash": settings["clustering_dvc_hash"],
        },
        "sample": {
            "fit_rows": len(ids),
            "weeks": int(len(np.unique(weeks))),
            "id_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
        },
        "reproduction": reproduction,
        "spaces": spaces,
        "rule": {
            "minimum_relative_gain": settings["minimum_relative_gain"],
            "bootstrap_draws": settings["bootstrap_draws"],
            "bootstrap_unit": "fit week of the comparison rows",
            "change_ari": settings["change_ari"],
        },
        **public_report(decision),
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
    report_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_table(decision["candidates"]).to_csv(candidates_path, index=False)
    write_json(report_path, report)
    save_offline_record(config, report)
    print_summary(report)


def save_offline_record(config: dict[str, Any], report: dict[str, Any]) -> None:
    paths = config["paths"]
    metrics = {
        f"lift_{entry['key'].replace('/', '_')}": float(entry["neighbor_lift"])
        for entry in report["selected"]
    }
    metrics["replaces_reference"] = float(report["replaces_reference"])
    record = build_run_record(
        config=config,
        run_name="m7s-space-sensitivity",
        stage="M7S",
        target="A1",
        split="fit_sample_calibration_assignment",
        view="semantic_patterns",
        features=["BGE", "PCA", "UMAP"],
        parameters={
            "plan": report["plan"],
            "reference": report["reference"]["key"],
            "winner": str(report["winner"]),
            "spaces": json.dumps(list(report["spaces"])),
        },
        metrics=metrics,
        artifacts=[
            paths["space_sensitivity_report"],
            paths["space_sensitivity_candidates"],
        ],
    )
    save_run_record(
        record,
        PROJECT_ROOT / paths["offline_runs"] / "m7s" / "space_sensitivity" / "run.json",
    )


def print_summary(report: dict[str, Any]) -> None:
    reference = report["reference"]
    print(f"Reference: {reference['key']} lift={reference['neighbor_lift']:.4f}")
    for entry in report["selected"]:
        print(
            f"{entry['key']}: accepted={entry['accepted']} "
            f"lift={entry['neighbor_lift']:.4f} ari={entry['ari_vs_reference']:.3f} "
            f"failures={entry['final_failures']}"
        )
    for comparison in report["comparisons"]:
        difference = comparison["difference"]
        print(
            f"{comparison['challenger']}: gain={comparison['relative_gain']:+.1%} "
            f"95% [{difference['p025']:+.4f}, {difference['p975']:+.4f}] "
            f"wins={comparison['wins']}"
        )
    for space, change in report["space_change"].items():
        print(
            f"{space}: ARI={change['ari_same_candidate']:.3f} "
            f"changes={change['changes']}"
        )
    print(f"Winner: {report['winner']}")


def smoke(config: dict[str, Any]) -> None:
    """Every step on a few rows, in memory, so the full run fails less often."""
    settings = config["space_sensitivity"]
    seed = config["experiment"]["seed"]
    inputs = subset_inputs(load_inputs(config), SMOKE_ROWS, seed)
    positions = fixed_sample_positions(SMOKE_ROWS["fit"], SMOKE_ROWS["sample"], seed)
    context = build_context(
        inputs["frames"]["fit"],
        inputs["fit_bge"],
        positions,
        SMOKE_ROWS["silhouette"],
        config,
        seed,
    )
    results = {}
    for name in space_names(settings):
        space = settings["spaces"][name]
        started = time.perf_counter()
        fit_values, calibration_values, _ = build_space(
            space, inputs, config["clustering"]["umap"], seed
        )
        print(f"{name}: space in {time.perf_counter() - started:.1f}s", flush=True)
        for algorithm in ALGORITHMS:
            records, labels = run_algorithm(
                algorithm,
                fit_values[positions],
                calibration_values,
                context,
                clustering_settings(config, space),
                seed,
                settings["threads"],
            )
            results[task_key(name, algorithm)] = {
                "space": name,
                "records": records,
                "labels": labels,
            }
            seconds = ", ".join(
                f"{record['id']}={record['elapsed_seconds']:.1f}s" for record in records
            )
            print(f"  {algorithm}: {seconds}", flush=True)
    for result in results.values():
        # The full run writes these records as JSON without NaN.
        json.dumps(result["records"], allow_nan=False)
    ids = context["frame"][ID_COLUMN].astype(str).to_numpy()
    weeks = load_weeks(PROJECT_ROOT / config["paths"]["input_data"], ids)
    decision = decide(
        results, context["neighbors"], weeks, settings, SMOKE_BOOTSTRAP_DRAWS, seed
    )
    report = public_report(decision)
    json.dumps(report, allow_nan=False)
    print_summary(report)
    print("M7S smoke finished; nothing was saved.")


def main() -> None:
    args = parse_args()
    config = load_experiment_config()
    set_seed(config["experiment"]["seed"])
    check_sources(config)
    if args.step == "prepare":
        prepare(config, task_index(args.index))
    elif args.step == "cluster":
        cluster(config, task_index(args.index))
    elif args.step == "summarize":
        summarize(config)
    else:
        smoke(config)


if __name__ == "__main__":
    main()
