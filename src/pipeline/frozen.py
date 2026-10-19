"""The frozen artifacts that `score` loads and `fit` checks (models_plan.md §29).

Every path comes from `configs/modeling.yaml`; the run keys are the ones the
pre-registration log records as current. The manifest lists each artifact
with the MD5 of its DVC pointer, so a score run can say exactly which frozen
models produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import yaml

from src.evaluation.experiment import PROJECT_ROOT
from src.models.semantic_space import load_dvc_hash

M9_RUN_ROOTS = {
    "report": "reports/modeling/weekly_counts",
    "artifact": "artifacts/models/weekly_counts",
}
M10_RUN_ROOTS = {
    "report": "reports/modeling/weekly_composition",
    "artifact": "artifacts/models/weekly_composition",
}
DAILY_RUN_ROOTS = {
    "report": "reports/modeling/daily_counts",
    "artifact": "artifacts/models/daily_counts",
}


@dataclass(frozen=True)
class FrozenArtifacts:
    """Where each frozen component lives, resolved from the config."""

    input_data: Path
    tfidf_dir: Path
    representation_dir: Path
    semantic_dir: Path
    clustering_dir: Path
    patterns_dir: Path
    bge_full_dir: Path
    cluster_summary: Path
    weekly_counts: Path
    daily_counts: Path
    m9_report: Path
    m9_artifact: Path
    m10_artifact: Path
    m11_report: Path
    daily_report: Path
    daily_artifact: Path
    foundation_report: Path
    tabpfn_weights: Path
    dvc_pointers: dict[str, Path] = field(default_factory=dict)


def frozen_artifacts(config: dict[str, Any]) -> FrozenArtifacts:
    paths = config["paths"]
    m11 = config["persistent_change"]
    m10 = config["mixture_signal"]
    daily = config["daily_change"]
    root = PROJECT_ROOT
    pointers = {
        name: root / paths[f"{name}_dvc"]
        for name in (
            "bge_full_artifacts",
            "semantic_artifacts",
            "weekly_patterns_artifacts",
        )
    }
    pointers["tfidf"] = root / "artifacts/models/tfidf.dvc"
    pointers["representation_comparison"] = (
        root / "artifacts/models/representation_comparison.dvc"
    )
    pointers["clustering_comparison"] = root / paths["clustering_artifacts_dvc"]
    pointers["weekly_counts"] = root / "artifacts/models/weekly_counts.dvc"
    pointers["weekly_composition"] = root / "artifacts/models/weekly_composition.dvc"
    pointers["daily_counts"] = root / "artifacts/models/daily_counts.dvc"
    return FrozenArtifacts(
        input_data=root / paths["input_data"],
        tfidf_dir=root / paths["tfidf_models"],
        representation_dir=root / paths["representation_artifacts"],
        semantic_dir=root / paths["semantic_artifacts"],
        clustering_dir=root / paths["clustering_artifacts"],
        patterns_dir=root / paths["weekly_patterns_artifacts"],
        bge_full_dir=root / paths["bge_full_artifacts"],
        cluster_summary=root / paths["cluster_summary"],
        weekly_counts=root / paths["weekly_counts"],
        daily_counts=root / paths["daily_counts"],
        m9_report=root / M9_RUN_ROOTS["report"] / m11["m9_model"] / m11["m9_run_key"],
        m9_artifact=root
        / M9_RUN_ROOTS["artifact"]
        / m11["m9_model"]
        / m11["m9_run_key"],
        m10_artifact=root
        / M10_RUN_ROOTS["artifact"]
        / m10["m10_model"]
        / m10["m10_run_key"],
        m11_report=root / m11["report_dir"] / "results.json",
        daily_report=root
        / DAILY_RUN_ROOTS["report"]
        / daily["m9d_model"]
        / daily["m9d_run_key"],
        daily_artifact=root
        / DAILY_RUN_ROOTS["artifact"]
        / daily["m9d_model"]
        / daily["m9d_run_key"],
        foundation_report=root / config["foundation_t1"]["report"],
        tabpfn_weights=Path(config["foundation_t1"]["tabpfn_model_path"]).expanduser(),
        dvc_pointers=pointers,
    )


REQUIRED_FILES = {
    "tfidf_dir": ("vectorizer.joblib", "product_encoder.joblib"),
    "representation_dir": (
        "tfidf_text_t2.joblib",
        "tfidf_text_t3.joblib",
        "bge_product_t1.joblib",
    ),
    "semantic_dir": ("pca.joblib", "faiss_fit.index", "faiss_fit_ids.parquet"),
    "clustering_dir": ("umap_15d.joblib",),
    "patterns_dir": ("kmeans.joblib", "novelty_thresholds.npy"),
    "bge_full_dir": ("manifest.parquet",),
    "m9_artifact": ("posterior.nc",),
    "m9_report": ("results.json",),
    "m10_artifact": ("posterior.nc",),
    "daily_artifact": ("posterior.nc", "model_spec.json"),
    "daily_report": ("results.json",),
}
REQUIRED_PATHS = (
    "input_data",
    "cluster_summary",
    "weekly_counts",
    "daily_counts",
    "m11_report",
    "foundation_report",
)


def missing_artifacts(frozen: FrozenArtifacts) -> list[str]:
    """Paths that `score` needs and are not on disk (`dvc pull` fetches them)."""
    missing = []
    for attribute, names in REQUIRED_FILES.items():
        directory: Path = getattr(frozen, attribute)
        for name in names:
            if not (directory / name).exists():
                missing.append(str((directory / name).relative_to(PROJECT_ROOT)))
    for attribute in REQUIRED_PATHS:
        path: Path = getattr(frozen, attribute)
        if not path.exists():
            missing.append(str(path.relative_to(PROJECT_ROOT)))
    if not frozen.tabpfn_weights.exists():
        missing.append(
            f"{frozen.tabpfn_weights} (TabPFN weights; the linear T1 is used instead)"
        )
    return missing


def run_name(settings: dict[str, Any], model_key: str, run_key: str) -> str:
    return f"{settings[model_key]}/{settings[run_key]}"


def manifest(frozen: FrozenArtifacts, config: dict[str, Any]) -> dict[str, Any]:
    """What a score run was produced with: run keys and DVC hashes."""
    hashes = {}
    for name, pointer in frozen.dvc_pointers.items():
        hashes[name] = load_dvc_hash(pointer) if pointer.exists() else None
    return {
        "m9_run": run_name(config["persistent_change"], "m9_model", "m9_run_key"),
        "m10_run": run_name(config["mixture_signal"], "m10_model", "m10_run_key"),
        "daily_run": run_name(config["daily_change"], "m9d_model", "m9d_run_key"),
        "m11_thresholds": str(frozen.m11_report.relative_to(PROJECT_ROOT)),
        "tabpfn_context_rows": config["foundation_t1"]["context_rows"],
        "tabpfn_pca_components": config["foundation_t1"]["pca_components"],
        "dvc_hashes": hashes,
    }


def load_model_spec(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "model_spec.json").read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))
