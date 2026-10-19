"""`fit`: the once-only chain of the modeling stages (models_plan.md §29).

The stages already exist as DVC stages and Khipu jobs; `fit` does not retrain
anything by itself. It checks which frozen artifacts are on disk, prints the
ordered chain with the command of each step, and writes the manifest that
`score` will stamp on its outputs. With `--run-preparation` it runs the DVC
preparation stages (P1-P6) locally; the modeling jobs run on Khipu.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from src.evaluation.experiment import PROJECT_ROOT
from src.pipeline.frozen import frozen_artifacts, manifest, missing_artifacts

CHAIN: tuple[tuple[str, str, str], ...] = (
    (
        "P1-P6",
        "Preparación: esquema, texto, taxonomía, objetivos, tabla y diccionario",
        "uv run dvc repro",
    ),
    (
        "M4",
        "TF-IDF y lineales de T1-T4 (vectorizador, codificador de producto)",
        "sbatch scripts/hpc/m4_tfidf.slurm",
    ),
    (
        "M8A",
        "Embeddings BGE-large de todas las narrativas",
        "sbatch scripts/hpc/m8a_bge_full.slurm",
    ),
    (
        "M5B",
        "Comparación de representaciones; clasificadores congelados T1-T3",
        "sbatch scripts/hpc/m5b_representation_comparison.slurm",
    ),
    (
        "M6",
        "PCA 256, UMAP 2D e índice FAISS de ajuste",
        "sbatch scripts/hpc/m6_semantic_space.slurm",
    ),
    (
        "M7",
        "UMAP 15 y comparación de agrupamientos (k-means k=40)",
        "sbatch scripts/hpc/m7_clustering.slurm",
    ),
    (
        "M8B",
        "Asignación de patrones, umbrales de novedad y conteos semanales",
        "sbatch scripts/hpc/m8b_weekly_patterns.slurm",
    ),
    (
        "M5F",
        "TabPFN-3.5 con contexto fijo (pesos locales)",
        "sbatch scripts/hpc/m5f_foundation_t1.slurm",
    ),
    (
        "M9",
        "C-A: binomial negativa jerárquica con memoria que decae (full)",
        "MODEL_CONFIG=configs/weekly_counts/nb_discounted_hierarchical_v1.yaml "
        "sbatch scripts/hpc/m9_weekly_counts.slurm",
    ),
    (
        "M10",
        "DM-R4: Dirichlet-multinomial de 4 semanas (full)",
        "MODEL_CONFIG=configs/weekly_composition/"
        "dirichlet_multinomial_rolling_4_v1.yaml "
        "sbatch scripts/hpc/m10_weekly_composition.slurm",
    ),
    (
        "M11",
        "Umbrales de CUSUM y regla semanal por simulación",
        "sbatch scripts/hpc/m11_persistent_change.slurm",
    ),
    (
        "M9D",
        "Panel diario y D-D (full)",
        "MODEL_CONFIG=configs/daily_counts/zinb_daily_hierarchical_v1.yaml "
        "sbatch scripts/hpc/m9d_daily_counts.slurm",
    ),
)


def run_preparation() -> None:
    subprocess.run(
        ["uv", "run", "--no-sync", "dvc", "repro"], cwd=PROJECT_ROOT, check=True
    )


def fit_report(config: dict[str, Any], run_preparation_stages: bool) -> dict[str, Any]:
    if run_preparation_stages:
        run_preparation()
    frozen = frozen_artifacts(config)
    missing = missing_artifacts(frozen)
    return {
        "plan": "models_plan.md §29",
        "chain": [{"stage": s, "what": w, "command": c} for s, w, c in CHAIN],
        "missing": missing,
        "ready": not [m for m in missing if "TabPFN weights" not in m],
        "tabpfn_available": frozen.tabpfn_weights.exists(),
        "manifest": manifest(frozen, config),
    }


def print_fit_report(report: dict[str, Any]) -> None:
    print("Cadena de ajuste (una vez):")
    for step in report["chain"]:
        print(f"  {step['stage']:<6} {step['what']}\n         {step['command']}")
    if report["missing"]:
        print("\nArtefactos ausentes (dvc pull, o correr la etapa):")
        for item in report["missing"]:
            print(f"  - {item}")
    ready = "sí" if report["ready"] else "no"
    tabpfn = "sí" if report["tabpfn_available"] else "no (T1 lineal)"
    print(f"\nListo para score: {ready} · TabPFN: {tabpfn}")
    print("Manifiesto:", json.dumps(report["manifest"], indent=2, ensure_ascii=False))


def write_fit_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
