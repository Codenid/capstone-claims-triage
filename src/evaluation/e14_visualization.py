"""Prepara una vista UMAP 2D de E14 exclusivamente para comunicación.

El clustering de E14 permanece definido en el espacio UMAP de 12 dimensiones.
Esta etapa reutiliza los embeddings y las asignaciones ya calculadas, ajusta dos
proyecciones 2D independientes (original y empresa enmascarada) y guarda una
muestra estratificada pequeña que Quarto puede graficar sin leer matrices grandes.

Uso:
    uv run python -m src.evaluation.e14_visualization
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import umap
from sklearn.model_selection import train_test_split
from umap import UMAP

from src.data.prepare_eda import cargar_parametros
from src.paths import PROCESSED, REPORTS, asegurar

R1_EMBEDDINGS = PROCESSED / "r1_embeddings.npy"
R1_INDICE = PROCESSED / "r1_hash_index.parquet"
R1_MASKED = PROCESSED / "e14_masked_embeddings.npy"
E14_INDICE = PROCESSED / "e14_train_index.parquet"
E14_ASIGNACIONES = PROCESSED / "e14_assignments.parquet"

SALIDA_VISUAL = REPORTS / "artefactos" / "e14_visualization.csv"
SALIDA_MANIFIESTO = PROCESSED / "e14_visualization_manifest.json"


def seleccionar_muestra_estratificada(
    labels: np.ndarray,
    n: int,
    semilla: int,
) -> np.ndarray:
    """Selecciona hasta ``n`` filas preservando la proporción de clusters y ruido."""
    indices = np.arange(len(labels), dtype=np.int64)
    if n >= len(indices):
        return indices
    seleccion, _ = train_test_split(
        indices,
        train_size=n,
        random_state=semilla,
        shuffle=True,
        stratify=labels,
    )
    return np.sort(np.asarray(seleccion, dtype=np.int64))


def _proyectar(matriz: np.ndarray, config: dict, semilla: int) -> np.ndarray:
    modelo = UMAP(
        n_components=2,
        n_neighbors=int(config["n_neighbors"]),
        min_dist=float(config["min_dist"]),
        metric=str(config["metric"]),
        random_state=semilla,
        n_jobs=1,
        low_memory=True,
        verbose=True,
    )
    return np.asarray(modelo.fit_transform(matriz), dtype=np.float32)


def _asignacion(asignaciones: pd.DataFrame, run_id: str, sufijo: str) -> pd.DataFrame:
    vista = asignaciones.loc[
        asignaciones["run_id"] == run_id,
        ["e14_row", "cluster", "is_noise", "membership_probability"],
    ].copy()
    if vista["e14_row"].duplicated().any():
        raise ValueError(f"La corrida {run_id} contiene e14_row duplicados")
    return vista.rename(
        columns={
            "cluster": f"cluster{sufijo}",
            "is_noise": f"is_noise{sufijo}",
            "membership_probability": f"membership_probability{sufijo}",
        }
    )


def main() -> None:
    parametros = cargar_parametros()
    semilla = int(parametros["random_seed"])
    config = parametros["visualization"]
    base_mcs = int(parametros["clustering"]["hdbscan"]["min_cluster_size"])
    run_original = f"r1_umap_seed{semilla}_mcs{base_mcs}"
    run_masked = f"r1_masked_seed{semilla}_mcs{base_mcs}"

    indice = pd.read_parquet(E14_INDICE)
    r1_indice = pd.read_parquet(R1_INDICE).set_index("hash_narrativa")["embedding_row"]
    filas_r1 = r1_indice.reindex(indice["hash_narrativa"])
    if filas_r1.isna().any():
        raise ValueError("Hay hashes de E14 ausentes del cache R1")

    r1_cache = np.load(R1_EMBEDDINGS, mmap_mode="r")
    original = np.asarray(r1_cache[filas_r1.astype("int64").to_numpy()], dtype=np.float32)
    masked = np.asarray(np.load(R1_MASKED, mmap_mode="r"), dtype=np.float32)
    if original.shape != masked.shape or len(original) != len(indice):
        raise ValueError("Embeddings e índice E14 no están alineados")

    print("E14 visual: R1 original -> UMAP 2D", flush=True)
    original_2d = _proyectar(original, config, semilla)
    print("E14 visual: R1 con empresa enmascarada -> UMAP 2D", flush=True)
    masked_2d = _proyectar(masked, config, semilla)

    asignaciones = pd.read_parquet(E14_ASIGNACIONES)
    datos = indice[
        [
            "e14_row",
            "hash_narrativa",
            "n_eventos_train",
            "product_unique",
            "issue_unique",
            "mask_changed",
        ]
    ].copy()
    datos = datos.merge(_asignacion(asignaciones, run_original, ""), on="e14_row", validate="one_to_one")
    datos = datos.merge(
        _asignacion(asignaciones, run_masked, "_masked"),
        on="e14_row",
        validate="one_to_one",
    )
    if len(datos) != len(indice):
        raise ValueError("Las asignaciones no cubren todos los hashes de E14")

    datos["umap_2d_x"] = original_2d[:, 0]
    datos["umap_2d_y"] = original_2d[:, 1]
    datos["masked_umap_2d_x"] = masked_2d[:, 0]
    datos["masked_umap_2d_y"] = masked_2d[:, 1]

    seleccion = seleccionar_muestra_estratificada(
        datos["cluster"].to_numpy(),
        int(config["plot_sample_size"]),
        semilla,
    )
    muestra = datos.iloc[seleccion].copy()
    manifiesto = {
        "purpose": "visualization_only",
        "clustering_space": "UMAP 12D; unchanged by this stage",
        "visualization_space": "independent UMAP 2D fits for original and company-masked R1",
        "fit_split": "train",
        "unit": "unique normalized narrative hash",
        "n_fit_rows": len(datos),
        "n_plot_rows": len(muestra),
        "sample_policy": "stratified by primary unmasked HDBSCAN cluster including noise",
        "seed": semilla,
        "umap": {
            "n_components": 2,
            "n_neighbors": int(config["n_neighbors"]),
            "min_dist": float(config["min_dist"]),
            "metric": str(config["metric"]),
        },
        "runs": {"original": run_original, "company_masked": run_masked},
        "umap_learn_version": umap.__version__,
    }

    asegurar(SALIDA_VISUAL.parent, PROCESSED)
    muestra.to_csv(SALIDA_VISUAL, index=False)
    SALIDA_MANIFIESTO.write_text(json.dumps(manifiesto, indent=2), encoding="utf-8")
    print(
        f"E14 visual listo: {len(datos):,} hashes ajustados; "
        f"{len(muestra):,} filas para Quarto",
        flush=True,
    )


if __name__ == "__main__":
    main()
