"""E13: comparación controlada R0 TF-IDF frente a R1 MiniLM.

Usa exactamente las mismas filas, targets, splits, pesos y clasificador lineal.
Compara señal predictiva, memoria de artefacto/RAM y latencia. No selecciona el
modelo final ni incorpora F1/FX como features.

Uso:
    uv run python -m src.evaluation.compare_embeddings
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, f1_score

from src.data.prepare_eda import cargar_parametros
from src.models.baseline import (
    SPLITS_EVALUACION,
    _top_k_accuracy,
    ajustar_lineal,
    mascara_vista,
    metricas_binarias,
)
from src.paths import PROCESSED, REPORTS, asegurar

R0_MATRIZ = PROCESSED / "r0_tfidf.npz"
R0_INDICE = PROCESSED / "r0_index.parquet"
R0_VECTORIZADORES = PROCESSED / "r0_vectorizers.joblib"
R1_MATRIZ = PROCESSED / "r1_embeddings.npy"
R1_INDICE = PROCESSED / "r1_hash_index.parquet"
SALIDA = REPORTS / "artefactos" / "e13_embeddings.csv"


def expandir_r1(indice_eventos: pd.DataFrame) -> np.ndarray:
    """Expande el cache por hash al mismo orden de filas usado por R0."""
    indice_hash = pd.read_parquet(R1_INDICE).set_index("hash_narrativa")["embedding_row"]
    posiciones = indice_hash.reindex(indice_eventos["hash_narrativa"])
    if posiciones.isna().any():
        raise ValueError("Hay eventos sin embedding R1")
    cache = np.load(R1_MATRIZ, mmap_mode="r")
    return np.asarray(cache[posiciones.astype("int64").to_numpy()], dtype=np.float32)


def memoria_mib(matriz: np.ndarray | sp.csr_matrix) -> float:
    if sp.issparse(matriz):
        csr = matriz.tocsr()
        return float((csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes) / 1024**2)
    return float(np.asarray(matriz).nbytes / 1024**2)


def _agregar_metricas(
    filas: list[dict[str, Any]],
    base: dict[str, Any],
    y: np.ndarray,
    metricas: dict[str, float],
) -> None:
    numerico = np.issubdtype(y.dtype, np.number)
    comun = {
        **base,
        "n": int(len(y)),
        "n_positive": int(y.sum()) if numerico else pd.NA,
        "prevalence": float(y.mean()) if numerico else pd.NA,
    }
    filas.extend({**comun, "metric": nombre, "value": valor} for nombre, valor in metricas.items())


def _evaluar_representacion(
    nombre: str,
    matriz: np.ndarray | sp.csr_matrix,
    indice: pd.DataFrame,
    config: dict[str, Any],
    semilla: int,
    artifact_mib: float,
) -> list[dict[str, Any]]:
    train = indice["split"].astype(str).eq("train").to_numpy()
    hashes_train = set(indice.loc[train, "hash_narrativa"].dropna().astype(str))
    filas: list[dict[str, Any]] = []
    ram_mib = memoria_mib(matriz)

    tareas = [
        ("T1", "target_t1_issue_raw", "issue_raw"),
        ("T2", "target_t2_relief", "main_all"),
        ("T3", "target_t3_monetary", "direct"),
        ("T4", "target_t4_late", "direct"),
    ]
    for desplazamiento, (target, columna, variante) in enumerate(tareas):
        y_serie = indice[columna]
        train_valido = train & y_serie.notna().to_numpy()
        y_train = y_serie.loc[train_valido]
        if target == "T1":
            y_train_np = y_train.astype(str).to_numpy()
        else:
            y_train_np = y_train.astype("int8").to_numpy()

        inicio = time.perf_counter()
        modelo = ajustar_lineal(
            matriz[train_valido], y_train_np, config, semilla + desplazamiento
        )
        fit_seconds = time.perf_counter() - inicio

        inicio = time.perf_counter()
        probabilidad = modelo.predict_proba(matriz)
        predict_seconds = time.perf_counter() - inicio
        inference_ms_per_1000 = 1000 * predict_seconds / (len(indice) / 1000)

        for split in SPLITS_EVALUACION:
            for vista in ("operational", "purged"):
                mascara = mascara_vista(indice, split, vista, hashes_train)
                mascara &= y_serie.notna().to_numpy()
                if target == "T1":
                    y = y_serie.loc[mascara].astype(str).to_numpy()
                    proba = probabilidad[mascara]
                    pred = modelo.classes_[np.argmax(proba, axis=1)]
                    metricas = {
                        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                        "weighted_f1": float(
                            f1_score(y, pred, average="weighted", zero_division=0)
                        ),
                        "accuracy": float(accuracy_score(y, pred)),
                        "top_3_accuracy": _top_k_accuracy(y, proba, modelo.classes_, 3),
                    }
                else:
                    y = y_serie.loc[mascara].astype("int8").to_numpy()
                    metricas = metricas_binarias(y, probabilidad[mascara, 1])

                _agregar_metricas(
                    filas,
                    {
                        "target": target,
                        "variant": variante,
                        "representation": nombre,
                        "split": split,
                        "view": vista,
                        "fit_seconds": fit_seconds,
                        "inference_ms_per_1000": inference_ms_per_1000,
                        "artifact_mib": artifact_mib,
                        "ram_matrix_mib": ram_mib,
                    },
                    y,
                    metricas,
                )
    return filas


def _artifact_mib(paths: list[Path]) -> float:
    return sum(path.stat().st_size for path in paths) / 1024**2


def main() -> None:
    parametros = cargar_parametros()
    indice = pd.read_parquet(R0_INDICE)
    if not (indice["row_number"].to_numpy() == np.arange(len(indice))).all():
        raise ValueError("El índice R0 no conserva row_number")

    print("E13 R0: ajustando cuatro sondas", flush=True)
    r0 = sp.load_npz(R0_MATRIZ).tocsr()
    filas = _evaluar_representacion(
        "R0_word_char",
        r0,
        indice,
        parametros["baseline"],
        int(parametros["random_seed"]),
        _artifact_mib([R0_MATRIZ, R0_INDICE, R0_VECTORIZADORES]),
    )
    del r0

    print("E13 R1: ajustando cuatro sondas", flush=True)
    r1 = expandir_r1(indice)
    filas += _evaluar_representacion(
        "R1_MiniLM",
        r1,
        indice,
        parametros["baseline"],
        int(parametros["random_seed"]),
        _artifact_mib([R1_MATRIZ, R1_INDICE]),
    )

    print("E13 R0+R1: midiendo señal incremental", flush=True)
    r0 = sp.load_npz(R0_MATRIZ).tocsr()
    fusion = sp.hstack([r0, sp.csr_matrix(r1)], format="csr", dtype=np.float32)
    del r0
    filas += _evaluar_representacion(
        "R0_plus_R1",
        fusion,
        indice,
        parametros["baseline"],
        int(parametros["random_seed"]),
        _artifact_mib(
            [R0_MATRIZ, R0_INDICE, R0_VECTORIZADORES, R1_MATRIZ, R1_INDICE]
        ),
    )
    del fusion, r1

    resultados = pd.DataFrame(filas)
    claves = ["target", "variant", "split", "view", "metric"]
    referencia = (
        resultados.loc[resultados["representation"] == "R0_word_char", claves + ["value"]]
        .rename(columns={"value": "r0_value"})
    )
    resultados = resultados.merge(referencia, on=claves, how="left", validate="many_to_one")
    resultados["delta_vs_r0"] = resultados["value"] - resultados["r0_value"]

    asegurar(SALIDA.parent)
    resultados.to_csv(SALIDA, index=False)
    principal = resultados[
        (resultados["split"] == "test")
        & (resultados["view"] == "purged")
        & (
            ((resultados["target"] == "T1") & resultados["metric"].isin(["macro_f1", "top_3_accuracy"]))
            | ((resultados["target"] != "T1") & (resultados["metric"] == "average_precision"))
        )
    ]
    print(principal[["target", "representation", "metric", "value", "delta_vs_r0"]].to_string(index=False))
    print(f"E13 -> {SALIDA}")


if __name__ == "__main__":
    main()
