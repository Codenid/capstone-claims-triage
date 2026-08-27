"""Construcción de R0: TF-IDF sparse de palabras y caracteres.

El vocabulario y los pesos IDF se ajustan exclusivamente con el split train.
La matriz conserva el orden de ``eda_sample.parquet`` y se acompaña de un índice
Parquet y un manifiesto para impedir joins por posición sin validación.

Uso:
    uv run python -m src.features.text
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.prepare_eda import cargar_parametros
from src.paths import INTERIM, PROCESSED, asegurar

ENTRADA = INTERIM / "eda_sample.parquet"
SALIDA_MATRIZ = PROCESSED / "r0_tfidf.npz"
SALIDA_INDICE = PROCESSED / "r0_index.parquet"
SALIDA_VECTORIZADORES = PROCESSED / "r0_vectorizers.joblib"
SALIDA_MANIFIESTO = PROCESSED / "r0_manifest.json"

COLUMNAS_INDICE = [
    "complaint_id",
    "date_received",
    "split",
    "hash_narrativa",
    "target_t1_issue_raw",
    "target_t2_relief",
    "target_t3_monetary",
    "target_t4_late",
    "t2_ambiguous",
    "company_response",
]


def _config_vectorizador(config: dict[str, Any]) -> dict[str, Any]:
    """Convierte listas YAML a tuplas y fija un dtype sparse compacto."""
    resultado = dict(config)
    resultado["ngram_range"] = tuple(resultado["ngram_range"])
    resultado["dtype"] = np.float32
    return resultado


def construir_r0(
    textos: pd.Series,
    splits: pd.Series,
    config: dict[str, Any],
) -> tuple[sp.csr_matrix, dict[str, TfidfVectorizer]]:
    """Ajusta R0 en train y transforma todas las filas en el orden recibido."""
    textos = textos.fillna("").astype(str)
    mascara_train = splits.astype(str).eq("train").to_numpy()
    if not mascara_train.any():
        raise ValueError("No hay filas train para ajustar TF-IDF")

    word = TfidfVectorizer(**_config_vectorizador(config["word"]))
    char = TfidfVectorizer(**_config_vectorizador(config["char"]))
    textos_train = textos.loc[mascara_train]

    word.fit(textos_train)
    char.fit(textos_train)
    matriz = sp.hstack(
        [word.transform(textos), char.transform(textos)],
        format="csr",
        dtype=np.float32,
    )
    return matriz, {"word": word, "char": char}


def _manifiesto(
    matriz: sp.csr_matrix,
    vectorizadores: dict[str, TfidfVectorizer],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "representation": "R0",
        "fit_split": "train",
        "row_order": "data/processed/r0_index.parquet:row_number",
        "shape": list(matriz.shape),
        "nnz": int(matriz.nnz),
        "dtype": str(matriz.dtype),
        "features": {
            nombre: len(vectorizador.vocabulary_)
            for nombre, vectorizador in vectorizadores.items()
        },
        "scikit_learn_version": sklearn.__version__,
        "params": config,
    }


def main() -> None:
    parametros = cargar_parametros()
    muestra = pd.read_parquet(ENTRADA)
    matriz, vectorizadores = construir_r0(
        muestra["narrative"], muestra["split"], parametros["tfidf"]
    )

    indice = muestra[COLUMNAS_INDICE].copy()
    indice.insert(0, "row_number", np.arange(len(indice), dtype=np.int64))
    if matriz.shape[0] != len(indice):
        raise ValueError("La matriz y el índice tienen distinto número de filas")

    asegurar(PROCESSED)
    sp.save_npz(SALIDA_MATRIZ, matriz, compressed=True)
    indice.to_parquet(SALIDA_INDICE, index=False, compression="zstd")
    joblib.dump(vectorizadores, SALIDA_VECTORIZADORES, compress=3)
    SALIDA_MANIFIESTO.write_text(
        json.dumps(_manifiesto(matriz, vectorizadores, parametros["tfidf"]), indent=2),
        encoding="utf-8",
    )

    print(f"R0: {matriz.shape[0]:,} x {matriz.shape[1]:,}; nnz={matriz.nnz:,}")
    print(f"Matriz -> {SALIDA_MATRIZ}")


if __name__ == "__main__":
    main()
