"""Construcción de R1 con embeddings locales MiniLM cacheados por hash.

Cada narrativa normalizada se codifica una sola vez. Los eventos repetidos se
reconstruyen mediante ``r1_hash_index.parquet``; el texto no sale de la máquina
y no se usa ninguna API de pago.

Uso:
    uv run python -m src.features.embeddings
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer

from src.data.prepare_eda import cargar_parametros, normalizar_narrativa
from src.paths import INTERIM, PROCESSED, asegurar

ENTRADA = INTERIM / "eda_sample.parquet"
SALIDA_EMBEDDINGS = PROCESSED / "r1_embeddings.npy"
SALIDA_INDICE = PROCESSED / "r1_hash_index.parquet"
SALIDA_MANIFIESTO = PROCESSED / "r1_manifest.json"
PARCIAL_EMBEDDINGS = PROCESSED / "r1_embeddings.partial.npy"
ESTADO_EMBEDDINGS = PROCESSED / "r1_embeddings.partial.json"


def textos_unicos(muestra: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una fila determinista por hash y su frecuencia de eventos."""
    if muestra["hash_narrativa"].isna().any():
        raise ValueError("No se pueden cachear narrativas sin hash")

    representantes = (
        muestra[["hash_narrativa", "narrative"]]
        .sort_values("hash_narrativa", kind="stable")
        .drop_duplicates("hash_narrativa", keep="first")
        .reset_index(drop=True)
    )
    representantes["texto_normalizado"] = representantes["narrative"].map(
        normalizar_narrativa
    )
    conteos = muestra["hash_narrativa"].value_counts()
    representantes["n_eventos"] = (
        representantes["hash_narrativa"].map(conteos).astype("int32")
    )
    representantes.insert(0, "embedding_row", np.arange(len(representantes), dtype=np.int64))
    return representantes


def _guardar_estado(path: Path, estado: dict[str, Any]) -> None:
    temporal = path.with_suffix(".tmp")
    temporal.write_text(json.dumps(estado, indent=2), encoding="utf-8")
    temporal.replace(path)


def codificar(
    textos: pd.Series,
    config: dict[str, Any],
) -> tuple[tuple[int, int], str]:
    """Codifica por checkpoints y reanuda una ejecución interrumpida."""
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = SentenceTransformer(
        config["model"],
        revision=config["revision"],
        device=dispositivo,
    )
    modelo.max_seq_length = int(config["max_seq_length"])
    dimension = modelo.get_sentence_embedding_dimension()
    if dimension is None:
        raise ValueError("No se pudo determinar la dimensión del embedding")

    n_filas = len(textos)
    forma = (n_filas, int(dimension))
    firma = {
        "model": config["model"],
        "revision": config["revision"],
        "max_seq_length": int(config["max_seq_length"]),
        "normalize": bool(config["normalize"]),
        "shape": list(forma),
    }
    inicio = 0
    modo = "w+"
    if PARCIAL_EMBEDDINGS.exists() and ESTADO_EMBEDDINGS.exists():
        estado = json.loads(ESTADO_EMBEDDINGS.read_text(encoding="utf-8"))
        if estado.get("signature") == firma:
            inicio = int(estado["next_row"])
            modo = "r+"
            print(f"Reanudando R1 desde la fila {inicio:,}/{n_filas:,}", flush=True)
        else:
            PARCIAL_EMBEDDINGS.unlink()
            ESTADO_EMBEDDINGS.unlink()

    destino = np.lib.format.open_memmap(
        PARCIAL_EMBEDDINGS,
        mode=modo,
        dtype=np.float32,
        shape=forma,
    )
    checkpoint = int(config["checkpoint_size"])
    lista_textos = textos.astype(str).tolist()
    for bloque_inicio in range(inicio, n_filas, checkpoint):
        bloque_fin = min(bloque_inicio + checkpoint, n_filas)
        bloque = modelo.encode(
            lista_textos[bloque_inicio:bloque_fin],
            batch_size=int(config["batch_size"]),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=bool(config["normalize"]),
        )
        bloque = np.asarray(bloque, dtype=np.float32)
        if bool(config["normalize"]):
            normas = np.linalg.norm(bloque, axis=1)
            if not np.allclose(normas, 1.0, atol=1e-4):
                raise ValueError("El modelo no devolvió embeddings normalizados")
        destino[bloque_inicio:bloque_fin] = bloque
        destino.flush()
        _guardar_estado(
            ESTADO_EMBEDDINGS,
            {"signature": firma, "next_row": bloque_fin},
        )
        print(f"Checkpoint R1: {bloque_fin:,}/{n_filas:,}", flush=True)

    del destino
    PARCIAL_EMBEDDINGS.replace(SALIDA_EMBEDDINGS)
    ESTADO_EMBEDDINGS.unlink(missing_ok=True)
    return forma, dispositivo


def main() -> None:
    parametros = cargar_parametros()
    config = parametros["embedding"]
    muestra = pd.read_parquet(ENTRADA, columns=["hash_narrativa", "narrative"])
    unicos = textos_unicos(muestra)
    asegurar(PROCESSED)
    forma, dispositivo = codificar(unicos["texto_normalizado"], config)

    if forma[0] != len(unicos):
        raise ValueError("El número de embeddings no coincide con el índice de hashes")

    indice = unicos[["embedding_row", "hash_narrativa", "n_eventos"]]
    manifiesto = {
        "representation": "R1",
        "cache_key": "hash_narrativa",
        "model": config["model"],
        "revision": config["revision"],
        "max_seq_length": int(config["max_seq_length"]),
        "normalize_embeddings": bool(config["normalize"]),
        "shape": list(forma),
        "dtype": "float32",
        "device_used": dispositivo,
        "sentence_transformers_version": sentence_transformers.__version__,
        "torch_version": torch.__version__,
    }

    indice.to_parquet(SALIDA_INDICE, index=False, compression="zstd")
    SALIDA_MANIFIESTO.write_text(json.dumps(manifiesto, indent=2), encoding="utf-8")
    print(
        f"R1: {forma[0]:,} hashes x {forma[1]:,} dimensiones "
        f"en {dispositivo}"
    )


if __name__ == "__main__":
    main()
