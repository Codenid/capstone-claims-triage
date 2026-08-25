"""Carga del parquet crudo del CFPB.

Regla del proyecto: nadie vuelve a escribir ``pd.read_parquet`` con una ruta
absoluta. Todo pasa por aquí, y por defecto se cargan solo las columnas que se
piden — el archivo completo son ~5.1 GB en memoria y 1.6 GB en disco.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

from src.paths import PARQUET_CRUDO

# Contrato F0/F1/FX de la decisión D2 del plan. Se importa desde otros módulos
# para que la prueba de no-fuga tenga una única fuente de verdad.
COLUMNAS_F0 = [
    "Date received",
    "Consumer complaint narrative",
    "Company",
    "State",
    "ZIP code",
    "Tags",
]
COLUMNAS_F1 = ["Product", "Sub-product", "Issue", "Sub-issue"]
COLUMNAS_FX = [
    "Date sent to company",
    "Company response to consumer",
    "Company public response",
    "Timely response?",
]
COLUMNA_ID = "Complaint ID"
COLUMNAS_DESCARTADAS = ["Submitted via"]  # varianza cero: un solo valor, "Web"


def esquema_crudo() -> dict[str, str]:
    """Nombre -> tipo Arrow, leyendo solo los metadatos (no carga datos)."""
    esquema = pq.read_schema(PARQUET_CRUDO)
    return {n: str(t) for n, t in zip(esquema.names, esquema.types)}


def n_filas() -> int:
    """Número de filas según los metadatos del parquet."""
    return pq.ParquetFile(PARQUET_CRUDO).metadata.num_rows


def cargar(
    columnas: Iterable[str] | None = None,
    n: int | None = None,
) -> pd.DataFrame:
    """Carga el parquet crudo.

    Args:
        columnas: subconjunto a leer. ``None`` carga las 16 (cuidado: ~5.1 GB).
        n: si se indica, devuelve solo las primeras ``n`` filas — para iterar
            rápido. No es una muestra aleatoria ni estratificada; la muestra de
            desarrollo de la decisión D5 se construye en ``src/data/sample.py``.
    """
    columnas = list(columnas) if columnas is not None else None

    if n is None:
        tabla = pq.read_table(PARQUET_CRUDO, columns=columnas)
    else:
        archivo = pq.ParquetFile(PARQUET_CRUDO)
        lotes = []
        acumuladas = 0
        for lote in archivo.iter_batches(batch_size=65_536, columns=columnas):
            lotes.append(lote)
            acumuladas += lote.num_rows
            if acumuladas >= n:
                break
        import pyarrow as pa

        tabla = pa.Table.from_batches(lotes).slice(0, n)

    return tabla.to_pandas(types_mapper=pd.ArrowDtype)
