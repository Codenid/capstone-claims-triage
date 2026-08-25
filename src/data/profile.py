"""E1 — Perfil de tipos, nulos, cardinalidad y memoria.

Entregable del plan: ``reports/artefactos/e01_tabla_tipos.csv`` (16 filas), la tabla de calidad
de datos que va a la slide y que fija el punto de partida del contrato de tipos
de la Fase 2.

Dos pasadas, ambas por lotes: el parquet completo son ~5.1 GB en memoria y la
narrativa sola no cabe en un ``value_counts``.

- Pasada 1 (todas las columnas, por lotes): nulos, longitudes y cardinalidad
  sobre un **hash** de los valores. Nunca materializa la columna entera.
- Pasada 2 (solo columnas de baja cardinalidad): valor más frecuente.

Uso:
    uv run python -m src.data.profile
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data.load_raw import (
    COLUMNA_ID,
    COLUMNAS_DESCARTADAS,
    COLUMNAS_F0,
    COLUMNAS_F1,
    COLUMNAS_FX,
)
from src.paths import PARQUET_CRUDO, REPORTS, asegurar

ARTEFACTOS = REPORTS / "artefactos"

TAM_LOTE = 262_144
# Por encima de esta cardinalidad no se busca el valor más frecuente: no dice
# nada útil y la segunda pasada dejaría de ser barata.
MAX_CARD_PARA_TOP = 5_000


def _tier(columna: str) -> str:
    """Tier del contrato de disponibilidad (decisión D2 del plan)."""
    if columna in COLUMNAS_F0:
        return "F0"
    if columna in COLUMNAS_F1:
        return "F1"
    if columna in COLUMNAS_FX:
        return "FX"
    if columna == COLUMNA_ID:
        return "ID"
    if columna in COLUMNAS_DESCARTADAS:
        return "FUERA"
    return "SIN_CLASIFICAR"


def _pasada_1(nombre: str) -> dict:
    """Nulos, longitudes y cardinalidad por hash, leyendo por lotes."""
    archivo = pq.ParquetFile(PARQUET_CRUDO)

    n_total = 0
    n_nulos = 0
    n_bytes = 0
    hashes: list[np.ndarray] = []
    longitudes: list[np.ndarray] = []

    for lote in archivo.iter_batches(batch_size=TAM_LOTE, columns=[nombre]):
        col = lote.column(0)
        n_total += len(col)
        n_nulos += col.null_count
        n_bytes += col.nbytes

        validos = col.drop_null()
        if len(validos) == 0:
            continue

        longitudes.append(pc.utf8_length(validos).to_numpy(zero_copy_only=False))
        valores = pd.Series(validos.to_numpy(zero_copy_only=False), dtype="object")
        hashes.append(pd.util.hash_array(valores.to_numpy(), categorize=True))

    n_no_nulos = n_total - n_nulos
    fila = {
        "columna": nombre,
        "tier": _tier(nombre),
        "n_no_nulos": n_no_nulos,
        "pct_nulos": round(100 * n_nulos / n_total, 2) if n_total else 0.0,
        "mb_arrow": round(n_bytes / 1024**2, 1),
    }

    if not hashes:
        return fila | {
            "n_unicos": 0,
            "pct_unicos": 0.0,
            "long_media": None,
            "long_p50": None,
            "long_p99": None,
        }

    todas_long = np.concatenate(longitudes)
    n_unicos = int(np.unique(np.concatenate(hashes)).size)

    return fila | {
        "n_unicos": n_unicos,
        "pct_unicos": round(100 * n_unicos / n_no_nulos, 2),
        "long_media": round(float(todas_long.mean()), 1),
        "long_p50": int(np.percentile(todas_long, 50)),
        "long_p99": int(np.percentile(todas_long, 99)),
    }


def _pasada_2(nombre: str, n_no_nulos: int) -> dict:
    """Valor más frecuente. Solo para columnas de baja cardinalidad."""
    col = pq.read_table(PARQUET_CRUDO, columns=[nombre]).column(nombre)
    conteos = col.drop_null().value_counts()
    valores = conteos.field("values").to_pylist()
    cuentas = np.asarray(conteos.field("counts"))
    i = int(cuentas.argmax())
    top = valores[i]
    return {
        "valor_top": top[:60] if isinstance(top, str) else top,
        "freq_top": int(cuentas[i]),
        "pct_top": round(100 * int(cuentas[i]) / n_no_nulos, 2) if n_no_nulos else 0.0,
    }


def perfilar() -> pd.DataFrame:
    """Perfila las 16 columnas y escribe ``reports/artefactos/e01_tabla_tipos.csv``."""
    esquema = pq.read_schema(PARQUET_CRUDO)
    tipos = {n: str(t) for n, t in zip(esquema.names, esquema.types)}
    total = pq.ParquetFile(PARQUET_CRUDO).metadata.num_rows
    print(f"{PARQUET_CRUDO.name}: {total:,} filas × {len(tipos)} columnas\n")

    filas = []
    for nombre in esquema.names:
        print(f"  [1/2] {nombre}", flush=True)
        fila = _pasada_1(nombre)
        fila["tipo_crudo"] = tipos[nombre]

        if 0 < fila["n_unicos"] <= MAX_CARD_PARA_TOP:
            print(f"  [2/2] {nombre} (card. {fila['n_unicos']})", flush=True)
            fila |= _pasada_2(nombre, fila["n_no_nulos"])
        else:
            fila |= {"valor_top": None, "freq_top": None, "pct_top": None}

        filas.append(fila)

    columnas = [
        "columna", "tier", "tipo_crudo", "n_no_nulos", "pct_nulos", "n_unicos",
        "pct_unicos", "valor_top", "freq_top", "pct_top", "long_media",
        "long_p50", "long_p99", "mb_arrow",
    ]
    orden_tier = {"F0": 0, "F1": 1, "FX": 2, "ID": 3, "FUERA": 4, "SIN_CLASIFICAR": 5}
    perfil = (
        pd.DataFrame(filas)[columnas]
        .assign(_o=lambda d: d["tier"].map(orden_tier))
        .sort_values(["_o", "pct_nulos"])
        .drop(columns="_o")
        .reset_index(drop=True)
    )

    asegurar(ARTEFACTOS)
    destino = ARTEFACTOS / "e01_tabla_tipos.csv"
    perfil.to_csv(destino, index=False, encoding="utf-8")
    print(f"\n→ {destino}")
    print(f"memoria Arrow de las 16 columnas: {perfil['mb_arrow'].sum() / 1024:.2f} GB")
    return perfil


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n" + perfilar().to_string(index=False))
