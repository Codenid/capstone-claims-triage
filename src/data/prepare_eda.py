"""Prepara la muestra temporal reproducible usada por E12-E14.

La etapa hace dos pasadas por el parquet crudo. La primera lee solo fecha e ID
para seleccionar reclamos de forma reproducible dentro de cada corte temporal.
La segunda recupera únicamente esas filas, deriva los targets y calcula un
SHA-1 estable de la narrativa normalizada. Así se evita materializar los 3.8 M
de textos en memoria.

Uso:
    uv run python -m src.data.prepare_eda
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from src.data.agregados import REGLA_RESPUESTA
from src.data.load_raw import COLUMNA_ID, cargar
from src.paths import INTERIM, PARQUET_CRUDO, RAIZ, REPORTS, asegurar

COL_FECHA = "Date received"
COL_TEXTO = "Consumer complaint narrative"
COL_RESPUESTA = "Company response to consumer"
COL_PLAZO = "Timely response?"

COLUMNAS_MUESTRA = [
    COLUMNA_ID,
    COL_FECHA,
    COL_TEXTO,
    "Company",
    "State",
    "ZIP code",
    "Tags",
    "Product",
    "Sub-product",
    "Issue",
    "Sub-issue",
    COL_RESPUESTA,
    COL_PLAZO,
]

RENOMBRES = {
    COLUMNA_ID: "complaint_id",
    COL_FECHA: "date_received",
    COL_TEXTO: "narrative",
    "Company": "company",
    "State": "state",
    "ZIP code": "zip_code",
    "Tags": "tags",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    COL_RESPUESTA: "company_response",
    COL_PLAZO: "timely_response",
}

SALIDA_MUESTRA = INTERIM / "eda_sample.parquet"
SALIDA_GRUPOS = INTERIM / "eda_hash_groups.parquet"
SALIDA_AUDITORIA = REPORTS / "artefactos" / "eda_sample_hash_audit.csv"

_RE_ESPACIOS = re.compile(r"\s+")


def cargar_parametros(path: Path | None = None) -> dict[str, Any]:
    """Lee la configuración versionada del pipeline."""
    ruta = path or RAIZ / "params.yaml"
    with ruta.open(encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


def normalizar_narrativa(texto: object) -> str | None:
    """Aplica la normalización estable ``lower-whitespace-v1``."""
    if texto is None or pd.isna(texto):
        return None
    return _RE_ESPACIOS.sub(" ", str(texto).strip()).lower()


def hash_narrativa(texto: object, algoritmo: str = "sha1") -> str | None:
    """Devuelve un hash criptográfico estable del texto normalizado."""
    normalizado = normalizar_narrativa(texto)
    if normalizado is None:
        return None
    return hashlib.new(algoritmo, normalizado.encode("utf-8")).hexdigest()


def asignar_split(fechas: pd.Series, splits: dict[str, dict[str, str]]) -> pd.Series:
    """Asigna cortes temporales inclusivos; lo no cubierto queda nulo."""
    fechas = pd.to_datetime(fechas, errors="coerce")
    resultado = pd.Series(pd.NA, index=fechas.index, dtype="string")
    for nombre, limites in splits.items():
        inicio = pd.Timestamp(limites["start"])
        fin = pd.Timestamp(limites["end"])
        mascara = fechas.between(inicio, fin, inclusive="both")
        if (resultado.notna() & mascara).any():
            raise ValueError(f"El split {nombre!r} se solapa con otro corte")
        resultado.loc[mascara] = nombre
    return resultado


def derivar_targets(respuesta: pd.Series, plazo: pd.Series) -> pd.DataFrame:
    """Deriva T2/T3/T4 y marca los negativos ambiguos de T2."""
    conocidas = set(REGLA_RESPUESTA)
    desconocidas = set(respuesta.dropna().astype(str).unique()) - conocidas
    if desconocidas:
        raise ValueError(f"Respuestas de empresa sin regla: {sorted(desconocidas)}")

    mapa_t2 = {clave: valor[0] for clave, valor in REGLA_RESPUESTA.items()}
    mapa_t3 = {clave: valor[1] for clave, valor in REGLA_RESPUESTA.items()}
    t4 = plazo.map({"Yes": 0, "No": 1})

    return pd.DataFrame(
        {
            "target_t2_relief": respuesta.map(mapa_t2).astype("Int8"),
            "target_t3_monetary": respuesta.map(mapa_t3).astype("Int8"),
            "target_t4_late": t4.astype("Int8"),
            "t2_ambiguous": respuesta.isin(["Untimely response", "Closed"]),
        },
        index=respuesta.index,
    )


def _seleccionar_ids(parametros: dict[str, Any]) -> tuple[set[str], pd.DataFrame]:
    """Selecciona IDs por split leyendo únicamente dos columnas del parquet."""
    indice = cargar([COLUMNA_ID, COL_FECHA])
    indice = pd.DataFrame(
        {
            "complaint_id": indice[COLUMNA_ID].astype("string"),
            "date_received": pd.to_datetime(indice[COL_FECHA], errors="coerce"),
        }
    )
    indice["split"] = asignar_split(indice["date_received"], parametros["splits"])

    semilla = int(parametros["random_seed"])
    partes: list[pd.DataFrame] = []
    for desplazamiento, nombre in enumerate(parametros["splits"]):
        candidatos = indice.loc[indice["split"] == nombre]
        solicitado = int(parametros["sample"][f"{nombre}_size"])
        if len(candidatos) < solicitado:
            raise ValueError(
                f"{nombre}: se solicitaron {solicitado:,} filas y solo hay "
                f"{len(candidatos):,}"
            )
        partes.append(
            candidatos.sample(
                n=solicitado,
                replace=False,
                random_state=semilla + desplazamiento,
            )
        )

    seleccion = pd.concat(partes, ignore_index=True)
    if seleccion["complaint_id"].duplicated().any():
        raise ValueError("Complaint ID no es único en la muestra")
    return set(seleccion["complaint_id"].astype(str)), seleccion


def _leer_filas(ids: set[str], batch_size: int) -> pd.DataFrame:
    """Recupera las filas seleccionadas en una segunda pasada por lotes."""
    archivo = pq.ParquetFile(PARQUET_CRUDO)
    valores = pa.array(sorted(ids), type=pa.large_string())
    tablas: list[pa.Table] = []

    opciones = pc.SetLookupOptions(value_set=valores)
    for lote in archivo.iter_batches(batch_size=batch_size, columns=COLUMNAS_MUESTRA):
        mascara = pc.call_function("is_in", [lote.column(COLUMNA_ID)], options=opciones)
        filtrado = lote.filter(mascara)
        if filtrado.num_rows:
            tablas.append(pa.Table.from_batches([filtrado]))

    if not tablas:
        raise ValueError("La selección no recuperó filas del parquet crudo")
    tabla = pa.concat_tables(tablas)
    if tabla.num_rows != len(ids):
        raise ValueError(f"Se esperaban {len(ids):,} filas y se recuperaron {tabla.num_rows:,}")
    return tabla.to_pandas(types_mapper=pd.ArrowDtype)


def preparar_muestra(parametros: dict[str, Any]) -> pd.DataFrame:
    """Construye la muestra tipada con targets, split y hash estable."""
    ids, seleccion = _seleccionar_ids(parametros)
    muestra = _leer_filas(ids, int(parametros["sample"]["batch_size"]))
    muestra = muestra.rename(columns=RENOMBRES)

    muestra["complaint_id"] = pd.to_numeric(muestra["complaint_id"], errors="raise").astype(
        "int64"
    )
    muestra["date_received"] = pd.to_datetime(muestra["date_received"], errors="raise")
    muestra["split"] = asignar_split(muestra["date_received"], parametros["splits"])
    muestra["hash_narrativa"] = muestra["narrative"].map(
        lambda texto: hash_narrativa(texto, parametros["text"]["hash_algorithm"])
    ).astype("string")

    targets = derivar_targets(muestra["company_response"], muestra["timely_response"])
    muestra = pd.concat([muestra, targets], axis=1)
    muestra["target_t1_issue_raw"] = muestra["issue"].astype("string")

    esperados = seleccion["split"].value_counts().sort_index()
    obtenidos = muestra["split"].value_counts().sort_index()
    if not esperados.equals(obtenidos):
        raise ValueError(f"Conteos por split inesperados: {obtenidos.to_dict()}")

    orden_split = {nombre: i for i, nombre in enumerate(parametros["splits"])}
    muestra["_orden_split"] = muestra["split"].map(orden_split).astype("int8")
    muestra = muestra.sort_values(
        ["_orden_split", "date_received", "complaint_id"], kind="stable"
    ).drop(columns="_orden_split")
    return muestra.reset_index(drop=True)


def auditar_hashes(muestra: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resume repeticiones y conflictos de etiqueta por narrativa normalizada."""
    validas = muestra[muestra["hash_narrativa"].notna()].copy()
    grupos = (
        validas.groupby("hash_narrativa", observed=True)
        .agg(
            n_eventos=("complaint_id", "size"),
            n_splits=("split", "nunique"),
            n_empresas=("company", "nunique"),
            n_productos=("product", "nunique"),
            n_issues=("issue", "nunique"),
            n_t2=("target_t2_relief", "nunique"),
            n_t3=("target_t3_monetary", "nunique"),
            n_t4=("target_t4_late", "nunique"),
        )
        .reset_index()
    )
    grupos["repetida"] = grupos["n_eventos"] > 1
    grupos["cruza_splits"] = grupos["n_splits"] > 1
    grupos["conflicto_issue"] = grupos["n_issues"] > 1
    grupos["conflicto_t2"] = grupos["n_t2"] > 1
    grupos["conflicto_t3"] = grupos["n_t3"] > 1
    grupos["conflicto_t4"] = grupos["n_t4"] > 1

    hashes_repetidos = set(grupos.loc[grupos["repetida"], "hash_narrativa"])
    filas_repetidas = int(validas["hash_narrativa"].isin(hashes_repetidos).sum())
    total = len(muestra)

    metricas = [
        ("filas_muestra", total, 100.0),
        ("hashes_unicos", len(grupos), 100 * len(grupos) / total),
        ("filas_en_hash_repetido", filas_repetidas, 100 * filas_repetidas / total),
        ("hashes_que_cruzan_splits", int(grupos["cruza_splits"].sum()), None),
        ("hashes_con_conflicto_issue", int(grupos["conflicto_issue"].sum()), None),
        ("hashes_con_conflicto_t2", int(grupos["conflicto_t2"].sum()), None),
        ("hashes_con_conflicto_t3", int(grupos["conflicto_t3"].sum()), None),
        ("hashes_con_conflicto_t4", int(grupos["conflicto_t4"].sum()), None),
    ]
    auditoria = pd.DataFrame(metricas, columns=["metrica", "n", "pct_filas"])
    auditoria["pct_filas"] = auditoria["pct_filas"].astype("Float64").round(4)
    return grupos, auditoria


def main() -> None:
    parametros = cargar_parametros()
    muestra = preparar_muestra(parametros)
    grupos, auditoria = auditar_hashes(muestra)

    asegurar(INTERIM, SALIDA_AUDITORIA.parent)
    muestra.to_parquet(SALIDA_MUESTRA, index=False, compression="zstd")
    grupos.to_parquet(SALIDA_GRUPOS, index=False, compression="zstd")
    auditoria.to_csv(SALIDA_AUDITORIA, index=False)

    print(f"Muestra: {len(muestra):,} filas -> {SALIDA_MUESTRA}")
    print(f"Hashes: {len(grupos):,} grupos -> {SALIDA_GRUPOS}")
    print(auditoria.to_string(index=False))


if __name__ == "__main__":
    main()
