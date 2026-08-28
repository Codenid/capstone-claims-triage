"""Fase 2: tipado y validación del corpus CFPB completo.

Transforma las 3.8 M filas por lotes y conserva cada evento. El Parquet usa
tipos físicos estables, codificación dictionary para categóricas y Zstandard.
El proceso falla ante columnas, dominios, fechas o IDs que rompan el contrato.

Uso:
    uv run python -m src.data.typing
"""

from __future__ import annotations

from collections.abc import Iterable
import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from src.data.agregados import REGLA_RESPUESTA
from src.paths import INTERIM, PARQUET_CRUDO, RAIZ, REPORTS, asegurar

CONFIG_PATH = RAIZ / "configs" / "dtypes.yaml"
PARAMS_PATH = RAIZ / "params.yaml"
SALIDA = INTERIM / "tipado.parquet"
SALIDA_TEMPORAL = INTERIM / "tipado.tmp.parquet"
SALIDA_AUDITORIA = REPORTS / "artefactos" / "f02_tipado_audit.csv"
SIN_DATO = "SIN_DATO"

CATEGORICAS = [
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "company_public_response",
    "company",
    "state",
    "zip_code",
    "company_response",
]

SCHEMA_SALIDA = pa.schema(
    [
        pa.field("complaint_id", pa.int64(), nullable=False),
        pa.field("date_received", pa.timestamp("ns"), nullable=False),
        pa.field("narrative", pa.large_string(), nullable=False),
        pa.field("company", pa.string(), nullable=False),
        pa.field("state", pa.string(), nullable=False),
        pa.field("zip_code", pa.string(), nullable=False),
        pa.field("is_servicemember", pa.bool_(), nullable=False),
        pa.field("is_older_adult", pa.bool_(), nullable=False),
        pa.field("product", pa.string(), nullable=False),
        pa.field("sub_product", pa.string(), nullable=False),
        pa.field("issue", pa.string(), nullable=False),
        pa.field("sub_issue", pa.string(), nullable=False),
        pa.field("date_sent_to_company", pa.timestamp("ns"), nullable=False),
        pa.field("company_response", pa.string(), nullable=False),
        pa.field("company_public_response", pa.string(), nullable=False),
        pa.field("timely_response", pa.bool_(), nullable=False),
        pa.field("missing_sub_issue", pa.bool_(), nullable=False),
        pa.field("missing_company_public_response", pa.bool_(), nullable=False),
        pa.field("target_t1_issue_raw", pa.string(), nullable=False),
        pa.field("target_t2_relief", pa.int8()),
        pa.field("target_t3_monetary", pa.int8()),
        pa.field("target_t4_late", pa.int8(), nullable=False),
        pa.field("t2_ambiguous", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_version": b"1.0", b"source": b"CFPB"},
)


def _leer_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


def _validar_dominio(columna: pa.Array, permitidos: Iterable[str], nombre: str) -> None:
    observados = set(pc.unique(columna).drop_null().to_pylist())
    inesperados = observados - set(permitidos)
    if inesperados:
        muestra = sorted(map(str, inesperados))[:10]
        raise ValueError(f"{nombre}: valores fuera del dominio: {muestra}")


def _texto(columna: pa.Array, *, llenar: bool = True) -> pa.Array:
    resultado = pc.cast(columna, pa.string())
    if llenar:
        resultado = pc.fill_null(resultado, SIN_DATO)
    return resultado


def _fecha(columna: pa.Array, nombre: str) -> pa.Array:
    resultado = pc.strptime(columna, format="%Y-%m-%d", unit="ns", error_is_null=True)
    if resultado.null_count != columna.null_count:
        raise ValueError(f"{nombre}: existen fechas que no cumplen %Y-%m-%d")
    return resultado


def _mapear_target(respuesta: pa.Array, posicion: int) -> pa.Array:
    claves = list(REGLA_RESPUESTA)
    valores = [REGLA_RESPUESTA[clave][posicion] for clave in claves]
    opciones = pc.SetLookupOptions(value_set=pa.array(claves, type=respuesta.type))
    indices = pc.call_function("index_in", [respuesta], options=opciones)
    return pc.take(pa.array(valores, type=pa.int8()), indices)


def transformar_lote(lote: pa.RecordBatch, config: dict[str, Any]) -> pa.Table:
    """Valida y transforma un lote crudo al esquema físico de Fase 2."""
    for nombre, permitidos in config["domains"].items():
        _validar_dominio(lote.column(nombre), permitidos, nombre)
    for nombre in config["required_non_null"]:
        if lote.column(nombre).null_count:
            raise ValueError(f"{nombre}: contiene nulos y el contrato lo exige")

    tags = lote.column("Tags")
    respuesta = lote.column("Company response to consumer")
    plazo = lote.column("Timely response?")
    issue = _texto(lote.column("Issue"))
    sub_issue_crudo = lote.column("Sub-issue")
    respuesta_publica_cruda = lote.column("Company public response")

    is_servicemember = pc.fill_null(pc.match_substring(tags, "Servicemember"), False)
    is_older_adult = pc.fill_null(pc.match_substring(tags, "Older American"), False)
    timely_response = pc.fill_null(pc.equal(plazo, "Yes"), False)
    target_t4 = pc.cast(pc.fill_null(pc.equal(plazo, "No"), False), pa.int8())
    t2_ambiguous = pc.fill_null(
        pc.match_substring_regex(respuesta, r"^(Untimely response|Closed)$"), False
    )

    columnas = {
        "complaint_id": pc.cast(lote.column("Complaint ID"), pa.int64()),
        "date_received": _fecha(lote.column("Date received"), "Date received"),
        "narrative": pc.cast(lote.column("Consumer complaint narrative"), pa.large_string()),
        "company": _texto(lote.column("Company")),
        "state": _texto(lote.column("State")),
        "zip_code": _texto(lote.column("ZIP code")),
        "is_servicemember": is_servicemember,
        "is_older_adult": is_older_adult,
        "product": _texto(lote.column("Product")),
        "sub_product": _texto(lote.column("Sub-product")),
        "issue": issue,
        "sub_issue": _texto(sub_issue_crudo),
        "date_sent_to_company": _fecha(
            lote.column("Date sent to company"), "Date sent to company"
        ),
        "company_response": _texto(respuesta),
        "company_public_response": _texto(respuesta_publica_cruda),
        "timely_response": timely_response,
        "missing_sub_issue": pc.is_null(sub_issue_crudo),
        "missing_company_public_response": pc.is_null(respuesta_publica_cruda),
        "target_t1_issue_raw": issue,
        "target_t2_relief": _mapear_target(respuesta, 0),
        "target_t3_monetary": _mapear_target(respuesta, 1),
        "target_t4_late": target_t4,
        "t2_ambiguous": t2_ambiguous,
    }
    tabla = pa.Table.from_pydict(columnas, schema=SCHEMA_SALIDA)
    if tabla.num_rows != lote.num_rows:
        raise ValueError("La transformación alteró el número de filas")
    return tabla


def _rango_fecha(columna: pa.ChunkedArray | pa.Array) -> tuple[pd.Timestamp, pd.Timestamp]:
    limites = pc.min_max(columna)
    return pd.Timestamp(limites["min"].as_py()), pd.Timestamp(limites["max"].as_py())


def _memoria_no_texto(config: dict[str, Any]) -> tuple[float, float]:
    """Mide memoria profunda aplicando las categorías definidas por contrato."""
    columnas_raw = [
        nombre
        for nombre in pq.read_schema(PARQUET_CRUDO).names
        if nombre != "Consumer complaint narrative"
    ]
    crudo = pd.read_parquet(PARQUET_CRUDO, columns=columnas_raw)
    memoria_crudo = float(crudo.memory_usage(deep=True).sum() / 1024**2)
    del crudo
    gc.collect()

    columnas_tipadas = [nombre for nombre in SCHEMA_SALIDA.names if nombre != "narrative"]
    tipado = pd.read_parquet(SALIDA, columns=columnas_tipadas)
    categoricas = list(config["categorical_columns"]) + ["target_t1_issue_raw"]
    tipado[categoricas] = tipado[categoricas].astype("category")
    memoria_tipado = float(tipado.memory_usage(deep=True).sum() / 1024**2)
    del tipado
    gc.collect()
    return memoria_crudo, memoria_tipado


def _tamano_comprimido_sin_texto(path: Path, columna_texto: str) -> float:
    archivo = pq.ParquetFile(path)
    total = 0
    for row_group in range(archivo.metadata.num_row_groups):
        grupo = archivo.metadata.row_group(row_group)
        total += sum(
            grupo.column(i).total_compressed_size
            for i in range(grupo.num_columns)
            if grupo.column(i).path_in_schema != columna_texto
        )
    return total / 1024**2


def tipar() -> pd.DataFrame:
    """Recorre el raw, escribe el Parquet tipado y devuelve su auditoría."""
    config = _leer_yaml(CONFIG_PATH)
    parametros = _leer_yaml(PARAMS_PATH)["typing"]
    archivo = pq.ParquetFile(PARQUET_CRUDO)
    columnas_crudas = archivo.schema_arrow.names
    if columnas_crudas != config["source_columns"]:
        raise ValueError("El orden o conjunto de columnas crudas cambió")

    asegurar(INTERIM, SALIDA_AUDITORIA.parent)
    SALIDA_TEMPORAL.unlink(missing_ok=True)
    writer = pq.ParquetWriter(
        SALIDA_TEMPORAL,
        SCHEMA_SALIDA,
        compression="zstd",
        compression_level=3,
        use_dictionary=CATEGORICAS,
        write_statistics=[c for c in SCHEMA_SALIDA.names if c != "narrative"],
    )

    ids: list[np.ndarray] = []
    n_filas = 0

    min_fecha: pd.Timestamp | None = None
    max_fecha: pd.Timestamp | None = None

    try:
        for numero, lote in enumerate(
            archivo.iter_batches(batch_size=int(parametros["batch_size"])), start=1
        ):
            tabla = transformar_lote(lote, config)
            writer.write_table(tabla)
            n_filas += tabla.num_rows
            ids.append(tabla.column("complaint_id").to_numpy())

            inicio, fin = _rango_fecha(tabla.column("date_received"))
            min_fecha = inicio if min_fecha is None else min(min_fecha, inicio)
            max_fecha = fin if max_fecha is None else max(max_fecha, fin)
            if numero % 8 == 0:
                print(f"Tipado: {n_filas:,} filas", flush=True)
    finally:
        writer.close()

    esperadas = archivo.metadata.num_rows
    if n_filas != esperadas:
        raise ValueError(f"Se esperaban {esperadas:,} filas y se escribieron {n_filas:,}")

    todos_ids = np.concatenate(ids)
    n_ids_unicos = int(np.unique(todos_ids).size)
    if n_ids_unicos != n_filas:
        raise ValueError(f"Complaint ID no es único: {n_ids_unicos:,}/{n_filas:,}")
    if min_fecha is None or max_fecha is None:
        raise ValueError("No se pudo determinar el rango de fechas")
    if min_fecha < pd.Timestamp(parametros["date_min"]):
        raise ValueError(f"Fecha anterior al contrato: {min_fecha}")
    if max_fecha > pd.Timestamp(parametros["date_max"]):
        raise ValueError(f"Fecha posterior al contrato: {max_fecha}")

    SALIDA.unlink(missing_ok=True)
    SALIDA_TEMPORAL.replace(SALIDA)
    memoria_crudo, memoria_tipado = _memoria_no_texto(config)
    reduccion = 100 * (1 - memoria_tipado / memoria_crudo)
    minimo = float(parametros["min_memory_reduction_pct"])
    if reduccion < minimo:
        raise ValueError(f"La reducción de memoria {reduccion:.2f}% es menor a {minimo:.2f}%")
    fisico_raw = _tamano_comprimido_sin_texto(
        PARQUET_CRUDO, "Consumer complaint narrative"
    )
    fisico_tipado = _tamano_comprimido_sin_texto(SALIDA, "narrative")
    auditoria = pd.DataFrame(
        [
            ("filas_preservadas", "ok", n_filas, esperadas),
            ("complaint_id_unico", "ok", n_ids_unicos, n_filas),
            ("fecha_min", "ok", min_fecha.date().isoformat(), parametros["date_min"]),
            ("fecha_max", "ok", max_fecha.date().isoformat(), parametros["date_max"]),
            ("columnas_salida", "ok", len(SCHEMA_SALIDA), len(SCHEMA_SALIDA)),
            ("memoria_raw_sin_texto_mib", "info", round(memoria_crudo, 2), None),
            ("memoria_tipado_categorico_mib", "info", round(memoria_tipado, 2), None),
            ("reduccion_memoria_sin_texto_pct", "ok", round(reduccion, 2), minimo),
            ("parquet_raw_sin_texto_mib", "info", round(fisico_raw, 2), None),
            ("parquet_tipado_sin_texto_mib", "info", round(fisico_tipado, 2), None),
            ("tamano_parquet_total_mib", "info", round(SALIDA.stat().st_size / 1024**2, 2), None),
        ],
        columns=["check", "status", "value", "expected"],
    )
    auditoria.to_csv(SALIDA_AUDITORIA, index=False)
    print(auditoria.to_string(index=False))
    print(f"Tipado completo -> {SALIDA}")
    return auditoria


if __name__ == "__main__":
    tipar()
