"""Document the prepared table: data dictionary and cleaning report (PB-14).

Both outputs are produced from the data, not written by hand:

- docs/diccionario/diccionario.csv: one row per column of
  data/processed/prepared.parquet with its type, the stage that created it,
  its role in the modeling contract, counts measured on the table, an
  example and the business description kept in docs/diccionario/
  descripciones.yaml.
- reports/preparation/informe_limpieza.json: rows and columns before and
  after each stage of dvc.yaml, with the hashes dvc.lock recorded, and one
  entry per data-quality finding of the EDA with the treatment it received
  and the figure measured on the prepared table.

The stage never transforms data. Run it with `dvc repro document_prepared`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

# pyarrow.compute builds its functions at import time; the stubs do not list them.
compute = cast(Any, pc)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
LOCK_PATH = PROJECT_ROOT / "dvc.lock"
DESCRIPTIONS_PATH = PROJECT_ROOT / "docs/diccionario/descripciones.yaml"
DICTIONARY_PATH = PROJECT_ROOT / "docs/diccionario/diccionario.csv"
REPORT_PATH = PROJECT_ROOT / "reports/preparation/informe_limpieza.json"
PREPARED_PATH = PROJECT_ROOT / "data/processed/prepared.parquet"
MISSING_DESCRIPTION = "PENDIENTE: describir en docs/diccionario/descripciones.yaml"


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["documentar"]


def load_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["stages"]


def column_stage(name: str, params: dict[str, Any]) -> str:
    """The dvc.yaml stage that created a column, from the declared mapping."""
    for stage, columns in params["columnas_por_etapa"].items():
        if name in columns:
            return stage
    raise ValueError(f"Column {name!r} is not assigned to a stage in params.yaml.")


def column_role(name: str, params: dict[str, Any]) -> str:
    """The role of a column in the modeling contract, from the declared lists."""
    for role, columns in params["roles"].items():
        if name in columns:
            return role
    raise ValueError(f"Column {name!r} has no role in params.yaml.")


def example_value(column: pa.ChunkedArray, name: str, params: dict[str, Any]) -> str:
    if name in params["columnas_sin_ejemplo"]:
        return "(texto libre)"
    valid = compute.drop_null(column)
    if len(valid) == 0:
        return ""
    text = str(valid[0].as_py())
    limit = int(params["ejemplo_max_caracteres"])
    return text if len(text) <= limit else text[: limit - 1] + "…"


def column_summary(
    name: str,
    column: pa.ChunkedArray,
    params: dict[str, Any],
    descriptions: dict[str, str],
) -> dict[str, Any]:
    rows = len(column)
    non_null = rows - column.null_count
    return {
        "columna": name,
        "tipo": str(column.type),
        "etapa": column_stage(name, params),
        "rol": column_role(name, params),
        "no_nulos": non_null,
        "pct_nulos": round(100 * (rows - non_null) / rows, 2) if rows else 0.0,
        "valores_unicos": distinct(column),
        "ejemplo": example_value(column, name, params),
        "descripcion": descriptions.get(name, MISSING_DESCRIPTION),
    }


def dictionary(
    table: pq.ParquetFile,
    params: dict[str, Any],
    descriptions: dict[str, str],
) -> list[dict[str, Any]]:
    """One summary per column, read one column at a time to bound memory."""
    names = table.schema_arrow.names
    declared = {
        column
        for columns in params["columnas_por_etapa"].values()
        for column in columns
    }
    if set(names) != declared:
        missing = sorted(set(names) - declared)
        extra = sorted(declared - set(names))
        raise ValueError(
            f"params.yaml and the table disagree: missing {missing}, extra {extra}."
        )
    return [
        column_summary(name, table.read(columns=[name]).column(0), params, descriptions)
        for name in names
    ]


def write_dictionary(rows: list[dict[str, Any]], path: Path = DICTIONARY_PATH) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def stage_report(params: dict[str, Any], lock: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows and columns before and after each transforming stage of dvc.yaml."""
    report = []
    columns_before = 0
    rows_before = None
    for stage, new_columns in params["columnas_por_etapa"].items():
        if stage not in lock:
            continue
        output = lock[stage]["outs"][0]
        path = PROJECT_ROOT / output["path"]
        rows = pq.ParquetFile(path).metadata.num_rows if path.exists() else None
        columns_after = columns_before + len(new_columns)
        report.append(
            {
                "etapa": stage,
                "descripcion": params["descripcion_etapas"][stage],
                "salida": output["path"],
                "md5": output["md5"],
                "bytes": output["size"],
                "filas_entrada": rows_before,
                "filas_salida": rows,
                "columnas_entrada": columns_before,
                "columnas_salida": columns_after,
                "columnas_nuevas": list(new_columns),
            }
        )
        columns_before = columns_after
        rows_before = rows
    return report


def count_true(column: pa.ChunkedArray) -> int:
    flags = compute.cast(compute.fill_null(column, False), pa.int64())
    return int(compute.sum(flags).as_py() or 0)


def distinct(column: pa.ChunkedArray) -> int:
    return int(compute.count_distinct(column, mode="only_valid").as_py())


def value_counts(column: pa.ChunkedArray) -> dict[str, int]:
    counts = compute.value_counts(column)
    return {
        str(item["values"]): int(item["counts"])
        for item in sorted(counts.to_pylist(), key=lambda item: str(item["values"]))
    }


def problems_report(table: pq.ParquetFile) -> list[dict[str, Any]]:
    """Each EDA data-quality finding, its treatment and the figure on the table."""

    def read(*names: str) -> pa.Table:
        return table.read(columns=list(names))

    rows = table.metadata.num_rows
    period = read("period").column(0)
    hashes = read("Consumer complaint narrative SHA-256").column(0)
    hash_counts = compute.value_counts(hashes)
    counts = hash_counts.field("counts")
    repeated = compute.filter(hash_counts, compute.greater(counts, 1))
    shared_rows = int(compute.sum(repeated.field("counts")).as_py() or 0)
    known_pair = read("known_product_issue_pair").column(0)
    unknown_pair_by_period = value_counts(
        compute.filter(period, compute.invert(compute.fill_null(known_pair, True)))
    )
    t2_known = read("known_T2").column(0)
    t2 = read("T2").column(0)
    t3 = read("T3").column(0)
    t4 = read("T4").column(0)
    train = compute.equal(period, "train_2023_2024")
    validation = compute.equal(period, "validation_2025_h1")

    def positive_rate(target: pa.ChunkedArray, mask: pa.ChunkedArray) -> float:
        selected = compute.filter(target, compute.fill_null(mask, False))
        selected = compute.drop_null(selected)
        if len(selected) == 0:
            return 0.0
        return round(100 * count_true(selected) / len(selected), 3)

    narrative = read("Consumer complaint narrative").column(0)
    return [
        {
            "problema": "El archivo cambia con el tiempo: 2025 reúne cerca de un "
            "tercio de los reclamos y 2026 es parcial",
            "tratamiento": "Periodos temporales en la columna period; ningún reparto "
            "al azar entre entrenamiento y evaluación",
            "columnas": ["period"],
            "cifra": {"filas_por_periodo": value_counts(period)},
        },
        {
            "problema": "La taxonomía histórica cambió de nombres entre periodos",
            "tratamiento": "Mapa canónico congelado al 2025-06-30 (Product canonical, "
            "Issue canonical) y banderas de categorías y pares conocidos",
            "columnas": [
                "Product canonical",
                "Issue canonical",
                "known_product",
                "known_issue",
                "known_product_issue_pair",
            ],
            "cifra": {
                "productos_originales": distinct(read("Product").column(0)),
                "productos_canonicos": distinct(read("Product canonical").column(0)),
                "motivos_originales": distinct(read("Issue").column(0)),
                "motivos_canonicos": distinct(read("Issue canonical").column(0)),
                "filas_con_par_desconocido_por_periodo": unknown_pair_by_period,
            },
        },
        {
            "problema": "Pocas categorías concentran los casos y hay motivos raros",
            "tratamiento": "Sin transformación: la métrica principal de T1 es "
            "Macro-F1 y la elegibilidad por objetivo se declara en columnas",
            "columnas": ["T1", "eligible_T1_complete"],
            "cifra": {
                "motivos_T1_en_entrenamiento": distinct(
                    compute.filter(read("T1").column(0), train)
                )
            },
        },
        {
            "problema": "T3 y T4 tienen pocos positivos",
            "tratamiento": "Sin remuestreo: métrica principal average precision y "
            "umbral fijado en calibración",
            "columnas": ["T3", "T4"],
            "cifra": {
                "pct_positivos_T2_conocidos": positive_rate(t2, t2_known),
                "pct_positivos_T3_conocidos": positive_rate(t3, t2_known),
                "pct_positivos_T4": positive_rate(t4, read("known_T4").column(0)),
            },
        },
        {
            "problema": "Las narrativas tienen longitudes muy distintas y marcas XXXX",
            "tratamiento": "Se conservan íntegras; la normalización no recorta ni "
            "elimina filas",
            "columnas": ["Consumer complaint narrative"],
            "cifra": {"narrativas_no_nulas": rows - narrative.null_count},
        },
        {
            "problema": "Mucha reutilización textual entre reclamos",
            "tratamiento": "SHA-256 del texto normalizado para identificar grupos; "
            "vista de evaluación sin textos compartidos con los periodos de "
            "referencia; no se elimina ninguna fila",
            "columnas": [
                "Consumer complaint narrative SHA-256",
                "no_shared_reference_text",
                "eligible_T1_no_shared_text",
            ],
            "cifra": {
                "textos_normalizados_distintos": len(hash_counts),
                "filas_que_comparten_texto": shared_rows,
                "pct_filas_que_comparten_texto": round(100 * shared_rows / rows, 2),
                "filas_sin_texto_compartido_de_referencia": count_true(
                    read("no_shared_reference_text").column(0)
                ),
            },
        },
        {
            "problema": "Los resultados T2–T4 cambian con el tiempo",
            "tratamiento": "Evaluación en periodos posteriores al entrenamiento; las "
            "respuestas ambiguas quedan como desconocidas (known_T2, known_T3)",
            "columnas": ["known_T2", "known_T3", "period"],
            "cifra": {
                "pct_T2_entrenamiento": positive_rate(
                    t2, compute.and_(train, t2_known)
                ),
                "pct_T2_validacion": positive_rate(
                    t2, compute.and_(validation, t2_known)
                ),
                "filas_con_respuesta_desconocida": rows - count_true(t2_known),
            },
        },
        {
            "problema": "Los 25,000 identificadores revisados antes no se recuperaron",
            "tratamiento": "2025-H2 bloqueado como evaluación final; banderas de "
            "identificador revisado y de grupo de texto excluido",
            "columnas": [
                "previously_reviewed_id",
                "excluded_by_reviewed_text_group",
                "holdout_status",
            ],
            "cifra": {
                "identificadores_revisados": count_true(
                    read("previously_reviewed_id").column(0)
                ),
                "filas_excluidas_por_grupo_revisado": count_true(
                    read("excluded_by_reviewed_text_group").column(0)
                ),
                "holdout_status": value_counts(read("holdout_status").column(0)),
            },
        },
    ]


def eligibility_report(table: pq.ParquetFile) -> dict[str, dict[str, int]]:
    names = [name for name in table.schema_arrow.names if name.startswith("eligible_")]
    period = table.read(columns=["period"]).column(0)
    report: dict[str, dict[str, int]] = {}
    for name in names:
        column = table.read(columns=[name]).column(0)
        mask = compute.fill_null(column, False)
        report[name] = value_counts(compute.filter(period, mask))
    return report


def build_report(
    table: pq.ParquetFile,
    rows: list[dict[str, Any]],
    params: dict[str, Any],
    lock: dict[str, Any],
) -> dict[str, Any]:
    final = lock["finalize_prepared"]["outs"][0]
    missing = sum(1 for item in rows if item["descripcion"] == MISSING_DESCRIPTION)
    return {
        "tabla": {
            "ruta": final["path"],
            "filas": table.metadata.num_rows,
            "columnas": table.metadata.num_columns,
            "md5": final["md5"],
            "bytes": final["size"],
        },
        "etapas": stage_report(params, lock),
        "problemas": problems_report(table),
        "elegibilidad_por_periodo": eligibility_report(table),
        "diccionario": {
            "columnas": len(rows),
            "sin_descripcion": missing,
            "cobertura_pct": round(100 * (len(rows) - missing) / len(rows), 1),
        },
    }


def main() -> None:
    params = load_params()
    lock = load_lock()
    descriptions = yaml.safe_load(DESCRIPTIONS_PATH.read_text(encoding="utf-8")) or {}
    with pq.ParquetFile(PREPARED_PATH) as table:
        rows = dictionary(table, params, descriptions)
        write_dictionary(rows)
        report = build_report(table, rows, params, lock)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{len(rows)} columns -> {DICTIONARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"report -> {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
