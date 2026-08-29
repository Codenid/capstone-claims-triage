"""Fase 3: canonicalización determinista y auditable de Product e Issue."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from src.paths import INTERIM, RAIZ, REPORTS, asegurar

CONFIG_PATH = RAIZ / "configs" / "taxonomia.yaml"
ENTRADA = INTERIM / "tipado.parquet"
SALIDA = INTERIM / "canonico.parquet"
TEMPORAL = INTERIM / "canonico.tmp.parquet"
AUDITORIA = REPORTS / "artefactos" / "f03_taxonomia.csv"
TRAZABILIDAD = REPORTS / "artefactos" / "f03_taxonomia_mapping.csv"
SEPARADOR = "\x1f"


def cargar_taxonomia(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as archivo:
        return yaml.safe_load(archivo)


def invertir_grupos(grupos: dict[str, list[str]], nombre: str) -> dict[str, str]:
    salida: dict[str, str] = {}
    for canonica, crudas in grupos.items():
        for cruda in crudas:
            if cruda in salida:
                raise ValueError(f"{nombre}: etiqueta repetida: {cruda}")
            salida[cruda] = canonica
    return salida


def mapas(config: dict[str, Any]) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    issues = invertir_grupos(config["issue_groups"], "issue_groups")
    productos = invertir_grupos(config["product_groups"], "product_groups")
    pares: dict[str, str] = {}
    for canonica, por_producto in config["product_subproduct_groups"].items():
        for producto, subproductos in por_producto.items():
            for subproducto in subproductos:
                clave = f"{producto}{SEPARADOR}{subproducto}"
                if clave in pares:
                    raise ValueError(f"product_subproduct_groups: combinación repetida: {clave}")
                pares[clave] = canonica
    if set(productos) & {clave.split(SEPARADOR)[0] for clave in pares}:
        raise ValueError("Un producto no puede tener regla directa y regla por subproducto")
    return issues, productos, pares


def _mapear_arrow(columna: pa.ChunkedArray, mapa: dict[str, str], nombre: str) -> pa.Array:
    crudos = list(mapa)
    indices = pc.index_in(columna, value_set=pa.array(crudos, type=pa.string()))
    resultado = pc.take(pa.array([mapa[x] for x in crudos], type=pa.string()), indices)
    if resultado.null_count:
        no_mapeados = sorted(set(pc.filter(columna, pc.is_null(resultado)).to_pylist()))[:10]
        raise ValueError(f"{nombre}: valores no mapeados: {no_mapeados}")
    return resultado


def canonicalizar_tabla(tabla: pa.Table, config: dict[str, Any]) -> pa.Table:
    """Añade columnas canónicas sin eliminar ni modificar columnas de origen."""
    issues, productos, pares = mapas(config)
    motivo = _mapear_arrow(tabla["issue"], issues, "issue")
    productos_crudos = tabla["product"].to_pylist()
    subproductos = tabla["sub_product"].to_pylist()
    canonicos = []
    for producto, subproducto in zip(productos_crudos, subproductos):
        valor = productos.get(producto)
        if valor is None:
            valor = pares.get(f"{producto}{SEPARADOR}{subproducto}")
        if valor is None:
            raise ValueError(f"product/sub_product no mapeado: {(producto, subproducto)}")
        canonicos.append(valor)
    producto_canonico = pa.array(canonicos, type=pa.string())
    version = pa.array([config["version"]] * tabla.num_rows, type=pa.string())
    return (
        tabla.append_column("producto_canonico", producto_canonico)
        .append_column("motivo_canonico", motivo)
        .append_column("target_t1_motivo_canonico", motivo)
        .append_column("taxonomia_version", version)
    )


def _split(fecha: pd.Timestamp) -> str:
    if pd.Timestamp("2023-01-01") <= fecha <= pd.Timestamp("2024-12-31"):
        return "train"
    if pd.Timestamp("2025-01-01") <= fecha <= pd.Timestamp("2025-06-30"):
        return "validation"
    if pd.Timestamp("2025-07-01") <= fecha <= pd.Timestamp("2025-12-31"):
        return "test"
    if pd.Timestamp("2026-01-01") <= fecha <= pd.Timestamp("2026-12-31"):
        return "ood_2026"
    return "historical"


def tabla_trazabilidad(config: dict[str, Any]) -> pd.DataFrame:
    issues, productos, pares = mapas(config)
    filas = [
        {"dimension": "issue", "raw_product": None, "raw_sub_product": None,
         "raw_issue": cruda, "canonical": canonica}
        for cruda, canonica in sorted(issues.items())
    ]
    filas += [
        {"dimension": "product", "raw_product": cruda, "raw_sub_product": None,
         "raw_issue": None, "canonical": canonica}
        for cruda, canonica in sorted(productos.items())
    ]
    filas += [
        {"dimension": "product_subproduct", "raw_product": clave.split(SEPARADOR)[0],
         "raw_sub_product": clave.split(SEPARADOR)[1], "raw_issue": None,
         "canonical": canonica}
        for clave, canonica in sorted(pares.items())
    ]
    return pd.DataFrame(filas)


def canonicalizar() -> pd.DataFrame:
    config = cargar_taxonomia()
    archivo = pq.ParquetFile(ENTRADA)
    asegurar(INTERIM, AUDITORIA.parent)
    TEMPORAL.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    conteos: Counter[tuple[str, str, str]] = Counter()
    filas = 0
    try:
        for numero, lote in enumerate(archivo.iter_batches(batch_size=131072), start=1):
            tabla = canonicalizar_tabla(pa.Table.from_batches([lote]), config)
            if writer is None:
                metadata = dict(tabla.schema.metadata or {})
                metadata[b"taxonomy_version"] = str(config["version"]).encode()
                writer = pq.ParquetWriter(TEMPORAL, tabla.schema.with_metadata(metadata),
                                          compression="zstd", compression_level=3,
                                          use_dictionary=True)
            writer.write_table(tabla)
            fechas = pd.to_datetime(tabla["date_received"].to_pandas())
            splits = fechas.map(_split)
            for dimension in ("producto_canonico", "motivo_canonico"):
                valores = tabla[dimension].to_pandas()
                for (canonica, split), n in pd.DataFrame(
                    {"canonical": valores, "split": splits}
                ).value_counts().items():
                    conteos[(dimension, str(canonica), str(split))] += int(n)
            filas += tabla.num_rows
            if numero % 8 == 0:
                print(f"Canonicalización: {filas:,} filas", flush=True)
    finally:
        if writer is not None:
            writer.close()

    if filas != archivo.metadata.num_rows:
        raise ValueError(f"Filas alteradas: {filas:,}/{archivo.metadata.num_rows:,}")
    filas_audit = []
    for dimension, grupos in [
        ("producto_canonico", config["product_groups"] | config["product_subproduct_groups"]),
        ("motivo_canonico", config["issue_groups"]),
    ]:
        for canonica in grupos:
            fila = {"dimension": dimension, "canonical": canonica}
            for split in ("historical", "train", "validation", "test", "ood_2026"):
                fila[split] = conteos[(dimension, canonica, split)]
            fila["total"] = sum(fila[s] for s in ("historical", "train", "validation", "test", "ood_2026"))
            filas_audit.append(fila)
    auditoria = pd.DataFrame(filas_audit)
    sin_soporte = auditoria[
        (auditoria[["train", "validation", "test"]] == 0).any(axis=1)
    ]
    if len(sin_soporte):
        TEMPORAL.unlink(missing_ok=True)
        detalle = sin_soporte[["dimension", "canonical", "train", "validation", "test"]]
        raise ValueError("Categorías sin soporte temporal:\n" + detalle.to_string(index=False))

    SALIDA.unlink(missing_ok=True)
    TEMPORAL.replace(SALIDA)
    auditoria.to_csv(AUDITORIA, index=False)
    tabla_trazabilidad(config).to_csv(TRAZABILIDAD, index=False)
    print(auditoria.to_string(index=False))
    print(f"Canonicalización completa -> {SALIDA}")
    return auditoria


if __name__ == "__main__":
    canonicalizar()
