"""Prepara E14: hashes únicos de train y ablación de nombres de empresa.

La condición enmascarada reutiliza R1 cuando el texto no cambia o cuando el hash
enmascarado ya existe en el cache. Solo codifica localmente los textos nuevos.
Validation, test y OOD no se leen para tomar decisiones de clustering.

Uso:
    uv run python -m src.features.e14_prepare
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np
import pandas as pd
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer

from src.data.prepare_eda import cargar_parametros, normalizar_narrativa
from src.paths import INTERIM, PROCESSED, REPORTS, asegurar

MUESTRA = INTERIM / "eda_sample.parquet"
R1_EMBEDDINGS = PROCESSED / "r1_embeddings.npy"
R1_INDICE = PROCESSED / "r1_hash_index.parquet"
SALIDA_INDICE = PROCESSED / "e14_train_index.parquet"
SALIDA_MASKED = PROCESSED / "e14_masked_embeddings.npy"
SALIDA_MANIFIESTO = PROCESSED / "e14_prepare_manifest.json"
SALIDA_AUDITORIA = REPORTS / "artefactos" / "e14_company_ablation.csv"

SUFIJOS_EMPRESA = re.compile(
    r"(?:,\s*)?\b(?:incorporated|corporation|company|holdings|financial services|"
    r"national association|n\.?\s*a\.?|inc\.?|llc|l\.?l\.?c\.?|corp\.?|ltd\.?).*$",
    flags=re.IGNORECASE,
)
ALIAS_GENERICOS = {
    "bank",
    "company",
    "financial",
    "financial services",
    "credit union",
    "mortgage",
}
RE_ESPACIOS = re.compile(r"\s+")


def _unico_o_nulo(serie: pd.Series) -> str | None:
    valores = serie.dropna().astype(str).unique()
    return str(valores[0]) if len(valores) == 1 else None


def construir_indice_train(muestra: pd.DataFrame) -> pd.DataFrame:
    """Colapsa eventos de train por hash sin imponer etiqueta mayoritaria."""
    train = muestra.loc[muestra["split"].astype(str) == "train"].copy()
    grupos = (
        train.groupby("hash_narrativa", sort=True, observed=True)
        .agg(
            n_eventos_train=("complaint_id", "size"),
            narrative=("narrative", "first"),
            companies=("company", lambda s: tuple(sorted(set(s.dropna().astype(str))))),
            n_companies=("company", "nunique"),
            n_products=("product", "nunique"),
            n_issues=("issue", "nunique"),
            product_unique=("product", _unico_o_nulo),
            issue_unique=("issue", _unico_o_nulo),
        )
        .reset_index()
    )
    grupos.insert(0, "e14_row", np.arange(len(grupos), dtype=np.int64))
    grupos["texto_normalizado"] = grupos["narrative"].map(normalizar_narrativa)
    if grupos["texto_normalizado"].isna().any():
        raise ValueError("E14 encontró narrativa nula en train")
    return grupos


def aliases_empresa(empresas: tuple[str, ...], min_chars: int) -> tuple[str, ...]:
    """Genera aliases conservadores: nombre legal y prefijo sin sufijo societario."""
    aliases: set[str] = set()
    for empresa in empresas:
        normalizada = normalizar_narrativa(empresa)
        if not normalizada:
            continue
        candidatos = {normalizada, SUFIJOS_EMPRESA.sub("", normalizada).strip(" ,.-")}
        for alias in candidatos:
            if len(alias) >= min_chars and alias not in ALIAS_GENERICOS:
                aliases.add(alias)
    return tuple(sorted(aliases, key=lambda x: (-len(x), x)))


def enmascarar_empresa(
    texto: str,
    empresas: tuple[str, ...],
    min_chars: int,
) -> tuple[str, int]:
    """Reemplaza menciones exactas de aliases conocidos y devuelve cuántos aplicó."""
    resultado = texto
    aplicados = 0
    for alias in aliases_empresa(empresas, min_chars):
        patron = re.compile(re.escape(alias), flags=re.IGNORECASE)
        resultado, n = patron.subn(" company ", resultado)
        aplicados += n
    return RE_ESPACIOS.sub(" ", resultado).strip(), aplicados


def _sha1(texto: str) -> str:
    return hashlib.sha1(texto.encode("utf-8")).hexdigest()


def _codificar(textos: list[str], config: dict[str, Any]) -> tuple[np.ndarray, str]:
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = SentenceTransformer(
        config["model"], revision=config["revision"], device=dispositivo
    )
    modelo.max_seq_length = int(config["max_seq_length"])
    embeddings = modelo.encode(
        textos,
        batch_size=int(config["batch_size"]),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=bool(config["normalize"]),
    )
    return np.asarray(embeddings, dtype=np.float32), dispositivo


def main() -> None:
    parametros = cargar_parametros()
    embedding_config = parametros["embedding"]
    mask_config = parametros["e14"]["company_mask"]
    columnas = [
        "complaint_id",
        "split",
        "hash_narrativa",
        "narrative",
        "company",
        "product",
        "issue",
    ]
    muestra = pd.read_parquet(MUESTRA, columns=columnas)
    indice = construir_indice_train(muestra)

    resultados = indice.apply(
        lambda fila: enmascarar_empresa(
            str(fila["texto_normalizado"]),
            fila["companies"],
            int(mask_config["min_alias_chars"]),
        ),
        axis=1,
    )
    indice["texto_enmascarado"] = [valor[0] for valor in resultados]
    indice["n_alias_matches"] = np.asarray([valor[1] for valor in resultados], dtype=np.int16)
    indice["mask_changed"] = indice["texto_enmascarado"] != indice["texto_normalizado"]
    indice["masked_hash"] = indice["texto_enmascarado"].map(_sha1).astype("string")

    r1_indice = pd.read_parquet(R1_INDICE)
    fila_por_hash = r1_indice.set_index("hash_narrativa")["embedding_row"]
    filas_r1 = fila_por_hash.reindex(indice["hash_narrativa"])
    if filas_r1.isna().any():
        raise ValueError("Hay hashes de train ausentes del cache R1")
    r1 = np.load(R1_EMBEDDINGS, mmap_mode="r")
    masked = np.asarray(r1[filas_r1.astype("int64").to_numpy()], dtype=np.float32).copy()

    cambiadas = indice["mask_changed"].to_numpy()
    hashes_masked = indice.loc[cambiadas, "masked_hash"]
    filas_reutilizables = fila_por_hash.reindex(hashes_masked)
    puede_reutilizar = filas_reutilizables.notna().to_numpy()
    posiciones_cambiadas = np.flatnonzero(cambiadas)
    if puede_reutilizar.any():
        masked[posiciones_cambiadas[puede_reutilizar]] = r1[
            filas_reutilizables.loc[filas_reutilizables.notna()].astype("int64").to_numpy()
        ]

    por_codificar = indice.loc[
        posiciones_cambiadas[~puede_reutilizar], ["masked_hash", "texto_enmascarado"]
    ]
    unicos_nuevos = por_codificar.drop_duplicates("masked_hash", keep="first")
    dispositivo = "reused_only"
    if len(unicos_nuevos):
        nuevos, dispositivo = _codificar(
            unicos_nuevos["texto_enmascarado"].astype(str).tolist(), embedding_config
        )
        vector_por_hash = {
            hash_: nuevos[i]
            for i, hash_ in enumerate(unicos_nuevos["masked_hash"].astype(str))
        }
        for posicion in posiciones_cambiadas[~puede_reutilizar]:
            masked[posicion] = vector_por_hash[str(indice.at[posicion, "masked_hash"])]

    normas = np.linalg.norm(masked, axis=1)
    if not np.allclose(normas, 1.0, atol=1e-4):
        raise ValueError("La matriz enmascarada no quedó normalizada")

    n_masked_unicos = int(indice["masked_hash"].nunique())
    auditoria = pd.DataFrame(
        [
            ("train_event_rows", int(indice["n_eventos_train"].sum())),
            ("train_unique_hashes", len(indice)),
            ("hashes_with_product_conflict", int((indice["n_products"] > 1).sum())),
            ("hashes_with_issue_conflict", int((indice["n_issues"] > 1).sum())),
            ("texts_changed_by_mask", int(cambiadas.sum())),
            ("pct_texts_changed", round(100 * float(cambiadas.mean()), 4)),
            ("changed_reused_from_r1", int(puede_reutilizar.sum())),
            ("new_masked_texts_encoded", len(unicos_nuevos)),
            ("unique_masked_hashes", n_masked_unicos),
            ("hashes_merged_after_mask", len(indice) - n_masked_unicos),
        ],
        columns=["metric", "value"],
    )
    manifiesto = {
        "fit_split": "train",
        "unit": "unique_hash",
        "rows": len(indice),
        "model": embedding_config["model"],
        "revision": embedding_config["revision"],
        "max_seq_length": int(embedding_config["max_seq_length"]),
        "device_for_new_embeddings": dispositivo,
        "sentence_transformers_version": sentence_transformers.__version__,
        "torch_version": torch.__version__,
        "mask_policy": {
            "method": "exact legal name and conservative suffix-stripped alias",
            "replacement": "company",
            **mask_config,
        },
    }

    salida_indice = indice[
        [
            "e14_row",
            "hash_narrativa",
            "masked_hash",
            "n_eventos_train",
            "n_companies",
            "n_products",
            "n_issues",
            "product_unique",
            "issue_unique",
            "n_alias_matches",
            "mask_changed",
        ]
    ]
    asegurar(PROCESSED, SALIDA_AUDITORIA.parent)
    np.save(SALIDA_MASKED, masked, allow_pickle=False)
    salida_indice.to_parquet(SALIDA_INDICE, index=False, compression="zstd")
    SALIDA_MANIFIESTO.write_text(json.dumps(manifiesto, indent=2), encoding="utf-8")
    auditoria.to_csv(SALIDA_AUDITORIA, index=False)
    print(auditoria.to_string(index=False))
    print(f"E14 preparado en {dispositivo}: {len(indice):,} hashes únicos de train")


if __name__ == "__main__":
    main()
