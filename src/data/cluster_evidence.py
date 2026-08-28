"""E14: estructura latente del texto usando únicamente hashes únicos de train.

Compara un control TF-IDF/SVD/KMeans con MiniLM/UMAP/HDBSCAN, evalúa
sensibilidad predeclarada, produce topics c-TF-IDF y repite la corrida principal
con nombres de empresa enmascarados. Product e Issue son referencias externas
imperfectas; los hashes conflictivos se excluyen de la vista uniforme.

Uso:
    uv run python -m src.data.cluster_evidence
"""

from __future__ import annotations

import gc
import json
import math
import time
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
import umap
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize
from umap import UMAP

from src.data.prepare_eda import cargar_parametros
from src.features.e14_prepare import construir_indice_train, enmascarar_empresa
from src.features.text import construir_r0
from src.paths import INTERIM, PROCESSED, REPORTS, asegurar

MUESTRA = INTERIM / "eda_sample.parquet"
E14_INDICE = PROCESSED / "e14_train_index.parquet"
R1_EMBEDDINGS = PROCESSED / "r1_embeddings.npy"
R1_INDICE = PROCESSED / "r1_hash_index.parquet"
R1_MASKED = PROCESSED / "e14_masked_embeddings.npy"

SALIDA_ASIGNACIONES = PROCESSED / "e14_assignments.parquet"
SALIDA_PROYECCION = PROCESSED / "e14_projection.parquet"
SALIDA_MANIFIESTO = PROCESSED / "e14_manifest.json"
SALIDA_METRICAS = REPORTS / "artefactos" / "e14_clusters.csv"
SALIDA_ESTABILIDAD = REPORTS / "artefactos" / "e14_stability.csv"
SALIDA_TOPICS = REPORTS / "artefactos" / "e14_topics.csv"


def _pureza(etiquetas: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    tabla = pd.crosstab(pd.Series(clusters, name="cluster"), pd.Series(etiquetas, name="label"))
    if tabla.empty:
        return math.nan, math.nan
    por_cluster = tabla.max(axis=1) / tabla.sum(axis=1)
    ponderada = float(tabla.max(axis=1).sum() / tabla.to_numpy().sum())
    return float(por_cluster.mean()), ponderada


def metricas_etiqueta(
    etiquetas: pd.Series,
    clusters: np.ndarray,
    prefijo: str,
) -> dict[str, float | int]:
    """Calcula asociación solo donde la referencia es no ambigua y no hay ruido."""
    validas = etiquetas.notna().to_numpy() & (clusters >= 0)
    cobertura = float(validas.mean())
    if validas.sum() < 2 or np.unique(clusters[validas]).size < 2:
        return {
            f"{prefijo}_coverage": cobertura,
            f"{prefijo}_n": int(validas.sum()),
            f"{prefijo}_nmi": math.nan,
            f"{prefijo}_ari": math.nan,
            f"{prefijo}_purity_macro": math.nan,
            f"{prefijo}_purity_weighted": math.nan,
        }
    y = etiquetas.loc[validas].astype(str).to_numpy()
    c = clusters[validas]
    pureza_macro, pureza_ponderada = _pureza(y, c)
    return {
        f"{prefijo}_coverage": cobertura,
        f"{prefijo}_n": int(validas.sum()),
        f"{prefijo}_nmi": float(normalized_mutual_info_score(y, c)),
        f"{prefijo}_ari": float(adjusted_rand_score(y, c)),
        f"{prefijo}_purity_macro": pureza_macro,
        f"{prefijo}_purity_weighted": pureza_ponderada,
    }


def _muestra_estratificada(labels: np.ndarray, n: int, semilla: int) -> np.ndarray:
    rng = np.random.default_rng(semilla)
    validos = np.flatnonzero(labels >= 0)
    clusters = np.unique(labels[validos])
    if len(validos) <= n:
        return validos
    cuota = max(2, math.ceil(n / max(1, len(clusters))))
    partes = []
    for cluster in clusters:
        candidatos = np.flatnonzero(labels == cluster)
        partes.append(rng.choice(candidatos, size=min(cuota, len(candidatos)), replace=False))
    indices = np.concatenate(partes)
    if len(indices) > n:
        indices = rng.choice(indices, size=n, replace=False)
    return np.sort(indices)


def _silhouette(
    matriz: np.ndarray | sp.spmatrix,
    labels: np.ndarray,
    n: int,
    semilla: int,
) -> tuple[float, int]:
    indices = _muestra_estratificada(labels, n, semilla)
    if len(indices) < 3 or np.unique(labels[indices]).size < 2:
        return math.nan, len(indices)
    return float(silhouette_score(matriz[indices], labels[indices], metric="cosine")), len(indices)


def _metricas_eventos(
    eventos: pd.DataFrame,
    hash_a_cluster: pd.Series,
    columna: str,
    prefijo: str,
) -> dict[str, float | int]:
    clusters = eventos["hash_narrativa"].map(hash_a_cluster)
    validas = clusters.notna() & clusters.ge(0) & eventos[columna].notna()
    if validas.sum() < 2 or clusters.loc[validas].nunique() < 2:
        return {f"{prefijo}_coverage": float(validas.mean()), f"{prefijo}_n": int(validas.sum())}
    y = eventos.loc[validas, columna].astype(str).to_numpy()
    c = clusters.loc[validas].astype(int).to_numpy()
    macro, weighted = _pureza(y, c)
    return {
        f"{prefijo}_coverage": float(validas.mean()),
        f"{prefijo}_n": int(validas.sum()),
        f"{prefijo}_nmi": float(normalized_mutual_info_score(y, c)),
        f"{prefijo}_ari": float(adjusted_rand_score(y, c)),
        f"{prefijo}_purity_macro": macro,
        f"{prefijo}_purity_weighted": weighted,
    }


def evaluar_run(
    run: dict[str, Any],
    labels: np.ndarray,
    probabilidades: np.ndarray,
    fuente: np.ndarray | sp.spmatrix,
    indice: pd.DataFrame,
    eventos: pd.DataFrame,
    config: dict[str, Any],
    segundos: float,
) -> dict[str, Any]:
    clusters_validos, conteos = np.unique(labels[labels >= 0], return_counts=True)
    silhouette, silhouette_n = _silhouette(
        fuente,
        labels,
        int(config["silhouette_sample_size"]),
        int(run["seed"]),
    )
    hash_a_cluster = pd.Series(labels, index=indice["hash_narrativa"].astype(str))
    fila: dict[str, Any] = {
        **run,
        "n_hashes": len(labels),
        "n_clusters": len(clusters_validos),
        "noise_rate": float((labels < 0).mean()),
        "largest_cluster_share": float(conteos.max() / len(labels)) if len(conteos) else math.nan,
        "mean_membership_probability": float(probabilidades[labels >= 0].mean())
        if (labels >= 0).any()
        else math.nan,
        "silhouette_cosine": silhouette,
        "silhouette_n": silhouette_n,
        "fit_seconds": round(segundos, 3),
    }
    fila |= metricas_etiqueta(indice["product_unique"], labels, "product_hash")
    fila |= metricas_etiqueta(indice["issue_unique"], labels, "issue_hash")
    fila |= _metricas_eventos(eventos, hash_a_cluster, "product", "product_event")
    fila |= _metricas_eventos(eventos, hash_a_cluster, "issue", "issue_event")
    return fila


def _ctfidf_terms(
    textos: pd.Series,
    labels: np.ndarray,
    max_features: int,
    top_n: int,
) -> dict[int, list[tuple[str, float]]]:
    """Calcula c-TF-IDF: TF por cluster × log(1 + palabras medias/frecuencia)."""
    frame = pd.DataFrame({"texto": textos.astype(str), "cluster": labels})
    frame = frame.loc[frame["cluster"] >= 0]
    documentos = frame.groupby("cluster", sort=True)["texto"].agg(" ".join)
    vectorizador = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=1,
    )
    conteos = vectorizador.fit_transform(documentos)
    tf = normalize(conteos, norm="l1", axis=1)
    frecuencia = np.asarray(conteos.sum(axis=0)).ravel()
    palabras_medias = float(np.asarray(conteos.sum(axis=1)).mean())
    idf = np.log1p(palabras_medias / np.maximum(frecuencia, 1))
    puntajes = tf.multiply(idf).tocsr()
    terminos = vectorizador.get_feature_names_out()

    salida: dict[int, list[tuple[str, float]]] = {}
    for fila, cluster in enumerate(documentos.index.astype(int)):
        datos = puntajes.getrow(fila)
        orden = np.argsort(datos.data)[::-1][:top_n]
        salida[int(cluster)] = [
            (str(terminos[datos.indices[i]]), float(datos.data[i])) for i in orden
        ]
    return salida


def _representante(matriz: np.ndarray, miembros: np.ndarray) -> int:
    centro = matriz[miembros].mean(axis=0)
    norma = np.linalg.norm(centro)
    if norma:
        centro = centro / norma
    similitud = matriz[miembros] @ centro
    return int(miembros[int(np.argmax(similitud))])


def construir_topics(
    condicion: str,
    labels: np.ndarray,
    matriz: np.ndarray,
    indice: pd.DataFrame,
    textos: pd.Series,
    n_eventos: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    terminos = _ctfidf_terms(
        textos,
        labels,
        int(config["topic_max_features"]),
        int(config["top_n_words"]),
    )
    filas = []
    for cluster in sorted(terminos):
        miembros = np.flatnonzero(labels == cluster)
        representante = _representante(matriz, miembros)
        productos = indice.loc[miembros, "product_unique"].dropna().value_counts()
        issues = indice.loc[miembros, "issue_unique"].dropna().value_counts()
        filas.append(
            {
                "condition": condicion,
                "cluster": cluster,
                "n_hashes": len(miembros),
                "n_events": int(n_eventos[miembros].sum()),
                "top_terms": " | ".join(t for t, _ in terminos[cluster]),
                "top_term_scores": " | ".join(f"{s:.6f}" for _, s in terminos[cluster]),
                "top_product": productos.index[0] if len(productos) else None,
                "top_product_share": float(productos.iloc[0] / productos.sum()) if len(productos) else None,
                "top_issue": issues.index[0] if len(issues) else None,
                "top_issue_share": float(issues.iloc[0] / issues.sum()) if len(issues) else None,
                "representative_hash": indice.at[representante, "hash_narrativa"],
                "representative_excerpt": str(textos.iloc[representante])[:180],
            }
        )
    return pd.DataFrame(filas)


def _ajustar_hdbscan(reducido: np.ndarray, config: dict[str, Any], min_cluster_size: int):
    modelo = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=int(config["min_samples"]),
        metric=str(config["metric"]),
        n_jobs=-1,
    )
    labels = modelo.fit_predict(reducido).astype(np.int32)
    probabilidades = np.asarray(modelo.probabilities_, dtype=np.float32)
    return labels, probabilidades


def _umap(matriz: np.ndarray, config: dict[str, Any], semilla: int) -> np.ndarray:
    modelo = UMAP(
        n_components=int(config["n_components"]),
        n_neighbors=int(config["n_neighbors"]),
        min_dist=float(config["min_dist"]),
        metric=str(config["metric"]),
        random_state=semilla,
        n_jobs=1,
        low_memory=True,
        verbose=True,
    )
    return np.asarray(modelo.fit_transform(matriz), dtype=np.float32)


def _estabilidad(run_ids: list[str], asignaciones: dict[str, np.ndarray], familia: str) -> list[dict]:
    filas = []
    for a, b in combinations(run_ids, 2):
        la, lb = asignaciones[a], asignaciones[b]
        comunes = (la >= 0) & (lb >= 0)
        filas.append(
            {
                "family": familia,
                "run_a": a,
                "run_b": b,
                "n_hashes": len(la),
                "n_shared_non_noise": int(comunes.sum()),
                "ari_with_noise": float(adjusted_rand_score(la, lb)),
                "ari_shared_non_noise": float(adjusted_rand_score(la[comunes], lb[comunes]))
                if comunes.sum() > 1
                else math.nan,
            }
        )
    return filas


def main() -> None:
    parametros = cargar_parametros()
    e14 = parametros["e14"]
    cluster_cfg = parametros["clustering"]
    semilla = int(parametros["random_seed"])

    columnas = ["complaint_id", "split", "hash_narrativa", "narrative", "company", "product", "issue"]
    muestra = pd.read_parquet(MUESTRA, columns=columnas)
    eventos = muestra.loc[muestra["split"].astype(str) == "train"].copy()
    indice_texto = construir_indice_train(muestra)
    indice = pd.read_parquet(E14_INDICE)
    if not indice["hash_narrativa"].equals(indice_texto["hash_narrativa"]):
        raise ValueError("El índice E14 no coincide con la reconstrucción de train")

    r1_idx = pd.read_parquet(R1_INDICE).set_index("hash_narrativa")["embedding_row"]
    filas_r1 = r1_idx.reindex(indice["hash_narrativa"]).astype("int64").to_numpy()
    r1_cache = np.load(R1_EMBEDDINGS, mmap_mode="r")
    r1 = np.asarray(r1_cache[filas_r1], dtype=np.float32)
    r1_masked = np.asarray(np.load(R1_MASKED, mmap_mode="r"), dtype=np.float32)
    if r1.shape != r1_masked.shape:
        raise ValueError("R1 original y enmascarado no están alineados")

    mask_cfg = e14["company_mask"]
    textos_masked = indice_texto.apply(
        lambda fila: enmascarar_empresa(
            str(fila["texto_normalizado"]), fila["companies"], int(mask_cfg["min_alias_chars"])
        )[0],
        axis=1,
    )

    asignaciones: dict[str, np.ndarray] = {}
    probabilidades: dict[str, np.ndarray] = {}
    definiciones: dict[str, dict[str, Any]] = {}
    metricas: list[dict[str, Any]] = []
    proyeccion = indice[["e14_row", "hash_narrativa"]].copy()

    print("E14 control: TF-IDF unique-fit -> SVD -> KMeans", flush=True)
    splits_train = pd.Series("train", index=indice_texto.index, dtype="string")
    r0, _ = construir_r0(indice_texto["texto_normalizado"], splits_train, parametros["tfidf"])
    svd = TruncatedSVD(
        n_components=int(cluster_cfg["svd_components"]),
        n_iter=5,
        random_state=semilla,
    )
    inicio = time.perf_counter()
    r0_reducido = normalize(svd.fit_transform(r0), norm="l2").astype(np.float32)
    tiempo_svd = time.perf_counter() - inicio
    proyeccion["svd_0"] = r0_reducido[:, 0]
    proyeccion["svd_1"] = r0_reducido[:, 1]
    del r0
    gc.collect()

    kmeans_ids = []
    for k in e14["kmeans_clusters"]:
        run_id = f"r0_kmeans_k{k}"
        inicio = time.perf_counter()
        modelo = KMeans(n_clusters=int(k), n_init=10, random_state=semilla)
        labels = modelo.fit_predict(r0_reducido).astype(np.int32)
        segundos = time.perf_counter() - inicio + tiempo_svd
        probs = np.ones(len(labels), dtype=np.float32)
        run = {
            "run_id": run_id,
            "representation": "R0_unique_fit",
            "condition": "unmasked",
            "algorithm": "SVD_KMeans",
            "seed": semilla,
            "parameter": f"k={k}",
        }
        asignaciones[run_id], probabilidades[run_id], definiciones[run_id] = labels, probs, run
        metricas.append(evaluar_run(run, labels, probs, r0_reducido, indice, eventos, e14, segundos))
        kmeans_ids.append(run_id)

    print("E14 semántico: R1 -> UMAP -> HDBSCAN", flush=True)
    umap_primary = None
    unmasked_seed_ids = []
    base_mcs = int(cluster_cfg["hdbscan"]["min_cluster_size"])
    for seed in e14["umap_seeds"]:
        inicio = time.perf_counter()
        reducido = _umap(r1, cluster_cfg["umap"], int(seed))
        tiempo_umap = time.perf_counter() - inicio
        if int(seed) == semilla:
            umap_primary = reducido
            for componente in range(reducido.shape[1]):
                proyeccion[f"r1_umap_{componente}"] = reducido[:, componente]
        inicio_hdb = time.perf_counter()
        labels, probs = _ajustar_hdbscan(reducido, cluster_cfg["hdbscan"], base_mcs)
        segundos = tiempo_umap + time.perf_counter() - inicio_hdb
        run_id = f"r1_umap_seed{seed}_mcs{base_mcs}"
        run = {
            "run_id": run_id,
            "representation": "R1",
            "condition": "unmasked",
            "algorithm": "UMAP_HDBSCAN",
            "seed": int(seed),
            "parameter": f"min_cluster_size={base_mcs}",
        }
        asignaciones[run_id], probabilidades[run_id], definiciones[run_id] = labels, probs, run
        metricas.append(evaluar_run(run, labels, probs, r1, indice, eventos, e14, segundos))
        unmasked_seed_ids.append(run_id)

    if umap_primary is None:
        raise ValueError("La semilla principal no está en e14.umap_seeds")

    sensibilidad_ids = [f"r1_umap_seed{semilla}_mcs{base_mcs}"]
    for mcs in e14["hdbscan_min_cluster_sizes"]:
        if int(mcs) == base_mcs:
            continue
        inicio = time.perf_counter()
        labels, probs = _ajustar_hdbscan(umap_primary, cluster_cfg["hdbscan"], int(mcs))
        segundos = time.perf_counter() - inicio
        run_id = f"r1_umap_seed{semilla}_mcs{mcs}"
        run = {
            "run_id": run_id,
            "representation": "R1",
            "condition": "unmasked",
            "algorithm": "UMAP_HDBSCAN",
            "seed": semilla,
            "parameter": f"min_cluster_size={mcs}",
        }
        asignaciones[run_id], probabilidades[run_id], definiciones[run_id] = labels, probs, run
        metricas.append(evaluar_run(run, labels, probs, r1, indice, eventos, e14, segundos))
        sensibilidad_ids.append(run_id)

    print("E14 ablación: R1 masked -> UMAP -> HDBSCAN", flush=True)
    inicio = time.perf_counter()
    umap_masked = _umap(r1_masked, cluster_cfg["umap"], semilla)
    tiempo_umap_masked = time.perf_counter() - inicio
    for componente in range(umap_masked.shape[1]):
        proyeccion[f"masked_umap_{componente}"] = umap_masked[:, componente]
    inicio_hdb = time.perf_counter()
    labels_masked, probs_masked = _ajustar_hdbscan(
        umap_masked, cluster_cfg["hdbscan"], base_mcs
    )
    run_masked = f"r1_masked_seed{semilla}_mcs{base_mcs}"
    definicion_masked = {
        "run_id": run_masked,
        "representation": "R1",
        "condition": "company_masked",
        "algorithm": "UMAP_HDBSCAN",
        "seed": semilla,
        "parameter": f"min_cluster_size={base_mcs}",
    }
    asignaciones[run_masked] = labels_masked
    probabilidades[run_masked] = probs_masked
    definiciones[run_masked] = definicion_masked
    metricas.append(
        evaluar_run(
            definicion_masked,
            labels_masked,
            probs_masked,
            r1_masked,
            indice,
            eventos,
            e14,
            tiempo_umap_masked + time.perf_counter() - inicio_hdb,
        )
    )

    primary_unmasked = f"r1_umap_seed{semilla}_mcs{base_mcs}"
    topics = pd.concat(
        [
            construir_topics(
                "unmasked",
                asignaciones[primary_unmasked],
                r1,
                indice,
                indice_texto["texto_normalizado"],
                indice["n_eventos_train"].to_numpy(),
                e14 | {"top_n_words": cluster_cfg["bertopic"]["top_n_words"]},
            ),
            construir_topics(
                "company_masked",
                asignaciones[run_masked],
                r1_masked,
                indice,
                textos_masked,
                indice["n_eventos_train"].to_numpy(),
                e14 | {"top_n_words": cluster_cfg["bertopic"]["top_n_words"]},
            ),
        ],
        ignore_index=True,
    )

    estabilidad = []
    estabilidad += _estabilidad(kmeans_ids, asignaciones, "kmeans_k")
    estabilidad += _estabilidad(unmasked_seed_ids, asignaciones, "umap_seeds")
    estabilidad += _estabilidad(sensibilidad_ids, asignaciones, "hdbscan_min_cluster_size")
    estabilidad += _estabilidad([primary_unmasked, run_masked], asignaciones, "company_mask")

    partes_asignacion = []
    for run_id, labels in asignaciones.items():
        run = definiciones[run_id]
        partes_asignacion.append(
            pd.DataFrame(
                {
                    "e14_row": indice["e14_row"],
                    "hash_narrativa": indice["hash_narrativa"],
                    "run_id": run_id,
                    "representation": run["representation"],
                    "condition": run["condition"],
                    "cluster": labels,
                    "is_noise": labels < 0,
                    "membership_probability": probabilidades[run_id],
                }
            )
        )
    tabla_asignaciones = pd.concat(partes_asignacion, ignore_index=True)
    tabla_metricas = pd.DataFrame(metricas)
    tabla_estabilidad = pd.DataFrame(estabilidad)

    manifiesto = {
        "fit_split": "train",
        "unit": "unique normalized narrative hash",
        "n_train_events": len(eventos),
        "n_train_hashes": len(indice),
        "conflict_policy": "Product/Issue metrics exclude conflicting hashes; event view reported separately",
        "r0_policy": "TF-IDF refit on unique train hashes",
        "ctfidf_formula": "L1 class TF * log(1 + mean class words / corpus term frequency)",
        "max_seq_length": int(parametros["embedding"]["max_seq_length"]),
        "params": {"e14": e14, "clustering": cluster_cfg, "tfidf": parametros["tfidf"]},
        "versions": {
            "scikit_learn": sklearn.__version__,
            "umap_learn": umap.__version__,
        },
        "runs": list(definiciones.values()),
    }

    asegurar(PROCESSED, SALIDA_METRICAS.parent)
    tabla_asignaciones.to_parquet(SALIDA_ASIGNACIONES, index=False, compression="zstd")
    proyeccion.to_parquet(SALIDA_PROYECCION, index=False, compression="zstd")
    tabla_metricas.to_csv(SALIDA_METRICAS, index=False)
    tabla_estabilidad.to_csv(SALIDA_ESTABILIDAD, index=False)
    topics.to_csv(SALIDA_TOPICS, index=False)
    SALIDA_MANIFIESTO.write_text(json.dumps(manifiesto, indent=2), encoding="utf-8")
    print(tabla_metricas[["run_id", "n_clusters", "noise_rate", "silhouette_cosine"]].to_string(index=False))
    print(f"E14 completo: {len(tabla_asignaciones):,} asignaciones, {len(topics):,} topics")


if __name__ == "__main__":
    main()
