"""E12: piso de señal con R0 y modelos lineales escalables.

No selecciona el modelo final. Mide cuánto permiten predecir las narrativas con
TF-IDF antes de invertir en embeddings o modelos más costosos. Reporta vistas
operacional y purgada de hashes vistos en train, sensibilidad de T2 y tres
formulaciones jerárquicas de T3.

Uso:
    uv run python -m src.models.baseline
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from src.data.prepare_eda import cargar_parametros
from src.paths import PROCESSED, REPORTS, asegurar

MATRIZ_R0 = PROCESSED / "r0_tfidf.npz"
INDICE_R0 = PROCESSED / "r0_index.parquet"
SALIDA_METRICAS = REPORTS / "artefactos" / "e12_piso_senal.csv"
SALIDA_PREDICCIONES = PROCESSED / "e12_predictions.parquet"

SPLITS_EVALUACION = ("validation", "test", "ood_2026")
CAPACIDADES = (0.01, 0.05, 0.10)


def ajustar_lineal(
    x: sp.csr_matrix,
    y: np.ndarray,
    config: dict[str, Any],
    semilla: int,
) -> SGDClassifier:
    """Ajusta regresión logística SGD con pesos balanceados acotados."""
    pesos = compute_sample_weight(class_weight="balanced", y=y)
    pesos = np.minimum(pesos, float(config["class_weight_cap"]))
    modelo = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(config["alpha"]),
        max_iter=int(config["max_iter"]),
        tol=float(config["tol"]),
        average=bool(config["average"]),
        random_state=semilla,
        n_jobs=-1,
    )
    modelo.fit(x, y, sample_weight=pesos)
    return modelo


def mascara_vista(
    indice: pd.DataFrame,
    split: str,
    vista: str,
    hashes_train: set[str],
) -> np.ndarray:
    """Crea vista operacional o excluye narrativas ya vistas en train."""
    mascara = indice["split"].astype(str).eq(split).to_numpy(copy=True)
    if vista == "operational":
        return mascara
    if vista == "purged":
        vistos = indice["hash_narrativa"].astype("string").isin(hashes_train).to_numpy()
        return mascara & ~vistos
    raise ValueError(f"Vista desconocida: {vista}")


def _ece(y: np.ndarray, probabilidad: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error con intervalos de probabilidad iguales."""
    bordes = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.digitize(probabilidad, bordes[1:-1]), n_bins - 1)
    error = 0.0
    for bin_id in range(n_bins):
        mascara = bins == bin_id
        if mascara.any():
            error += mascara.mean() * abs(y[mascara].mean() - probabilidad[mascara].mean())
    return float(error)


def metricas_binarias(y: np.ndarray, probabilidad: np.ndarray) -> dict[str, float]:
    """Métricas robustas al desbalance y métricas por capacidad operativa."""
    y = np.asarray(y, dtype=np.int8)
    probabilidad = np.clip(np.asarray(probabilidad, dtype=float), 1e-7, 1 - 1e-7)
    prevalencia = float(y.mean())
    metricas = {
        "average_precision": float(average_precision_score(y, probabilidad)),
        "roc_auc": float(roc_auc_score(y, probabilidad)) if np.unique(y).size == 2 else math.nan,
        "brier": float(brier_score_loss(y, probabilidad)),
        "log_loss": float(log_loss(y, probabilidad, labels=[0, 1])),
        "ece_10": _ece(y, probabilidad),
    }
    orden = np.argsort(probabilidad)[::-1]
    positivos = int(y.sum())
    for capacidad in CAPACIDADES:
        k = max(1, math.ceil(len(y) * capacidad))
        seleccion = y[orden[:k]]
        precision = float(seleccion.mean())
        recall = float(seleccion.sum() / positivos) if positivos else math.nan
        sufijo = f"at_{int(capacidad * 100)}pct"
        metricas[f"precision_{sufijo}"] = precision
        metricas[f"recall_{sufijo}"] = recall
        metricas[f"lift_{sufijo}"] = precision / prevalencia if prevalencia else math.nan
    return metricas


def _agregar(
    filas: list[dict[str, Any]],
    target: str,
    variante: str,
    split: str,
    vista: str,
    y: np.ndarray,
    metricas: dict[str, float],
) -> None:
    base = {
        "target": target,
        "variant": variante,
        "representation": "R0_word_char",
        "split": split,
        "view": vista,
        "n": int(len(y)),
        "n_positive": int(y.sum()) if np.issubdtype(y.dtype, np.number) else pd.NA,
        "prevalence": float(y.mean()) if np.issubdtype(y.dtype, np.number) else pd.NA,
    }
    filas.extend({**base, "metric": nombre, "value": valor} for nombre, valor in metricas.items())


def _evaluar_binario(
    filas: list[dict[str, Any]],
    indice: pd.DataFrame,
    target: str,
    variante: str,
    y_col: str,
    probabilidades: np.ndarray,
    hashes_train: set[str],
    excluir_ambiguos: bool = False,
) -> None:
    for split in SPLITS_EVALUACION:
        for vista in ("operational", "purged"):
            mascara = mascara_vista(indice, split, vista, hashes_train)
            mascara &= indice[y_col].notna().to_numpy()
            if excluir_ambiguos:
                mascara &= ~indice["t2_ambiguous"].astype(bool).to_numpy()
            y = indice.loc[mascara, y_col].astype("int8").to_numpy()
            _agregar(
                filas,
                target,
                variante,
                split,
                vista,
                y,
                metricas_binarias(y, probabilidades[mascara]),
            )


def _top_k_accuracy(y: np.ndarray, proba: np.ndarray, clases: np.ndarray, k: int) -> float:
    k = min(k, proba.shape[1])
    top = np.argpartition(proba, -k, axis=1)[:, -k:]
    return float(np.any(clases[top] == y[:, None], axis=1).mean())


def main() -> None:
    parametros = cargar_parametros()
    config = parametros["baseline"]
    semilla = int(parametros["random_seed"])
    x = sp.load_npz(MATRIZ_R0).tocsr()
    indice = pd.read_parquet(INDICE_R0)
    if x.shape[0] != len(indice) or not (indice["row_number"].to_numpy() == np.arange(len(indice))).all():
        raise ValueError("R0 y su índice no están alineados")

    train = indice["split"].astype(str).eq("train").to_numpy()
    hashes_train = set(indice.loc[train, "hash_narrativa"].dropna().astype(str))
    filas: list[dict[str, Any]] = []
    predicciones = indice[["row_number", "complaint_id", "split", "hash_narrativa"]].copy()

    print("T1: Issue crudo", flush=True)
    y_t1 = indice["target_t1_issue_raw"].astype("string")
    train_t1 = train & y_t1.notna().to_numpy()
    modelo_t1 = ajustar_lineal(x[train_t1], y_t1.loc[train_t1].astype(str).to_numpy(), config, semilla)
    for split in SPLITS_EVALUACION:
        for vista in ("operational", "purged"):
            mascara = mascara_vista(indice, split, vista, hashes_train) & y_t1.notna().to_numpy()
            y = y_t1.loc[mascara].astype(str).to_numpy()
            proba = modelo_t1.predict_proba(x[mascara])
            pred = modelo_t1.classes_[np.argmax(proba, axis=1)]
            metricas = {
                "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
                "accuracy": float(accuracy_score(y, pred)),
                "top_3_accuracy": _top_k_accuracy(y, proba, modelo_t1.classes_, 3),
            }
            _agregar(filas, "T1", "issue_raw", split, vista, y, metricas)

    print("T2: principal y sensibilidad", flush=True)
    y_t2 = indice["target_t2_relief"]
    train_t2 = train & y_t2.notna().to_numpy()
    modelo_t2 = ajustar_lineal(x[train_t2], y_t2.loc[train_t2].astype("int8").to_numpy(), config, semilla + 1)
    p_t2 = modelo_t2.predict_proba(x)[:, 1]
    predicciones["p_t2_main"] = p_t2.astype("float32")
    _evaluar_binario(filas, indice, "T2", "main_all", "target_t2_relief", p_t2, hashes_train)
    _evaluar_binario(
        filas, indice, "T2", "main_nonambiguous", "target_t2_relief", p_t2, hashes_train, True
    )

    train_sens = train_t2 & ~indice["t2_ambiguous"].astype(bool).to_numpy()
    modelo_t2_sens = ajustar_lineal(
        x[train_sens], y_t2.loc[train_sens].astype("int8").to_numpy(), config, semilla + 2
    )
    p_t2_sens = modelo_t2_sens.predict_proba(x)[:, 1]
    predicciones["p_t2_sensitivity"] = p_t2_sens.astype("float32")
    _evaluar_binario(
        filas,
        indice,
        "T2",
        "sensitivity_nonambiguous",
        "target_t2_relief",
        p_t2_sens,
        hashes_train,
        True,
    )

    print("T3: directo, cascada y tres clases", flush=True)
    y_t3 = indice["target_t3_monetary"]
    train_t3 = train & y_t3.notna().to_numpy()
    modelo_t3 = ajustar_lineal(x[train_t3], y_t3.loc[train_t3].astype("int8").to_numpy(), config, semilla + 3)
    p_t3_direct = modelo_t3.predict_proba(x)[:, 1]
    predicciones["p_t3_direct"] = p_t3_direct.astype("float32")
    _evaluar_binario(filas, indice, "T3", "direct", "target_t3_monetary", p_t3_direct, hashes_train)

    train_relief = train_t3 & y_t2.eq(1).fillna(False).to_numpy(dtype=bool)
    modelo_condicional = ajustar_lineal(
        x[train_relief], y_t3.loc[train_relief].astype("int8").to_numpy(), config, semilla + 4
    )
    p_t3_cascade = p_t2 * modelo_condicional.predict_proba(x)[:, 1]
    predicciones["p_t3_cascade"] = p_t3_cascade.astype("float32")
    _evaluar_binario(filas, indice, "T3", "cascade", "target_t3_monetary", p_t3_cascade, hashes_train)

    valores_t2 = y_t2.to_numpy(dtype=np.int8, na_value=-1)
    valores_t3 = y_t3.to_numpy(dtype=np.int8, na_value=-1)
    y_tres = np.where(valores_t2 == 0, 0, np.where(valores_t3 == 1, 2, 1))
    modelo_tres = ajustar_lineal(x[train_t3], y_tres[train_t3], config, semilla + 5)
    proba_tres = modelo_tres.predict_proba(x)
    pos_monetario = int(np.flatnonzero(modelo_tres.classes_ == 2)[0])
    p_t3_tres = proba_tres[:, pos_monetario]
    predicciones["p_t3_three_class"] = p_t3_tres.astype("float32")
    _evaluar_binario(filas, indice, "T3", "three_class", "target_t3_monetary", p_t3_tres, hashes_train)

    print("T4: riesgo de respuesta tardía", flush=True)
    y_t4 = indice["target_t4_late"]
    train_t4 = train & y_t4.notna().to_numpy()
    modelo_t4 = ajustar_lineal(x[train_t4], y_t4.loc[train_t4].astype("int8").to_numpy(), config, semilla + 6)
    p_t4 = modelo_t4.predict_proba(x)[:, 1]
    predicciones["p_t4"] = p_t4.astype("float32")
    _evaluar_binario(filas, indice, "T4", "direct", "target_t4_late", p_t4, hashes_train)

    for split in SPLITS_EVALUACION:
        for vista in ("operational", "purged"):
            mascara = mascara_vista(indice, split, vista, hashes_train) & y_t3.notna().to_numpy()
            violacion = float((p_t3_direct[mascara] > p_t2[mascara]).mean())
            _agregar(
                filas,
                "T3",
                "direct_vs_t2",
                split,
                vista,
                y_t3.loc[mascara].astype("int8").to_numpy(),
                {"hierarchy_violation_rate": violacion},
            )

    resultados = pd.DataFrame(filas)
    asegurar(SALIDA_METRICAS.parent, PROCESSED)
    resultados.to_csv(SALIDA_METRICAS, index=False)
    predicciones.to_parquet(SALIDA_PREDICCIONES, index=False, compression="zstd")
    print(f"E12: {len(resultados):,} métricas -> {SALIDA_METRICAS}")


if __name__ == "__main__":
    main()
