# 0006 · Representaciones y clasificadores de T1–T4

| | |
|---|---|
| **Fecha** | 2026-10-07 (registro retrospectivo; decisiones del 2026-09-22 al 2026-10-06) |
| **Etapa CRISP-DM** | 4 · Modelado |
| **Issue** | #17 (PB-17), #12 (PB-12) |
| **Responsable** | @pipaber (DS) |
| **Estado** | aceptada |

## Contexto

Las entradas permitidas son la narrativa normalizada y el producto canónico
([0001](0001-alcance-y-fugas.md)); empresa, estado y fecha quedan como
candidatas. Los baselines de frecuencia dan Macro-F1 0.006 (T1) y la regla por
producto 0.076. T3 y T4 tienen 2.6 % y 1.1 % de positivos.

## Decisión

Comparar cinco representaciones con la regla de [0005](0005-protocolo-de-evaluacion.md),
en calibración sin texto compartido, con modelos lineales (regresión logística):

| Tarea | Métrica | Elegido en M5B | Calibración | 2025-H1 |
|---|---|---|---:|---:|
| T1 motivo (90 clases) | Macro-F1 | BGE-large (1,024) + producto | 0.240 | 0.227 |
| T2 alguna solución | average precision | TF-IDF del texto | 0.609 | 0.570 |
| T3 compensación monetaria | average precision | TF-IDF del texto | 0.364 | 0.270 |
| T4 respuesta fuera de plazo | average precision | TF-IDF + producto | 0.055 | 0.078 (no confirmado) |

T4 no superó a la regla por producto en validación y se descarta: en producción
se usa la regla.

El 2026-10-06, en la ronda pre-registrada `models_plan.md` §25.4, **TabPFN-3.5**
(aprendizaje en contexto con 20,000 filas de ajuste, 100 componentes del PCA
del embedding BGE + producto como categórica, 8 estimadores) reemplazó al T1
lineal: Macro-F1 0.271 contra 0.240 (+12.9 %, IC 95 % [+0.027, +0.035]);
en 2025-H1, 0.270 contra 0.227. Kumo Tabular salió por memoria. Una ablación
de entrada (256 y 1,024 columnas) quedó pre-registrada en §26 el 2026-10-07.

## Alternativas consideradas

- Ajustar modelos no lineales sobre TF-IDF — M5 mostró que la representación
  pesa más que el clasificador; se priorizó cambiar la entrada.
- Reentrenar BGE (fine-tuning) — fuera del alcance de cómputo de la rama.
- Remuestrear T3/T4 — la métrica average precision no lo necesita y el
  umbral se fija en calibración.

## Dónde vive

| | |
|---|---|
| **Parámetro** | `configs/modeling.yaml` → `tfidf`, `bge`, `representation`, `foundation_t1` |
| **Código** | `src/models/tfidf_models.py`, `src/models/bge_full.py`, `src/models/representation_comparison.py`, `src/models/foundation_t1.py` |
| **Resultados** | `reports/modeling/representation_results.json`, `reports/modeling/foundation_t1/results.json`, `reports/modeling/final_confirmation.json` |
| **Artefactos** | `artifacts/models/tfidf`, `artifacts/models/bge_full`, `artifacts/models/foundation_t1` (DVC) |

## Consecuencias

TabPFN-3.5 tiene licencia no comercial: sirve para la evaluación académica y un
experto debe validar esa lectura antes de usarlo con un banco. Cada consulta
exige volver a cargar el contexto de 20,000 filas (≈ 6 minutos en A100 para
168,000 reclamos). El campeón con alias (PB-20) sigue pendiente del MLE.
