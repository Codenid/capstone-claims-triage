# M5B: representaciones de T1–T4 con todo `fit`

<!-- markdownlint-disable MD013 -->

## Pregunta

¿Qué representación del texto conviene en cada objetivo cuando los modelos se
entrenan con todas las filas de `fit`? La regla de `models_plan.md` §21 se
aprobó el 2026-09-30, antes de entrenar.

## Ejecución

| Campo | Valor |
|---|---|
| Código | Commit `267135a`, `src/models/representation_comparison.py` |
| Prueba rápida | SLURM 53902: 20,000 filas de `fit`, 2 min 38 s, 10.2 GB; no guarda resultados |
| Run completo | SLURM 53903: 44 min 29 s, 25.8 GB |
| Reporte | `reports/modeling/representation_results.json` |
| DVC | `artifacts/models/representation_comparison.dvc`, `ecf6c9262d0e9a44ad9ea97617e316ba.dir` (18 archivos) |
| MLflow | T1 `c98a902b…`, T2 `bb51ccdd…`, T3 `243949bf…`, T4 `c2feae99…` |
| Filas de `fit` | 1,067,194 para T1 y T4; 1,066,288 para T2 y T3 |
| Filas de calibración | Sin texto compartido: 167,973 para T1 y T4, 167,738 para T2 y T3; 14 semanas |

Controles:

- M3 y M4 reprodujeron sus métricas de calibración publicadas (tolerancia
  1e-9), así que las filas y los embeddings están alineados.
- Los embeddings de M8A cubren todas las filas elegibles de T1–T4 en `fit` y
  calibración, en el mismo orden del dataset.
- La validación no se abrió.

## Resultados

Métrica principal en calibración sin texto compartido. En negrita, la
representación elegida:

| Objetivo | Regla por producto | TF-IDF texto | TF-IDF + producto | BGE texto | BGE + producto |
|---|---:|---:|---:|---:|---:|
| T1 — Macro-F1 | 0.0762 | 0.1406 | 0.2142 | 0.1582 | **0.2397** |
| T2 — precisión promedio | 0.4536 | **0.6094** | 0.6119 | 0.6056 | 0.6104 |
| T3 — precisión promedio | 0.1739 | **0.3640** | 0.3699 | 0.3347 | 0.3512 |
| T4 — precisión promedio | 0.0313 | 0.0471 | **0.0551** | 0.0305 | 0.0417 |

Pasos de la regla. La diferencia es retador menos elegida hasta ese momento,
con su IC bootstrap 95% por semanas:

| Objetivo | Retador | Frente a | Ganancia relativa | Diferencia [IC 95%] | ¿Reemplaza? |
|---|---|---|---:|---|---|
| T1 | TF-IDF texto | Regla por producto | +84.6% | +0.0645 [+0.0611, +0.0690] | Sí |
| T1 | TF-IDF + producto | TF-IDF texto | +52.4% | +0.0736 [+0.0707, +0.0779] | Sí |
| T1 | BGE texto | TF-IDF + producto | −26.2% | −0.0561 [−0.0611, −0.0527] | No |
| T1 | BGE + producto | TF-IDF + producto | +11.9% | +0.0255 [+0.0229, +0.0286] | Sí |
| T2 | TF-IDF texto | Regla por producto | +34.4% | +0.1558 [+0.1478, +0.1636] | Sí |
| T2 | TF-IDF + producto | TF-IDF texto | +0.4% | +0.0026 [+0.0017, +0.0034] | No: menos de 5% |
| T2 | BGE texto | TF-IDF texto | −0.6% | −0.0038 [−0.0080, +0.0011] | No |
| T2 | BGE + producto | TF-IDF texto | +0.2% | +0.0010 [−0.0031, +0.0059] | No |
| T3 | TF-IDF texto | Regla por producto | +109.3% | +0.1901 [+0.1771, +0.2030] | Sí |
| T3 | TF-IDF + producto | TF-IDF texto | +1.6% | +0.0059 [+0.0015, +0.0121] | No: menos de 5% |
| T3 | BGE texto | TF-IDF texto | −8.0% | −0.0292 [−0.0394, −0.0170] | No |
| T3 | BGE + producto | TF-IDF texto | −3.5% | −0.0128 [−0.0253, +0.0023] | No |
| T4 | TF-IDF texto | Regla por producto | +50.7% | +0.0159 [+0.0041, +0.0251] | Sí |
| T4 | TF-IDF + producto | TF-IDF texto | +16.8% | +0.0079 [+0.0026, +0.0151] | Sí |
| T4 | BGE texto | TF-IDF + producto | −44.5% | −0.0245 [−0.0312, −0.0195] | No |
| T4 | BGE + producto | TF-IDF + producto | −24.3% | −0.0134 [−0.0190, −0.0081] | No |

Control de T1 frente a M5: en las mismas filas, BGE + producto con todo `fit`
obtuvo Macro-F1 0.2397, y el modelo de la muestra de M5, 0.2396. Pasa el
control por estimación puntual, como se aprobó, pero la diferencia es mínima:
entrenar con casi nueve veces más filas prácticamente no cambió el resultado.

## Decisión

| Objetivo | Modelo elegido | Archivo |
|---|---|---|
| T1 | BGE + producto, todo `fit` | `bge_product_t1.joblib` |
| T2 | TF-IDF solo texto | `tfidf_text_t2.joblib`, con el vectorizador de M4 |
| T3 | TF-IDF solo texto | `tfidf_text_t3.joblib`, con el vectorizador de M4 |
| T4 | TF-IDF + producto (M4) | `artifacts/models/tfidf/t4_model.joblib` |

- En T2 y T3 el producto mejora poco (0.4% y 1.6%) y no alcanza la ganancia
  mínima, así que se queda la representación más simple.
- BGE solo gana en T1. En T2–T4 no justifica su costo: necesita calcular un
  embedding por reclamo.
- Frente a la decisión posterior a M5, que se tomó mirando validación, T2 y T3
  pasan de TF-IDF + producto a TF-IDF solo texto, y T4 pasa de la regla por
  producto a TF-IDF + producto. T1 sigue con BGE + producto.

### T4: validación no independiente

M4 ya evaluó este mismo modelo en validación, antes de que existiera la regla.
En 2025-H1 sin texto compartido obtuvo AP 0.0781, frente a 0.1208 de la regla
por producto, y por eso el README lo había descartado. El 2026-09-30 el usuario
decidió:

- Mantener la elección de la regla, sin usar validación para cambiarla.
- Marcar que la validación de T4 no es una prueba independiente.
- En la confirmación final, reportar T4 junto a la regla por producto y decidir
  ahí su uso en el triaje.

La señal de T4 es débil: 0.56% de positivos en calibración y AP de 0.055.

En MLflow, los 4 runs tienen `validation_used_for_selection=false` y el de T4
además tiene `validation_independent=false`. Ambas etiquetas se agregaron
después de publicar.

## Siguiente paso

M10. La validación se usará una sola vez, en la confirmación final, después de
congelar M10.
