# Índice de `reports/modeling`

Qué es cada archivo y carpeta, a qué etapa pertenece y si sigue vigente. El
catálogo de modelos probados y ganadores está en
[`docs/modelos.md`](../../docs/modelos.md); la bitácora completa de decisiones
está en [`docs/registro/models_plan.md`](../../docs/registro/models_plan.md).

Los archivos no se movieron: muchas rutas están fijadas en
`configs/modeling.yaml` y en el código que verifica resultados congelados.

## Estados

| Estado | Significado |
|---|---|
| **Vigente** | Es el modelo o resultado que se usa hoy |
| Confirmado | Pasó la confirmación única en validación (§22) y luego fue reemplazado por un ganador de la ronda §25 |
| Rechazado | Corrió y no cumplió su regla de aceptación |
| Histórico | Versión anterior, conservada solo como registro |
| Insumo | Datos o contratos que usan otras etapas |

## Mapa por etapa

| Etapa | Archivos | Qué es | Estado |
|---|---|---|---|
| M1 | `input_contract.json` | Hash DVC y filas del parquet de entrada | Insumo |
| M2 | `evaluation_contract.json` | Periodos, vistas, métricas y umbrales congelados | Insumo |
| M3 | `baseline_results.json`, `baseline_t1_per_class.csv` | Referencias por frecuencia global y por producto (T1–T4) | Referencia |
| M4 | `tfidf_results.json`, `tfidf_t1_per_class.csv` | TF-IDF + producto (T1–T4) | Confirmado para T2 y T3 como TF-IDF solo texto |
| M5 | `bge_sample_results.json` | BGE vs TF-IDF en una muestra de 120,000 filas | Histórico (M5B lo supera) |
| M5B | `representation_results.json`, `representation_decision.md` | Cinco representaciones de T1–T4 con todo `fit`; elige T1 = BGE + producto, T2 y T3 = TF-IDF solo texto | Confirmado |
| M5F (bloque B) | `foundation_t1/results.json` | TabPFN-3.5 contra el T1 congelado | **Vigente para T1** |
| M5F (ablación §26) | `foundation_t1/ablation_results.json` | TabPFN con 256 componentes y con BGE 1,024 contra TabPFN-100 | Ninguna gana; 100 componentes bastan |
| M10S (§27) | `weekly_composition/mixture_signal/results.json` | p-valor predictivo posterior de la mezcla semanal de M10 | Señal semanal junto a M11; 0 de 12 en calibración |
| M6 | `semantic_space_results.json`, `semantic_pca_variance.png`, `semantic_umap_2d.png` | PCA 1,024→256, UMAP 2D para visualizar, índice FAISS exacto | **Vigente** |
| M7 | `clustering_results.json`, `clustering_candidates.csv`, `clustering_*.png` | 12 configuraciones de clustering sobre UMAP 15D; gana k-means `k=40` | **Vigente** |
| M7S (bloque A) | `space_sensitivity/results.json`, `space_sensitivity/candidates.csv` | 9 espacios × 15 candidatos; ninguno supera a M7 | Sensibilidad |
| M8A | `bge_full_results.json` | Embeddings BGE de todas las filas elegibles | Insumo |
| M8B | `weekly_patterns_results.json`, `weekly_counts.csv`, `cluster_summary.csv`, `weekly_patterns.png`, `weekly_cluster_sizes.png` | 40 patrones, regla de novedad y conteos semanales por patrón | **Vigente** |
| M9 (antiguo) | `negative_binomial_*.{json,csv,png}` | NB-V2 (`normalized_softmax_v2`) del runner inicial; rechazado | Histórico |
| M9 | `weekly_counts/` | Un subdirectorio por candidato de conteos semanales (tabla abajo) | Ver tabla |
| M9 bloque C | `weekly_counts/challenges/*.json` | Cada candidato contra NB-R4-H v3 en calibración; `*_validation.json` es el reporte único en 2025-H1 | Decisión |
| M10 | `weekly_composition/` | Composición semanal Dirichlet-Multinomial (tabla abajo) | Ver tabla |
| M11 | `persistent_change/` | CUSUM sobre NB-R4-H v3: diseño, alertas, detección y reporte de 2025-H1 | Histórico |
| M11 | `persistent_change_c_a/` | CUSUM sobre C-A, rehecho tras el bloque C | **Vigente** |
| §22 | `final_confirmation.json`, `final_confirmation.md` | Confirmación única en validación de T1–T4, M9 y M10 (2026-10-01) | Registro |
| Gráficos | `model_graphs/*.png` | Grafos PyMC de los modelos de M9 y M10 | Referencia |
| Registros | `runs/` (ignorado por Git) | `run.json` de cada ejecución, lo que se publica en MLflow | Local |
| Logs | `slurm-*.out`, `*.log` (ignorados por Git) | Salida de SLURM y de DVC | Local |

## Cómo leer una carpeta de modelo

```
weekly_counts/<modelo>/
  decision.md                      reglas fijadas antes de correr y decisión
  prior_checks/<run_key>.json      prior predictive
  <run_key>/                       un run: <fecha>-<modo>-<config8>-<fuentes8>
    results.json                   diagnósticos, métricas y comparación
    predictions.csv                predicciones por semana y patrón
    metrics.csv, bootstrap.json    métricas por patrón y bootstrap semanal
    posterior_summary.csv          resumen del posterior
    backtest.png, coverage.png, residuals.png, prior.png, trace.png
    run.json                       registro para MLflow
    _SUCCESS                       el run terminó completo
```

El modo es `prior`, `pilot` o `full`. Solo un run `full` decide; los pilotos
son pruebas técnicas cortas. Los posteriores (`posterior.nc`) viven en
`artifacts/models/weekly_counts/<modelo>/<run_key>/` y se versionan con DVC.

## Candidatos de M9 en `weekly_counts/`

WIS de calibración (menor es mejor); referencia B1-R4 = 43.12.

| Carpeta | Modelo | Run que decide | WIS | Estado |
|---|---|---|---:|---|
| `poisson_static_pymc_v1` | B1 PyMC, Poisson con participaciones fijas | `20260927T003211…-full` | 138.75 | Rechazado: cobertura 16% |
| `b1_rolling_4_v1` | B1-R4, Poisson con participaciones de 4 semanas | `20260927T042119…-full` | 43.12 | Referencia de M9 |
| `b1_rolling_13_v1` | B1-R13, ídem con 13 semanas | `20260927T042119…-full` | 64.84 | Referencia |
| `nb_static_global_v3` | NB-V3, binomial negativa estática, una dispersión | `20260927T011144…-full` | 92.57 | Rechazado |
| `nb_rolling_4_global_v1` | NB-R4, participaciones de 4 semanas, una dispersión | `20260927T153922…-full` | 47.72 | Rechazado: no mejora B1-R4 |
| `nb_rolling_4_hierarchical_v1` | NB-R4-H v1, dispersión por patrón, no centrada | solo piloto | 32.49 | Histórico: ESS insuficiente |
| `nb_rolling_4_hierarchical_v2` | NB-R4-H v2, `ZeroSumNormal` | solo prior | — | Histórico: JAX falló con cadenas paralelas |
| `nb_rolling_4_hierarchical_v3` | NB-R4-H v3, dispersión por patrón, centrada | `20260927T161316…-full` | 32.42 | Confirmado en §22; reemplazado por C-A |
| `nb_discounted_hierarchical_v1` | C-A: v3 con memoria que decae (δ = 0.5) | `20261005T200606…-full` | **30.30** | **Vigente** (bloque C) |
| `nb_state_space_v1` | C-B: paseo aleatorio t de Student en las participaciones | `20261006T012448…-full` | 31.95 | Rechazado: no supera a v3 con claridad |
| `chronos2_zero_shot_v1` | Chronos-2 sin entrenamiento | `20261006T161509…-full` | 36.02 | Rechazado |
| `timesfm3_zero_shot_v1` | TimesFM 3.0 sin entrenamiento (solo intervalos 50% y 80%) | `20261006T161506…-full` | 40.70 vs 41.38 | Rechazado |

`nb_discounted_hierarchical_v1/share_selection.json` es la búsqueda de δ y del
recorte en `fit`. NB-V1 (`nb_independent_linear_v1`) y NB-V2
(`nb_softmax_linear_v2`) no tienen carpeta: quedaron en MLflow y, NB-V2, en los
`negative_binomial_*` de la raíz.

## Candidatos de M10 en `weekly_composition/`

Log score conjunto de calibración (mayor es mejor); referencia B2-R4 = −460.9.

| Carpeta | Modelo | Run que decide | Log score | Estado |
|---|---|---|---:|---|
| `dirichlet_multinomial_static_v1` | DM-V1, participaciones fijas | `20261001T015447…-full` | −246.0 | Rechazado: falló una comprobación predictiva |
| `dirichlet_multinomial_rolling_4_v1` | DM-R4, participaciones de 4 semanas, κ = 826 | `20261001T015410…-full` | **−210.1** | **Vigente**; confirmado en §22 |

## Alertas de M11

| Carpeta | Línea base | Umbral CUSUM | Alertas al mes (`fit`, calibración) | Estado |
|---|---|---:|---|---|
| `persistent_change/` | NB-R4-H v3 | 3.368 | 3.9 y 2.2 | Histórico; incluye `decision.md` y el reporte de 2025-H1 |
| `persistent_change_c_a/` | C-A | 3.368 | 2.2 y 1.4 | **Vigente**; `validation_*` es su reporte de 2025-H1 |

## Bloque diario en `daily_counts/`

Panel `daily_counts.csv` (DVC; 912 días × 40 patrones, SHA-256 fijado en cada
config). WIS diario de calibración (menor es mejor); referencia D-B1 Poisson
de 7 días = 9.263. Las tablas CSV de cada run están en DVC; los posteriores en
`artifacts/models/daily_counts/` (una carpeta DVC).

| Carpeta | Modelo | Run que decide | WIS | Estado |
|---|---|---|---:|---|
| `nb_daily_hierarchical_v1` | D-A, binomial negativa con memoria (δ = 0.8) y día de semana | `20261008T031603…-full` | 6.959 | Aceptado; `share_selection.json` es la búsqueda de δ en ajuste |
| `nb_daily_no_dow_v1` | D-B2, D-A sin día de semana | `20261008T031603…-full` | 7.696 | Aceptado |
| `nb_daily_fourier_v1` | D-C, Fourier (2 armónicos) y tendencia local | `20261008T034814…-full` | 7.173 | Aceptado |
| `zinb_daily_hierarchical_v1` | D-D, D-A con inflación de ceros | `20261008T040100…-full` | **6.946** | **Ganador diario** (§28.2) |
| `nb_daily_state_space_v1` | D-B, paseo aleatorio t de Student diario | solo pilotos | 7.82 (7 días) | No evaluado en full: el muestreador no converge en 8 h |
| `dirichlet_multinomial_daily_v1` | D-E, Dirichlet-multinomial diaria, κ = 261 | `20261008T180858…-full` | log score −151.6 | Aceptado (+78 vs multinomial de 7 días); `daily_surprise.csv` es la señal diaria |

`challenge/` guarda la elección del ganador (`*-calibration.json`, bootstrap
pareado del ganador contra cada aceptado) y su único reporte de 2025-H1
(`*-validation.json`, ya consultado: WIS 14.56 vs 19.31).

## Alertas diarias D-11 en `daily_change/`

| Archivo | Qué contiene |
|---|---|
| `results.json` | Umbrales (h = 3.90, z* = 2.72), detección por escenario, alarmas reales, decisión: **no adoptada** (detección máxima 44%, se exigía 50%) |
| `alerts.csv` (DVC) | Exceso, CUSUM y alarmas de cada día y patrón en `fit` y calibración |
| `detection.csv` (DVC) | Retraso de cada regla en cada ráfaga inyectada |

No hay `validation_*`: sin regla adoptada no se consulta 2025-H1.

## Publicado en MLflow

Un run por decisión (DagsHub, experimento `claims-triage-modeling`). Los
`run.json` de `runs/` son la fuente; estos son los de la ronda §25:

| Run | Qué registra |
|---|---|
| `5ca3b797784d4a968fac0a95d7904cff` | M7S, bloque A |
| `b29a0b2a36b047e0a37a17d6217bc1e4` | M5F, bloque B |
| `d07b98c6fa3644269b69254021b87552` | M5F, ablación de entrada de TabPFN (§26) |
| `2e6d9a23cbf24ab7be73a6b9c355e133` | M10S, señal semanal de mezcla rara (§27) |
| `21ec486b702948fd9969e8b80e5357e2` | Bloque C, los 4 candidatos contra v3 |
| `bd71bb91`, `60c6c456`, `ff8450aa`, `a76cf32a` | M9D, fulls de D-A, D-B2, D-C y D-D (§28.2) |
| `8c4b1cd5a80b46f09be1cdcc4e24f636` | M10D, full de D-E (§28.2) |
| `2fedeefaf2d446d5a46e78f954f5a051` | M9D, challenge diario: ganador D-D (§28.2) |
| `c3bd8c4ad23c4f59b3caa965acfd5210` | M11D, regla diaria D-11 sobre D-D: no adoptada (§28.3) |
