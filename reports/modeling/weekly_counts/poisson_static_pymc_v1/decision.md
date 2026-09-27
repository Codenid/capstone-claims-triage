# B1 PyMC: `poisson_static_pymc_v1`

<!-- markdownlint-disable MD013 -->

## Pregunta del experimento

¿El pipeline PyMC de M9 reproduce el baseline Poisson analítico con
participaciones fijas? B1 PyMC es un control del pipeline
(`candidate_role=pipeline_baseline`), no un candidato a promoción. El baseline
analítico (`fixed_poisson_reference.py`) se conserva sin cambios.

## Modelo y versión

| Campo | Valor |
|---|---|
| Modelo | `poisson_static_pymc_v1`; fuente y configuración sin cambios desde `0fed42d` |
| Fórmula | `rate ~ Dirichlet(1, ..., 1)`; `mu[c,t] = weekly_total[t] * rate[c]`; `y[c,t] ~ Poisson(mu[c,t])` |
| Estrategia del prior | Uniforme; la fuente congelada no declara `prior_strategy`, por eso MLflow muestra `not_recorded` |
| Commit | `713ee344ba5be5532868cf9084a4627ee10aacc1` |
| Hash de configuración | `b487e2fbc8d050051c83406ac384750c064ecd21531caff6b38d5a9cd597d0ae` |
| Hash de datos | `weekly_counts.csv`: `35c5503b0b9b2a4b9e47984294683d27987b3c7824ae41de642c0a47cefd1ce1`; DVC de patrones: `87be2a97daa3b169bbd65850a64c89a6.dir` |
| Panel | 40 clusters; 91 semanas de `fit`, 12 de calibración y 25 de validación |
| Ajuste | Solo `fit`; calibración y validación se predicen sin actualizar el posterior |

## Ejecuciones en Khipu

Clon limpio: `/home/piero.palacios/capstone-claims-triage-runs`.

| Nivel | Run key | SLURM | MLflow | Tiempo del job | MaxRSS SLURM | Pico del proceso |
|---|---|---|---|---|---|---|
| `prior` | `20260927T003014.154155Z-prior-b487e2fb-1f4fefe6` | Nodo de acceso | No aplica | Segundos | No aplica | No medido |
| `pilot` | `20260927T003054.260416Z-pilot-b487e2fb-1f4fefe6` | 53177 | `d5219aa9f6eb4c848cbb0551a533afd2`, borrado de MLflow el 2026-09-27; sus salidas siguen en Git | 37 s | 1.19 GiB | 0.98 GiB |
| `full` | `20260927T003211.742368Z-full-b487e2fb-1f4fefe6` | 53178 | `0ddb96b9f9c645bdb8e8a687cedf09cc` | 47 s | 1.15 GiB | 1.41 GiB |

Los posteriores están en `artifacts/models/weekly_counts.dvc`
(`f590c4b1a8e70db5e243c22e56837f80.dir`), subido a DagsHub.

## Prior predictive

Con 500 draws sobre `fit` no hubo conteos negativos ni mayores que el total
semanal. El share máximo semanal del prior tuvo un intervalo 95% de
[0.069, 0.179]; en `fit` fue [0.141, 0.202]. El prior no se modificó.

## Diagnósticos del full

| R-hat máximo | ESS bulk mínimo | ESS tail mínimo | Divergencias | BFMI mínimo | Profundidad máxima |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 7858 | 4556 | 0 | 0.94 | 5, sin topes |

## Reproducción del baseline analítico

Las medias esperadas de B1 PyMC difieren de las analíticas en 7.45e-4 como
máximo relativo (mediana 1.15e-4), por el pseudo-conteo del prior Dirichlet. Las
medias suman el total semanal con error máximo 8.9e-16.

| Split | Modelo | WIS | WAPE | Cobertura 80% | Cobertura 95% |
|---|---|---:|---:|---:|---:|
| `fit` | B1 PyMC | 56.69 | 23.32% | 17.25% | 26.70% |
| `fit` | B1 analítico | 56.73 | 23.33% | 17.14% | 26.76% |
| Calibración | B1 PyMC | 138.75 | 34.48% | 9.17% | 16.25% |
| Calibración | B1 analítico | 138.89 | 34.48% | 9.17% | 16.46% |

El runner también calcula métricas de validación, pero no se abrieron para esta
decisión.

## Decisión

- Gate de B1 del plan: el pipeline reproduce el baseline analítico. El pipeline
  PyMC queda validado.
- Estado automático: `rejected_predictive`; criterios fallidos
  `wape,coverage_80,coverage_95`. Poisson subestima fuertemente la dispersión:
  su cobertura 95% en calibración es 16%. El WAPE empata con el analítico a
  cuatro decimales.
- B1 PyMC queda registrado como baseline del pipeline y no se promueve.
- `historical_run_id` sigue en `null` para no modificar la fuente congelada; el
  run publicado de referencia es `0ddb96b9f9c645bdb8e8a687cedf09cc`.

## Siguiente acción permitida

Continuar con NB-V3. B1-R4, B1-R13 y B2 siguen pendientes.
