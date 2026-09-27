# NB-R4-H v2: `nb_rolling_4_hierarchical_v2`

<!-- markdownlint-disable MD013 -->

## Pregunta

La misma de NB-R4-H v1: ¿una `alpha` por cluster, con la media de las 4 semanas
anteriores, supera a B1-R4 según la regla de `models_plan.md` §12.4?

## Cambio respecto de v1

El piloto de v1 falló el gate de ESS: `log_alpha_sigma` tuvo ESS bulk 58 con la
parametrización no centrada. Con aprobación del usuario del 2026-09-27, v2 cambia
solo la parametrización, que pasa a ser centrada:

```text
log_alpha_deviation ~ ZeroSumNormal(log_alpha_sigma)   # suma cero, centrada
alpha[c] = exp(log_alpha_global + log_alpha_deviation[c])
```

La media, los priors (`log_alpha_global ~ Normal(2.302585093, 1)` y
`log_alpha_sigma ~ HalfNormal(0.75)`), los datos y el muestreo son iguales a
v1. Esto se aparta de la parametrización no centrada que pedía §7.5.

## Gates y regla

Son los mismos de v1 (`../nb_rolling_4_hierarchical_v1/decision.md`), fijados
antes de ejecutar: prior con como máximo 1% de conteos imposibles, piloto con 0
divergencias, R-hat `<= 1.05`, ESS `>= 100` y sin topes de profundidad. El full
sigue la regla del 2026-09-27 frente a B1-R4. Los pilotos no se publican en
MLflow.

## Ejecuciones

Commit `b04b068`. El prior predictive pasó
(`20260927T160149.911429Z-prior-82c5ccd9-ccba3b59`: conteos imposibles 2.7e-5).
Los dos pilotos (SLURM 53278 y 53280) terminaron con `Segmentation fault` (código
139) a los 14 s, al compilar el modelo en JAX. Sus carpetas `.inprogress` quedan
en Khipu y DVC las ignora.

Reproducción con los datos reales, en Khipu y en Windows:

| Modelo | Cadenas en paralelo | Cadenas secuenciales |
|---|---|---|
| v2 (`pm.ZeroSumNormal`) | Segmentation fault | OK |
| v1 (contrastes con matriz) | OK | No probado |

## Decisión

v2 no se puede ejecutar con cadenas en paralelo por un fallo de JAX/XLA con
`pm.ZeroSumNormal`. No es un problema del modelo ni de los datos. Se reemplaza
por v3, que mantiene la parametrización centrada aprobada, usa las mismas
operaciones de v1 y define el mismo modelo que v1.
