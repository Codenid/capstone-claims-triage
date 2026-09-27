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

Pendiente.
