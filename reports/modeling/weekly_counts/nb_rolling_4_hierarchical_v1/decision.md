# NB-R4 con `alpha` por cluster: `nb_rolling_4_hierarchical_v1`

<!-- markdownlint-disable MD013 -->

## Pregunta

NB-R4 superó a B1-R4 en clusters pequeños y medianos, pero una sola `alpha`
dejó intervalos demasiado anchos en los grandes. ¿Una `alpha` por cluster, con
la misma media, supera a B1-R4 según la regla de `models_plan.md` §12.4?

## Modelo

```text
mu[c,t] = weekly_total[t] * recent_share[c,t]          # misma media que NB-R4
y[c,t] ~ NegativeBinomial(mu[c,t], alpha[c])
log_alpha[c] = log_alpha_global + log_alpha_sigma * contraste_suma_cero[c]
log_alpha_global ~ Normal(2.302585093, 1)
log_alpha_sigma ~ HalfNormal(0.75)                     # priors de NB-V2
```

- Parametrización no centrada, con los contrastes de suma cero de NB-V2.
- El ajuste usa las semanas 5 a 91 de `fit`; `alpha` no se actualiza después.

## Gates del piloto y regla del full

Son los mismos de NB-R4 (`../nb_rolling_4_global_v1/decision.md`), fijados
antes de ejecutar: prior con como máximo 1% de conteos imposibles, 0
divergencias, R-hat `<= 1.05`, ESS `>= 100`, sin topes de profundidad y
normalización menor que `1e-10`. El piloto no se publica en MLflow.

Para el full se aplica la regla del 2026-09-27: WIS de calibración al menos 5%
menor que el de B1-R4 en el mismo run (cerca de 43.1, así que cerca de 40.9 o
menos), con el intervalo bootstrap de la diferencia entero por debajo de 0. WAPE y MAE rechazan solo si su intervalo
bootstrap queda entero por encima de 0. Además, cobertura en rango, R-hat
`<= 1.01`, ESS `>= 400` y 0 divergencias.

## Ejecuciones

Pendiente.
