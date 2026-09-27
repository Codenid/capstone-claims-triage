# NB-R4-H v3: `nb_rolling_4_hierarchical_v3`

<!-- markdownlint-disable MD013 -->

## Pregunta

La misma de v1 y v2: ¿una `alpha` por cluster, con la media de las 4 semanas
anteriores, supera a B1-R4 según la regla de `models_plan.md` §12.4?

## Cambio respecto de v1 y v2

- v1, no centrada: el piloto falló el gate de ESS (`log_alpha_sigma`, 58).
- v2, centrada con `pm.ZeroSumNormal`: JAX se cae con cadenas en paralelo.
- v3, centrada con los contrastes de v1:

```text
log_alpha_contrast ~ Normal(0, log_alpha_sigma)          # centrada
log_alpha[c] = log_alpha_global + (zero_sum_basis @ log_alpha_contrast)[c]
```

Define el mismo modelo que v1, con la misma media, priors, datos y muestreo;
solo cambia la parametrización. Mantiene la decisión del usuario del 2026-09-27
de usar una parametrización centrada. Antes de congelarla se comprobó en Windows
que compila y muestrea con cadenas en paralelo y datos reales; el piloto lo
comprueba en Khipu.

## Gates y regla

Son los mismos de v1 (`../nb_rolling_4_hierarchical_v1/decision.md`), fijados
antes de ejecutar. Los pilotos no se publican en MLflow.

## Ejecuciones

Pendiente.
