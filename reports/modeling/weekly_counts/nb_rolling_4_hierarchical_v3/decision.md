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

Commit `544d45e`, clon limpio de Khipu.

| Nivel | Run key | Resultado |
|---|---|---|
| Prior | `20260927T161027.833236Z-prior-a3ecfab3-9105ce1f` | Conteos imposibles 2.4e-5, ninguno negativo |
| Piloto | `20260927T161101.453104Z-pilot-a3ecfab3-9105ce1f` | SLURM 53282, 1 min 30 s, MaxRSS 0.73 GiB; 0 divergencias, R-hat 1.020, ESS bulk 467, ESS tail 224; `log_alpha_sigma` con ESS 686; normalización 5.7e-15 |

La parametrización centrada resolvió la mezcla de v1 y se ejecutó sin fallos con
cadenas en paralelo en Khipu. Todos los gates del piloto se cumplen. No se
revisaron métricas de calibración del piloto, y el piloto no se publica en
MLflow.
