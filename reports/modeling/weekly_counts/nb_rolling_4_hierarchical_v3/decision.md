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

### Full

| Campo | Valor |
|---|---|
| Run key | `20260927T161316.494762Z-full-a3ecfab3-9105ce1f` |
| Commit | `2298783` |
| SLURM | 53283, `COMPLETED`, 5 min, MaxRSS 1.37 GiB |
| MLflow | `ed9fa77b50c1458ea5261916da97c5c0` |
| DVC | `artifacts/models/weekly_counts.dvc`, `5da37969ff08350871a1ab6b65d0d009.dir` |
| Diagnósticos | R-hat 1.002, ESS bulk 8317, ESS tail 5343, 0 divergencias |
| `alpha` por cluster | De 0.83 a 178, mediana 6.7 |

Calibración frente al mejor baseline, B1-R4:

| Métrica | NB-R4-H v3 | B1-R4 | Diferencia, IC bootstrap 95% |
|---|---:|---:|---|
| WIS | 32.42 | 43.06 | -10.64, [-13.65, -7.62]; mejora de 24.7% |
| WAPE | 12.66% | 12.46% | +0.20 puntos, [-0.21, 0.60] |
| MAE | 56.61 | 55.72 | +0.89, [-0.92, 2.75] |
| Cobertura 80% y 95% | 85.4% y 95.6% | 29.2% y 42.1% | Dentro de rango |

Por tercil de volumen en calibración:

| Tercil | Cobertura 80% | Cobertura 95% | WIS NB-R4-H v3 | WIS B1-R4 |
|---|---:|---:|---:|---:|
| Pequeños | 88.7% | 96.4% | 14.45 | 18.91 |
| Medianos | 87.2% | 94.2% | 30.11 | 35.58 |
| Grandes | 80.1% | 96.2% | 54.07 | 76.53 |

## Decisión

- Estado automático: `accepted`. Cumple todos los criterios de la regla del
  2026-09-27: el WIS mejora 24.7% con un intervalo bootstrap enteramente
  negativo; WAPE y MAE no empeoran de forma significativa; coberturas y
  convergencia están en rango.
- **NB-R4-H v3 es el candidato elegido de M9.** Mejora a B1-R4 en los tres
  terciles de volumen.
- La validación no se abrió. Queda para la confirmación final única, después
  de congelar T1–T4 y M10.
