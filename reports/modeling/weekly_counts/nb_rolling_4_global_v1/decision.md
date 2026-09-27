# NB-R4: `nb_rolling_4_global_v1`

<!-- markdownlint-disable MD013 -->

## Pregunta

¿Una Negative Binomial alrededor de las participaciones de las 4 semanas
anteriores, con una sola dispersión, supera al mejor baseline (B1-R4) según la
regla de promoción de `models_plan.md` §12.4?

## Modelo

```text
recent_share[c,t] = (conteos de c en las 4 semanas anteriores + 1) / (total de esas semanas + 40)
mu[c,t] = weekly_total[t] * recent_share[c,t]
y[c,t] ~ NegativeBinomial(mu[c,t], alpha)        # variance = mu + mu**2 / alpha
log_alpha ~ Normal(2.302585093, 1)               # mismo prior que NB-V2 y NB-V3
```

- `recent_share` es un dato, no un parámetro: PyMC estima solo `alpha`.
- El ajuste usa las semanas 5 a 91 de `fit` (3,480 filas), porque las cuatro
  primeras no tienen ventana completa.
- En calibración, `recent_share` usa las semanas anteriores ya observadas;
  `alpha` no se actualiza después de `fit`.

## Gates técnicos del piloto, fijados antes de ejecutarlo

El piloto usa 2 cadenas, 250 de tune y 250 draws. No se publica en MLflow y sus
métricas de calibración no se usan. El full se envía solo si:

- las pruebas pasan en Linux;
- el prior predictive tiene como máximo 1% de conteos imposibles;
- el error de normalización de medias es menor que `1e-10`;
- hay 0 divergencias, R-hat `<= 1.05` y ESS bulk y tail `>= 100`;
- no hay topes de profundidad del árbol;
- el tiempo extrapolado al full es menor que 3 h y la memoria menor que 32 GB.

## Regla del full

Se aplica la regla congelada el 2026-09-27:

- WIS de calibración al menos 5% menor que el del mejor baseline. Hoy es B1-R4,
  con 43.11, así que el objetivo es 40.95 o menos.
- El intervalo bootstrap 95% de la diferencia de WIS debe quedar entero por
  debajo de 0.
- WAPE y MAE solo rechazan si su intervalo bootstrap queda entero por encima
  de 0.
- Cobertura 80% entre 0.70 y 0.90, cobertura 95% entre 0.88 y 0.99, R-hat
  `<= 1.01`, ESS `>= 400` y 0 divergencias.

## Ejecuciones

Clon limpio en Khipu: `/home/piero.palacios/capstone-claims-triage-runs`,
commit `e807869`.

| Nivel | Run key | Resultado |
|---|---|---|
| Prior | `20260927T153630.891988Z-prior-34f4684f-e1410d83` | Conteos imposibles 5.2e-6, ninguno negativo; share máximo semanal 95% en [0.107, 0.339] frente a [0.141, 0.195] en `fit` |
| Piloto | `20260927T153714.395019Z-pilot-34f4684f-e1410d83` | SLURM 53270, 1 min 15 s, MaxRSS 0.70 GiB; R-hat 1.006, ESS bulk 293, ESS tail 170, 0 divergencias, profundidad máxima 3; normalización 5.7e-15; `alpha` 5.83, HDI 94% [5.55, 6.11] |

Todos los gates del piloto se cumplen. No se revisaron métricas de calibración
ni de validación del piloto, y el piloto no se publica en MLflow.

### Full

| Campo | Valor |
|---|---|
| Run key | `20260927T153922.164206Z-full-34f4684f-e1410d83` |
| Commit | `8304231` |
| SLURM | 53271, `COMPLETED`, 3 min 3 s, MaxRSS 1.3 GB |
| MLflow | `5ecad8c8482d4bfa8ecac5d6afd7cc11` |
| DVC | `artifacts/models/weekly_counts.dvc`, `632c07e66b63a73ee6f28c68f02896dd.dir` |
| Diagnósticos | R-hat 1.002, ESS bulk 2855, ESS tail 2754, 0 divergencias |

Calibración, con B1-R4 como mejor baseline:

| Métrica | NB-R4 | B1-R4 | Diferencia, IC bootstrap 95% |
|---|---:|---:|---|
| WIS | 47.72 | 43.06 | +4.66, [-0.09, 9.45] |
| WAPE | 13.68% | 12.46% | +1.22 puntos, [0.52, 1.90] |
| MAE | 61.17 | 55.72 | +5.45, [2.32, 8.51] |
| Cobertura 80% y 95% | 89.2% y 95.6% | — | Dentro de rango |

Por tercil de volumen en calibración:

| Tercil | Cobertura 95% NB-R4 | WIS NB-R4 | WIS B1-R4 |
|---|---:|---:|---:|
| Pequeños | 91.7% | 13.88 | 18.91 |
| Medianos | 95.5% | 27.62 | 35.58 |
| Grandes | 100% | 104.26 | 76.53 |

## Decisión

- Estado automático: `rejected_no_practical_gain`; criterios fallidos
  `wis_gain,wis_bootstrap,wape,mae`. NB-R4 no se acepta.
- Con una sola `alpha` (5.8), los clusters grandes reciben una dispersión
  relativa cercana a 41%, pero su error real ronda 10%. Sus intervalos son
  demasiado anchos y su mediana queda baja. En los clusters pequeños y
  medianos NB-R4 ya supera a B1-R4.
- La cobertura cambia sistemáticamente con el volumen: se cumple el gate de
  §5 para NB-R4 con `alpha` por cluster.
