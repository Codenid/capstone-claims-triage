# DM-R4: `dirichlet_multinomial_rolling_4_v1`

<!-- markdownlint-disable MD013 -->

## Pregunta

¿Una Dirichlet-Multinomial centrada en las participaciones de las 4 semanas
anteriores, con una sola concentración `kappa`, predice la composición semanal
conjunta de los 40 grupos mejor que B2 y B2-R4, según la regla de
`models_plan.md` §12.3?

## Modelo

```text
counts[t] ~ DirichletMultinomial(weekly_total[t], kappa * recent_share[t])
recent_share = participación de las 4 semanas anteriores, con +1 por grupo
log_kappa ~ Normal(log 100, 1);  rho = 1 / (kappa + 1)
```

`recent_share` es un dato y PyMC estima solo `kappa`. El ajuste usa las 87
semanas de `fit` con ventana completa. B2 y B2-R4 se calculan dentro del mismo
run: B2 de forma exacta, porque es conjugada, y B2-R4 sin parámetros.

## Gates y regla

Fijados antes de ejecutar:

- Prior: cada draw suma el total semanal y no hay conteos negativos ni
  composiciones imposibles.
- Piloto: 0 divergencias, R-hat `<= 1.05`, ESS `>= 100` y sin topes de
  profundidad.
- Full: mejorar el log score conjunto medio por semana del mejor baseline
  (B2 o B2-R4), con el IC bootstrap semanal pareado 95% entero sobre 0;
  cobertura marginal 80% en [0.70, 0.90] y 95% en [0.88, 0.99]; R-hat
  `<= 1.01`, ESS `>= 400` y 0 divergencias.

## Ejecuciones

Código del commit `c10fde1`, en el clon limpio de Khipu.

| Nivel | Run key | Resultado |
|---|---|---|
| Prior | `20261001T015135.911288Z-prior-cff27014-1fa6f95c` | Los draws suman el total; `kappa` del prior: mediana 102, IC [18, 653] |
| Piloto | `20261001T015224.747306Z-pilot-cff27014-1fa6f95c` | SLURM 53917, 21 s, 0.74 GB; R-hat 1.004, ESS 262/184, 0 divergencias |

El prior supone más variación semanal que la observada en `fit`, pero no produce
composiciones imposibles y su rango cubre lo observado; se mantuvo. El log del
piloto también imprimió el log score de calibración; no se usó, y el full corrió
con la misma configuración.

### Full

| Campo | Valor |
|---|---|
| Run key | `20261001T015410.667626Z-full-cff27014-1fa6f95c` |
| Commit | `3327da9`: el código de `c10fde1` más los reportes del piloto |
| SLURM | 53919, `COMPLETED`, 38 s, MaxRSS 1.31 GB |
| MLflow | `90ac1c09ab3c4276b68f5072f6ec32f9` |
| DVC | `artifacts/models/weekly_composition.dvc`, `b73cc2d94925f1cbacbb81f84c79a5ef.dir` |
| Diagnósticos | R-hat 1.002, ESS bulk 2851, ESS tail 2501, 0 divergencias |
| `kappa` | 826, HDI 94% [786, 866]; `rho` = 0.0012 |

Calibración, 12 semanas:

| Métrica | DM-R4 | B2-R4 | B2 |
|---|---:|---:|---:|
| Log score conjunto por semana | −210.14 | −460.93 | −1770.24 |
| Cobertura marginal 80% y 95% | 89.2% y 95.4% | 28.5% y 41.3% | 9.4% y 16.7% |
| WIS marginal | 31.18 | 43.39 | 139.21 |
| Variación total de la composición | 0.063 | 0.063 | 0.172 |

Frente a B2-R4, el mejor baseline, el log score mejora 250.79 nats por semana,
con IC bootstrap 95% [205.13, 295.84].

Por tercil de volumen en calibración:

| Tercil | Cobertura 80% | Cobertura 95% | WIS DM-R4 | WIS B2-R4 |
|---|---:|---:|---:|---:|
| Pequeños | 89.9% | 95.8% | 14.19 | 18.92 |
| Medianos | 87.8% | 94.2% | 25.88 | 35.70 |
| Grandes | 89.7% | 96.2% | 54.77 | 77.44 |

El WIS marginal no es un gate de M10. Solo como referencia: NB-R4-H v3 obtuvo en
M9 un WIS de 32.42 y coberturas de 85.4% y 95.6%.

## Decisión

- Estado automático: `accepted`. Cumple todos los criterios de §12.3.
- **DM-R4 es el modelo elegido de M10.**
- Un `kappa` global alcanza: la cobertura es estable en los tres terciles,
  porque en la Dirichlet-Multinomial la variación relativa ya es mayor en los
  grupos con menos participación. Por eso no hace falta la Multinomial
  logística-normal de la escalera aprobada el 2026-09-30.
- La cobertura de 80% (89.2%) está cerca del límite superior: esos intervalos
  son algo anchos.
- La validación no se abrió. Queda para la confirmación final única.
