# DM-V1: `dirichlet_multinomial_static_v1`

<!-- markdownlint-disable MD013 -->

## Pregunta

Es una referencia, no un candidato: ¿basta una composición fija, aprendida con
todo `fit`, más una dispersión Dirichlet global?

## Modelo

```text
share ~ Dirichlet(1)
counts[t] ~ DirichletMultinomial(weekly_total[t], kappa * share)
log_kappa ~ Normal(log 100, 1);  rho = 1 / (kappa + 1)
```

El ajuste usa las 91 semanas de `fit`. Se evalúa en las mismas semanas que
DM-R4, con los mismos gates y la misma regla de §12.3.

## Ejecuciones

Código del commit `c10fde1`, en el clon limpio de Khipu.

| Nivel | Run key | Resultado |
|---|---|---|
| Prior | `20261001T015140.352069Z-prior-f9a73151-bc87c1f1` | Los draws suman el total y no hay conteos negativos |
| Piloto | `20261001T015245.009873Z-pilot-f9a73151-bc87c1f1` | SLURM 53918, 26 s, 0.77 GB; R-hat 1.032, ESS 482/236, 0 divergencias |
| Full | `20261001T015447.101243Z-full-f9a73151-bc87c1f1` | SLURM 53920, 88 s, 1.38 GB; R-hat 1.002, ESS 8216/4956, 0 divergencias |

El full registra el commit `3327da9`. MLflow `55832eccec284ffdb8fd7bfeca0dda43`;
DVC `b73cc2d94925f1cbacbb81f84c79a5ef.dir`.

Calibración, 12 semanas:

| Métrica | DM-V1 | DM-R4 | B2-R4 |
|---|---:|---:|---:|
| Log score conjunto por semana | −246.04 | −210.14 | −460.93 |
| Cobertura marginal 80% y 95% | 66.3% y 87.1% | 89.2% y 95.4% | 28.5% y 41.3% |
| WIS marginal | 96.43 | 31.18 | 43.39 |
| Variación total de la composición | 0.191 | 0.063 | 0.063 |
| `kappa` | 292, HDI 94% [278, 306] | 826 | No aplica |

## Decisión

- Estado automático: `rejected_predictive`. Su log score supera a B2-R4 (+214.89,
  IC [166.61, 263.19]), pero las coberturas de 80% y 95% quedan bajo el rango.
- Confirma lo que mostró M9: la composición cambia de una semana a otra. Con la
  mezcla fija, el centro se aleja de lo observado y el modelo compensa con más
  dispersión (`kappa` 292 frente a 826), sin llegar a una cobertura correcta.
- No se usa en el triaje.
