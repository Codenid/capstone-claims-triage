# NB-V3: `nb_static_global_v3`

<!-- markdownlint-disable MD013 -->

## Pregunta del experimento

¿Una Negative Binomial con participaciones estáticas y una sola dispersión
global captura la sobredispersión que Poisson (B1) no captura, sin añadir
tendencia temporal?

## Modelo y versión

```text
share ~ Dirichlet(share_concentration)
log_alpha ~ Normal(log_alpha_mean, log_alpha_sigma)
alpha = exp(log_alpha)
mu[c,t] = weekly_total[t] * share[c]
y[c,t] ~ NegativeBinomial(mu[c,t], alpha)      # variance = mu + mu**2 / alpha
```

- Participaciones estáticas, sin tendencia y una sola `alpha` escalar.
- Las medias suman `weekly_total` porque `share` es un simplex; los draws
  independientes no están obligados a sumarlo.
- Configuración full: 4 cadenas, 2000 de tune, 2000 draws,
  `target_accept=0.9` y `max_treedepth=12`.
- `candidate_role=candidate`; fuente y configuración congeladas en
  `configs/weekly_counts/releases.json`.

## Decisión del prior, tomada antes de mirar calibración

Solo se usaron las 91 semanas de `fit`, con 500 draws y semilla 42. Antes de
ver resultados se fijó este criterio: rechazar una alternativa si produce con
frecuencia conteos imposibles (`y > weekly_total` en más de 1% de las celdas) o
una proporción de ceros claramente incompatible con `fit`; entre las
razonables, elegir la más simple.

Alternativas comparadas para `share`:

- uniforme: `Dirichlet(1, ..., 1)`;
- empirical Bayes: `Dirichlet(tau * p0)`, con `p0` igual a las participaciones
  de `fit` y `tau=40` (misma concentración total que la uniforme) o `tau=400`.

Las tres usan para `log_alpha` el mismo prior global de NB-V2:
`Normal(2.302585093, 1)`.

| Prior de `share` | Conteos imposibles | Ceros por cluster, mediana y p97.5 | Ceros del cluster más pequeño | Share máximo semanal, 95% |
|---|---:|---:|---:|---:|
| Uniforme | 0 | 0.47%; 0.88% | 0.7% | [0.070, 0.247] |
| Empirical Bayes, `tau=40` | 1.2e-5 | 15.5%; 58.1% | 64.0% | [0.114, 0.434] |
| Empirical Bayes, `tau=400` | 1.6e-6 | 0.05%; 4.6% | 10.1% | [0.108, 0.340] |
| Observado en `fit` | No aplica | 0%; 7.8% | 2.2% | [0.141, 0.202] |

Para `log_alpha`, el prior produce `alpha` entre 1.4 y 67 (95%) y una razón
varianza/media entre 2.5 y 74 para una media de 100. Es amplio pero plausible,
por lo que se conserva el prior de NB-V2 para mantener la comparabilidad.

Decisión: **prior uniforme**, `prior_strategy=uniform`.

- Es razonable: no produce conteos imposibles y sus ceros y participaciones
  cubren lo observado en `fit`.
- Es la opción más simple: no requiere elegir `tau` ni reutilizar `fit` para
  construir el prior.
- Coincide con el prior de B1, así que B1 y NB-V3 difieren solo en la
  verosimilitud.
- Empirical Bayes con `tau=40` genera demasiados ceros en clusters pequeños.
  Con `tau=400` es razonable, pero añade un parámetro ajustado con los mismos
  datos.

Evidencia: `prior_choice.json`. Los draws de la comparación se simularon con
NumPy usando las mismas fórmulas del modelo y se resumieron con
`prior_check_summary` de `src/models/weekly_counts/diagnostics.py`:

```python
rng = np.random.default_rng(42)
share = rng.dirichlet(concentration, size=500)
alpha = np.exp(rng.normal(2.302585093, 1.0, size=500))
mu = weekly_total[None, :] * share[:, cluster_id]
observed = rng.negative_binomial(alpha[:, None], alpha[:, None] / (alpha[:, None] + mu))
```

El prior predictive oficial del módulo PyMC se genera con
`--run-mode prior` en `prior_checks/`.

## Gates técnicos del piloto, fijados antes de ejecutarlo

El piloto usa 2 cadenas, 250 de tune y 250 draws. Es solo técnico: sus
métricas de calibración y validación no se usan para decidir. El full se envía
solo si se cumplen todas estas condiciones:

- pruebas específicas y suite completa en `OK` en Linux;
- error máximo de normalización de medias menor que `1e-10`;
- 0 divergencias;
- R-hat máximo `<= 1.05`, umbral de piloto; el full exige `<= 1.01`;
- ESS bulk y tail mínimos `>= 100`, umbral de piloto; el full exige `>= 400`;
- ningún tope de profundidad del árbol;
- tiempo extrapolado al full, ocho veces más iteraciones por cadena, menor que
  3 h, y memoria menor que 32 GB. El límite de SLURM es 6 h y 64 GB.

## Criterios del full

Se aplican los criterios congelados en la configuración: R-hat `<= 1.01`, ESS
`>= 400`, 0 divergencias, cobertura 80% entre 0.70 y 0.90, cobertura 95% entre
0.88 y 0.99, y WIS y WAPE no peores que el baseline Poisson fijo en
calibración. Un estado `accepted` sería provisional: la mejora práctica mínima,
el bootstrap semanal y la comparación con B1-R4, B1-R13 y B2 siguen pendientes.

## Ejecuciones

Pendiente.
