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

Clon limpio en Khipu: `/home/piero.palacios/capstone-claims-triage-runs`.

### Prior predictive oficial

Run key `20260927T010447.709571Z-prior-45c908d1-e06833de`, commit
`a099a30f175c515a71356ae3c49068f74c2f2144`, nodo de acceso de Khipu. Los
resultados coinciden exactamente con los de Windows: conteos imposibles
4.9e-6, ninguno negativo, ceros por cluster con 95% en [0.21%, 0.85%], share
máximo semanal en [0.070, 0.242] y `alpha` en [1.8, 65]. Cumple el criterio.

### Piloto

| Campo | Valor |
|---|---|
| Run key | `20260927T010530.820879Z-pilot-45c908d1-e06833de` |
| Commit | `a099a30f175c515a71356ae3c49068f74c2f2144` |
| SLURM | 53181, `COMPLETED`, 1 min 43 s, MaxRSS 0.74 GiB; pico del proceso 1.00 GiB |
| MLflow | `7e5d90f0748f4e0f87a9af3f7c87f23a`, borrado el 2026-09-27 por la política de §14.5 del plan; sus salidas siguen en Git |
| Muestreo | 2 cadenas, 250 de tune y 250 draws |
| Diagnósticos | R-hat máximo 1.03, ESS bulk mínimo 512, ESS tail mínimo 214, 0 divergencias, BFMI mínimo 0.85, profundidad máxima 5 sin topes |
| Normalización de medias | Error máximo 4.4e-16 |
| Posterior de `alpha`, solo `fit` | 2.47, HDI 94% [2.36, 2.58] |

Todos los gates del piloto se cumplen. El tiempo extrapolado al full es menor
que 15 min (103 s x 8) y la memoria se mantiene cerca de 1 a 2 GiB. No se
revisaron métricas de calibración ni de validación del piloto.

El piloto reveló que `az.summary(..., round_to=None)` redondeaba R-hat a dos
decimales y ESS a enteros. Antes del full se cambió a `round_to="none"` para
que el gate `R-hat <= 1.01` use el valor exacto. Esto no cambia ninguna
decisión del piloto: su R-hat real está entre 1.025 y 1.035, por debajo de
1.05.

### Full

| Campo | Valor |
|---|---|
| Run key | `20260927T011144.681956Z-full-45c908d1-f3a4936c` |
| Commit | `81be140a13a14a6ad6f4d220481b3c7fb9652f6c` |
| Comando | `sbatch --export=ALL,MODEL_CONFIG=configs/weekly_counts/nb_static_global_v3.yaml,RUN_MODE=full scripts/hpc/m9_weekly_count.slurm` |
| SLURM | 53184, `COMPLETED`, 5 min 41 s, MaxRSS 1.18 GiB; pico del proceso 1.44 GiB |
| MLflow | `c5dae8eeb3cf435292b351846f7a1653` |
| DVC | `artifacts/models/weekly_counts.dvc`, `7136b408b674cca7038f0bf21413725d.dir`, subido a DagsHub |
| Muestreo | 4 cadenas, 2000 de tune y 2000 draws, en CPU con NumPyro |
| Diagnósticos | R-hat máximo 1.003, ESS bulk mínimo 8613, ESS tail mínimo 4666, 0 divergencias, BFMI mínimo 0.94, profundidad máxima 5 sin topes |
| Normalización de medias | Error máximo 8.9e-16 |
| Posterior de `alpha` | 2.47, HDI 94% [2.35, 2.58] |

Métricas frente al baseline Poisson fijo. WAPE, MAE y sesgo usan la mediana
predictiva:

| Split | Modelo | WIS | WAPE | MAE | Sesgo de la mediana | Cobertura 80% | Cobertura 95% |
|---|---|---:|---:|---:|---:|---:|---:|
| `fit` | NB-V3 | 51.54 | 25.12% | 73.44 | -13.4% | 81.3% | 93.5% |
| `fit` | Poisson fijo | 56.73 | 23.33% | 68.20 | -0.05% | 17.1% | 26.8% |
| Calibración | NB-V3 | 92.57 | 34.69% | 155.13 | -13.5% | 77.3% | 94.0% |
| Calibración | Poisson fijo | 138.89 | 34.48% | 154.16 | -0.04% | 9.2% | 16.5% |

La media posterior no tiene sesgo agregado porque las medias suman el total
semanal. El sesgo proviene de la mediana: con una sola `alpha` cercana a 2.5 la
distribución es asimétrica y su mediana queda por debajo de la media.

Terciles de clusters definidos por su volumen en `fit`:

| Tercil | Clusters | Cobertura 80%, `fit` y calibración | Cobertura 95%, `fit` y calibración | WAPE de calibración, NB-V3 y Poisson |
|---|---:|---:|---:|---:|
| Pequeños | 14 | 71.6%; 67.3% | 90.7%; 89.3% | 72.8%; 76.9% |
| Medianos | 13 | 75.6%; 73.1% | 90.4%; 92.9% | 44.7%; 39.9% |
| Grandes | 13 | 97.4%; 92.3% | 99.6%; 100% | 30.1%; 30.4% |

Los clusters grandes concentran cerca de 80% de los conteos. Para ellos la
dispersión global es excesiva: sus intervalos son demasiado anchos y su mediana
queda baja. En los pequeños la cobertura 80% queda cerca o por debajo de 70%.

El runner calcula métricas de validación, pero no se abrieron.

## Decisión

- Estado automático: `rejected_no_practical_gain`; criterio fallido `wape`.
  Pasan todos los criterios de convergencia y cobertura, y el WIS de
  calibración mejora 33.3%, pero el WAPE es 34.69% frente a 34.48% del Poisson
  fijo. Los criterios no se modificaron después de ver los resultados.
- **NB-V3 no se acepta.** El baseline Poisson fijo sigue siendo la referencia
  de M9.
- La cobertura cambia sistemáticamente con el volumen del cluster, con
  intervalos demasiado anchos en los grandes y estrechos en los pequeños. Es la
  evidencia de dispersión heterogénea que el plan exige para considerar NB-V4.
- Como referencia histórica, NB-V2 obtuvo en calibración WIS 89.06 y WAPE
  31.34%, pero fue rechazado por ESS y cobertura.

## Siguiente acción permitida

- NB-V4, `nb_static_hierarchical_v4`, queda justificado por la evidencia
  anterior, pero su implementación requiere aprobación explícita y no se hizo
  en esta sesión.
- Siguen pendientes B1-R4, B1-R13, B2, el bootstrap semanal y el umbral de
  mejora práctica mínima.
