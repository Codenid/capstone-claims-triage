# Catálogo de modelos

Qué se probó, con qué regla se decidió y qué quedó vigente. Es un resumen:
las reglas registradas antes de cada corrida y el detalle de cada decisión están
en [`docs/registro/models_plan.md`](registro/models_plan.md); el mapa de los archivos de resultados
está en [`reports/modeling/README.md`](../reports/modeling/README.md).

Los datos son los reclamos públicos de la CFPB (EE. UU., narrativas en inglés).
Sirven para demostrar el método; las etiquetas no equivalen a las de un banco
peruano.

## Reglas comunes

| Periodo | Fechas | Uso |
|---|---|---|
| Ajuste (`fit`) | 2023-01-01 a 2024-09-30 | Entrenar; 1,067,194 reclamos elegibles |
| Calibración | 2024-10-01 a 2024-12-31 | Elegir modelos, umbrales y priors; 167,973 reclamos sin texto compartido |
| Validación | 2025-01-01 a 2025-06-30 | Consultada una sola vez el 2026-10-01 (§22); 564,813 sin texto compartido |

- **Vista "sin texto compartido"**: se excluyen las narrativas idénticas a
  alguna del periodo anterior, para que las plantillas no inflen las métricas.
- **Regla para reemplazar un modelo**: el retador mejora la métrica principal
  al menos 5% relativo y el intervalo bootstrap 95% de la diferencia, pareado
  por semanas (2,000 remuestreos, semilla 42), queda entero a su favor.
- 2025-H2 está bloqueado (contaminado por un experimento anterior) y 2026 es
  parcial. No queda otro periodo limpio: los resultados de validación posteriores
  al 2026-10-01 se reportan como **ya consultados**.

## Modelos vigentes

| Componente | Modelo vigente | Calibración | Validación (2025-H1) | Dónde |
|---|---|---|---|---|
| T1, motivo del reclamo (90 clases) | **TabPFN-3.5** en contexto con 20,000 filas de PCA 100 + producto | Macro-F1 0.271, top-3 95.5% | Macro-F1 0.270, top-3 95.9% (ya consultado) | `reports/modeling/foundation_t1/` |
| T2, alguna solución registrada | TF-IDF solo texto, logística calibrada; umbral 0.300 | AP 0.609 | AP 0.570 | `artifacts/models/representation_comparison/` |
| T3, compensación monetaria | TF-IDF solo texto; umbral 0.182 | AP 0.364 | AP 0.270 | ídem |
| T4, respuesta no oportuna | Sale del triaje; lo reemplaza una regla de plazo (15 o 45 días hábiles) | — | AP 0.078, peor que la regla | `configs/peru_holidays.yaml` |
| Reclamos parecidos | FAISS exacto sobre BGE (1,024 dim.) | vecino con similitud mediana 0.924 | 0.918 | `artifacts/models/semantic_space/` |
| Patrones semánticos | PCA 256 → UMAP 15 → k-means `k=40`; novedad si distancia > p99 | 0.88% novedosos en `fit` | 0.77% | `artifacts/models/weekly_patterns/` |
| Conteo semanal por patrón (M9) | **C-A**: binomial negativa jerárquica con memoria que decae (δ = 0.5) | WIS 30.30 | WIS 129.6 (ya consultado) | `reports/modeling/weekly_counts/nb_discounted_hierarchical_v1/` |
| Composición semanal (M10) | DM-R4: Dirichlet-multinomial con participaciones de 4 semanas, κ = 826 | log score −210.1 | −251.7 | `reports/modeling/weekly_composition/dirichlet_multinomial_rolling_4_v1/` |
| Mezcla rara semanal (M10, §27) | p-valor predictivo posterior del log score de DM-R4; aviso si p < 0.01 | 0 de 12 semanas | 7 de 25 (ya consultado; incluye la ráfaga del 13 de enero) | `reports/modeling/weekly_composition/mixture_signal/` |
| Alertas persistentes (M11) | CUSUM (k = 0.5, h = 3.37) sobre los excesos de C-A | 1.4 alertas al mes | 30 alertas en 25 semanas (ya consultado) | `reports/modeling/persistent_change_c_a/` |

TabPFN-3.5 tiene licencia no comercial: sirve para esta evaluación académica,
pero usarlo con el banco requiere una licencia de Prior Labs. Un experto debe
validar esa lectura de la licencia.

## T1–T4: clasificadores

Métrica principal en calibración sin texto compartido: Macro-F1 para T1,
precisión promedio (AP) para T2–T4. Mayor es mejor.

| Etapa | Candidato | T1 | T2 | T3 | T4 | Resultado |
|---|---|---:|---:|---:|---:|---|
| M3 | Frecuencia global | 0.006 | 0.391 | 0.022 | 0.006 | Referencia inferior |
| M3 | Regla por producto | 0.076 | 0.454 | 0.174 | 0.031 | Referencia |
| M4 | TF-IDF + producto, lineal calibrado | 0.214 | 0.612 | 0.370 | 0.055 | Supera a la regla en T1–T3 |
| M5 | BGE + producto, muestra de 120,000 filas | 0.237 | 0.567 | 0.258 | 0.062 | Solo T1 mejora frente a TF-IDF en la misma muestra |
| M5B | TF-IDF solo texto, todo `fit` | 0.141 | 0.609 | 0.364 | 0.047 | **Elegido para T2 y T3**: el producto no suma 5% |
| M5B | BGE solo texto, todo `fit` | 0.158 | 0.606 | 0.335 | 0.031 | Rechazado |
| M5B | BGE + producto, todo `fit` | 0.240 | 0.610 | 0.351 | 0.042 | **Elegido para T1** (+11.9% sobre TF-IDF + producto) |
| M5B | TF-IDF + producto (T4) | — | — | — | 0.055 | Elegido para T4, luego no confirmado |
| §22 | Confirmación en validación | 0.227 vs 0.068 | 0.570 vs 0.451 | 0.270 vs 0.127 | 0.078 vs 0.121 | T1–T3 confirmados; T4 rechazado |
| Bloque B | **TabPFN-3.5**, 20,000 filas de contexto, 8 estimadores | **0.271** | — | — | — | **Gana a BGE + producto** (+12.9%, IC [+0.027, +0.035]); en 2025-H1, 0.270 vs 0.227 |
| Ablación §26 | TabPFN-3.5 con 256 componentes PCA | 0.268 | — | — | — | Empata con TabPFN-100 (−1.0%, IC [−0.006, +0.001]): no reemplaza |
| Ablación §26 | TabPFN-3.5 con BGE 1,024 sin PCA | 0.192 | — | — | — | Pierde (−29%): cada estimador ve 768 de 1,025 columnas |
| Bloque B | Kumo Tabular (NVIDIA) | — | — | — | — | No corrió: sin memoria con 50,000 filas en GPU ni con 20,000 en RAM |

Regla de desempate de M5B: de la representación más simple a la más compleja,
cada una reemplaza a la anterior solo si cumple la regla común. Por eso T2 y T3
quedaron con TF-IDF solo texto aunque TF-IDF + producto tuviera una décima más.

## Patrones semánticos

| Etapa | Qué se probó | Resultado |
|---|---|---|
| M5/M8A | Embeddings BGE-large (1,024 dim., normalizados) para 1,067,194 filas de `fit` y todas las de calibración y validación | Insumo de todo lo semántico |
| M6 | PCA ajustado solo con 120,000 filas de `fit` | 256 componentes conservan 92.4% de la varianza (128: 82.9%; 512: 98.3%) |
| M6 | UMAP 2D | Solo para visualizar |
| M6 | FAISS `IndexFlatIP`, 120,000 filas de `fit` | Vecino más cercano: 82.7% mismo producto, 47.7% mismo motivo (calibración) |
| M7 | 12 configuraciones sobre 20,000 filas en UMAP 15: k-means (`k` = 20, 40, 80), HDBSCAN (6) y CURE (3) | **k-means `k=40`**: silhouette 0.138, estabilidad ARI 0.626, 99% de cobertura futura. HDBSCAN dejaba 84% como ruido; CURE juntaba casi todo en un grupo |
| Bloque A (M7S) | 9 espacios (PCA 128/256/512; UMAP 5/10/15/30; UMAP 15 con semillas 43 y 44) × 15 candidatos (los 12 de M7 + GMM `k` = 20, 40, 80) | **Ninguno supera a M7** por lift de vecinos (0.637). Sin UMAP no se pasa ninguna regla. Hallazgo: con otro espacio o semilla salen grupos distintos de calidad parecida (ARI 0.19–0.63); los 40 patrones no son categorías fijas |
| M8B | k-means `k=40` reajustado con las 1,067,194 filas; novedad si la distancia al centro supera el p99 de su patrón en `fit` | 40 patrones; 0.88% / 1.08% / 0.77% de reclamos novedosos por periodo; 5,120 semanas-patrón completas |

## M9: conteo semanal por patrón

WIS (*weighted interval score*: error de la mediana más penalización por
intervalos anchos o que no cubren) en las 12 semanas de calibración. Menor es
mejor. Cobertura: fracción de conteos dentro del intervalo 80% y 95%; se exige
70–90% y 88–99%.

| Candidato | Idea | WIS | Cobertura 80 / 95 | Resultado |
|---|---|---:|---|---|
| B1 Poisson fijo | Participación histórica fija | 138.9 | 9% / 16% | Referencia inicial; intervalos demasiado estrechos |
| B1-R4 | Poisson con la participación de las 4 semanas anteriores | 43.1 | 29% / 42% | **Referencia de M9** |
| B1-R13 | Ídem con 13 semanas | 64.8 | 21% / 32% | Peor que B1-R4 |
| NB-V1, NB-V2 | Binomial negativa con tendencia lineal por patrón | 89.1 (V2) | — | Rechazados; archivados |
| NB-V3 | Binomial negativa estática, una dispersión | 92.6 | 77% / 94% | Rechazado |
| NB-R4 | Participación de 4 semanas, una dispersión | 47.7 | 89% / 96% | Rechazado: no mejora B1-R4 |
| NB-R4-H v1, v2 | Dispersión por patrón, parametrizaciones no centrada y `ZeroSumNormal` | 32.5 (piloto) | — | Problemas de muestreo; reemplazadas por v3 |
| NB-R4-H v3 | Dispersión por patrón, centrada | 32.4 | 85% / 96% | Aceptado (−24.7% vs B1-R4) y confirmado en §22 |
| **C-A** | v3 con memoria que decae: cada semana pesa 0.5 veces la anterior | **30.3** | 85% / 97% | **Gana a v3** (−6.5%, IC [−3.36, −0.84]) |
| C-B | Paseo aleatorio t de Student sobre las log-participaciones | 32.0 | 88% / 98% | Rechazado: −1.4%, el IC cruza 0 |
| Chronos-2 | Fundacional de series, sin entrenamiento, total semanal como covariable | 36.0 | 77% / 96% | Rechazado |
| TimesFM 3.0 | Ídem; solo deciles, comparado con intervalos 50% y 80% | 40.7 vs 41.4 | 76% / — | Rechazado |
| NB-V4, DM-V2 | Variantes estáticas y con tendencia | — | — | Descartados sin correr |

Para C-A, la memoria (δ) y un recorte de semanas atípicas se eligieron solo en
`fit`: ganó δ = 0.5 sin recorte, así que C-A no corrige el eco de las ráfagas
(ver límites).

## M10: composición semanal

Log score conjunto en calibración: logaritmo de la probabilidad que el modelo
asigna a la composición observada de cada semana, sumado. Mayor es mejor.

| Candidato | Idea | Log score | Resultado |
|---|---|---:|---|
| B2 estática | Multinomial con participaciones fijas | −1,770 | Referencia inferior |
| B2-R4 | Multinomial con participaciones de 4 semanas | −461 | **Referencia** |
| DM-V1 | Dirichlet-multinomial con participaciones fijas | −246 | Rechazado: falló una comprobación predictiva |
| **DM-R4** | Dirichlet-multinomial con participaciones de 4 semanas, κ = 826 | **−210** | **Aceptado** (+251 vs B2-R4, IC [205, 296]) y confirmado en §22 |

κ es la concentración: cuanto mayor, más se parece la composición de cada
semana a la esperada. 826 implica mucha más variación que una multinomial.

## M11: alertas de aumento persistente

Cada semana se calcula el exceso de cada patrón como el *score* normal de la
mid-PIT (posición del conteo observado dentro de la predictiva de M9). CUSUM lo
acumula; avisa cuando supera h y reinicia. La regla alternativa avisa si el
exceso de una sola semana supera z*. Umbrales fijados por simulación para 1
falsa alarma al mes en total: h = 3.37, z* = 2.53.

| Línea base | Detecta crecimiento 10% / 20% semanal en 8 semanas | Avisa sin aumento | Alertas reales al mes (`fit`, calibración) | 2025-H1 (ya consultado) |
|---|---|---:|---|---|
| NB-R4-H v3 | CUSUM 50.8% / 79.7%; regla semanal 36.1% / 58.4% | 11.5% | 3.9 y 2.2 | 42 alertas en 25 semanas, 25 en el episodio del 27-ene al 17-feb |
| **C-A (vigente)** | CUSUM 42.3% / 70.9%; regla semanal 32.0% / 54.0% | 6.5% | 2.2 y 1.4 | 30 alertas, 16 en el episodio |

CUSUM se eligió en ambos casos porque detecta al menos 5 puntos más que la
regla semanal en los dos crecimientos. Con C-A hay menos alertas y menos falsas,
pero también menos detección: su memoria corta alcanza antes a un patrón que
crece.

## Descartados sin correr

| Modelo | Por qué |
|---|---|
| TabPFN-TS | Exige pandas < 3 y trae telemetría activada |
| Kumo Tabular (NVIDIA) | Sin memoria en Khipu; pendiente probarlo con más memoria, quizá vía NVIDIA NIM |
| Kumo Relational (KumoRFM-2) | Requiere armar tablas relacionales; pendiente junto al anterior |
| Mezcla de von Mises–Fisher, mezclas t, procesos de Dirichlet | El bloque A mostró que el límite es el espacio, no la familia de mezclas |
| Fine-tuning y ensamble de TabPFN | Mejoras posibles; no se probaron |

## Límites conocidos

- **Eco de ráfagas.** La semana del 2025-01-13 un patrón recibió 58% de los
  reclamos. Como M9 usa las semanas anteriores, en las siguientes esperó de
  más para ese patrón y de menos para los otros 39, y M11 avisó por eso. C-A lo
  atenúa (16 alertas en el episodio contra 25) pero no lo corrige.
- **Patrones.** Son una partición útil, no categorías del banco: otro espacio
  u otra semilla de UMAP da grupos distintos de calidad parecida.
- **Validación consultada.** Todo lo decidido después del 2026-10-01 se eligió
  con `fit` + calibración; sus cifras de 2025-H1 son informativas, no evidencia
  limpia.
- **Licencias.** TabPFN-3.5 y TimesFM 3.0 son de uso no comercial.
- **Etiquetas.** T2–T4 son aproximaciones de la CFPB, no resultados bancarios;
  ninguna detecta fraude.

## Publicado en MLflow

Un run por decisión en DagsHub (`claims-triage-modeling`, 51 de 100 runs).
Los identificadores de la ronda §25 están en
[`reports/modeling/README.md`](../reports/modeling/README.md).
