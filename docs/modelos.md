# Propuesta de modelos para el triaje

<!-- markdownlint-disable MD013 MD060 -->

Este documento registra la propuesta de modelado. No significa que todos los
modelos serán implementados. Cada opción deberá superar una comparación simple
y demostrar valor con datos que no fueron usados para entrenarla.

## Alcance real

El sistema busca **apoyar al agente humano**, no reemplazarlo. Con los datos
CFPB disponibles podemos construir un prototipo para:

- sugerir el motivo de un reclamo;
- estimar resultados históricos registrados por CFPB;
- recuperar reclamos con significado parecido;
- descubrir grupos de narrativas relacionadas;
- alertar cuando un patrón aumenta de forma inusual.

No podemos afirmar que detectamos fraude confirmado, pérdidas económicas, el
tiempo real de resolución de un banco ni su plazo interno de 15 días. Esas
etiquetas no existen en el archivo.

## Diccionario de resultados

| Nombre | Resultado propuesto | Límite |
|---|---|---|
| **T1** | Motivo CFPB (`Issue`) más probable | No equivale al equipo interno de un banco |
| **T2** | Probabilidad de alguna solución o compensación registrada | Puede usarse como señal auxiliar |
| **T3** | Probabilidad de compensación monetaria registrada | Es una aproximación a reembolso, no un reembolso bancario confirmado |
| **T4** | Probabilidad de respuesta no oportuna según CFPB | No mide resolución ni el plazo bancario de 15 días |
| **A1** | Alerta de patrón semántico emergente | Un patrón atípico no implica fraude |

Una **variable de entrada** es información disponible cuando llega el reclamo.
Un **objetivo** es el resultado histórico que intentamos predecir.

`Issue`, `Company response to consumer`, `Company public response` y
`Timely response?` no pueden entrar como variables del modelo que intenta
predecirlas, porque revelarían el resultado.

## Valor esperado

Los centros de atención suelen usar reglas para clasificar reclamos conocidos.
La propuesta añade una segunda capa:

1. Sugiere categorías para reducir clasificación manual.
2. Muestra probabilidades y reclamos históricos parecidos.
3. Identifica casos que no encajan bien en patrones conocidos.
4. Detecta aumentos de grupos semánticos relacionados.
5. Entrega evidencia al agente para que tome la decisión final.

El principal valor adicional frente a reglas estáticas es encontrar reclamos
parecidos aunque usen palabras diferentes y advertir cuándo aumentan juntos.

## Flujo propuesto

```mermaid
flowchart TD
    A[Nuevo reclamo] --> B[Narrativa, producto, empresa y fecha]

    B --> C0[TF-IDF: comparación sencilla]
    B --> C1[BGE: representación semántica]

    C0 --> D[T1: motivo probable]
    C1 --> D
    C0 --> E[T2 y T3: solución y compensación monetaria]
    C1 --> E
    C0 --> F[T4: respuesta no oportuna CFPB]
    C1 --> F

    C1 --> G[FAISS: buscar reclamos parecidos]
    G --> H{¿Encaja en un patrón conocido?}
    H -->|Sí| I[Asignar grupo semántico]
    H -->|No| J[Reservorio de patrones nuevos]

    I --> K[Conteo semanal por patrón]
    J --> K

    K --> L[Negative Binomial: volumen esperado]
    K --> M[Dirichlet-Multinomial: composición esperada]
    L --> N[CUSUM por patrón]

    N --> O{¿Cambio persistente?}
    O -->|No| P[Triaje normal]
    O -->|Sí| Q[Alerta de patrón emergente]
    M -->|contexto: participación esperada| Q

    D --> R[Recomendación de triaje]
    E --> R
    F --> R
    P --> R
    Q --> R

    R --> S[Agente humano revisa y decide]
```

En el archivo actual, `Submitted via` siempre vale `Web`. Por tanto, ese campo
no aporta información al modelo. Podría ser útil en una futura aplicación si
el banco dispone de varios canales reales.

## Representación del texto

### TF-IDF: comparación inicial

**TF-IDF** representa cada narrativa mediante la importancia de sus palabras.
Es rápido, económico y suele funcionar bien cuando ciertas palabras están muy
relacionadas con una categoría.

Se evaluó con un clasificador logístico lineal entrenado por descenso de
gradiente. En 2025-H1 sin texto compartido mejoró la regla por producto para
T1–T3, pero no para T4. Un modelo posterior solo se justifica si mejora estas
referencias por objetivo.

### BGE: representación semántica

Un **embedding** es una lista de números que intenta representar el significado
del texto. Narrativas con significado parecido deberían quedar cerca aunque no
usen exactamente las mismas palabras.

Modelo candidato: `BAAI/bge-large-en-v1.5`.

La A100 disponible se utilizó para generar embeddings sobre una muestra
temporal. M5 comparó BGE y TF-IDF sobre las mismas filas y periodos.

En la validación sin texto compartido, BGE mejoró Macro-F1 de 0.1968 a 0.2367
para T1. En T2 prácticamente empató con TF-IDF y en T3–T4 quedó por debajo. La
decisión es usar BGE como candidato para T1 y para similitud semántica, no como
reemplazo general de TF-IDF.

Las narrativas largas requieren decidir si se recortan o se dividen en
fragmentos. E5 encontró que 8.43% supera 400 palabras.

## Modelos por componente

| Componente | Referencia | Decisión actual | Razón |
|---|---|---|---|
| T1 — motivo | TF-IDF + producto | BGE + producto, todo `fit` | Superó a TF-IDF + producto en 11.9% (M5B) |
| T2 — alguna solución | Regla por producto | TF-IDF solo texto | Ni el producto ni BGE mejoraron 5% (M5B) |
| T3 — compensación monetaria | Regla por producto | TF-IDF solo texto | El producto sumó 1.6% y BGE fue peor (M5B) |
| T4 — no oportuna CFPB | Regla por producto | Sale del triaje | Sin señal estable; en M12 la reemplaza una alerta de 15 días hábiles |
| Reducción y visualización | Embedding BGE | PCA y UMAP 2D | Reducir ruido y revisar visualmente el espacio semántico |
| Vecinos similares | No aplica | FAISS | Búsqueda rápida entre millones de embeddings |
| Grupos semánticos | HDBSCAN y CURE | MiniBatchKMeans con `k=40` | Único método que pasó separación, estabilidad, tamaño y asignación futura |
| Descripción del grupo | Palabras frecuentes | c-TF-IDF | Explicar cada grupo con términos representativos |
| Volumen temporal | Media histórica | Negative Binomial con PyMC | Estimar el conteo esperado y su incertidumbre |
| Composición temporal | Proporción histórica | Dirichlet-Multinomial con PyMC | Detectar cambios relativos entre todos los grupos |
| Cambio persistente | Regla por exceso semanal | CUSUM sobre los excesos de M9 | Detectó 14.6 y 21.3 puntos más de crecimientos artificiales (M11) |

Un modelo **calibrado** produce probabilidades interpretables. Por ejemplo, de
100 casos con probabilidad cercana a 20%, aproximadamente 20 deberían resultar
positivos.

### Decisión después de M5

| Objetivo | Modelo conservado | Razón |
|---|---|---|
| T1 | BGE + producto | Mejoró Macro-F1 en 0.0399 sobre la misma muestra |
| T2 | TF-IDF + producto | BGE mejoró menos de 0.001 y cuesta más |
| T3 | TF-IDF + producto | Superó a BGE en precisión promedio |
| T4 | Regla por producto | Tanto TF-IDF como BGE quedaron por debajo de M3 |

La decisión se toma por objetivo. No se fuerza un único modelo para todo el
triaje. BGE continúa hacia M6 porque los embeddings también permiten recuperar
vecinos y formar grupos; esa utilidad no depende de ganar T2–T4.

El experimento usó 120,000 filas de ajuste, 40,000 de calibración y 80,000 de
validación. Los clasificadores convergieron y el artefacto está en DVC con hash
`009e3b35e25d9df095cf753e0a05f041.dir`.

### Decisión después de M5B

La decisión anterior miró la validación. M5B la reemplaza usando solo
calibración: entrenó cada representación con las 1,067,194 filas de `fit` y
aplicó la regla de `models_plan.md` §21, aprobada antes de entrenar. Una
representación más compleja se elige solo si mejora la métrica al menos 5% y el
intervalo bootstrap por semanas queda entero sobre 0.

| Objetivo | Modelo elegido | Métrica en calibración | Razón |
|---|---|---:|---|
| T1 | BGE + producto | Macro-F1 0.2397 | 11.9% sobre TF-IDF + producto |
| T2 | TF-IDF solo texto | AP 0.6094 | El producto sumó 0.4% y BGE no mejoró |
| T3 | TF-IDF solo texto | AP 0.3640 | El producto sumó 1.6% y BGE fue peor |
| T4 | TF-IDF + producto | AP 0.0551 | 16.8% sobre TF-IDF solo texto |

En T4, M4 ya había evaluado ese modelo en validación, donde quedó por debajo de
la regla por producto. Por decisión del usuario, se mantiene la elección de la
regla, pero su validación no cuenta como prueba independiente; en la
confirmación final se reporta junto a la regla por producto. El detalle está en
`reports/modeling/representation_decision.md`.

La confirmación final en validación (2025-H1) confirmó T1, T2 y T3 frente a la
regla por producto. T4 no se confirmó (precisión promedio 0.0781 frente a
0.1208). El detalle está en `reports/modeling/final_confirmation.md`.

Después, por decisión del usuario, T4 salió del triaje. Responder a tiempo
depende del proceso del banco y no del texto del reclamo, y en el banco el
plazo se conoce desde el registro: 15 días hábiles, o 45 con extensión, sin
contar feriados nacionales. M12 mostrará los días hábiles que faltan en lugar
de una predicción (`models_plan.md` §23).

## Descubrimiento de patrones

### FAISS y vecinos cercanos

FAISS permite buscar rápidamente qué reclamos históricos tienen embeddings
más cercanos al reclamo nuevo.

Esto aporta dos señales:

- **similitud:** el reclamo se parece a un patrón conocido;
- **novedad:** está lejos de los reclamos históricos comparables.

También ofrece una explicación útil al agente: puede mostrar ejemplos
históricos similares sin depender únicamente de una probabilidad.

### PCA y UMAP

M6 reutilizó los embeddings generados en M5 y no volvió a ejecutar BGE. PCA se
ajustó con las 120,000 filas de ajuste y transformó después las 40,000 de
calibración y 80,000 de validación.

**PCA** reduce las dimensiones de los embeddings y elimina parte del ruido. Se
ajustará únicamente con el periodo de ajuste y luego transformará los periodos
posteriores sin volver a aprender.

**UMAP** creará una visualización bidimensional sobre una muestra fija. El
resultado permite observar solapamientos, casos aislados y grupos dominados por
plantillas, pero no demuestra por sí solo que un clustering sea correcto. Las
distancias del gráfico 2D están distorsionadas y no se usarán como único insumo
del clustering.

Si HDBSCAN necesita una reducción adicional, se comparará:

- clustering directamente sobre PCA;
- clustering sobre PCA más UMAP de varias dimensiones.

El UMAP de dos dimensiones permanecerá reservado para visualización.

### Resultado de M6

PCA redujo los embeddings de 1,024 a 256 dimensiones y conservó 92.42% de la
variación. Con 128 componentes conservaba 82.86%. M7 comenzará con las 256
dimensiones para comparar todos los métodos sobre la misma representación.

La visualización UMAP mostró amplia superposición entre ajuste, calibración y
validación. También mostró zonas relacionadas con productos y casos aislados,
pero no se usará como prueba de que existe un cluster.

FAISS indexó las 120,000 filas de ajuste y recuperó vecinos para 1,000 consultas
sin texto compartido de calibración y 1,000 de validación. La similitud mediana
del vecino más cercano fue 0.9245 y 0.9177, respectivamente. En validación, el
77.1% compartió producto y el 37.4% compartió T1.

Esto justifica conservar FAISS para mostrar evidencia y estudiar novedad, pero
no usar el vecino más cercano como sustituto del clasificador. El artefacto está
en DVC con hash `08662a82971a71666b06c9ccf8120d7f.dir`.

### Comparación de grupos semánticos

Un **grupo semántico** reúne narrativas con significado parecido. Se evaluarán
tres alternativas sobre las mismas filas:

- **MiniBatchKMeans con k-means++:** referencia escalable que exige un número de
  grupos y asigna todos los reclamos;
- **HDBSCAN:** permite grupos con densidades distintas y puede dejar casos sin
  asignar como ruido;
- **CURE:** representa un grupo con varios puntos y puede capturar formas
  irregulares, pero primero debe demostrar que puede escalar y asignar reclamos
  futuros.

La comparación incluirá:

1. gráfico de silhouette y promedio sobre una muestra fija;
2. estabilidad al cambiar la muestra o semilla;
3. distribución de tamaños y porcentaje de ruido;
4. ejemplos cercanos y términos c-TF-IDF por grupo;
5. presencia de plantillas exactas o casi exactas;
6. tiempo, memoria y facilidad para asignar nuevos reclamos.

El **silhouette** compara qué tan cerca está un caso de su propio grupo frente a
los otros grupos. Valores cercanos a 1 indican separación, cerca de 0 indican
solapamiento y valores negativos sugieren una asignación dudosa. No será el
único criterio porque suele favorecer grupos compactos y puede penalizar las
formas irregulares de HDBSCAN o CURE.

### Resultado de M7

Se compararon 12 configuraciones sobre las mismas 20,000 filas de ajuste y un
UMAP común de 15 dimensiones.

MiniBatchKMeans con `k=40` fue el único método aceptado. Obtuvo silhouette
0.1376, estabilidad ARI 0.6260, ningún caso sin grupo y 98.99% de cobertura al
aplicar su regla de novedad en calibración. El grupo más grande representa
13.89% de la muestra.

HDBSCAN obtuvo silhouette 0.5455, pero formó solo 2 grupos y marcó 48.91% como
ruido. El valor alto describe únicamente la parte que decidió agrupar; no
compensa la baja cobertura.

CURE obtuvo silhouette 0.4093, pero colocó 99.21% de los reclamos en un solo
grupo. Su estabilidad ARI fue 0.4307. Por eso también fue rechazado.

La elección de k-means es operacional, no una afirmación de que existan 40
categorías naturales. Su silhouette es moderado y 28.96% de los casos evaluados
tiene silhouette negativo. M8 generó los embeddings faltantes, ajustó `k=40`
con todos los reclamos elegibles del periodo de ajuste y conservó una señal
separada de novedad.

El artefacto M7 está en DVC con hash
`425408aa84d0e2a44b8c5becd467e724.dir`.

### Resultado de M8

M8A completó 1,996,978 embeddings BGE: reutilizó 240,000 de M5 y generó
1,756,978. El artefacto está en DVC con hash
`3ff36a299da1f2e19a00a4bce49f18ad.dir`.

M8B ajustó MiniBatchKMeans con `k=40` solo con las 1,067,194 filas de ajuste.
Después asignó 234,600 filas de calibración y 695,184 de validación sin volver a
aprender. La tasa de novedad fue 0.88% en ajuste, 1.08% en calibración y 0.77%
en validación.

Ningún grupo quedó vacío. El grupo mayor representa 16.56% del ajuste. Se
crearon 5,360 filas semanales, de las cuales 5,120 pertenecen a semanas
completas y 240 a seis semanas parciales. Las combinaciones sin reclamos se
completaron con cero.

El artefacto de patrones está en DVC con hash
`87be2a97daa3b169bbd65850a64c89a6.dir`. Estos grupos son una partición
operativa para medir cambios; no son categorías naturales confirmadas.

c-TF-IDF describe cada grupo mediante las palabras que lo distinguen de los
demás. No determina fraude ni reemplaza la revisión humana.

E6 mostró que muchos textos son plantillas exactas o casi exactas. Antes de
generar alertas, se comprobará si esas plantillas dominan un grupo. Se
reportarán tanto el conteo de reclamos como el número de textos normalizados
diferentes.

## Modelos temporales

Los modelos temporales se aplican **después** de crear grupos semánticos. No
procesan directamente millones de embeddings: reciben conteos por grupo y
semana.

Su implementación común está en `src/models/weekly_counts/`:

| Módulo | Responsabilidad |
|---|---|
| `models/*.py` | Definiciones PyMC versionadas e independientes del muestreo |
| `registry.py` | Registro explícito de los modelos candidatos |
| `contracts.py` | Columnas, splits, cuantiles e intervalos compartidos |
| `data.py` | Validación y preparación del panel semana-cluster |
| `negative_binomial.py` | Base matemática para parámetros con suma cero |
| `fixed_poisson_reference.py` | Referencia analítica histórica del backtest; no es un candidato |
| `sampling.py` | Muestreo común con PyMC y NumPyro y predicciones posteriores |
| `prediction.py` | Predicciones y cuantiles con un formato común |
| `metrics.py` | WIS, WAPE, cobertura, sesgo y aceptación |
| `diagnostics.py` | R-hat, ESS, divergencias, BFMI y profundidad |
| `reporting.py` | Gráficos, reportes y descriptor MLflow |
| `run.py` | Orquestación reproducible del experimento |

Cada candidato tiene un módulo PyMC y una configuración YAML con el mismo ID.
`configs/weekly_counts/releases.json` congela sus hashes, procedencia y run
histórico. Un cambio matemático o de configuración requiere un ID nuevo en vez
de modificar una versión publicada. Cada ejecución usa una carpeta nueva y
guarda un snapshot del código, la configuración y el lock de dependencias.

El entry point canónico es `python -m src.models.weekly_counts.run`; cada modelo
se elige con `--config configs/weekly_counts/<modelo>.yaml`. Multinomial y
Dirichlet-Multinomial se añadirán a este paquete después de comparar las
variantes estáticas de M9.

`--run-mode` indica el nivel de ejecución: `prior` solo revisa el prior
predictive con `fit`, `pilot` hace un muestreo corto y puramente técnico, y
`full` usa la configuración congelada. MLflow registra `run_mode`,
`candidate_role` y `candidate_status`: `pilot_only`, `accepted`,
`rejected_convergence`, `rejected_predictive` o
`rejected_no_practical_gain`, junto con los criterios fallidos en
`rejection_reason`.

### Negative Binomial

La distribución **Negative Binomial** modela cuántos reclamos esperamos para un
grupo. Permite más variación que una distribución Poisson simple, algo común en
reclamos con picos y campañas.

Pregunta principal:

> ¿Este patrón tiene más reclamos de los esperados en términos absolutos,
> considerando el volumen general?

PyMC puede estimar el valor esperado, diferencias entre grupos y la
incertidumbre de la estimación.

M9 usó una media normalizada:

```text
eta[c,t] = log_rate[c] + annual_trend[c] * time_years[t]
share[c,t] = softmax(eta[:,t])[c]
mu[c,t] = weekly_total[t] * share[c,t]
count[c,t] ~ NegativeBinomial(mu[c,t], alpha[c])
```

`softmax` transforma los valores de los 40 clusters en participaciones que suman
uno. Por ello, sus medias esperadas suman exactamente el total semanal. Los draws
Negative Binomial continúan siendo independientes y no están obligados a sumar el
total; M10 evaluará una distribución conjunta para la composición.

### Resultado de M9

Se probaron dos versiones. V1 se reconstruyó desde la especificación conservada
en MLflow porque el run histórico no guardó su código fuente exacto. La primera
permitió que las medias de los clusters sumaran más que el total y también
presentó convergencia insuficiente. V2 tiene una implementación verificada como
equivalente a su run histórico y corrigió la media con `softmax` y contrastes de
suma cero.

V2 no tuvo divergencias, obtuvo R-hat máximo 1.01 y superó al baseline Poisson en
calibración: WIS 89.06 frente a 138.89 y WAPE 31.34% frente a 34.48%. Sin
embargo, su ESS bulk mínimo fue 263, menor al requisito de 400, y su cobertura
95% fue 87.71%, menor al requisito de 88%.

La validación no se utilizó para seleccionar el modelo. En ese periodo V2 mejoró
1.99% el WIS, pero empeoró el WAPE de 53.08% a 65.14%. Por tanto, M9 está
completado como evaluación, pero el modelo no se acepta y el baseline sigue
siendo la referencia. Ambos intentos permanecen en MLflow; el posterior V2 está
en DVC con hash `ada511fc7ac612613faf9f02133fc2a6.dir`.

### Continuación de M9: B1 PyMC y NB-V3

B1 PyMC (`poisson_static_pymc_v1`) se ejecutó en Khipu solo como control del
pipeline. Reprodujo el baseline analítico: WIS de calibración 138.75 frente a
138.89 y WAPE 34.48% en ambos. Su cobertura 95% fue 16%, por lo que Poisson
queda como referencia y no como modelo. El run está en MLflow con ID
`0ddb96b9f9c645bdb8e8a687cedf09cc`; ver
`reports/modeling/weekly_counts/poisson_static_pymc_v1/decision.md`.

NB-V3 (`nb_static_global_v3`) conserva las participaciones estáticas de B1 y
cambia solo la verosimilitud: Negative Binomial con una dispersión global. El
prior de participaciones es uniforme; se eligió solo con `fit` y prior
predictive, antes de abrir calibración.

NB-V3 convergió sin divergencias (R-hat máximo 1.003 y ESS mínimo 4,666) y fue
el primer candidato de M9 con coberturas dentro del rango: 77.3% y 94.0% en
calibración. Además redujo el WIS de 138.89 a 92.57. Sin embargo, su WAPE fue
34.69% frente a 34.48% del baseline, por lo que no se acepta
(`rejected_no_practical_gain`). La cobertura cambia con el tamaño del cluster:
los clusters grandes tienen intervalos demasiado anchos y los pequeños,
demasiado estrechos. Esa es la evidencia que el plan exige antes de considerar
una dispersión por cluster (NB-V4). La validación no se abrió. El detalle está
en `reports/modeling/weekly_counts/nb_static_global_v3/decision.md`.

Los baselines móviles B1-R4 y B1-R13 usan las participaciones de las 4 o 13
semanas anteriores ya observadas. En calibración, B1-R4 obtuvo WIS 43.11 y WAPE
12.47%, muy por debajo de los modelos con participaciones estáticas (NB-V3:
92.57 y 34.69%). Sus intervalos Poisson son demasiado estrechos (cobertura 95%
de 42.1%), pero su predicción puntual muestra que la composición entre
clusters cambia de una semana a otra. Desde el 2026-09-27, un candidato nuevo
debe reducir al menos 5% el WIS del mejor baseline, hoy B1-R4, con un
bootstrap semanal que lo confirme.

NB-R4 (`nb_rolling_4_global_v1`) usa esas mismas participaciones de las 4
semanas anteriores y estima con PyMC una sola dispersión. Convergió y tuvo
cobertura dentro de rango (89.2% y 95.6%), pero su WIS de calibración fue 47.72
frente a 43.06 de B1-R4, por lo que no se acepta. Supera a B1-R4 en clusters
pequeños y medianos; en los grandes, la dispersión global produce intervalos
demasiado anchos. El detalle está en
`reports/modeling/weekly_counts/nb_rolling_4_global_v1/decision.md`.

NB-R4-H v3 (`nb_rolling_4_hierarchical_v3`) mantiene esa media y estima una
dispersión por cluster, centrada y con suma cero. Es el candidato elegido de
M9: en calibración obtuvo WIS 32.42 frente a 43.06 de B1-R4 (24.7% mejor, con
un intervalo bootstrap entero por debajo de 0), sin empeorar de forma
significativa el WAPE (12.66% frente a 12.46%), con coberturas de 85.4% y
95.6% y mejoras en los tres terciles de volumen. La validación sigue cerrada
hasta la confirmación final. El detalle está en
`reports/modeling/weekly_counts/nb_rolling_4_hierarchical_v3/decision.md`.

### Dirichlet-Multinomial

La **Dirichlet-Multinomial** modela cómo se reparte el total semanal entre los
diferentes grupos.

Pregunta principal:

> ¿Cambió la proporción que representa este patrón dentro de todos los
> reclamos?

No crea los grupos. Recibe grupos ya definidos mediante embeddings y analiza
su composición conjunta.

### Resultado de M10

M10 comparó, en calibración, dos Multinomiales sin dispersión adicional (B2 con
la mezcla fija y B2-R4 con las 4 semanas anteriores) y dos Dirichlet-Multinomial
con una concentración `kappa`. La regla de `models_plan.md` §12.3 exige mejorar
el log score conjunto del mejor baseline, con un bootstrap semanal entero sobre
0, y una cobertura marginal en rango.

| Modelo | Log score por semana | Cobertura 80% y 95% | Decisión |
|---|---:|---:|---|
| B2 | −1770.2 | 9.4% y 16.7% | Baseline |
| B2-R4 | −460.9 | 28.5% y 41.3% | Mejor baseline |
| DM-V1, mezcla fija | −246.0 | 66.3% y 87.1% | Rechazado por cobertura |
| **DM-R4, 4 semanas anteriores** | **−210.1** | **89.2% y 95.4%** | **Aceptado** |

DM-R4 mejora a B2-R4 en 250.8 por semana (IC bootstrap 95% [205.1, 295.8]),
con `kappa` = 826. La cobertura es estable en los grupos pequeños, medianos y
grandes, así que una sola concentración alcanza y no hizo falta la versión
logística-normal. El detalle está en
`reports/modeling/weekly_composition/<modelo>/decision.md`.

### Por qué se usan juntas

| Modelo | Pregunta |
|---|---|
| Negative Binomial | ¿El conteo del patrón es mayor que el esperado? |
| Dirichlet-Multinomial | ¿El patrón ocupa una proporción inusual del total? |

Ejemplo:

- Históricamente llegan 10,000 reclamos semanales.
- Un patrón representa 0.5%, aproximadamente 50 reclamos.
- Esta semana aparecen 120, es decir, 1.2% del total.

Negative Binomial detectaría el exceso de 50 a 120. Dirichlet-Multinomial
confirmaría que la participación aumentó de 0.5% a 1.2%.

Si el volumen total se duplicara a 20,000 y el patrón subiera a 100, su
participación seguiría siendo 0.5%. Al considerar el volumen total, ninguno de
los dos modelos debería tratar ese aumento proporcional como una alerta por sí
solo.

### CUSUM y detección de cambio

**CUSUM** acumula desviaciones pequeñas respecto de lo esperado. Puede detectar
un aumento persistente que no parece extremo en una sola semana.

Otra opción es **Bayesian Online Change-Point Detection (BOCPD)**, que estima la
probabilidad de que haya comenzado un comportamiento nuevo. Se comparará solo
después de construir una referencia temporal sencilla.

### Resultado de M11

M11 aplicó un CUSUM por patrón al exceso de cada semana frente a NB-R4-H v3, y
lo comparó con la regla por exceso semanal con el mismo presupuesto: 1 falsa
alerta al mes en total si M9 estuviera bien calibrado (`models_plan.md` §24).
Con aumentos artificiales en `fit` + calibración, CUSUM detectó dentro de 8
semanas el 50.8% de los crecimientos de 10% semanal y el 79.7% de los de 20%,
frente a 36.1% y 58.4% de la regla semanal. M12 usará CUSUM.

Como lo esperado sale de las 4 semanas anteriores, un salto pequeño que luego se
estabiliza se vuelve lo normal en unas 4 semanas; M11 busca sobre todo
crecimientos que continúan. En los datos reales avisa más que el presupuesto,
unas 2 a 4 veces al mes, porque hay más semanas extremas y rachas que las que
supone M9. En 2025-H1 dio 42 alertas en 25 semanas, el 60% en un episodio de
febrero de 2025 en patrones de reportes de crédito. BOCPD no se probó, porque
CUSUM ya superó a la referencia. El detalle está en
`reports/modeling/persistent_change/decision.md`.

M10 no genera alertas. Para un patrón, DM-R4 espera lo mismo que M9
($N_t r_{c,t}$) y solo cambia la varianza, así que un CUSUM con M10 sería casi
el mismo. Su aporte es la vista conjunta: en M12, cada alerta mostrará la
participación observada frente a la esperada por DM-R4, con su intervalo de 95%
(`models_plan.md` §23).

## Cómo una alerta cambia el triaje

| Situación | Acción sugerida |
|---|---|
| Patrón conocido y estable | Triaje normal |
| Reclamo aislado y novedoso | Revisión manual; no elevar automáticamente |
| Patrón conocido con crecimiento anormal | Agrupar casos y alertar a un equipo especializado |
| Grupo dominado por una plantilla habitual | Evitar una alerta automática hasta validar su importancia |
| Patrón nuevo que continúa creciendo | Priorizar investigación y considerar una nueva regla |

Una alerta no afirma fraude. Informa al agente que varios reclamos pueden estar
relacionados y que conviene revisarlos juntos.

## Orden de experimentación

1. Terminar EDA, transformaciones y divisiones temporales.
2. Entrenar TF-IDF + regresión logística para T1–T4.
3. Comparar BGE usando las mismas filas y métricas.
4. Reutilizar los embeddings de M5 y validar vecinos FAISS.
5. Ajustar PCA en el periodo de ajuste y crear una visualización UMAP.
6. Comparar MiniBatchKMeans++, HDBSCAN y CURE con la misma muestra.
7. Elegir y congelar grupos usando silhouette, estabilidad, coherencia y costo.
8. Generar los embeddings faltantes, asignar el corpus elegible y crear conteos
   semanales completos.
9. Probar Negative Binomial como modelo temporal principal.
10. Probar Dirichlet-Multinomial como comprobación de composición.
11. Calibrar CUSUM para detectar cambios persistentes.
12. Integrar las señales en una recomendación revisada por una persona.

## Evaluación

Las divisiones respetarán el tiempo:

- ajuste: 2023-01-01 a 2024-09-30;
- calibración: 2024-10-01 a 2024-12-31;
- validación temporal: 2025-01-01 a 2025-06-30.

Se reportará una vista completa y otra que excluye textos ya vistos en periodos
usados para aprender. La vista sin texto compartido será la principal para
elegir modelos.

- **T1:** Macro-F1, top-3 y F1 por motivo.
- **T2–T4:** average precision, precisión, cobertura y Brier score.
- **Clustering:** silhouette sobre una muestra fija, estabilidad, tamaños,
  porcentaje de ruido, coherencia semántica, tiempo y memoria.
- **Patrones temporales:** revisión humana, tiempo hasta detectar un crecimiento
  y cantidad de falsas alertas por semana.

Para T2–T4, el umbral se elegirá maximizando F1 en la calibración sin texto
compartido. Luego permanecerá fijo durante la validación.

A partir de esta nueva rama, ningún modelo será elegido usando 2025-H2 o 2026.
Sin embargo, un experimento anterior ya consultó una muestra de 25,000 casos de
2025-H2. Esos IDs y todos los registros que compartan sus grupos de texto
deberán excluirse de una evaluación final intacta. El tamaño elegible se
calculará después de esa exclusión. Si la lista no puede recuperarse, 2025-H2
se declarará contaminado y no se presentará como evaluación final intacta.

2026 seguirá separado porque su cobertura es parcial. Como la fecha de
extracción no está documentada, T2–T4 no se evaluarán allí hasta definir cuánto
tiempo debe pasar para considerar que la respuesta histórica ya maduró. T1 y la
disponibilidad de entradas sí podrán estudiarse.

## Aplicación web futura

La aplicación puede mostrar:

- motivo probable y alternativas;
- probabilidades T2–T4 con sus límites;
- reclamos históricos similares;
- grupo semántico y evolución semanal;
- explicación de la alerta;
- decisión y comentario del agente humano.

M12 ya muestra estas piezas como demo estática, con una ficha por reclamo y un
panel semanal, en `reports/modeling/triage_demo/`. La app web usará las mismas
funciones de `src/triage/`, que devuelven JSON (`models_plan.md` §25).

Después del triaje, un agente construido con LangGraph podría redactar una
respuesta sugerida usando información aprobada. Esa respuesta siempre deberá
ser revisada por una persona antes de enviarse.
