# PB-24: teorías de Asta Theorizer sobre los resultados cerrados

Corrido el 2026-10-21, después del cierre del modelado (§32 del registro).
Theorizer (Ai2, vía `asta generate-theories`) busca papers, extrae hallazgos
con un esquema por pregunta, forma teorías con evidencia a favor y en contra
y evalúa la novedad de cada afirmación contra la literatura recuperada. Se
corrió **por etapas**, validando con el usuario la lista de papers y las
teorías antes de cada paso siguiente; 30 papers por pregunta; objetivo
«accuracy-focused». Los resultados crudos (tiendas de teorías, extracciones,
índices) están en `.asta/generate-theories/2026-10-21-*/` fuera de Git.

Las preguntas salen de los límites declarados en §32.3 y de los modelos
bayesianos de §32.2. Ninguna teoría cambia un resultado: entran como
hipótesis de trabajo futuro (§32.4), con cita, sin corrida.

Este documento se generó con asistencia de IA (Claude Code sobre Theorizer);
las teorías son síntesis de un modelo de lenguaje sobre papers recuperados y
deben leerse como hipótesis, no como hechos establecidos, salvo donde la
evaluación de novedad las marca como «ya establecidas».

## Cómo leer la novedad

Theorizer juzga cada teoría en siete dimensiones (fenómeno, mecanismo,
unificación, generalización, restricción, reencuadre conceptual, síntesis
empírica) con cuatro grados: **ya establecido** (algún paper lo dice
explícitamente), **periférico** (mencionado de pasada), **derivable pero no
enunciado** (los ingredientes están, nadie lo escribe así) y **genuinamente
nuevo**. Lo «derivable» es lo que podemos citar como síntesis nuestra con
apoyo; lo «nuevo» es lo que habría que probar.

## Pregunta 1. Eco de las ráfagas: diseño de la línea base de M11

**Pregunta.** Cómo el diseño de la línea base (longitud de la memoria y
reponderación de periodos anómalos) afecta la tasa de falsas alarmas y el
retraso de detección de CUSUM y de la regla de una semana en series cortas y
ruidosas de conteos por categoría que comparten un total conocido. Contexto
enviado: C-A con memoria δ = 0.5, el eco, y el resultado negativo de
C-M11-FARR (§31.2).

**Evidencia.** 30 papers, 20 con texto completo, 19 con hallazgos (74
registros). Corrida `gen-20261021-c09f9734`.

**Teorías** (6):

| # | Teoría | Idea central | Hallazgos |
|---|---|---|---:|
| 1 | Inestabilidad por realimentación en memorias cortas | Con memoria de pocas semanas la línea base es parte del lazo cerrado; reponderar o recortar según el mismo residuo que juzga la alarma crea realimentación. Lo que ayuda con ventanas largas (Farrington, Noufaily) **cambia de signo** con olvido rápido. | 12 |
| 2 | Sensibilidad presupuestada por escala temporal | A presupuesto fijo, la detección de crecimientos sostenidos de 10–20 % se maximiza cuando línea base y estadístico integran en la misma escala que el cambio. | 4 |
| 3 | Grados de libertad de la línea base y dispersión correcta | El coste en falsas alarmas lo fijan tanto la varianza de la línea base (ventanas cortas) como el umbral; regresión estacional de años, GLM binomial negativa o filtros de estado permiten límites más ajustados. | 11 |
| 4 | Contaminación y reinyección de anomalías | Las semanas anómalas que entran a media y varianza cambian los umbrales siguientes; se estabiliza separando detección de aprendizaje. | 5 |
| 5 | Banda de guarda + almacenamiento winsorizado gana a la reponderación continua | En memorias cortas, la cuarentena discreta (no usar las últimas semanas sospechosas; guardar el valor límite de alarma en lugar del pico) supera a los pesos continuos. | 9 |
| 6 | Agregación multiescala como memoria larga implícita | Sumas móviles de L semanas aumentan la potencia para crecimientos sostenidos; correr dos reglas (corta y de L semanas) con supresión de conteos mínimos. | 11 |

**Novedad** (41 juicios sobre las teorías 1, 3, 4 y 5): 28 derivables no
enunciados, 6 ya establecidos (la contaminación de la línea base por brotes
recientes, teoría 4; la descomposición de varianza de la línea base, teoría
3), 5 periféricos, 3 genuinamente nuevos: el cambio de signo de la
robustificación según la memoria efectiva (teoría 1, como reencuadre y como
generalización entre detectores) y el mecanismo de dependencia multiplicativa
por el que la reponderación continua se desestabiliza en memorias cortas
(teoría 5).

**Lectura para el proyecto.** El resultado negativo de C-M11-FARR es el caso
previsto por la teoría 1, no una rareza. Hipótesis futuras, por orden de
costo: (a) cuarentena discreta: banda de guarda de g semanas y, en semana con
alarma, almacenar en la memoria el valor límite de alarma en lugar del
observado (teoría 5; Dórea 2013 lo operacionaliza); (b) una segunda regla
sobre la suma móvil de 4 semanas junto a la semanal, con el mismo presupuesto
repartido (teoría 6); (c) separar nivel lento de ráfaga transitoria con un
modelo de estado (teorías 1 y 4), que es la vía C-B/D-B donde el muestreador
no convergió.

**Referencias clave.** Gopaluni et al. (2025), *Statistics in Medicine*,
10.1002/sim.70075 (métodos móviles ante aberraciones previas). Zareie et al.
(2023), *BMC Med Res Methodol*, 10.1186/s12874-023-02050-z (revisión de GLM
con guías). Bédubourg y Le Strat (2017), *PLoS ONE*,
10.1371/journal.pone.0181227 (comparación de métodos temporales). Dórea et al.
(2013), *PLoS ONE*, 10.1371/journal.pone.0082183 (vigilancia veterinaria con
sustitución del valor límite). Salmon, Schumacher y Höhle (2016), *JSS*,
10.18637/jss.v070.i10. Noufaily et al. (2019), *Bioinformatics*,
10.1093/bioinformatics/bty997 (vigilancia diaria). Hutwagner et al. (2005),
*EID*, 10.3201/eid1102.040587. Enki et al. (2016), *PLoS ONE*,
10.1371/journal.pone.0160759.

## Pregunta 2. Estabilidad del patrón: asignación fuera de muestra

**Pregunta.** Cómo el esquema de asignación fuera de muestra (transformación
UMAP, UMAP paramétrico, extensión de Nyström o por puntos de referencia,
centroide o k vecinos en el espacio original, agrupamiento incremental)
determina la estabilidad de las etiquetas de documentos nuevos, y cómo la
inestabilidad en las fronteras se propaga a los conteos por grupo y a las
alarmas. Contexto enviado: 9 % de reasignación según el lote (§29), el voto
de vecinos de §31.1 (83 % frente a 91 %), conteos que difieren hasta 8 %.

**Evidencia.** Una primera búsqueda trajo aplicaciones bibliométricas de UMAP
sin valor de método y se descartó; la segunda, con consulta técnica y una
restricción explícita, dio 30 papers, 23 con texto completo y 23 con
hallazgos. Corrida `gen-20261021-e8cd9fe2`.

**Teorías** (6):

| # | Teoría | Idea central |
|---|---|---|
| 1 | Inestabilidad de margen en la frontera y propagación como error de medida | La inestabilidad se concentra en los documentos de margen bajo; los conteos por grupo pasan por una matriz de confusión que, si varía con el lote, inventa no estacionariedad y dispara detectores calibrados sin ese ruido. |
| 2 | Determinismo anclado frente a deriva por extensión de variedad | Jerarquía de estabilidad: prototipos fijos en un espacio fijo (PCA) > codificador paramétrico (UMAP paramétrico) > extensiones no paramétricas tipo `umap.transform`. Nuestro k-means en el espacio UMAP está en el escalón más bajo. |
| 3 | Compromiso anclaje–adaptación | Anclar da reproducibilidad pero envejece con la deriva; actualizar anclas produce churn transitorio; la cadencia de actualización debe decidirse por la estabilidad de los conteos y las falsas alarmas. |
| 4 | La masa de frontera gobierna la fragilidad de las alarmas | La variable de control es la fracción de documentos cerca de fronteras por grupo: medirla, reducirla con la representación, o modelar el error de medida en los conteos. |
| 5 | Dependencia del lote en UMAP por perturbación del grafo de vecinos | Mecanismo de nuestro 9 %: el lote cambia los vecinos recuperados y sus pesos; los puntos interiores apenas se mueven, los de frontera cruzan celdas de Voronoi. |
| 6 | Dependencia del orden en agrupamiento en línea | Los esquemas incrementales tienen memoria de trayectoria; dos permutaciones del mismo conjunto dan etiquetas distintas. |

**Novedad** (98 juicios): 86 derivables no enunciados, 2 ya establecidos (la
dependencia del orden en agrupamiento en línea, teoría 6), 3 periféricos, 6
genuinamente nuevos: el mecanismo y el reencuadre de los conteos como canal
de error de medida con matriz de confusión variable (teoría 1), el mecanismo
y la generalización del compromiso anclaje–adaptación (teoría 3), y el
mecanismo y la síntesis empírica de la masa de frontera como variable de
control de las alarmas (teoría 4). La literatura casi no estudia la
propagación de la inestabilidad de etiquetas a conteos vigilados: ahí está el
hueco.

**Lectura para el proyecto.** El patrón blando de §32.3 queda explicado por
las teorías 2 y 5. Hipótesis futuras: (a) medir la masa de frontera por
patrón (margen entre el centro asignado y el segundo) y reportarla junto a
cada alarma de M11, sin cambiar el modelo (teoría 4); (b) anclar la
asignación a prototipos fijos en el espacio PCA 256 o a un codificador
paramétrico entrenado una vez (teoría 2), que debería pre-registrarse con el
criterio de §31.1 y, además, con cobertura futura y estabilidad entre
submuestras como en §19; (c) modelar el error de reasignación en los conteos
de M9 como ruido de medida (teoría 1), lo que explicaría parte de la
sobredispersión observada.

**Referencias clave.** Sainburg, McInnes y Gentner (2020), *Neural
Computation*, 10.1162/neco_a_01434 (UMAP paramétrico). Ghojogh et al.
(2021), arXiv 2109.02508 (tutorial de UMAP y variantes). Wassenaar et al.
(2024), 10.3217/978-3-99161-014-4-061 (UMAP aproximado en línea). Bahri et al.
(2020), 10.1007/978-3-030-44584-3_4 (UMAP incremental por lotes con k-NN).
Senanayake et al. (2019), *IEEE TNNLS*, 10.1109/TNNLS.2020.3023941 (SONG).
Long y Ferguson (2017), *ACHA*, 10.1016/j.acha.2017.08.004 (extensión fuera
de muestra con puntos de referencia). Herrmann et al. (2022), *DMKD*,
10.1007/s10618-023-00980-2 (análisis topológico del agrupamiento tras UMAP).
Tarekegn et al. (2024), arXiv 2406.10552 (índice de estabilidad de
agrupamiento). Gupta et al. (2023), arXiv 2312.02337 (deriva distribucional
con agrupamiento sobre embeddings). Liu et al. (2020), *IEEE Trans.
Cybernetics*, 10.1109/TCYB.2020.2983962 (detección de deriva con partición
k-means). Periti et al. (2022), 10.18653/v1/2022.lchange-1.4 (propagación de
afinidad incremental).

## Pregunta 3. T1: cabezal tabular en contexto frente a ajuste fino

**Pregunta.** Cómo los modelos tabulares en contexto sobre embeddings
reducidos (TabPFN sobre PCA de BGE + producto) se comparan con transformers
ajustados y con lineales sobre embeddings o TF-IDF, en Macro-F1 y
calibración, para texto con muchas clases desbalanceadas. Contexto enviado:
Macro-F1 0.271 con contexto de 20,000 filas y PCA 100, ablación 256 empata y
1,024 pierde, C-T1-FT sin correr.

**Evidencia.** Primera búsqueda de 30 papers con 11 con hallazgos; se
añadieron semillas (Hollmann 2025, TabICL 2025, TabPFN-2.5, Zhang 2026) y
quedó en 40 papers, 31 con texto completo, 23 con hallazgos. Corrida
`gen-20261021-0d334f20`.

**Teorías** (6):

| # | Teoría | Idea central |
|---|---|---|
| 1 | Cabezal bayesiano con cuello de botella | El cabezal en contexto gana cuando el preprocesado (PCA moderado, blanqueo, producto) lo lleva a su «soporte previo»; el ajuste fino gana cuando el cuello de botella descarta información de las clases de cola; híbrido: adaptar el codificador y cerrar con el cabezal tabular por calibración. Explica la ablación (256 empata, 1,024 pierde). |
| 2 | Descomposición de muchas clases y exposición de la cola | Con 90 clases y cola larga, lo que manda en Macro-F1 es la evidencia mínima por clase de cola dentro del contexto y cómo se descompone la decisión (jerárquica mejor que uno-contra-resto). |
| 3 | Descomposición frente a métricas de cola | Las soluciones para muchas clases pueden conservar top-1 y dañar calibración y Macro-F1 salvo que se recompongan en una distribución coherente y calibrada. |
| 4 | Alineación compresión–contexto | Explica el punto dulce de PCA y por qué un lineal sigue siendo competitivo cuando el embedding ya es separable. Casi todo ya establecido. |
| 5 | Rescate por blanqueo | La caída con más dimensiones viene de la anisotropía, no de la dimensión: blanqueo o eliminación de componentes comunes antes de TabPFN debería permitir 256–512 componentes sin colapso y mejorar NLL y Macro-F1. |
| 6 | Curación del contexto consciente de la cola | Con presupuesto fijo de filas, mínimos por clase y recuperación condicionada a la consulta deberían mejorar la cola. |

**Novedad** (77 juicios): 53 derivables no enunciados, 16 ya establecidos
(concentrados en la teoría 4, que es literatura conocida sobre TabPFN), 4
periféricos, 3 genuinamente nuevos: el mecanismo y el reencuadre de la
exposición de la cola bajo descomposición de clases (teoría 2) y la
generalización del rescate por blanqueo (teoría 5).

**Lectura para el proyecto.** Dos hipótesis baratas, comprobables con el
protocolo de §25.2 sin GPU nueva: (a) blanquear (ZCA) o quitar componentes
comunes a los embeddings antes de PCA y TabPFN, y repetir la ablación de 256
y 1,024 (teoría 5); (b) construir el contexto de 20,000 filas con mínimos por
clase y, para cada reclamo, vecinos recuperados por FAISS, en lugar de la
muestra fija (teoría 6). C-T1-FT (ajuste fino) queda como la vía cara cuando
el cuello de botella pierda información de cola (teoría 1); la literatura
no trae comparaciones directas con BERT ajustado en 90 clases de cola larga:
es un hueco, no un resultado.

**Referencias clave.** Hollmann et al. (2025), *Nature*,
10.1038/s41586-024-08328-6. Qu et al. (2025), ICML, arXiv 2502.05564
(TabICL). Zhang et al. (2026), arXiv 2607.11007 (TabPFN sobre embeddings
multimodales). Bergström et al. (2026), arXiv 2605.20674 (composición de
codificadores con cabezal tabular). Tanna et al. (2026), *The Web
Conference*, 10.1145/3774904.3792923 (ajuste fino de modelos tabulares
fundacionales). Kang et al. (2026), *The Web Conference*,
10.1145/3774904.3792477. Ma et al. (2024), NeurIPS (TabDPT). Gardner et al.
(2024), NeurIPS, arXiv 2406.12031 (TabuLa-8B). Erickson et al. (2025),
NeurIPS (TabArena). Jiang et al. (2025), *IEEE TPAMI* (survey).

## Pregunta 4. Conteos bayesianos: memoria corta frente a espacio de estados

**Pregunta.** Cuándo una binomial negativa jerárquica con memoria que decae
para la participación esperada supera a modelos dinámicos de estado
(caminata aleatoria logístico-normal con innovaciones gaussianas o t, filtros
de Kalman, multinomial o Dirichlet-multinomial dinámicas) en pronóstico
probabilístico de paneles cortos de conteos sobredispersos con total conocido,
juzgado por WIS y cobertura, y qué estructuras y parametrizaciones evitan que
NUTS sature la profundidad de árbol o pierda energía. Contexto enviado: C-A
(WIS 30.3), C-B que no ganó con claridad, D-B saturado (R-hat 1.87 centrado,
1.68 y BFMI 0.007 no centrado), D-D ganador diario, Chronos-2 y TimesFM
peores que la referencia bayesiana.

**Evidencia.** 30 papers, 27 con texto completo y 27 con hallazgos (137
registros), casi todos de método. Corrida `gen-20261021-a2026bc1`.

**Teorías** (7):

| # | Teoría | Idea central |
|---|---|---|
| 1 | Tamaño muestral efectivo frente a flexibilidad | En paneles cortos con decenas de categorías y muchas casi vacías, la verosimilitud condicionada al total informa poco sobre un estado latente de alta dimensión; el modelo latente gasta grados de libertad en trayectorias mal identificadas y da intervalos demasiado anchos o demasiado estrechos. La memoria que decae sustituye la trayectoria por un resumen regularizado: más sesgo, mucha menos varianza. Se invierte con T y totales grandes y participaciones suaves. |
| 2 | Dominio de la geometría de inferencia bajo NUTS | Los modelos dinámicos de participación con total conocido inducen curvatura y correlaciones fuertes entre estados, escalas de innovación y transformaciones del simplex: embudos y crestas; NUTS satura el árbol o pierde energía. Colapsar bloques conjugados o muestrear trayectorias enteras (FFBS con Pólya-gamma) restaura la geometría. |
| 3 | Geometría del muestreador en paneles cortos | Cuatro ingredientes juntos (simplex, dependencia casi no estacionaria, poca información por periodo, D−1 contrastes) crean el problema; cambiar centrado por no centrado es de segundo orden: la salida es estructural (marginalizar, FFBS, filtrado). Es exactamente el diagnóstico de D-B. |
| 4 | Dominio de la memoria corta en WIS | Con historias cortas y ráfagas, los modelos latentes sobreajustan fluctuaciones transitorias o difunden la incertidumbre; la memoria descontada es un regularizador implícito alineado con el objetivo de pronóstico a corto horizonte; la dispersión jerárquica absorbe la variabilidad extra. |
| 5 | Dominio en pronóstico de participaciones | Misma mecánica de sesgo–varianza–calibración, enunciada como síntesis empírica. |
| 6 | El mecanismo de inferencia pesa más que la clase de modelo | Para multinomial logístico-normal dinámica, la calidad del pronóstico la gobierna la geometría numérica (colapsar bloques, evitar covarianzas globales, explotar gaussianidad condicional), no la elección Gauss frente a t. Ya establecido en la literatura (Silverman 2019, Saxena 2024). |
| 7 | Principio de ajuste del descuento | El WIS en función del descuento es en U, con óptimo en la tasa de deriva real de las participaciones; más ráfagas piden memoria más corta, totales más pequeños piden memoria más larga; si el espacio de estados se reformula con FFBS o bloques conjugados, la brecha se cierra y puede ganar cuando el proceso real es una caminata y el panel es largo. |

**Novedad** (84 juicios): 64 derivables no enunciados, 11 ya establecidos (la
teoría 6 entera, y la unificación y generalización de la 2: es literatura
conocida de inferencia para modelos logístico-normales dinámicos), 5
periféricos, 4 genuinamente nuevos: la generalización del tamaño muestral
efectivo frente a flexibilidad (teoría 1), la unificación de la dominancia de
la memoria corta bajo WIS (teoría 4, dos jueces) y su síntesis empírica
(teoría 5).

**Lectura para el proyecto.** Las teorías 1, 4, 5 y 7 dicen que C-A ganó por
la razón correcta: en un panel de 100 semanas y 40 patrones con ráfagas, un
resumen regularizado de la historia reciente da mejor WIS que un estado
latente mal identificado. Las teorías 2, 3 y 6 dicen que D-B no falló por
mala suerte ni por la parametrización: con 39 contrastes y 639 días, NUTS
sobre trayectorias latentes es la herramienta equivocada; la vía es colapsar
bloques conjugados o FFBS con Pólya-gamma (Linderman 2015, Silverman 2019,
Saxena 2024) o un filtro, no otra parametrización. La teoría 7 predice una U
en el WIS frente al descuento, que nuestra rejilla de §25.4 puede confirmar o
refutar sin cómputo nuevo.

**Referencias clave.** Silverman et al. (2019), *JMLR*, arXiv 1903.11695
(multinomial logístico-normal con procesos matriz-t). Saxena et al. (2024),
AISTATS, arXiv 2410.05548 (inferencia escalable para DLM logístico-normal).
Linderman, Johnson y Adams (2015), NeurIPS, arXiv 1506.05843 (stick-breaking
con Pólya-gamma). Berry y West (2018), *JBES*,
10.1080/07350015.2019.1604372 (mezcla dinámica de conteos). Chapados (2014),
ICML, arXiv 1405.3738 (espacio de estados binomial negativo jerárquico).
Aktekin, Polson y Soyer (2016), *Bayesian Analysis*, 10.1214/17-BA1054
(Poisson-beta escalado secuencial). Das et al. (2022), UAI, arXiv 2204.10414
(proporciones Dirichlet). Yanchenko, Deng y West (2021), *JRSS C*,
10.1093/jrsssc/qlad002 (DGLM jerárquico). Zhang et al. (2023), *EJOR*,
10.1016/j.ejor.2024.05.024 (reconciliación discreta). Olivares et al.
(2021), *IJF*, 10.1016/j.ijforecast.2023.04.007 (mezclas de Poisson
jerárquicas). Abril-Pla et al. (2023), *PeerJ CS*, 10.7717/peerj-cs.1516
(PyMC; Dirichlet-multinomial marginalizada). Stoner y Economou (2019),
*Biometrics*, 10.1111/biom.13188 (Dirichlet-multinomial generalizada).

## Resumen: qué se encontró, qué hicimos bien, qué hicimos mal, qué probar

### Qué se encontró

Cuatro preguntas, 25 teorías, 300 juicios de novedad. El patrón se repite:
la mayoría de lo que Theorizer propone es **derivable de la literatura pero
no está enunciado** (231 de 300 juicios), poco es ya establecido (35) y poco
es genuinamente nuevo (16). Lo nuevo se concentra en tres ideas: la
robustificación de la línea base cambia de signo con memoria corta (P1), los
conteos por patrón son un canal de error de medida cuya matriz de confusión
depende del lote (P2), y la dominancia de la memoria corta sobre el estado
latente en paneles cortos bajo WIS (P4).

### Qué hicimos bien, según la literatura

- **M9 (C-A).** Elegir una memoria que decae sobre un estado latente es lo
  que las teorías 1, 4, 5 y 7 de P4 predicen para paneles cortos con
  ráfagas; la dispersión jerárquica por patrón es la pieza que absorbe la
  variabilidad extra. El descuento elegido en una rejilla de ajuste (§25.4)
  es el «ajuste del descuento» de la teoría 7.
- **Excluir D-B por regla.** La saturación de NUTS en D-B es el caso
  canónico de las teorías 2, 3 y 6 de P4; insistir con otra parametrización
  habría sido de segundo orden.
- **No adoptar C-M11-FARR.** La teoría 1 de P1 predice la inversión del
  efecto con δ = 0.5; nuestro resultado es su confirmación empírica. Haberlo
  pre-registrado y rechazado por regla es lo que la literatura de vigilancia
  llama evaluar con brotes inyectados y presupuesto fijo (Enki 2016,
  Bédubourg 2017).
- **Presupuesto de falsas alarmas y simulación N(0, 1).** Es el diseño
  estándar (Salmon 2016, Robertson 2016) y la teoría 3 de P1 explica por qué
  el umbral solo no basta sin controlar los grados de libertad de la línea
  base.
- **TabPFN sobre PCA 100 + producto.** La teoría 4 de P3 (casi toda ya
  establecida) dice que el punto dulce de compresión existe y que el lineal
  sigue siendo competitivo; nuestra ablación 100/256/1,024 lo midió antes de
  que la literatura lo dijera para texto.
- **Declarar el patrón blando y medirlo.** La teoría 2 de P2 coloca
  `umap.transform` + k-means en el peor escalón de estabilidad; nosotros lo
  encontramos con la prueba de punta a punta y lo cuantificamos (9 %).

### Qué hicimos mal, o a medias

- **El patrón vive en el espacio UMAP.** Según P2, anclar la asignación a
  prototipos fijos en el espacio PCA 256 (o a un codificador paramétrico)
  habría dado etiquetas reproducibles por construcción. Elegimos el espacio
  por calidad de agrupamiento (§19) sin un criterio de estabilidad fuera de
  muestra; el criterio de §31.1 llegó tarde y comparaba contra etiquetas que
  ya eran inestables.
- **Reponderar con memoria de dos semanas.** Trasladamos Farrington y
  Noufaily a un diseño para el que no fueron pensados (línea base de años
  ajustada por regresión). La literatura ya tenía la alternativa correcta
  para memorias cortas: cuarentena discreta con banda de guarda y
  sustitución del valor límite (Dórea 2013; teoría 5 de P1).
- **Una sola escala temporal en M11.** La teoría 6 de P1 (Gopaluni 2025,
  Bédubourg 2017) muestra que una segunda regla sobre la suma móvil de 4
  semanas detecta crecimientos lentos que la semanal pierde; nuestro 42 % en
  el crecimiento de 10 % es consistente con esa carencia.
- **Contexto de TabPFN por muestra fija.** La teoría 6 de P3 dice que, con
  presupuesto de 20,000 filas y 90 clases de cola larga, la muestra fija
  subrepresenta la cola; la recuperación por vecinos ya la calculamos para
  otra cosa y no la usamos para el contexto.
- **Embeddings sin blanquear.** La teoría 5 de P3 atribuye la caída con
  1,024 dimensiones a la anisotropía, no a la dimensión; no lo probamos.
- **Las alarmas de M11 no llevan su margen de asignación.** Según P2
  (teoría 4), una alarma en un patrón con mucha masa de frontera puede ser
  reasignación; hoy no se reporta.

### Qué valdría la pena probar, por costo

| Prioridad | Hipótesis | Origen | Costo | Qué cambia si funciona |
|---|---|---|---|---|
| 1 | Reportar junto a cada alarma de M11 la masa de frontera del patrón (margen entre primer y segundo centro) | P2, teoría 4 | Horas; sin reentrenar | Lectura de alarmas más honesta en el informe y el pipeline |
| 2 | Confirmar la U del WIS frente al descuento con la rejilla de §25.4 ya corrida | P4, teoría 7 | Horas; sin cómputo | Apoyo o refutación de una predicción nueva con datos propios |
| 3 | Blanqueo ZCA o eliminación de componentes comunes antes de PCA y TabPFN; repetir la ablación 256 y 1,024 | P3, teoría 5 | Un job de GPU | Si rescata 256–512 componentes, mejor Macro-F1 y calibración de T1 |
| 4 | Contexto de TabPFN con mínimos por clase y vecinos FAISS por reclamo | P3, teoría 6 | Un job de GPU | Mejor cola de T1 con el mismo presupuesto |
| 5 | M11 con cuarentena discreta: banda de guarda g y sustitución del valor límite en semanas con alarma | P1, teoría 5 | Un job de CPU, mismo protocolo que §31.2 | Menos eco sin realimentación; la regla de §31 ya está escrita |
| 6 | Alerta multiescala con un solo presupuesto: suma móvil de 3 días para ráfagas cortas (la D-11 v2 de §28.3, con umbrales empíricos), regla semanal, y suma móvil de 4 semanas para crecimientos lentos | P1, teorías 2 y 6 | Un job de CPU por escala | Más detección de ráfagas de días y de crecimientos lentos; el eco de D-D (δ = 0.8, unos cinco días) se trata con la misma cuarentena de la hipótesis 5 |
| 7 | Anclar el patrón a una **mezcla gaussiana en PCA 256** (propuesta del usuario): asignación determinista en un espacio fijo, probabilidad por patrón, conteos blandos opcionales y novedad por verosimilitud; criterios de §19 más acuerdo ≥ 91 % con las etiquetas congeladas y determinismo bajo permutación. Alternativa: UMAP paramétrico | P2, teorías 1, 2 y 4 | Rehacer M8B y los paneles | Etiquetas reproducibles y margen por reclamo; obliga a reajustar M9–M11. En el espacio UMAP la mezcla no resuelve nada: heredaría la inestabilidad de la proyección |
| 8 | Modelo de participaciones dinámico con FFBS y Pólya-gamma o filtro, en lugar de NUTS sobre trayectorias | P4, teorías 2, 3, 6 | Semanas de implementación | Solo si el panel crece; hoy C-A es el modelo correcto |

Resultados de las prioridades 1 y 2 (2026-10-21, §32.6): la U del WIS
frente al descuento existe en ajuste y en calibración, con mínimo en δ = 0.4
y δ = 0.5 a 0.3 % de él (C-A no cambia). La mezcla gaussiana sobre UMAP
define otra partición (42 % de acuerdo con k-means), así que el margen
utilizable es el geométrico: alrededor del 20 % de los reclamos están a
menos del 10 % de distancia relativa de un segundo centro, con una mediana
por patrón de 19 % y extremos de 0 % a 76 %. Esa tabla acompaña a las
alarmas de M11. D-11 v2 (prioridad del usuario, §32.5) se corrió y no se
adoptó: con umbrales empíricos el presupuesto se cumple, la ventana de 3
días es la mejor regla diaria para ráfagas de varios días (24 % en ×2/3 d
frente a 21 % de un día), pero ninguna llega al 50 %; a resolución diaria
la señal por patrón no alcanza con 4 falsas alarmas al mes. La parte
semanal de la hipótesis 6 (suma de 4 semanas) sigue pendiente.

Nada de esto se corre sin entrada propia en el registro (§32.4). Las
prioridades 1 y 2 no cambian ningún modelo y caben antes del informe final;
las 3 a 6 son candidatos pre-registrables de un ciclo cada uno; 7 y 8 son
rondas nuevas. Para la prioridad 1, una mezcla gaussiana ajustada sobre las
coordenadas UMAP de ajuste existentes, inicializada en los centros de
k-means, da la probabilidad por patrón con la que medir la masa de frontera
sin tocar ningún modelo.

Nota sobre las escalas: las preguntas se centraron en la escala semanal
(M11); el bloque diario entró como contexto en la pregunta 4. Hoy no hay
alarma diaria vigente (D-11 no se adoptó, §28.3); D-D es expectativa e
intervalos. Las teorías 2 y 6 de la pregunta 1 explican el resultado de D-11
(una regla de un día es buena para picos enormes y mala para ráfagas de tres
días) y señalan la suma móvil de 3 días como la versión diaria correcta.
