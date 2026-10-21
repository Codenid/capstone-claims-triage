# Revisión de literatura con Asta (PB-23)

Fecha: 2026-10-19. Herramientas: Asta Paper Finder (`asta literature
interactive`, cinco consultas, una por componente) y Semantic Scholar
(`asta papers search` / `get`) para referencias canónicas puntuales. Los
resultados crudos (JSON con relevancia, resúmenes y fragmentos) quedan en
`.asta/literature/threads/2026-10-19-*/` en la máquina local; no se versionan
porque superan el límite de 1 MB del guard del curso. Cada paper se cita con su
DOI o arXiv; las entradas nuevas están en `reports/informe/referencias.bib`.

Pregunta de PB-23: para cada componente del prototipo, ¿la literatura usa
enfoques como el nuestro, y existe una alternativa mejor documentada que deba
entrar como candidato pre-registrado?

Lectura general: el prototipo coincide con la práctica publicada en los cinco
frentes. Aparecen cuatro alternativas documentadas que valen un candidato
futuro; ninguna reemplaza nada sin correrse bajo la regla de §25.2. Se
registran en `docs/registro/models_plan.md` §31 sin ejecutarse.

## 1. Clasificación del motivo (T1) y resultados T2–T3

| Trabajo | Enfoque | Datos | Métrica | Relación con lo nuestro |
|---|---|---|---|---|
| Li y Shi (2026), IEEE Access, DOI 10.1109/ACCESS.2026.3716494 | Narrativas CFPB, embeddings por SVD y SVM lineal con pesos por clase; modelado Dirichlet de la evolución de quejas | CFPB, 5 clases | Macro-F1 | Mismo dato y misma métrica; 5 clases frente a nuestros 90 motivos |
| Gopinathan y Manzoor (2025), IJACSA, DOI 10.14569/ijacsa.2025.0161202 | Embeddings MiniLM congelados + red neuronal; baselines TF-IDF y Word2Vec | Quejas de consumidores, 5 productos | F1 | Misma arquitectura «codificador congelado + cabeza»; granularidad gruesa |
| Roumeliotis et al. (2025), Electronics, DOI 10.3390/electronics14061070 | LLM de razonamiento en cero disparos para clasificar quejas financieras | Quejas financieras | Accuracy/F1 | Alternativa sin entrenamiento; reporta solapamiento de categorías como dificultad |
| Bastani et al. (2019), ESWA, DOI 10.1016/J.ESWA.2019.03.001 | LDA sobre narrativas CFPB para extraer temas | CFPB | Coherencia | Antecedente directo de nuestros patrones sobre el mismo dato (230 citas) |
| Wang, Zhu y Chen (2026), arXiv 2606.22664 | Marco híbrido para predecir alivio monetario: texto + temas LDA + rasgos de texto + columnas estructuradas; binaria desbalanceada | CFPB | AP/F1 | Misma tarea que T3; añade columnas estructuradas (empresa, estado) que nosotros excluimos por contrato |
| Zhang et al. (2026), arXiv 2607.11007 | TabPFN como cabeza sin gradiente sobre codificadores congelados de imagen, texto y audio; 22,820 episodios, 14 datasets, 11 codificadores | Multimodal | Accuracy y calibración | Es exactamente nuestro T1 (TabPFN sobre embeddings BGE): TabPFN obtiene el mejor rango medio entre nueve cabezas y mejor calibración |
| Takeshita et al. (2025), EMNLP, arXiv 2508.17744 | Quitar al azar hasta 50 % de las dimensiones de un embedding de texto cuesta < 10 % en clasificación y recuperación | 6 codificadores, 26 tareas | Varias | Explica por qué PCA 100 funciona y por qué 256 no añade (§26) |
| Grinsztajn et al. (2025), arXiv 2511.08667 | Informe técnico de TabPFN-2.5: límites de filas y columnas, rendimiento en tabulares | Benchmarks tabulares | Varias | Referencia del modelo; documenta el límite de columnas que explica la caída con 1,024 valores |

Veredicto T1: **coincide con la literatura** en la arquitectura (embeddings
congelados + cabeza lineal o TabPFN) y en la métrica. No hay trabajos con
TabPFN sobre narrativas CFPB: nuestro resultado es, hasta donde encontramos,
inédito en este dominio, y Zhang et al. (2026) lo respalda en 14 datasets de
otros dominios. Las cifras publicadas (F1 > 0.9) son con 5–6 clases gruesas,
no comparables con 90 motivos.

Alternativas documentadas: (a) ajuste fino de un transformer (BERT/FinBERT)
sobre las narrativas, que varios trabajos usan con clases gruesas; (b) LLM en
cero disparos (Roumeliotis 2025). Ninguna reporta el régimen de 90 clases con
Macro-F1. Entran como **candidato futuro C-T1-FT** (§31) bajo la regla de 25.2;
el LLM en cero disparos no entra por costo y por la licencia de uso con datos
del banco.

Veredicto T2–T3: **coincide** (texto + modelo lineal es lo usual). Wang, Zhu y
Chen (2026) documentan una mejora al añadir columnas estructuradas (empresa,
estado, producto) a la predicción de alivio monetario. Nosotros excluimos
empresa y estado del contrato inicial (`evaluation_contract.json`, columnas
candidatas no usadas). Entra como **candidato futuro C-T3-ESTRUCT** (§31):
T2/T3 con producto + empresa + estado, con la misma regla.

## 2. Patrones semánticos y novedad

| Trabajo | Enfoque | Relación |
|---|---|---|
| Grootendorst (2022), arXiv 2203.05794 (BERTopic, 3,279 citas) | Embeddings → UMAP → HDBSCAN → c-TF-IDF por grupo | Nuestra tubería es la misma salvo el algoritmo final (k-means en lugar de HDBSCAN) y la descripción por términos del catálogo es un c-TF-IDF |
| Angelov (2020), Top2Vec, arXiv 2008.09470 | Embeddings conjuntos + UMAP + HDBSCAN; búsqueda de documentos similares | Antecedente de la combinación embeddings + UMAP + HDBSCAN |
| Pattnayak et al. (2026), IJCNLP-AACL, arXiv 2601.04388 | Conversaciones de soporte: embeddings + UMAP + HDBSCAN, emparejamiento incremental de casos nuevos y creación de grupos nuevos | Es la versión «en línea» de nuestra asignación + novedad |
| Asyaky y Mandala (2021), ICAICTA, DOI 10.1109/ICAICTA53211.2021.9640285 | HDBSCAN sobre textos cortos mejora con embeddings + UMAP | Confirma el papel de UMAP antes de agrupar |
| Milev y Bahov (2026), BDCC, DOI 10.3390/bdcc10090314 | Benchmark reproducible para detectar temas nuevos en flujos de texto | Marco para evaluar nuestra regla de novedad (p99 de distancia) |
| An et al. (2023), IEEE Access, DOI 10.1109/ACCESS.2023.3337808 | BERTopic con HDBSCAN y k-means; modelado dinámico de temas en el tiempo | Mismo par de algoritmos que comparamos |

Veredicto: **coincide**. La literatura dominante es BERTopic (HDBSCAN); en
nuestros datos HDBSCAN dejó 49 % de ruido y 63 % de cobertura futura y falló
los criterios pre-registrados, por lo que k-means fue la elección correcta
bajo esas reglas. Ningún trabajo encontrado evalúa estabilidad temporal ni
cobertura de documentos futuros como nosotros; la asignación incremental con
creación de grupos nuevos (Pattnayak 2026) es la extensión natural de nuestra
regla de novedad y entra como **candidato futuro C-PAT-INC** (§31) para el
pipeline, no como reemplazo del M8.

## 3. Conteo semanal y composición (M9, M10)

| Trabajo | Enfoque | Relación |
|---|---|---|
| Bracher et al. (2021), PLoS Comput Biol, DOI 10.1371/journal.pcbi.1008618 | WIS y cobertura para pronósticos por intervalos | Nuestra métrica y nuestros rangos de cobertura |
| Cramer et al. (2022), PNAS, DOI 10.1073/pnas.2113561119 | Evaluación de pronósticos de muertes por COVID-19 con WIS; el ensamble supera a todos los modelos individuales | Sugiere un ensamble como alternativa a elegir un solo candidato |
| Bosse et al. (2023), PLoS Comput Biol, DOI 10.1371/journal.pcbi.1011393 | Puntuar pronósticos en escala transformada (log) para que los patrones grandes no dominen el WIS | Relevante: nuestro WIS agregado está dominado por los patrones grandes |
| Nigri, Han y Ungolo (2026), arXiv 2603.00973 | Marco Dirichlet-multinomial-Poisson: total Poisson, composición multinomial con Dirichlet, pronóstico coherente por causa | Misma estructura que nuestro M9 + M10 (total × composición), aplicada a mortalidad por causa |
| Mills et al. (2025), R Soc Open Sci, DOI 10.1098/rsos.241870 | Varios modelos, incluido jerárquico bayesiano, evaluados con WIS y cobertura | Mismo protocolo de evaluación |
| Salmon, Schumacher y Höhle (2016), JSS, DOI 10.18637/jss.v070.i10 | Paquete `surveillance`: modelos de conteo y detección de aberraciones | Referencia de implementación de conteos y alertas |

Veredicto: **coincide** con la práctica de pronóstico epidemiológico (binomial
negativa, WIS, cobertura) y la descomposición total × composición tiene un
antecedente formal (Nigri 2026). Dos alternativas documentadas: (a) un
**ensamble** de los candidatos aceptados (Cramer 2022), **candidato futuro
C-M9-ENS** (§31); (b) **evaluar también en escala logarítmica** (Bosse 2023),
que no cambia los modelos pero sí qué patrones pesan en la decisión; entra
como métrica secundaria pre-registrada, no como criterio de reemplazo.

## 4. Alertas (M11, D-11)

| Trabajo | Enfoque | Relación |
|---|---|---|
| Farrington et al. (1996), JRSS A, DOI 10.2307/2983331 | Algoritmo de detección temprana de brotes con umbral sobre un modelo de referencia; **reduce el peso de semanas con brotes pasados** al estimar la línea base | Es la solución clásica al «eco de ráfagas» que declaramos como límite |
| Noufaily et al. (2013), Stat Med, DOI 10.1002/sim.5595 | Mejora de Farrington: ventana de referencia más larga, umbrales por cuantiles de la NB, menos falsas alarmas | Igual |
| Enki et al. (2016), PLoS ONE, DOI 10.1371/journal.pone.0160759 | Comparación de algoritmos de detección en sistemas múltiples de vigilancia | Protocolo de comparación con brotes inyectados, como el nuestro |
| Salmon, Schumacher y Höhle (2016), JSS, DOI 10.18637/jss.v070.i10 | CUSUM de razón de verosimilitud y reglas de un periodo para conteos | Mismas dos reglas que comparamos |
| O'Brien y Christie (1997), Public Health, DOI 10.1038/SJ.PH.1900353 | CUSUM en vigilancia rutinaria de enfermedades transmisibles | Antecedente del uso de CUSUM con líneas base históricas |
| Robertson y Yee (2016), PLoS ONE, DOI 10.1371/journal.pone.0165688 | CUSUM con umbral ajustado por ARL y comparación con un umbral fijo | Mismo diseño: presupuesto de falsas alarmas y comparación con regla de un periodo |
| Chen, Zeng y Yan (2009), Infectious Disease Informatics, DOI 10.1007/978-1-4419-1278-7_4 | Bandas de guarda y sensibilidad del umbral en CUSUM frente a reglas tipo Shewhart | Documenta el tratamiento de brotes recientes en la línea base |
| Valdez-Vivas et al. (2018), KDD, DOI 10.1145/3219819.3219858 | CUSUM secuencial que distingue picos transitorios de regresiones sostenidas | Apoya la lectura de D-11: a escala fina, una ráfaga corta queda dentro del ruido |

Veredicto: **coincide** en diseño (CUSUM frente a regla de un periodo con
presupuesto de falsas alarmas y brotes inyectados). La literatura de
vigilancia tiene una respuesta documentada al eco de ráfagas que nosotros no
probamos: **reducir el peso de las semanas con alarma al estimar la
participación esperada** (Farrington 1996, Noufaily 2013). Nuestro recorte de
semanas atípicas en ajuste (§25.5) no ganó, pero recortaba la memoria, no la
reponderaba. Entra como **candidato futuro C-M11-FARR** (§31). Para D-11, la
alternativa documentada es un umbral empírico sobre los excesos de ajuste en
lugar de la N(0, 1) simulada (Noufaily 2013 usa cuantiles de la NB); ya estaba
anotado como pendiente en §28.3 y queda formalizado en §31.

## 5. Modelos fundacionales de series y protocolo de validación

| Trabajo | Enfoque | Relación |
|---|---|---|
| Gupta (2025), arXiv 2501.00889 | TimesFM, Chronos y Moirai en cero disparos sobre series periódicas ruidosas frente a FFT y AR | Mismo hallazgo: los fundacionales no ganan a baselines simples en series cortas y ruidosas |
| Yu et al. (2025), CAiSE, arXiv 2512.07624 | Fundacionales en cero disparos sobre series dispersas de registros de eventos frente a baselines estadísticos | Igual |
| Alonso y Franklin (2026), arXiv 2606.27100 | Fundacionales sobre retornos financieros frente a paseo aleatorio | Ganancias escasas sobre el baseline |
| Meyer et al. (2025), KDD, DOI 10.1145/3770855.3817515 | TS-Arena: plataforma de **pre-registro** de pronósticos con evaluación en datos futuros | Formaliza lo que hacemos a mano: regla escrita antes, evaluación hacia adelante |
| Gan et al. (2026), Eng. Comput. Insights, DOI 10.67728/eci.2026.002 | Meta-análisis: validación temporal, fuga por proyecto o tiempo, recomienda pre-registrar particiones y planes de evaluación | Mismo protocolo |

Veredicto: **coincide**. Nuestro rechazo de Chronos-2 y TimesFM 3.0 frente a
la binomial negativa es lo que reportan los benchmarks recientes en series
cortas y ruidosas; no hay alternativa mejor documentada en este régimen. El
pre-registro y la validación hacia adelante son la práctica recomendada.

## Candidatos que pasan al pre-registro (§31), sin correr

| Id | Qué | Origen | Regla |
|---|---|---|---|
| C-T1-FT | Ajuste fino de un transformer (BERT o FinBERT) para T1 con 90 clases | Varios trabajos con clases gruesas | 25.2 contra TabPFN-3.5 en calibración sin texto compartido |
| C-T3-ESTRUCT | T2 y T3 con producto + empresa + estado además del texto | Wang, Zhu y Chen (2026) | 25.2 contra TF-IDF solo texto; empresa y estado se añaden al contrato antes |
| C-M9-ENS | Ensamble (promedio de cuantiles) de C-A y los candidatos aceptados | Cramer et al. (2022) | 25.2 contra C-A por WIS; se reporta también en escala log (Bosse 2023) |
| C-M11-FARR | Reponderar las semanas con alarma al estimar la participación esperada (Farrington/Noufaily) | Farrington (1996), Noufaily (2013) | Misma simulación de M11: detección ≥ la actual y menos alertas de eco en 2025-H1 (ya consultado, informativo) |
| C-PAT-INC | Asignación incremental con creación de grupos nuevos para el pipeline | Pattnayak et al. (2026) | Diseño del pipeline §29; no compite con M8 |
| D-11 v2 | Umbrales empíricos sobre los excesos de ajuste y ráfagas mayores | Noufaily (2013), §28.3 | Misma regla de adopción de §28.3 |

Ninguno de estos se corre hasta que el registro tenga su entrada con fecha y
regla, y todos se deciden en calibración.

Continuación: las teorías de Asta Theorizer sobre los resultados cerrados
(PB-24, 2026-10-21) están en [`theorizer_pb24.md`](theorizer_pb24.md), con
el balance de qué hicimos bien y mal según la literatura y qué probar.
