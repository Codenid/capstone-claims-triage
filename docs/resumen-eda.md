# Resumen ejecutivo del EDA

El análisis exploratorio de datos (EDA) revisa el archivo antes de transformarlo
o entrenar modelos. El detalle reproducible está en
[`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb).

## Pregunta de negocio

El prototipo busca apoyar el triaje de reclamos:

- sugerir el motivo del reclamo;
- estimar si históricamente se registró compensación monetaria;
- estimar el indicador de respuesta no oportuna de CFPB;
- encontrar reclamos relacionados y patrones que aumentan con el tiempo;
- entregar evidencia a una persona, que conserva la decisión final.

Los datos no permiten predecir fraude confirmado, pérdidas, tiempo real de
resolución bancaria ni el plazo interno de 15 días.

## Datos disponibles

- Fuente: Consumer Financial Protection Bureau (CFPB).
- 3,837,184 reclamos con narrativa pública.
- 16 columnas.
- Periodo de recepción: 2015-03-19 a 2026-07-27.
- Tamaño Parquet: 1.51 GiB.
- Todos los `Complaint ID` son únicos.

El archivo contiene únicamente reclamos con narrativa publicada. El texto está
anonimizado y puede no ser idéntico al texto original recibido por CFPB.

## Resultados que podrían modelarse

- **T1 — motivo:** clasificación asistida del `Issue` CFPB.
- **T2 — relief registrado:** compensación monetaria o no monetaria registrada.
- **T3 — relief monetario:** aproximación CFPB a una compensación monetaria.
- **T4 — respuesta no oportuna:** indicador `Timely response? = No` de CFPB.

T1 no equivale al equipo interno de un banco. T2 y T3 no demuestran
satisfacción o resolución. T4 no mide cierre ni el plazo bancario de 15 días.

## Hallazgos principales

### 1. El archivo cambia mucho con el tiempo

- 2025 contiene 1,222,056 reclamos, 31.85% del total.
- 2026 es parcial y presenta menor cobertura.
- Las categorías y las tasas T2–T4 cambian entre periodos.
- Por ello, los datos no deben repartirse al azar para evaluar modelos.

### 2. La taxonomía histórica cambió

- Hay 21 productos y 173 motivos originales.
- Categorías antiguas desaparecen mientras otras relacionadas aumentan.
- Entre 2023–2024 y 2025-H1, una categoría antigua de credit reporting cae
  17.26 puntos y otra relacionada sube 16.30 puntos.
- Se necesita un mapa explícito que agrupe nombres históricos equivalentes.

### 3. Pocas categorías concentran gran parte de los casos

- Las categorías relacionadas con credit reporting reúnen aproximadamente
  65.43% de los reclamos.
- Los 10 motivos más frecuentes reúnen 78.55%.
- 31 motivos tienen menos de 100 ejemplos.
- Las tres empresas principales concentran 60.72% de las filas.

Un porcentaje global de aciertos podría ocultar resultados deficientes en
motivos poco frecuentes.

### 4. T3 y T4 tienen pocos positivos

- T2: 1,273,365 positivos; 33.315% entre resultados conocidos.
- T3: 98,150 positivos; 2.568%.
- T4: 42,615 positivos; 1.111%.

Un modelo que siempre predice negativo alcanzaría 97.43% de aciertos en T3 y
98.89% en T4, pero no encontraría ningún positivo. Por eso no se usará el
porcentaje de aciertos como métrica principal.

### 5. Las narrativas tienen longitudes muy diferentes

- No hay narrativas nulas o vacías.
- La mitad tiene 117 palabras o menos.
- 95% tiene 518 palabras o menos.
- 8.43% supera 400 palabras.
- 67.59% contiene marcas `XXXX` de anonimización.

Los modelos con límite de texto deberán comparar recorte frente a división en
fragmentos.

### 6. Hay mucha reutilización textual

Con `text_normalizer_v1`:

- 1,642,513 filas, 42.81%, comparten texto normalizado con otro registro.
- Existen 281,943 grupos normalizados repetidos.
- 15,026 grupos cumplen una regla exploratoria de posible plantilla y reúnen
  860,019 reclamos.
- 468 grupos cumplen una regla estricta de posible ráfaga localizada.

IDs diferentes no prueban que sean eventos diferentes. Tampoco sabemos quién
originó las posibles plantillas. No se eliminarán filas: los grupos servirán
para controlar la evaluación.

### 7. Los resultados también cambian con el tiempo

De 2023–2024 a 2025-H1:

- T2 baja de 43.226% a 38.722%.
- T3 baja de 2.010% a 1.078%.
- T4 sube de 0.535% a 0.888%.

Un modelo debe evaluarse en periodos posteriores a los usados para aprender.

## Decisiones aprobadas

- Mantener el archivo original sin modificaciones.
- Conservar todas las filas.
- Definir el momento de predicción antes de cualquier respuesta empresarial.
- Excluir como entradas las columnas que revelan resultados posteriores.
- Usar `text_normalizer_v1` y SHA-256 para identificar grupos de texto.
- Congelar el mapa producto–motivo antes de evaluar periodos reservados.
- Crear elegibilidad separada para T1, T2, T3 y T4.
- Mantener 2015–2022 como contexto histórico.
- Usar 2023–2024 para entrenamiento y 2025-H1 para validación.
- Mantener 2025-H2 como candidato reservado y 2026 como periodo parcial.
- Reportar una evaluación con casos elegibles y otra sin grupos compartidos con
  periodos usados para aprender.
- Terminar esta rama con una tabla preparada mediante DVC.
- Desarrollar TF-IDF, BGE, FAISS y modelos temporales en `Modeling/pipaber`.

## Riesgos pendientes

- Falta la fecha oficial de extracción del archivo.
- Falta recuperar los 25,000 IDs de 2025-H2 consultados anteriormente.
- La exclusión deberá cerrarse por grupo de texto, no solo por ID.
- Si la lista no se recupera, 2025-H2 no podrá presentarse como evaluación final
  intacta.
- T2–T4 no se evaluarán en 2026 hasta definir cuánto deben madurar sus
  resultados.
- El mapa canónico será una propuesta mientras no exista revisión de negocio.

## Siguientes pasos

1. Recuperar la procedencia y fecha de extracción del archivo.
2. Recuperar los 25,000 IDs de 2025-H2 y cerrar sus grupos de texto.
3. Implementar tipado, mapa de categorías, objetivos y periodos.
4. Añadir pruebas automáticas del contrato E9.
5. Declarar las transformaciones en `dvc.yaml`.
6. Generar y publicar con DVC la tabla preparada para modelado.
7. Entregar esa tabla a `Modeling/pipaber`.

La propuesta posterior de modelos y alertas está en
[`docs/modelos.md`](modelos.md).
