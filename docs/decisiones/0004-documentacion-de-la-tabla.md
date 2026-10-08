# 0004 · Diccionario de datos e informe de limpieza como salidas del pipeline

| | |
|---|---|
| **Fecha** | 2026-10-07 |
| **Etapa CRISP-DM** | 3 · Preparación |
| **Issue** | #14 (PB-14), #11 (PB-11) |
| **Responsable** | @pipaber (DS) |
| **Estado** | aceptada |

## Contexto

La tabla preparada tiene 3,837,184 filas y 44 columnas, pero su documentación
vivía en notebooks y en README escritos a mano. El curso pide que el
diccionario cubra el 100 % de las columnas finales como `out` de una etapa y
que el informe de limpieza responda, problema por problema, al informe de
calidad del EDA (`docs/resumen-eda.md`, hallazgos 1–7).

## Decisión

Añadir la etapa `document_prepared` a `dvc.yaml`. Lee la tabla columna por
columna y escribe dos salidas sin transformar nada:

- `docs/diccionario/diccionario.csv`: una fila por columna con tipo, etapa
  que la creó, rol en el contrato de modelado, no nulos, porcentaje de nulos,
  valores únicos, ejemplo y descripción de negocio. Las descripciones las
  escribe el equipo en `docs/diccionario/descripciones.yaml`; las narrativas
  no llevan ejemplo, para no copiar texto de reclamos a Git.
- `reports/preparation/informe_limpieza.json`: filas y columnas antes y
  después de cada etapa, con el MD5 y los bytes que anota `dvc.lock`, y una
  entrada por hallazgo del EDA con su tratamiento y la cifra medida en la
  tabla (textos compartidos, pares desconocidos, respuestas ambiguas,
  identificadores revisados, positivos por objetivo, filas por periodo).

La etapa se detiene si las columnas de la tabla no coinciden con las que
`params.yaml` declara por etapa: la lista es la decisión y la tabla la prueba.

## Alternativas consideradas

- Escribir el diccionario a mano — se desactualiza y no mide nada.
- Usar la plantilla `etapa_3_preparacion/codigo/documentar.py` del curso —
  espera `train/valid/test.parquet` y un `params.yaml` con otra estructura;
  el pipeline activo tiene una sola tabla con periodos y elegibilidad.
- Declarar los intermedios como dependencias para medir filas por etapa — obliga
  a descargar cuatro tablas de 1.5 GB; el informe toma filas y MD5 de
  `dvc.lock` y mide lo demás en la tabla final.

## Dónde vive

| | |
|---|---|
| **Parámetro** | `params.yaml` → `documentar.columnas_por_etapa`, `documentar.roles` |
| **Código** | `src/data/document_prepared.py`, pruebas en `tests/test_document_prepared.py` |
| **Etapa** | `document_prepared` (su `desc` enlaza esta entrada) |
| **Commit** | el que añade la etapa; el `dvc.lock` se actualiza en Khipu |

## Consecuencias

El mismo día, por decisión del usuario, las constantes de P1–P5 pasaron a
`params.yaml` (`preparacion.<etapa>`) y el pipeline completo se reejecutó en
Khipu (SLURM 54844): todas las salidas conservaron su hash y la tabla su MD5
`d189a3ae…` (detalle en 0002). Sigue pendiente la verificación desde un clon limpio (PB-15).
