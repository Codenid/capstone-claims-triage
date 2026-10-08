# 0002 · Preparación conservadora y partición temporal

Registro retrospectivo: 2026-10-03. Responsable DS: Piero Palacios.
Responsable MLE: Winston Flores. Fuente: `EDA/pipaber`, commit `4adb309`.

## Decisiones existentes

- Conservar el crudo y todas sus filas; los `Complaint ID` son únicos.
- Tipar las fechas sin reemplazar las columnas originales por imputaciones.
- Añadir narrativa normalizada y su SHA-256 con `text_normalizer_v1`.
  Los textos compartidos identifican grupos para la evaluación: no justifican
  eliminar reclamos ni afirmar que proceden de la misma persona.
- Aplicar una taxonomía propuesta de 18 productos y 140 motivos canónicos,
  a partir de 21 productos y 173 motivos originales. El mapa y los 323 pares
  conocidos se congelan al 2025-06-30; categorías o pares posteriores
  desconocidos quedan señalados, sin ampliar el mapa al evaluar.
- Construir objetivos T1–T4 y máscaras independientes de elegibilidad.
- Reservar 2015–2022 para contexto, 2023–2024 para entrenamiento y
  2025-H1 para validación temporal. Comparar una vista completa y otra sin
  textos compartidos con los periodos usados como referencia.
- Bloquear 2025-H2 como evaluación final: no se recuperaron los 25,000 IDs
  revisados previamente. Mantener 2026 como periodo parcial y bloquear sus
  resultados T2–T4 hasta acordar la maduración necesaria.

## Implementación y evidencia

El [pipeline activo](../../dvc.yaml) y su [lock](../../dvc.lock) enlazan:

`type_data → normalize_text → apply_taxonomy → build_targets → finalize_prepared`.

Los originales se conservan en `data/raw/`; los intermedios, en
`data/interim/`; la entrega, en `data/processed/prepared.parquet`.
La salida histórica contiene 3,837,184 filas y 44 columnas, con MD5
`d189a3ae6ea7a8f9041fc8210968eda6`. Los bytes se guardan mediante DVC.

La evidencia está en el [notebook de preparación](../../notebooks/02_revision_preparacion.ipynb),
el [estado del holdout](../../configs/holdout_review_status.json) y los
[tests](../../tests). Los commits originales son `95bdfbd` (tipado),
`63d6adc` (normalización), `d80b821` (taxonomía) y `4adb309` (objetivos,
particiones y entrega).

## Validación de la importación

El 2026-10-03 pasaron las 25 pruebas originales y siete pruebas de la guardia en la copia de entrega y
`dvc dag` mostró las cinco etapas enlazadas. Se recreó el entorno con `uv sync --frozen`:
Python 3.12.12, pandas 3.0.5, PyArrow 25.0.1 y DVC 3.67.1.
Esta verificación no equivale a descargar los datos y ejecutar todo el
corpus desde un clon limpio.

## Adaptación al curso

El pipeline original mantiene sus rutas y hashes. La receta genérica del
curso queda en `etapa_3_preparacion/dvc.yaml.example` para evitar dos
pipelines activos con decisiones distintas.

El 2026-10-07 las decisiones de las cinco etapas pasaron a `params.yaml`
(`preparacion.<etapa>`): columnas esperadas y formato de fecha, versión y
reglas del normalizador, versión, estado y fecha de congelamiento de la
taxonomía, inicio de cada periodo, respuestas que definen T2–T4 y filas
esperadas. Cada script las lee al cargarse (`src/data/params.py`), cada
etapa las declara en `dvc.yaml` y `dvc.lock` las anota. El pipeline se
reejecutó en Khipu el 2026-10-07 (SLURM 54844, 13 minutos) con
`scripts/hpc/p1_5_preparation.slurm`: las seis salidas conservaron su hash y
la tabla preparada su MD5 `d189a3ae…`; `dvc.lock` registra ahora los
parámetros de cada etapa (commit `a05eebf`). El diccionario y el informe de limpieza son salidas del
pipeline desde esa fecha. Queda pendiente la reproducción independiente
desde un clon limpio (PB-15).

La importación añade `scripts/repro_preparacion.py` para fijar el directorio y
`PYTHONPATH` del pipeline: los scripts históricos con imports `src.data` fallan
al invocarse por ruta sin ese contexto. El lanzador conserva la receta y el lock.
La sincronización del crudo y las cinco salidas con DagsHub se comprobó en el
checkout de origen el 2026-10-03; no equivale a descargar desde un clon limpio.
