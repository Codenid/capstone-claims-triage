# Datos del proyecto

## RAW

- **Archivo:** `data/raw/cfpb_reclamos_narrativa.parquet`
- **Fuente:** Consumer Financial Protection Bureau (CFPB)
- **Entregado por:** docente del curso
- **Filas:** 3,837,184
- **Columnas:** 16
- **Contenido:** reclamos que incluyen narrativa del consumidor
- **Formato:** Parquet
- **Versionamiento:** DVC
- **Remote:** DagsHub

## Regla de tratamiento

Los archivos ubicados en `data/raw/` representan la fuente original y no deben modificarse manualmente.

Las operaciones de limpieza, filtrado, enriquecimiento o transformación deben producir nuevos artefactos en:

* `data/interim/`: datos intermedios.
* `data/processed/`: datos preparados para análisis o modelado.
