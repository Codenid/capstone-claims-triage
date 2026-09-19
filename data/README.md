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

Los archivos ubicados en `data/raw/` representan la fuente original y no deben
modificarse manualmente.

Las operaciones de limpieza, filtrado, enriquecimiento o transformación deben
producir nuevos artefactos en:

- `data/interim/`: datos intermedios.
- `data/processed/`: datos preparados para análisis o modelado.

## PROCESSED

- **Archivo:** `data/processed/prepared.parquet`
- **Filas:** 3,837,184
- **Columnas:** 44
- **Contenido:** narrativas normalizadas, taxonomía propuesta, objetivos T1–T4,
  periodos y reglas de elegibilidad
- **Entrada para:** rama `Modeling/pipaber`
- **Versionamiento:** DVC

`2025-H2` está incluido para conservar el registro, pero está bloqueado como
evaluación final porque no se recuperaron los IDs revisados previamente. Los
resultados T2–T4 de 2026 también están bloqueados hasta definir su maduración.
