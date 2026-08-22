# Datos del proyecto

## RAW v1

* **Archivo:** `data/raw/complaints.csv.zip`
* **Fuente:** Consumer Financial Protection Bureau (CFPB)
* **Dataset:** Consumer Complaint Database
* **URL:** https://www.consumerfinance.gov/data-research/consumer-complaints/
* **Formato original:** CSV
* **Tamaño aproximado:** ~1.4 GB
* **Estado:** datos originales, sin transformaciones
* **Versionamiento:** DVC
* **DVC Remote:** DagsHub

## Regla de tratamiento

Los archivos ubicados en `data/raw/` representan la fuente original y no deben modificarse manualmente.

Las operaciones de limpieza, filtrado, enriquecimiento o transformación deben producir nuevos artefactos en:

* `data/interim/`: datos intermedios.
* `data/processed/`: datos preparados para análisis o modelado.
