# Pipeline de inferencia: ajustar una vez, puntuar muchas veces

Planificado en `docs/registro/models_plan.md` §29 y construido en la rama
`pipeline/pipaber`. Responde a una pregunta operativa: cuando llegan reclamos
nuevos con las mismas columnas, ¿qué hay que correr y qué no? La respuesta es
que no se reajusta nada. Los componentes vigentes (clasificadores, espacio
semántico, patrones, M9, M10, M11 y D-D) están congelados en DVC con sus
claves de corrida, y `score` solo los aplica.

## Contrato de entrada

Una tabla Parquet o CSV con, como mínimo:

| Columna | Tipo | Nota |
|---|---|---|
| `Complaint ID` | texto | único |
| `Date received` | fecha `AAAA-MM-DD` | |
| `Product` | texto | producto de la CFPB; uno desconocido pasa a `__UNKNOWN__` |
| `Consumer complaint narrative` | texto | la narrativa |

Las otras doce columnas públicas de la CFPB son opcionales. `Issue` es
obligatoria solo para filas con fecha hasta el congelamiento de la taxonomía
(2025-06-30), porque P3 lo exige; `Company response to consumer` y `Timely
response?` permiten, si vienen, medir la deriva de T1 con etiquetas reales.

## Qué hace `score`

1. **P1–P3** sobre la tabla: esquema y fechas, normalización del texto
   (`<redacted>` para las marcas XXXX) y taxonomía canónica congelada.
2. **Embeddings BGE-large** con el modelo del caché local (sin descarga).
3. **Patrón y novedad**: PCA 256 → UMAP 15 → centro k-means más cercano;
   novedoso si la distancia supera el p99 congelado de su patrón.
4. **T2 y T3**: TF-IDF del texto y los modelos lineales congelados, con los
   umbrales elegidos en calibración.
5. **T1**: TabPFN-3.5 con el mismo contexto fijo de 20,000 filas (PCA 100 +
   producto) si los pesos están en `~/models/tabpfn_3_5/`; si no, el lineal
   BGE + producto. La columna `t1_model` dice cuál se usó.
6. **Vecinos**: los cinco reclamos de ajuste más parecidos en el índice FAISS.
7. **Conteos**: semanas y días de la tabla nueva sobre la historia congelada
   (`reports/modeling/weekly_counts.csv`, `daily_counts.csv`); los periodos
   que trae la tabla reemplazan a los congelados. Sobre ese panel extendido:
   expectativa e intervalos de M9 (C-A) y de D-D, exceso z, CUSUM y las dos
   reglas de M11 con sus umbrales, y el p-valor de mezcla rara de M10.
8. **Señales de reentrenamiento** (`signals` en `score.json`): fracción de
   novedosos (> 3 % avisa), cobertura 80 % de M9 en las últimas 12 semanas
   completas (< 70 % avisa), semanas con mezcla rara, y Macro-F1 de T1 si la
   tabla trae el motivo real (caída > 5 % relativo avisa). Regla por defecto:
   M9, M10 y umbrales cada trimestre con ventana móvil; clasificadores y
   patrones solo cuando una señal lo pida.

Salidas en la carpeta indicada: `complaints.parquet` y `.csv` (una fila por
reclamo: motivos top-3 con probabilidad, T2/T3, patrón, distancia, novedad,
vecinos y similitud), `weekly.csv`, `daily.csv`, `mixture.csv`, `score.json`
(resumen, manifiesto de artefactos con hashes de DVC, alertas y señales) y
`run.json` (registro para MLflow, corrida `pipeline-score`).

## Cómo correrlo

```bash
# Comprobar la cadena congelada y escribir el manifiesto
uv run --no-sync python -m src.pipeline fit

# Puntuar una tabla (CPU; en Khipu con GPU, el job de abajo)
uv run --no-sync python -m src.pipeline score --input tabla.parquet --output reports/pipeline/<nombre>

# Cortar una tabla de ejemplo de la tabla preparada (fechas [inicio, fin))
uv run --no-sync python -m src.pipeline sample --start 2024-11-04 --end 2024-11-18 --output data/pipeline_demo/tabla.parquet

# En Khipu, con GPU
INPUT=data/pipeline_demo/tabla.parquet OUTPUT=reports/pipeline/<nombre> sbatch scripts/hpc/pipeline_score.slurm
```

`fit` no reentrena: imprime la cadena de etapas en orden con el comando de
cada una (P1–P6 por DVC y los jobs de Khipu de M4 a M9D), dice qué artefacto
falta (`dvc pull` los trae) y escribe `reports/pipeline/fit.json`.

## Límites

- Las semanas y días incompletos de la tabla se puntúan igual; `is_complete`
  en `weekly.csv` dice cuáles son semanas enteras. La cobertura de la señal
  solo usa semanas completas.
- El CUSUM arrastra su estado desde la historia congelada; si la tabla nueva
  trae un hueco de semanas, las semanas ausentes cuentan como cero.
- La regla diaria D-11 no se adoptó (§28.3): `daily.csv` trae expectativa e
  intervalos, no alarmas.
- Un producto desconocido para M4 entra a TabPFN con el primer código de
  producto; la columna `known_product` lo marca.
