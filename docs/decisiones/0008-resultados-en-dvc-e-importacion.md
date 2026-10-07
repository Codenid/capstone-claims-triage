# 0008 · Tablas de resultados en DVC e importación del modelado al repositorio del curso

| | |
|---|---|
| **Fecha** | 2026-10-07 |
| **Etapa CRISP-DM** | 4–5 · Modelado y evaluación (integración) |
| **Issue** | #17 (PB-17), #18 (PB-18) |
| **Responsable** | @pipaber (DS) |
| **Estado** | aceptada |

## Contexto

El modelado vive en la rama `Modeling/pipaber` de `Codenid/capstone-claims-triage`
(146 commits desde `4adb309`). La guardia del curso
(`.github/scripts/verificar_sin_datos.py`) rechaza cualquier CSV fuera de tres
rutas permitidas y revisa también el historial de cada push y PR. La rama
tenía 73 CSV de resultados (predicciones semanales, métricas, alertas), todos
menores de 2.2 MB.

## Decisión

1. Las 73 tablas pasan a DVC con sus mismas rutas (`dvc add` por archivo,
   punteros `.dvc` en Git, bytes en DagsHub). El código lee las mismas rutas
   después de `dvc pull`. Las corridas nuevas que escriban CSV deben añadirlos
   a DVC antes de confirmar.
2. El repositorio del curso recibe la rama `modeling` como **importación en un
   commit** desde `main`: el historial original contiene los CSV y seguiría
   fallando en un PR; se conserva en Codenid y se cita por SHA (`ace47c7`).
   No se incluye `notebooks/03_modelos_base.ipynb` (trabajo en curso del
   autor, fuera de la entrega).
3. La portada del curso se mantiene en `README.md` con una sección de la
   entrega del modelado; el README técnico pasa a `docs/plan_modelado.md`.
   `.gitignore` conserva las barreras del curso y permite `reports/modeling`
   y `reports/evidence_card`.

## Alternativas consideradas

- Empujar el historial completo — el CI del push pasa el árbol pero todo PR
  a `main` fallaría por el historial.
- Pedir excepciones por nombre para los CSV — requiere revisión del docente
  (`CODEOWNERS`) y rompe la regla de «solo configuración de DVC en Git».
- Convertir los CSV a JSON — los de predicciones superan el límite de 1 MB
  para texto.

## Dónde vive

| | |
|---|---|
| **Código** | `src/evaluation/comparison_table.py` (tabla permitida del curso) |
| **Etiquetas** | `src/evaluation/experiment.py` añade `version_datos`, `datos_md5` y `tipo`; `src/evaluation/backfill_tags.py` las aplicó a las 51 corridas previas |
| **Commit** | `ace47c7` (Codenid, tablas a DVC) · `d60280e` (curso, importación) |

## Consecuencias

Ver un resultado tabular exige `dvc pull`. La rama `modeling` del curso se
actualiza copiando el árbol de Codenid y conservando `README.md`,
`.gitignore`, `.gitattributes` y `.dvcignore` propios; el CI «sin datos en Git»
debe quedar en verde en cada push.
