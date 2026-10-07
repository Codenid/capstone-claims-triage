# 0003 · Importación del historial de EDA y excepción de taxonomías

| | |
|---|---|
| **Fecha** | 2026-10-03 |
| **Etapas CRISP-DM** | 1–2 · Comprensión; 3 · Preparación parcial |
| **Historias relacionadas** | [PB-06 / #6](https://github.com/utec-dsia/pi1-262-g1/issues/6), [PB-09 / #9](https://github.com/utec-dsia/pi1-262-g1/issues/9), [PB-11 / #11](https://github.com/utec-dsia/pi1-262-g1/issues/11) |
| **Responsable** | Piero Palacios · Data Scientist |
| **Estado** | Propuesta; pendiente de revisión del docente `@pshiguihara`, responsable de `.github/` y `.gitignore` en CODEOWNERS |

## Contexto

El repositorio del curso y el repositorio de trabajo tienen historias independientes.
Se propone incorporar los 36 commits alcanzables desde la rama fuente `EDA/pipaber`,
con punta `4adb3098382334515fc29c72b51dafc986504ab3`. La rama de Modeling queda fuera
del alcance: sus commits exclusivos, modelos y resultados no forman parte de esta importación.

La auditoría con la guardia original del curso detectó tres CSV de configuración,
introducidos en `d80b821`. El curso permite ampliar `PERMITIDOS` para metadatos pequeños
mediante una justificación; el diagnóstico está en
[verificar_sin_datos.py](../../.github/scripts/verificar_sin_datos.py).
La revisión de esos archivos corresponde al responsable de CODEOWNERS.
Estos archivos contienen nombres de categorías del CFPB, sus equivalencias propuestas
y combinaciones de etiquetas observadas; no contienen narrativas, identificadores,
fechas, montos, datos personales ni filas individuales de reclamos.

| Ruta exacta | Contenido | Filas, sin cabecera | Bytes en la fuente |
|---|---|---:|---:|
| `configs/issue_taxonomy_v1.csv` | `raw_issue`, `canonical_issue`, `mapping_reason`, `review_status` | 173 | 17 869 |
| `configs/product_taxonomy_v1.csv` | `raw_product`, `canonical_product`, `mapping_reason`, `review_status` | 21 | 2 014 |
| `configs/valid_product_issue_pairs_v1.csv` | `raw_product`, `raw_issue` | 323 | 21 815 |

Los motivos de correspondencia son `identity` y `clear_rename`. Las dos tablas de
equivalencias conservan `proposed_no_business_review`: esta propuesta no las presenta
como aprobadas por negocio. La tabla de pares enumera combinaciones históricas; su
nombre no acredita que todas sean correctas para el futuro.

## Decisión propuesta

Unir las dos historias con un commit de merge que permita historias independientes,
sin rebase, squash ni filtrado del historial fuente. Así se conservan los objetos de
los 36 commits, incluidos sus hashes, autores y fechas. Los ajustes al formato del
curso se añaden en la integración, sin modificar los commits originales de EDA.

Ampliar la lista de metadatos permitidos **solo para las tres rutas exactas** anteriores.
La guardia exige CSV legible en UTF-8, cabeceras exactas y un máximo de 32 KiB por
archivo. Limita las filas a 256, 32 y 512, respectivamente, y cada celda a 200 caracteres
sin controles ni saltos de línea. Mantiene las comprobaciones generales de secretos,
modelos, rutas prohibidas, tamaños y punteros LFS. Otros CSV, incluidas copias con
otro nombre o dentro de otra carpeta, siguen rechazados. `.gitignore` refleja las
mismas tres excepciones y omite las salidas locales de `reports/` y las carpetas de
herramientas `.claude/` y `.codex/`.

## Alternativas consideradas

- Reescribir el historial para convertir las tablas en otro formato o extraerlas:
  cambiaría los hashes originales y perdería la continuidad solicitada.
- Copiar únicamente los archivos a una rama nueva: no conservaría el historial.
- Permitir todos los CSV de `configs/`: abriría una excepción demasiado amplia.
- Mantener la guardia original sin cambios: bloquearía estos tres metadatos en los
  commits históricos aunque se eliminasen del árbol final.

## Dónde vive

| | |
|---|---|
| **Configuración** | `TAXONOMIAS` y `LIMITE_TAXONOMIA` en `.github/scripts/verificar_sin_datos.py`; rutas exactas en `.gitignore` |
| **Código de EDA** | `src/data/apply_taxonomy.py`; archivos de configuración en `configs/` |
| **Historial fuente** | `EDA/pipaber` → `4adb3098382334515fc29c72b51dafc986504ab3` |
| **Etapa DVC** | No aplica: es una decisión de integración de Git, no una nueva transformación de datos |

## Consecuencias y revisión

La excepción permite revisar el EDA con toda su historia sin almacenar el dataset
en Git. Los datos y derivados siguen sujetos a DVC, y los modelos a MLflow.
Los límites de tamaño y esquema acotan la excepción, pero no reemplazan la revisión
humana del significado de nuevas etiquetas ni la aprobación de negocio.

El PR enlaza las historias relacionadas y deja pendiente la revisión de
`@pshiguihara` antes de fusionarse. La auditoría externa con la guardia original seguirá señalando
estas tres rutas hasta que el docente acepte la excepción. Que la guardia modificada
pase no equivale a dicha aprobación, ni declara completas las etapas 1–3 del curso.


La política de directorios de datos se aplica tanto a `datos/` (curso) como a
`data/` (fuente importada), incluidos `interim/` y `processed/`: tampoco se
admiten allí derivados registrados manualmente con punteros `.dvc`.
