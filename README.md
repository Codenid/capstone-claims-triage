# Triaje de reclamos — rama Modeling

Esta rama nace de `EDA/pipaber` después de completar E0–E10 y P1–P5. Puede
avanzar mientras el pull request de EDA hacia `develop` está en revisión.

## Objetivo

Construir un prototipo reproducible que ayude a una persona a:

- sugerir el motivo de un reclamo;
- estimar resultados históricos registrados por CFPB;
- encontrar reclamos con significado parecido;
- identificar patrones semánticos cuyo volumen o proporción aumenta;
- revisar la evidencia antes de tomar una decisión de triaje.

No afirmaremos detectar fraude confirmado, pérdidas económicas, resolución
bancaria ni el plazo interno de 15 días. Esas etiquetas no están disponibles.

## Entrada aprobada

- Archivo: `data/processed/prepared.parquet`.
- Filas: 3,837,184.
- Columnas: 44.
- Versionamiento de datos: DVC.
- Entrenamiento: `train_2023_2024`.
- Validación temporal: `validation_2025_h1`.
- `holdout_2025_h2`: bloqueado porque no se recuperaron los IDs revisados.
- `ood_2026_partial`: solo podrá usarse para estudiar T1 después de congelar el
  modelo; T2–T4 permanecen bloqueados.

## Objetivos

- **T1 — Motivo:** motivo canónico más probable.
- **T2 — Relief registrado:** compensación monetaria o no monetaria registrada.
- **T3 — Relief monetario:** compensación monetaria registrada.
- **T4 — Respuesta no oportuna:** `Timely response? = No` según CFPB.
- **A1 — Patrón emergente:** grupo semántico cuyo comportamiento temporal merece
  revisión.

## Herramientas y responsabilidades

<!-- markdownlint-disable MD013 -->

| Herramienta | Responsabilidad |
| --- | --- |
| Git | Código, configuración y documentación |
| DVC | Datos, embeddings, índices y otros artefactos grandes |
| MLflow | Parámetros, métricas, gráficos, modelos y comparación de experimentos |
| PyMC | Modelos probabilísticos de volumen y composición temporal |
| HPC con A100 | Embeddings BGE y tareas que demuestren beneficio al usar GPU |

<!-- markdownlint-enable MD013 -->

MLflow será obligatorio desde el primer baseline. Cada ejecución registrará,
como mínimo, el commit Git, hash DVC de los datos, objetivo, periodo, variables,
semilla, parámetros y métricas. Las credenciales y direcciones privadas se
configurarán mediante variables de entorno; no se guardarán en Git.

DVC y MLflow tendrán funciones distintas. DVC conservará los datos grandes y
MLflow permitirá comparar cómo se obtuvo cada resultado sin duplicar el
conjunto completo de reclamos.

## Plan de modelado

- [x] **M0 — Verificar la entrega:** revisar DVC, esquema y periodos; confirmar
  objetivos y restricciones antes de entrenar.
- [x] **M1 — Preparar experimentos:** configurar MLflow, semillas y ejecución
  reproducible local y en HPC.
- [x] **M2 — Congelar la evaluación:** definir el corte interno de calibración,
  las métricas y las vistas completa y sin texto compartido.
- [ ] **M3 — Crear referencias simples:** comparar contra frecuencias globales y
  reglas basadas en producto.
- [ ] **M4 — Entrenar TF-IDF:** evaluar modelos lineales para T1–T4.
- [ ] **M5 — Evaluar BGE:** generar una muestra en la A100 y compararla con
  TF-IDF sobre las mismas filas.
- [ ] **M6 — Buscar patrones semánticos:** crear vecinos FAISS, grupos y una
  medida de novedad.
- [ ] **M7 — Crear series semanales:** congelar grupos y construir conteos sin
  consultar periodos futuros.
- [ ] **M8 — Modelar volumen con PyMC:** usar Negative Binomial y revisar su
  incertidumbre y diagnóstico de muestreo.
- [ ] **M9 — Modelar composición y cambio:** usar Dirichlet-Multinomial como
  comprobación y CUSUM para cambios persistentes.
- [ ] **M10 — Integrar el triaje:** combinar predicciones, vecinos y alertas para
  revisión humana.

## Contrato de ejecución M1

`configs/modeling.yaml` define la semilla y las rutas comunes. Cada ejecución
crea un registro pequeño con:

- commit Git y hash DVC;
- objetivo, periodo y vista evaluada;
- variables de entrada y semilla;
- ejecución local o en Khipu;
- parámetros, métricas y artefactos.

Un **artefacto** es un archivo producido por una ejecución, por ejemplo un
gráfico o una tabla de resultados. MLflow guardará los artefactos pequeños. DVC
guardará los archivos grandes, como embeddings o índices FAISS.

Los nodos SLURM de Khipu no tienen internet. Allí se crea primero un registro
JSON en `reports/modeling/runs/`. Después, desde el nodo de acceso, se publica
en MLflow con `src/evaluation/publish_run.py`. Las credenciales permanecen en
`.env` y `.dvc/config.local`; ninguno de esos archivos se versiona.

## Contrato de evaluación M2

Los periodos quedan fijados antes de entrenar:

- **Ajuste:** 2023-01-01 a 2024-09-30; el modelo aprende aquí.
- **Calibración:** 2024-10-01 a 2024-12-31; aquí se eligen parámetros y el
  umbral de decisión.
- **Validación temporal:** 2025-01-01 a 2025-06-30; mide el comportamiento en
  datos posteriores.

Un **umbral** es el punto desde el cual una probabilidad se convierte en una
predicción positiva. Para T2–T4 se elegirá en calibración el umbral que maximice
F1, que equilibra precisión y cobertura. Después permanecerá fijo en
validación.

Se reportarán dos vistas del mismo modelo:

- **Completa:** incluye todos los casos elegibles.
- **Sin texto compartido:** excluye de cada evaluación las narrativas que ya
  aparecieron en periodos usados para aprender.

La vista sin texto compartido será la principal para elegir modelos, porque
reduce el beneficio artificial de memorizar plantillas. Conservamos la vista
completa porque las plantillas también existen en la operación real. El corte
deja 1,067,194 filas para ajuste, 167,973 para calibración sin texto compartido
y 564,813 para validación sin texto compartido.

T1 conserva 90 motivos en ajuste y no aparecen motivos nuevos en calibración o
validación. Sin embargo, algunos motivos tienen muy pocos ejemplos. Sus
resultados individuales se mostrarán como evidencia descriptiva, no como una
estimación estable.

Las métricas quedan definidas así:

- **Macro-F1 (T1):** calcula F1 por motivo y da el mismo peso a cada motivo.
- **Top-3 (T1):** revisa si el motivo correcto aparece entre tres sugerencias.
- **Average precision o precisión promedio (T2–T4):** resume la relación entre
  precisión y cobertura; es más útil que el porcentaje total de aciertos cuando
  hay pocos positivos.
- **Precisión:** de los casos marcados positivos, cuántos eran positivos.
- **Cobertura o recall:** de los positivos reales, cuántos encontró el modelo.
- **Brier score:** mide el error de las probabilidades; un valor menor es mejor.

Las entradas iniciales serán la narrativa normalizada y el producto canónico.
Empresa, estado y fecha quedan como candidatos que deberán demostrar valor. Se
excluyen `Issue`, respuestas de la empresa y cualquier resultado T1–T4 porque
revelarían lo que intentamos predecir.

El detalle reproducible está en
`reports/modeling/evaluation_contract.json`. Seguimos sin tener un test final
intacto: 2025-H1 es validación temporal del prototipo, mientras 2025-H2 y 2026
no se usarán para elegir modelos.

## Orden de comparación

```mermaid
flowchart TD
    A[Tabla preparada] --> B[Referencias simples]
    B --> C[TF-IDF]
    C --> D{¿BGE aporta valor?}
    D -->|Sí| E[Embeddings BGE en A100]
    D -->|No para clasificación| F[Conservar modelo simple]
    E --> G[FAISS y grupos semánticos]
    F --> G
    G --> H[Conteos semanales]
    H --> I[Negative Binomial con PyMC]
    H --> J[Dirichlet-Multinomial con PyMC]
    I --> K[CUSUM]
    J --> K
    K --> L[Recomendación revisada por una persona]
```

Dirichlet-Multinomial no creará los grupos. Recibirá grupos ya definidos y
comprobará si cambió su proporción conjunta. Negative Binomial medirá si el
volumen absoluto de un grupo es mayor de lo esperado.

## Evaluación

- T1: Macro-F1, top-3 y resultados por motivo.
- T2–T4: precisión promedio, precisión, cobertura y Brier score.
- Patrones: falsas alertas por semana, tiempo de detección, estabilidad y
  revisión de ejemplos.
- Todas las comparaciones usarán las mismas filas y periodos.
- La vista sin texto compartido excluirá narrativas vistas en periodos usados
  como referencia.

No existe un conjunto final intacto: 2025-H1 servirá para validación temporal y
2025-H2 está bloqueado. Los resultados se presentarán como validación de un
prototipo, no como rendimiento final garantizado.

## MLflow y PyMC

Los modelos PyMC requieren registrar más que una sola métrica. Cada ejecución
guardará en MLflow:

- fórmula y variables del modelo;
- priors, que son los supuestos iniciales de las distribuciones;
- número de cadenas, muestras y calentamiento;
- `R-hat`, tamaño efectivo de muestra y divergencias;
- revisión predictiva posterior;
- métricas del backtest temporal;
- resumen de las distribuciones estimadas.

La primera versión usará registro manual para que el código no dependa de una
integración específica entre MLflow y PyMC. Solo construiremos un adaptador
adicional si reduce trabajo real.

## Uso del HPC y la A100

- Los baselines pequeños se validarán localmente.
- La A100 se usará para BGE y, si aporta valor, FAISS con GPU.
- PyMC empezará con una implementación clara y verificable.
- Se probará el backend JAX/NumPyro en la A100 solo si mantiene los mismos
  resultados y mejora el tiempo de muestreo.
- Antes de ejecutar el corpus completo se medirá una muestra y se registrará el
  consumo de tiempo y memoria en MLflow.

Khipu usa SLURM. Los scripts reproducibles y sus instrucciones están en
`scripts/hpc/`.

## Forma de trabajo

1. Empezar por la comparación más simple.
2. Cambiar una decisión experimental a la vez.
3. Registrar cada experimento en MLflow.
4. No consultar H2 ni usar 2026 para escoger modelos.
5. Versionar artefactos grandes con DVC.
6. Ejecutar primero una muestra antes de usar el corpus completo o la A100.
7. No promover un modelo complejo si no demuestra valor.
8. Revisar cada etapa antes de continuar.

## Documentos relacionados

- [Resumen del EDA](docs/resumen-eda.md).
- [Propuesta detallada de modelos](docs/modelos.md).
- [Notebook de preparación](notebooks/02_revision_preparacion.ipynb).
- [Notebook de modelos base](notebooks/03_modelos_base.ipynb).

## Criterio para cerrar esta rama

La rama terminará con:

- experimentos reproducibles y comparables en MLflow;
- modelos evaluados temporalmente;
- artefactos grandes versionados con DVC;
- vecinos y patrones semánticos revisables;
- alertas temporales con incertidumbre explícita;
- límites documentados y decisión humana final.
