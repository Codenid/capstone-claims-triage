# 0005 · Protocolo de evaluación del modelado: periodos, regla y validación única

| | |
|---|---|
| **Fecha** | 2026-10-07 (registro retrospectivo; decisiones del 2026-09-20 al 2026-10-06) |
| **Etapa CRISP-DM** | 5 · Evaluación |
| **Issue** | #18 (PB-18), #17 (PB-17) |
| **Responsable** | @pipaber (DS) |
| **Estado** | aceptada |

## Contexto

El EDA mostró deriva temporal (2025 reúne 31.85 % de los reclamos; T2 baja de
43.2 % a 38.7 % entre 2023–2024 y 2025-H1) y reutilización textual (42.81 % de
las filas comparten texto). Un reparto al azar o una evaluación sobre textos ya
vistos sobreestimaría cualquier modelo. Además, 2025-H2 quedó bloqueado
([0002](0002-preparacion-y-particion.md)).

## Decisión

Tres periodos fijos y una sola regla para toda la rama de modelado:

- **Ajuste** 2023-01-01 a 2024-09-30 (1,067,194 filas elegibles para T1):
  entrena modelos o fija priors.
- **Calibración** 2024-10-01 a 2024-12-31 (167,973 filas sin texto compartido):
  elige modelos, umbrales y priors. Es la única evidencia de selección.
- **Validación** 2025-01-01 a 2025-06-30 (564,813 filas sin texto compartido):
  se abrió una sola vez, el 2026-10-01 (`models_plan.md` §22). Lo decidido
  después se reporta ahí marcado como «ya consultado».
- **Regla de reemplazo** (`models_plan.md` §25.2): un candidato sustituye al
  vigente solo con ≥ 5 % de ganancia relativa en la métrica principal y un
  intervalo bootstrap pareado por semana al 95 % (2,000 remuestreos, semilla 42)
  entero a su favor. Si varios ganan, el de mejor métrica.
- **Test final**: 2026 completo queda sellado hasta cerrar la cobertura
  (decisión del 2026-10-06); `ood_2026_partial` no se consulta.
- Dos vistas de evaluación: completa y sin texto compartido con los periodos de
  referencia. La principal es la segunda.

## Alternativas consideradas

- Validación cruzada aleatoria — mezcla periodos y textos repetidos; mide
  memoria, no generalización.
- Elegir con la validación 2025-H1 — es la práctica que el curso prohíbe
  (`best_score_` como resultado final); por eso existe la calibración.
- Sin umbral mínimo de ganancia — cualquier diferencia significativa justificaría
  un cambio de modelo con costo operativo.

## Dónde vive

| | |
|---|---|
| **Parámetro** | `configs/modeling.yaml` → `evaluation`, `foundation_t1.minimum_relative_gain`, `m9_challenge` |
| **Código** | `src/evaluation/freeze_evaluation.py`, `src/evaluation/final_confirmation.py`, `src/models/representation_comparison.py` |
| **Etapa** | contrato en `reports/modeling/evaluation_contract.json`; resultados en `reports/modeling/final_confirmation.json` |
| **Tabla** | `etapa_4_5_modelado_evaluacion/resultados/comparacion_modelos.csv` (`python -m src.evaluation.comparison_table`) |

## Consecuencias

Las cifras de 2025-H1 posteriores al 2026-10-01 (TabPFN, C-A, M11 rehecho) son
informativas, no evidencia limpia. Toda corrida de MLflow lleva
`version_datos`, `datos_md5` y `git_commit`; hay una sola versión de datos
(`datos-v1`, MD5 `d189a3ae…`), así que el criterio del curso de dos versiones
no se cumple y queda declarado.
