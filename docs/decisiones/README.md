# 📓 Bitácora de decisiones

Una entrada por decisión. **Ninguna herramienta la genera: la escribe el equipo.** Es el eslabón humano de la cadena de
trazabilidad; lo demás (código, parámetro, salida, resultado) lo anota DVC o MLflow en cada ejecución.

```text
DECISIÓN → JUSTIFICACIÓN → CÓDIGO → PARÁMETRO → SALIDA → RESULTADO
   Issue      esta bitácora    src/     params.yaml   dvc.lock   MLflow
```

**Tres referencias cruzadas obligatorias:** el commit cita el Issue · la entrada nombra el parámetro y el script que
aplican la decisión · la etapa de `dvc.yaml` apunta a la entrada en su `desc`. Sin los tres enlaces hay dos mitades
sueltas, no una cadena.

## Cómo se numera

`NNNN-titulo-corto.md`, correlativo, nunca se reutiliza. Las entradas 0001–0003
registran retrospectivamente el EDA y la preparación (fecha de registro
2026-10-03); 0005–0007 registran el modelado (2026-10-07) y citan el
pre-registro `docs/registro/models_plan.md`, donde cada decisión tiene fecha.

| Archivo | Decisión | Dónde se aplica |
|---|---|---|
| [0001-alcance-y-fugas.md](0001-alcance-y-fugas.md) | objetivos, momento de predicción y columnas excluidas como predictores | contrato del EDA y preparación |
| [0002-preparacion-y-particion.md](0002-preparacion-y-particion.md) | conservación, normalización, taxonomía, periodos y elegibilidad | etapas `type_data` a `finalize_prepared` de `../../dvc.yaml` |
| [0003-importacion-historial.md](0003-importacion-historial.md) | historia Git y excepción propuesta para tres metadatos | integración del repositorio; no transforma datos |
| [0004-documentacion-de-la-tabla.md](0004-documentacion-de-la-tabla.md) | diccionario de datos e informe de limpieza generados por el pipeline | etapa `document_prepared`; `params.yaml → documentar` |
| [0005-protocolo-de-evaluacion.md](0005-protocolo-de-evaluacion.md) | periodos de ajuste, calibración y validación; regla de reemplazo; validación única | `configs/modeling.yaml`, `reports/modeling/evaluation_contract.json` |
| [0006-clasificadores-t1-t4.md](0006-clasificadores-t1-t4.md) | representaciones y clasificadores de T1–T4; TabPFN-3.5 en T1; T4 descartado | `src/models/`, `reports/modeling/representation_results.json` |
| [0007-patrones-conteos-y-alertas.md](0007-patrones-conteos-y-alertas.md) | patrones semánticos, conteo y composición semanal, regla de alerta | `src/models/weekly_counts/`, `src/models/persistent_change.py` |
| [0008-resultados-en-dvc-e-importacion.md](0008-resultados-en-dvc-e-importacion.md) | tablas de resultados en DVC e importación en un commit al repositorio del curso | `.gitignore`, `src/evaluation/comparison_table.py` |

Copia `0000-plantilla.md` para cada nueva entrada.

Las seis etapas llevan `desc` hacia su entrada y declaran sus decisiones en
`params.yaml` (0002 y 0004). Los
ejemplos de selección/limpieza/construcción/partición permanecen en
`etapa_3_preparacion/dvc.yaml.example`; esa receta genérica no es el pipeline
activo de CFPB.
