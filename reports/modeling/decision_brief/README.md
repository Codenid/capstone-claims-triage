# Documento de decisión: once alertas, dos cambios

Simula la lectura del panel del lunes 10 de febrero de 2025 y explica por qué
solo 2 de las 11 alertas son aumentos reales: las demás son una ráfaga de
reclamos de transferencias de enero o su eco en la referencia de 4 semanas
(`models_plan.md` §26). Está escrito para quien decide, no para el equipo
técnico.

## Archivos

| Archivo | Qué es |
|---|---|
| `once_alertas.qmd` | Fuente en Quarto: el texto y las cifras, que se calculan al generar |
| `once_alertas.html` | Página generada, autocontenida |
| `brief_lib.py` | Formato de cifras (un solo redondeo, hacia arriba en el medio) y gráficos SVG |
| `brief_parts.py` | Tablas, tarjetas y recuadros en HTML |
| `brief.css`, `brief.js`, `brief-head.html`, `brief-after.html` | Estilos claro y oscuro, tooltips y fuentes |
| `data/facts.json` | Cifras a precisión completa, escritas por `src/triage/brief_facts.py` |
| `data/pattern_names.json`, `data/pattern_families.json` | Nombres y familias de los 40 patrones, propuestos con IA |

La ficha del reclamo se lee de `reports/modeling/triage_demo/cards.json`.

## Cómo se calculan las cifras

`src/triage/brief_facts.py` tiene dos pasos:

1. En Khipu, `--export DIR` escribe los reclamos de 2023 a 2025-H1 sin
   narrativas, con su patrón de M8, y el historial semanal de patrones de M12.
2. Con esas dos exportaciones, el posterior de M9 y los reportes congelados,
   escribe `data/facts.json`. No redondea nada.

`scripts/hpc/m12_brief_facts.slurm` hace los dos pasos (ver
`scripts/hpc/README.md`). La cuenta sin el eco está en `src/triage/echo.py`: el
mismo cálculo de M9 y M11, sin el patrón de la ráfaga. Con todos los patrones
reproduce la rareza del sistema con un error menor que 1e-14.

Una auditoría independiente recalculó las cifras desde los datos crudos, y una
revisión estadística y una prueba de lectura revisaron el argumento. El
criterio que separa el eco de los aumentos reales se definió después de ver el
episodio: es un diagnóstico, no un modelo validado.

## Cómo generar la página

Desde la raíz del repositorio, con el entorno del proyecto y Quarto 1.8:

```bash
QUARTO_PYTHON=.venv/Scripts/python.exe quarto render reports/modeling/decision_brief/once_alertas.qmd
```

En Linux, `QUARTO_PYTHON=.venv/bin/python`. No necesita Khipu ni internet,
salvo las fuentes de Google, que se cargan al abrir la página.

Preparado con apoyo de IA generativa: deben validarlo expertos del negocio y
de Legal antes de usarlo para decidir.
