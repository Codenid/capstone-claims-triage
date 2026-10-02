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
| `data/facts.json` | Cifras verificadas, a precisión completa |
| `data/pattern_names.json`, `data/pattern_families.json` | Nombres y familias de los 40 patrones, propuestos con IA |
| `data/faiss.json` | Dos métricas de FAISS de `semantic_space_results.json` |

La ficha del reclamo se lee de `reports/modeling/triage_demo/cards.json`.

## Cómo generarlo

Desde la raíz del repositorio, con el entorno del proyecto y Quarto 1.8:

```bash
QUARTO_PYTHON=.venv/Scripts/python.exe quarto render reports/modeling/decision_brief/once_alertas.qmd
```

En Linux, `QUARTO_PYTHON=.venv/bin/python`. No necesita Khipu ni internet,
salvo las fuentes de Google, que se cargan al abrir la página.

## De dónde salen las cifras

`data/facts.json` se calculó fuera del repositorio con tres scripts
exploratorios, a partir de:

- el panel semanal congelado de M11 y M12;
- las predicciones de M9 y sus 2,000 muestras de la dispersión (semilla 42);
- los reportes de M11 y de la confirmación final;
- una exportación de los reclamos sin narrativas.

La cuenta sin el eco reproduce la rareza del sistema con un error menor que
1e-14. Una auditoría independiente recalculó las cifras desde los datos
crudos, y una revisión estadística y una prueba de lectura revisaron el
argumento.

El criterio que separa el eco de los aumentos reales se definió después de
ver el episodio. Es un diagnóstico, no un modelo validado.

Preparado con apoyo de IA generativa: deben validarlo expertos del negocio y
de Legal antes de usarlo para decidir.
