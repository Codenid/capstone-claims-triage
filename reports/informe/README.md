# Informe del proyecto (Quarto)

Informe para asesor y jurado, en español, planificado en
`docs/registro/models_plan.md` §30. Una sola fuente (`informe.qmd` con una
sección por archivo en `secciones/`) produce el HTML y el PDF por Typst.

Toda cifra se lee al renderizar: `facts.py` carga `reports/modeling/*.json`,
la tabla comparativa del curso y el catálogo de patrones; ninguna se escribe a
mano. Las figuras se dibujan con matplotlib en el propio documento; las series
de los 40 patrones se copian de `reports/modeling/patterns/series/` a
`figuras/` porque Typst solo lee archivos dentro de la carpeta del proyecto.

## Renderizar

Desde la raíz del repositorio, con el entorno `uv` sincronizado y Quarto
≥ 1.8 (trae Typst):

```bash
cd reports/informe && QUARTO_PYTHON=../../.venv/Scripts/python.exe quarto render informe.qmd
```

En Linux o macOS el intérprete es `../../.venv/bin/python`. Salidas:
`informe.html` (con `informe_files/` y `figuras/`, no versionados) e
`informe.pdf` (versionado). Una sola fuente de datos vive en DVC: el resumen
posterior de M10, del que se lee κ; antes de renderizar, `dvc pull` de
`reports/modeling/weekly_composition/dirichlet_multinomial_rolling_4_v1/*/posterior_summary.csv.dvc`.

## Pruebas

`tests/test_informe_facts.py` comprueba que todas las fuentes cargan y que las
cifras que el texto cita existen con el tipo esperado.
