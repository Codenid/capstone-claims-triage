<div align="center">

# 🚀 Etapa 6 · Prototipo ejecutable por un tercero

**Sprint 3 · Etapas 4–5 (+6) · cierre sáb 31-oct · 23:59**

![CRISP-DM](https://img.shields.io/badge/CRISP--DM-etapa_6-2E7D32?style=for-the-badge) ![Pipeline](https://img.shields.io/badge/pipeline-fit_once_·_score_many-0194E2?style=for-the-badge)

</div>

[← Volver a la portada](https://github.com/utec-dsia/pi1-262-g1#readme)

**Meta.** Un pipeline que ajusta una vez los componentes vigentes y, con una
tabla nueva de reclamos que respete las mismas columnas públicas de la CFPB,
solo predice: motivo, solución y compensación, reclamos parecidos, patrón y
novedad, conteos semanales y diarios con sus intervalos, alertas semanales y
señales de reentrenamiento. Nada se reentrena al puntuar. Es el prototipo de
PB-21 (#21).

## Dónde está

El código vive fuera de esta carpeta porque comparte módulos con las etapas
4–5; esta carpeta es el mapa.

| Qué | Dónde |
|---|---|
| Código del pipeline | `src/pipeline/` (`contract.py`, `frozen.py`, `counts.py`, `signals.py`, `score.py`, `fit.py`, `__main__.py`) |
| Guía de uso y límites | [`docs/pipeline.md`](../docs/pipeline.md) |
| Diseño y decisiones | `docs/registro/models_plan.md` §29 (pre-registro, prueba de punta a punta, hallazgo del patrón blando) |
| Componentes congelados que usa | `docs/registro/models_plan.md` §32.2 y [`docs/decisiones/0005-modelado-y-cierre.md`](../docs/decisiones/0005-modelado-y-cierre.md) |
| Job de clúster | `scripts/hpc/pipeline_score.slurm` |
| Pruebas | `tests/test_pipeline_core.py` |
| Evidencia de la prueba de punta a punta | `reports/pipeline/demo_2024-11-04/score.json` (GPU, TabPFN) y `reports/pipeline/demo_2024-11-04_cpu/score.json` (CPU, lineal de respaldo) |

## Cómo se ejecuta

```bash
# 1. Entorno y artefactos congelados (DVC apunta a DagsHub; pide acceso al equipo)
uv sync --frozen
uv run dvc pull

# 2. Comprobar la cadena congelada y escribir el manifiesto (no reentrena)
uv run python -m src.pipeline fit

# 3. Cortar una tabla de demostración o traer la propia con las columnas de la CFPB
uv run python -m src.pipeline sample --start 2024-11-04 --end 2024-11-18 --output data/pipeline_demo/tabla.parquet

# 4. Puntuar (GPU recomendada para TabPFN; en CPU usa --skip-tabpfn)
uv run python -m src.pipeline score --input data/pipeline_demo/tabla.parquet --output reports/pipeline/demo --device cuda
```

Salidas en la carpeta indicada: `complaints.csv` (una fila por reclamo),
`weekly.csv`, `daily.csv`, `mixture.csv`, `score.json` (resumen y señales de
reentrenamiento) y `run.json` (registro para MLflow).

## Lo logrado

- Prueba de punta a punta en Khipu sobre dos semanas de calibración (35,265
  reclamos): 12 minutos en GPU con TabPFN, 3 horas en CPU con el clasificador
  lineal de respaldo; cobertura del intervalo de 80 % de M9 de 79 %;
  reproduce la alarma de M11 de esas semanas.
- Hallazgo documentado: la proyección UMAP cambia el patrón de alrededor del
  9 % de los reclamos según el lote; el sustituto determinista probado no
  superó su regla y el patrón queda declarado como etiqueta blanda, con su
  masa de frontera medida por patrón (§31.1, §32.6).
- Señales de reentrenamiento: novedosos > 3 %, cobertura 80 % de M9 < 70 %
  en 12 semanas, semanas de mezcla rara y caída de Macro-F1 > 5 % cuando la
  tabla trae el motivo real.

## Lo que no hace

- No decide por la persona: entrega evidencia; la revisión humana de cada
  alerta sigue siendo necesaria.
- No reentrena: cuando una señal lo pida, se repiten las etapas de
  `docs/pipeline.md` con el protocolo del registro.
- No incluye alerta diaria formal (no adoptada en §28.3 y §32.5): lo diario
  es expectativa, intervalos y una lista ordenada por exceso.
