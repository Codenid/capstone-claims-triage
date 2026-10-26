# 0005 · Modelado, evaluación y cierre: qué quedó vigente y por qué

| | |
|---|---|
| **Fecha** | 2026-10-26 (resume decisiones del 2026-09-23 al 2026-10-22) |
| **Etapa CRISP-DM** | 4 · Modelado, 5 · Evaluación, 6 · Despliegue |
| **Issue** | #17 (PB-17), #18 (PB-18), #21 (PB-21), #22 (PB-22), #24 (PB-23), #25 (PB-24) |
| **Responsable** | @pipaber (DS) |
| **Estado** | aceptada |

## Contexto

El prototipo entrega cinco salidas sobre los reclamos públicos de la CFPB:
motivo (T1), solución y compensación (T2, T3), reclamos parecidos y patrón
semántico, conteo semanal y diario por patrón con alertas. Cada candidato se
decidió con reglas escritas antes de correrlo en el registro de pre-registro
(`docs/registro/models_plan.md`, secciones numeradas y congeladas) y se
evaluó hacia adelante en el tiempo, en una vista que excluye el texto
reutilizado (24 % de las narrativas comparten texto con otra). Esta entrada
no repite el registro: lo resume y señala dónde está cada decisión.

## Decisión

Los componentes finales son los de la tabla de §32.2 del registro, con sus
corridas congeladas, hashes de DVC e identificadores de MLflow:

- **T1**: TabPFN-3.5 en contexto (20,000 filas de ajuste, PCA 100 de BGE-large
  más producto), Macro-F1 0.271 en calibración y 0.270 en 2025-H1, frente a
  0.076 de la regla por producto y 0.26 del mejor lineal entrenado. Decidido
  con la regla del 5 % y el bootstrap pareado por semanas (§25.2).
- **T2 y T3**: regresión logística calibrada sobre TF-IDF del texto, AP 0.609
  y 0.364; el producto y los embeddings no superaron el 5 % exigido. **T4**
  descartado al confirmarlo en validación (AP 0.078); lo reemplaza una regla
  de plazo con feriados.
- **Patrones**: PCA 256 → UMAP 15 → k-means de 40 grupos, elegido entre 28
  configuraciones con criterios de aceptación fijados antes (§19); novedad
  por percentil 99 de la distancia al centro.
- **M9**: binomial negativa jerárquica con memoria que decae (δ = 0.5,
  elegido en ajuste), WIS 30.3 en calibración, 25 % mejor que la referencia
  Poisson; ganó a un espacio de estados y a dos modelos fundacionales de
  series (§25.5). **M10**: Dirichlet-multinomial de cuatro semanas con señal
  de mezcla rara. **M11**: CUSUM sobre los excesos de M9 con presupuesto de
  una falsa alarma al mes, elegido por detección simulada (§24).
- **Bloque diario**: D-D (binomial negativa con inflación de ceros y día de
  semana) para la expectativa diaria; la alerta diaria no se adoptó en
  ninguna de sus dos versiones (§28.3, §32.5).

Tras el cierre (§32) se probaron por regla tres candidatos más y los tres se
rechazaron: el voto de vecinos como sustituto de la proyección UMAP
(§31.1), la memoria reponderada de M11 (§31.2) y la alerta diaria con
ventana de tres días (§32.5). El patrón queda declarado como etiqueta blanda
cerca de las fronteras, con su masa de frontera medida (§32.6).

## Alternativas consideradas

- Elegir los modelos en validación — descartado: validación se abrió una vez
  (2026-10-01) y todo lo posterior reporta 2025-H1 como «ya consultado».
- Adoptar candidatos que ganaban por poco (dos en el bloque C, uno en el
  diario) — descartado por la regla del bootstrap; quedan registrados.
- Corregir el eco de las ráfagas reponderando las semanas con alarma —
  probado y rechazado: con memoria de dos semanas se realimenta (§31.2).
- Una mezcla gaussiana anclada en PCA, el ajuste fino de TabPFN, el blanqueo
  de embeddings, la alerta multiescala — quedan como trabajo futuro con
  regla escrita (§32.4), sin corrida.

## Dónde vive

| | |
|---|---|
| **Registro** | `docs/registro/models_plan.md` §19–§32 (reglas, fechas, evidencia) |
| **Catálogo** | `docs/modelos.md` (qué se probó y qué quedó vigente); `docs/metodologia.md` (por qué) |
| **Configuración** | `configs/modeling.yaml` y `configs/{weekly_counts,weekly_composition,daily_counts}/*.yaml` |
| **Código** | `src/models/`, `src/evaluation/`, `src/pipeline/`; pruebas en `tests/` |
| **Resultados** | `reports/modeling/` (índice en su README), `etapa_4_5_modelado_evaluacion/resultados/comparacion_modelos.csv` (186 filas) |
| **Procedencia** | 67 corridas en MLflow (DagsHub) con `git_commit`, `datos_md5` y `version_datos`; artefactos por DVC |
| **Literatura** | `docs/literatura/revision_asta.md` (PB-23), `docs/literatura/theorizer_pb24.md` (PB-24) |
| **Informe** | `reports/informe/informe.pdf` (toda cifra se lee de los archivos al renderizar) |
| **Commit** | el que añade esta entrada; cada decisión cita el suyo en el registro |

## Consecuencias

El pipeline de inferencia (`python -m src.pipeline fit` / `score`, etapa 6)
usa exactamente esta cadena y no reajusta nada; cualquier cambio en un
componente exige una entrada nueva en el registro y un re-render del
informe. Riesgos declarados: etiquetas de la CFPB como aproximación de las de
un banco, licencia no comercial de TabPFN, eco de las ráfagas sin corregir,
patrón blando (alrededor del 9 % de reasignación), validación ya consultada
para las rondas posteriores al 2026-10-01; 2026 queda sellado como prueba
final.
