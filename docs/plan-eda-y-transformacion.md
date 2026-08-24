# Plan de EDA y transformación de datos — Triaje de reclamos financieros

**Objetivo:** llevar `data/raw/cfpb_reclamos_narrativa.parquet` (3,837,184 × 16, todo `str`)
a un dataset entrenable en `data/processed/`, con los targets que el caso de negocio
realmente puede pagar, sin fugas y con validación temporal.

**Caso de negocio (de las slides):** la jefa de atención al cliente necesita que cada
reclamo entrante llegue (a) clasificado por motivo, (b) priorizado por riesgo, (c) con
semáforo de plazo legal, (d) con borrador de respuesta. El oficial de cumplimiento
necesita causas raíz por producto y mes.

---

## 0. Lo que el negocio pide → lo que el dato permite entrenar

| Necesidad de negocio | Modelo | Target | Tasa base (2025) |
|---|---|---|---|
| Clasificar el motivo | **T1** multiclase | `motivo_canonico` (de `Issue`) | top-1 = 31.4% |
| Priorizar por riesgo de escalar | **T2** binaria | `relief` = `Company response to consumer` ∈ {monetary, non-monetary} | 36.8% |
| Costo económico esperado | **T3** binaria | `relief_monetario` | 1.2% |
| Semáforo de plazo legal | **T4** binaria | `fuera_plazo` = `Timely response? == "No"` | 0.9% |
| Borrador de respuesta + hechos | Agente LLM (no se entrena) | — | requiere set de evaluación con rúbrica |
| Causas raíz por producto/mes | Agregación, no modelo | — | — |

**Bloqueante de alcance (D1):** las slides prometen "riesgo de disputa" con el campo
*Disputa del consumidor*. Ese campo **no está** en las 16 columnas (el CFPB dejó de
publicar `Consumer disputed?` en 2017). O se pide la columna al docente, o el objetivo
se redefine como T2 (`relief` = la empresa tuvo que compensar) y se declara como
**proxy** en la slide. Recomendación: T2, y decirlo explícitamente.

---

## 1. Decisiones de diseño que condicionan todo lo demás

### D2 — Contrato de disponibilidad (anti-fuga). Es la decisión más importante.

| Tier | Campos | Uso permitido |
|---|---|---|
| **F0 — ingreso** | `Date received`, `Consumer complaint narrative`, `Company`, `State`, `ZIP code`, `Tags` | features de todos los modelos |
| **F1 — declarado en formulario** | `Product`, `Sub-product`, `Issue`, `Sub-issue` | **etiqueta** de T1; feature solo en variantes marcadas "con taxonomía declarada" (en la bandeja del banco no existe) |
| **FX — post-respuesta, PROHIBIDO como feature** | `Date sent to company`, `Company response to consumer`, `Company public response`, `Timely response?` | solo targets y diagnóstico |
| **Clave** | `Complaint ID` (3.8 M únicos) | trazabilidad, nunca feature |
| **Fuera** | `Submitted via` (un solo valor: `Web`) | varianza cero, se elimina |

Consecuencia: **no hay ninguna variable numérica de origen**. Toda feature numérica se
construye desde F0 (longitud de narrativa, densidad de `XXXX`, frecuencia histórica de
la empresa, mes, etc.). En particular, *días de gestión* (`Date sent to company` −
`Date received`) es FX: no se conoce al momento de triar, así que queda como variable de
diagnóstico, no como feature.

### D3 — Ventana de entrenamiento: régimen actual, no historia completa

La taxonomía y la tasa base se rompen entre 2022 y 2024:

- `Credit reporting` (2015-17) → `…credit repair services…` (2017-23) → `…or other personal consumer reports` (2023-26).
- `relief`: 20% (2015) → 11% (2020) → 46% (2024) → 37% (2025).
- Volumen: 2025 concentra 1.22 M filas (32% del total); 2023+ = 2.63 M (69%).

**Decisión:** dataset principal = **2023-01 en adelante** (2.63 M filas). La historia
2015-2022 se conserva como experimento de deriva con etiquetas canonicalizadas, no como
train por defecto.

### D4 — Split temporal y por grupo, nunca aleatorio

- `train` 2023-01 → 2024-12 · `val` 2025-01 → 2025-06 · `test` 2025-07 → 2026-07.
- **Group-aware:** el 39.1% de las filas comparte narrativa con otra fila (2.58 M textos
  únicos / 3.84 M filas; la plantilla más repetida aparece 27,496 veces). El hash de la
  narrativa define el grupo: un grupo nunca se parte entre splits.
- Verificar censura a la derecha en los últimos meses: en 2026 `relief` cae a 18.3% y
  `monetario` sube a 6.6%. Antes de usar 2026 como test hay que decidir si son
  respuestas aún pendientes o cambio de composición.

### D5 — Muestra de desarrollo

300 k filas estratificadas por (mes × producto canónico) para iterar; entrenamiento
final sobre el dataset completo. Ambas versionadas con DVC.

---

## 2. Fase 1 — EDA (`notebooks/01_eda_*.ipynb`, figuras a `reports/figures/`)

Cada bloque produce **una figura o tabla reutilizable en las slides**.

| # | Pregunta | Entregable |
|---|---|---|
| E1 | Perfil de tipos, nulos, cardinalidad, memoria | tabla `reports/tabla_tipos.csv` (16 filas) |
| E2 | Volumen por mes y cobertura temporal | serie 2015-2026 con los quiebres marcados |
| E3 | Deriva de taxonomía: `Product`/`Issue` × año | heatmap etiqueta-año; insumo del mapa canónico |
| E4 | Desbalance por nivel | Pareto: `Issue` top-1 31.4%, top-10 78.6%, 84 de 173 etiquetas con <1000 filas |
| E5 | Duplicados y plantillas | histograma de repeticiones + top-10 plantillas y su producto |
| E6 | Narrativa | longitud (media 1,022 car., p50 669, p99 5,873), densidad de `XXXX`, idioma, PII residual |
| E7 | Empresas | concentración: top-5 = 64.3%, top-50 = 84.3%; 3,908 de 6,831 con <10 reclamos |
| E8 | Geografía | `ZIP code`: 861 k valores con `X` (22%), 290 malformados → prefijo de 3 dígitos + flag |
| E9 | Targets candidatos y su deriva | tasa base de T2/T3/T4 por año y por producto |
| E10 | Auditoría de fuga | tabla F0/F1/FX acordada por el equipo |
| E11 | Sesgo | tasas por `Tags` y `State`, **controlando por producto**: `Older American` recibe compensación monetaria 9.6% vs 2.25% sin tag, pero su mezcla de productos es distinta — no concluir sin estratificar |
| E12 | Piso de señal | TF-IDF + LogReg sobre 100 k filas: macro-F1 de referencia antes de invertir en modelos grandes |

**Salida de fase:** `docs/data_card.md` con esquema, fuente, límites conocidos y
decisiones D1-D5 registradas.

---

## 3. Fase 2 — Tipado y contrato de esquema → `data/interim/tipado.parquet`

`src/data/typing.py` + `configs/dtypes.yaml`:

- `Date received`, `Date sent to company` → `datetime64[ns]` (formato `%Y-%m-%d`, validado).
- `Timely response?` → `bool`.
- 9 categóricas de baja cardinalidad → `category`.
- `Company`, `ZIP code` → `category` (alta cardinalidad; el encoding viene en features).
- `Complaint ID` → `int64` si es numérico puro, con test de unicidad.
- `Consumer complaint narrative` → `string[pyarrow]`.
- `Tags` → **dos booleanas** (`es_militar`, `es_adulto_mayor`): hoy empaqueta multi-etiqueta
  en un string (`"Older American, Servicemember"`).
- Nulos explícitos: `Tags` 90.4%, `Company public response` 46.8%, `Sub-issue` 8.7% →
  categoría `SIN_DATO` más un flag booleano donde el nulo sea informativo.

Efecto medido: las 15 columnas no-narrativa pasan de **1.54 GB → 0.29 GB (−81%)**.

Validación con `pandera`: tipos, rangos de fecha, dominio de categóricas, unicidad de ID.
El pipeline falla si el contrato se rompe.

---

## 4. Fase 3 — Canonicalización de taxonomía → `data/interim/canonico.parquet`

`src/data/canonicalize.py` + `configs/taxonomia.yaml` (mapa versionado, revisado a mano).

1. `producto_canonico`: 21 etiquetas crudas → ~11 (unifica las tres variantes de
   *credit reporting*, las dos de *credit card*, las de *payday/personal loan*).
2. `motivo_canonico`: 173 `Issue` → ~25-40 clases con volumen suficiente; las 84
   etiquetas con <1000 filas se agrupan o van a `OTRO` (documentar la regla).
3. Test de regresión obligatorio: **cada etiqueta canónica debe existir en train y en
   test**. Sin esto, entrenar en 2023-24 y evaluar en 2025 produce métricas absurdas.
4. Guardar la trazabilidad `etiqueta_cruda → canónica` para el tablero de cumplimiento.

---

## 5. Fase 4 — Deduplicación y splits → `data/interim/split.parquet`

`src/data/dedupe.py`, `src/data/split.py`:

1. `hash_narrativa` = SHA-1 del texto normalizado; `n_plantilla` = tamaño del grupo.
2. Dos vistas: **completa** (con `peso = 1/n_plantilla`) y **deduplicada** (una fila por
   grupo). Se entrena y reporta en ambas: si difieren mucho, el modelo aprendió plantillas.
3. Split temporal de D4 respetando grupos. Registrar filas descartadas por conflicto de
   grupo entre periodos.
4. Cerca-duplicados (MinHash/LSH) como extensión, solo si el gap completa-vs-dedup lo justifica.

---

## 6. Fase 5 — Features → `data/processed/features_{train,val,test}.parquet`

`src/features/`, todo `fit` **solo en train**, serializado en `models/encoders/`.

- **Texto** (`text.py`): normalización de `XXXX`, minúsculas, TF-IDF (1-2 gramas,
  `min_df=5`) como baseline; embeddings de oración como alternativa. Derivadas numéricas:
  `n_caracteres`, `n_palabras`, `ratio_mayusculas`, `n_xxxx`, `n_montos`, `n_fechas`,
  `tiene_amenaza_legal` (léxico), `n_plantilla`.
- **Categóricas** (`categorical.py`): one-hot para baja cardinalidad; para `Company` y
  `ZIP-3`, codificación por frecuencia y por target con *smoothing* y **calculada solo
  con datos anteriores al periodo de la fila** (evita fuga temporal); bucket `OTRA` para
  la cola de 3,908 empresas con <10 reclamos.
- **Temporales** (`temporal.py`): mes, día de semana, año, semanas desde el inicio del
  régimen, indicador de picos de volumen. No incluye *días de gestión* (es FX).
- **Ensamblado** (`build.py`): matriz por tier — `F0` (producción) y `F0+F1`
  (referencia superior, no desplegable).

Salida validada con `pandera` y una prueba automática que **falla si aparece cualquier
columna FX** en la matriz de features.

---

## 7. Fase 6 — Modelos y métricas

`src/models/`, `src/evaluation/`.

- **Baselines:** clase mayoritaria; reglas por palabra clave; TF-IDF + LogReg/SGD.
- **Siguiente nivel:** LightGBM sobre features tabulares + SVD del TF-IDF; transformer
  ligero (DistilBERT/DeBERTa) afinado si hay GPU.
- **Cascada:** T1 predice motivo → su probabilidad alimenta T2/T3/T4, para que el modelo
  de riesgo no dependa de una taxonomía que en producción no existe.
- **Desbalance:** pesos por clase, `scale_pos_weight`, umbral elegido por costo, no 0.5.

| Modelo | Métrica primaria | Secundarias |
|---|---|---|
| T1 motivo | macro-F1 | top-3 accuracy, F1 por clase, matriz de confusión de las 10 mayores |
| T2 relief | PR-AUC | precisión@10% de la cola, lift del decil superior, ECE (calibración) |
| T3 monetario | PR-AUC | recall al 20% de capacidad revisada |
| T4 plazo | recall con precisión ≥ 0.5 | el costo del falso negativo es la multa |

**Evaluación por cortes, obligatoria:** producto canónico, mes (deriva), estado, top-5
empresas vs cola, `Tags`. Un macro-F1 global que esconde una clase en 0.0 no sirve.

**Predicción selectiva:** curva de abstención → "% de reclamos automatizables con
precisión ≥ 0.90". Esa cifra responde a la pregunta abierta de la slide 2 (qué reclamos
van siempre a revisión humana).

---

## 8. Fase 7 — Métrica de negocio: simulación de la bandeja

Es la evidencia que cierra el caso y se puede calcular sobre el periodo de test:

1. Fijar capacidad: N reclamos atendibles por día (parámetro).
2. Política A = FIFO (el "hoy" de la slide). Política B = orden por riesgo del modelo.
3. Medir: % de reclamos de alto riesgo atendidos dentro del plazo, días hasta primer
   contacto del decil de riesgo alto, reclamos `fuera_plazo` evitados.
4. Traducir a dinero con un costo unitario de multa como parámetro explícito.

Salida: `reports/simulacion_bandeja.md` + figura. Con eso la slide financiera deja de ser
una afirmación y pasa a ser un número con supuestos declarados.

---

## 9. Estructura de código y reproducibilidad

```
configs/     dtypes.yaml · taxonomia.yaml · split.yaml · features.yaml · model_*.yaml
src/data/    load_raw.py · typing.py · canonicalize.py · dedupe.py · split.py
src/features/text.py · categorical.py · temporal.py · build.py
src/models/  baseline.py · train.py · cascade.py
src/evaluation/metrics.py · slices.py · calibration.py · simulation.py · report.py
tests/       test_schema.py · test_no_leakage.py · test_taxonomia.py · test_split.py
```

`dvc.yaml` con etapas encadenadas
`tipado → canonico → dedupe → split → features → train → evaluate → report`,
cada una con `deps`, `outs`, `params` y `metrics`. Así los 5.1 GB del notebook no se
recargan en cada corrida y los números de la slide son reproducibles.

---

## 10. Orden de ejecución y riesgos

**Orden:** D1 (decidir target) → E1-E12 → tipado → canonicalización → dedup/split →
features → baseline → modelos → cortes → simulación. Nada de features antes de tener el
mapa canónico: es la dependencia dura.

| Riesgo | Señal temprana | Mitigación |
|---|---|---|
| El target de la slide no existe | ya confirmado | D1: proxy `relief`, declarado como proxy |
| Fuga por campos post-respuesta | métricas sospechosamente altas | `test_no_leakage.py` |
| Fuga por plantillas duplicadas | gap grande completa vs dedup | split por grupo |
| Deriva de etiquetas y de tasa base | clases ausentes en test | régimen 2023+, split temporal, PSI mensual |
| Censura en los últimos meses | tasa base anómala en 2026 | excluir el margen sin respuesta consolidada |
| Sesgo por `Tags`/`State` | diferencias sin controlar por producto | cortes estratificados en cada evaluación |
| Costo de iteración (3.8 M filas) | notebooks que no terminan | muestra de 300 k + etapas DVC cacheadas |
