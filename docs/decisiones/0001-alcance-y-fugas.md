# 0001 · Alcance del triaje y fugas de información

Registro retrospectivo: 2026-10-03. Responsable DS: Piero Palacios.
Responsables de la validación de negocio: Gianmarco Mejía y Edgard Inga
(Product Lead). Fuente: rama `EDA/pipaber`, hasta `4adb309`.

## Decisión existente y evidencia

El proyecto apoya una decisión humana de triaje a partir de una narrativa
pública de CFPB. Los objetivos definidos en el EDA son motivo (T1), relief
monetario o no monetario registrado (T2), relief monetario (T3) y respuesta
no oportuna según CFPB (T4). No son etiquetas de fraude confirmado,
satisfacción, resolución bancaria ni cumplimiento de un plazo interno.

El momento de predicción se sitúa antes de la respuesta empresarial. Las
columnas `Company response to consumer`, `Timely response?` y
`Company public response` no deben entrar como predictores de esos
resultados. `Issue` y `Sub-issue` tampoco deben usarse para predecir el motivo.
Los identificadores sirven para trazabilidad, no como entrada predictiva.
La presencia de originales y objetivos en la tabla preparada no autoriza
entregar todas sus columnas a un modelo.

Evidencia: [notebook EDA, secciones E4 y E9](../../notebooks/01_eda.ipynb),
[resumen del EDA](../resumen-eda.md) y
[propuesta previa de modelos](../modelos.md). Los commits originales
`4107fbe` y `8f4ee4b` documentan las definiciones y el contrato de transformación;
`5413385` y `53b33fd` conservan sus aprobaciones históricas.

## Consecuencia

Los datos alcanzan para explorar un prototipo sobre los resultados históricos
de CFPB, con evaluación temporal y revisión humana. No prueban su rendimiento
en un banco peruano ni cuantifican el ahorro operativo. El desbalance hace
inadecuada la exactitud global como única métrica: el EDA propone Macro-F1 y
top-3 para T1 y métricas de precisión/cobertura y calibración para T2–T4.

## Pendientes de negocio

Los Product Lead deben confirmar el decisor operativo, el dolor cuantificado,
el KPI y umbral de aceptación, el pagador y la viabilidad financiera/comercial.
El one-pager y el pitch se incorporarán después, según lo indicado por Piero
el 2026-10-03. Esta bitácora recoge decisiones existentes; no constituye una
nueva aprobación de negocio ni un cierre retroactivo del Sprint 1.
