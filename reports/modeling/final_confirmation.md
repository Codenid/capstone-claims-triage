# Confirmación final en validación

<!-- markdownlint-disable MD013 -->

## Pregunta

¿Los modelos congelados siguen superando a sus referencias en validación
(2025-H1)? Las reglas de `models_plan.md` §22 se aprobaron antes de abrirla.

## Ejecución

| Campo | Valor |
|---|---|
| Código | Commit `5f9853d`, `src/evaluation/final_confirmation.py` |
| Ensayo en calibración | SLURM 53921: 5 min 13 s, 7.9 GB; reprodujo todos los resultados publicados y no guardó nada |
| Run en validación | SLURM 53923: 16 min 56 s, 16.3 GB |
| Reporte | `reports/modeling/final_confirmation.json` |
| MLflow | `8862d1ec8a9c429b8925ae48ec993616`, el único run de la confirmación |
| Datos | T1 y T4: 564,813 reclamos sin texto compartido; T2 y T3: 563,148; 27 semanas. M9 y M10: 25 semanas completas |
| Bootstrap | Pareado por semanas, 2000 remuestreos, semilla 42 |

## Resultados

Clasificadores, en validación sin texto compartido:

| Objetivo | Modelo congelado | Modelo | Regla por producto | Diferencia [IC 95%] | Resultado |
|---|---|---:|---:|---|---|
| T1 — Macro-F1 | BGE + producto | 0.2267 | 0.0684 | +0.1584 [+0.1557, +0.1631] | Confirmado |
| T2 — precisión promedio | TF-IDF solo texto | 0.5704 | 0.4509 | +0.1195 [+0.1036, +0.1380] | Confirmado |
| T3 — precisión promedio | TF-IDF solo texto | 0.2695 | 0.1272 | +0.1423 [+0.1161, +0.1755] | Confirmado |
| T4 — precisión promedio | TF-IDF + producto | 0.0781 | 0.1208 | −0.0427 [−0.0727, −0.0185] | No confirmado |

Métricas secundarias, que no deciden. Precisión y cobertura (recall) usan el
umbral elegido en calibración:

| Objetivo | Modelo | Regla por producto |
|---|---|---|
| T1 — motivo correcto en top-3 | 95.9% | 91.0% |
| T2 — precisión y cobertura | 50.5% y 80.7% | 44.7% y 92.2% |
| T3 — precisión y cobertura | 21.0% y 54.5% | 11.8% y 80.5% |
| T4 — precisión y cobertura | 13.4% y 12.2% | 18.4% y 41.4% |

Modelos semanales, en 25 semanas:

| Pieza | Modelo | Referencia | Diferencia [IC 95%] | Cobertura 80% y 95% | Resultado |
|---|---:|---:|---|---|---|
| M9 — WIS, NB-R4-H v3 | 133.63 | 164.22 (B1-R4) | −30.59 [−49.94, −17.07] | 76.7% y 88.7% | Confirmado |
| M10 — log score conjunto, DM-R4 | −251.7 | −3337.9 (B2-R4) | +3086 [+556, +7675] | 83.4% y 91.7% | Confirmado |

## Decisión

Según las reglas de §22:

- **T1, T2, T3, M9 y M10 se confirman** y se mantienen.
- **T4 no se confirma.** En producción se usa la regla por producto (M3). Es el
  resultado que M4 ya había visto en 2025-H1, por eso su validación no era
  independiente (§21).
- Ningún modelo se eligió ni se cambió mirando validación.

## Observaciones

- De calibración a validación los clasificadores bajan algo: T1 pasa de 0.240 a
  0.227, T2 de 0.609 a 0.570 y T3 de 0.364 a 0.270. Aun así mantienen una
  ventaja clara sobre la regla por producto.
- En M9, el WAPE sube de 12.7% a 26.4% en NB-R4-H v3 y de 12.5% a 26.4% en
  B1-R4: 2025-H1 fue más difícil de predecir para los dos. La ventaja de
  NB-R4-H v3 está en sus intervalos, porque su media es la de B1-R4.
- La cobertura de 95% de M9 (88.7%) quedó justo sobre el mínimo de 88%: en
  2025-H1 sus intervalos fueron algo estrechos. Conviene vigilarla en
  producción.
- En M10, B2-R4 falla con mucha seguridad en algunas semanas con cambios grandes
  de composición. Por eso su log score es tan bajo y el IC de la diferencia es
  ancho, aunque queda entero sobre 0.
- La validación ya está usada. Cualquier evaluación posterior necesita datos
  nuevos, como 2025-H2, que hoy está bloqueado.

## Siguiente paso

Por decisión del usuario del 2026-10-01, los modelos confirmados no se
reentrenan para producción (`models_plan.md` §22). Lo siguiente es M11 (CUSUM).
