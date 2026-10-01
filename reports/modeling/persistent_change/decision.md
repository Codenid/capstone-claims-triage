# M11 — Cambios persistentes

<!-- markdownlint-disable MD013 -->

## Pregunta

Cuando un patrón crece varias semanas, ¿avisa antes un CUSUM que la regla por
exceso semanal, con el mismo presupuesto de falsas alertas? Las reglas de
`models_plan.md` §24 se aprobaron antes de calcular nada.

## Ejecución

| Campo | Valor |
|---|---|
| Código | Commit `6ed379b`, `src/models/persistent_change.py` |
| Run de diseño | SLURM 53943: 7 min 53 s, 1.4 GB; solo `fit` + calibración |
| Reporte | `reports/modeling/persistent_change/results.json` |
| MLflow | `9fa81bf1241144d8af9500c8f8367b0c` |
| Lo esperado | NB-R4-H v3 congelado (M9), con sus 2000 muestras del posterior y semilla 42 |
| Comprobación con M9 | Lo esperado coincide (diferencia máxima 1.6e-10) y $P(Y \ge y)$ difiere como máximo 0.037, bajo el límite de 0.06 |
| Datos | 99 semanas con ventana completa (87 de `fit` y 12 de calibración) y 40 patrones |

## Umbrales

Con 1 falsa alerta al mes en total si M9 estuviera bien calibrado, es decir,
0.00575 por patrón y semana:

- CUSUM: $k = 0.5$ y $h = 3.37$, por simulación con 2000 series de 5000
  semanas.
- Regla semanal: $z^* = 2.53$.

## Detección de aumentos artificiales

Porcentaje de casos detectados dentro de 8 semanas y, entre paréntesis, la
mediana de semanas hasta avisar. Cada escenario tiene 3680 casos: 40 patrones
por 92 semanas de inicio.

| Escenario | CUSUM | Regla semanal | Diferencia |
|---|---:|---:|---:|
| Crecimiento 10% semanal | 50.8% (5) | 36.1% (4) | +14.6 |
| Crecimiento 20% semanal | 79.7% (4) | 58.4% (3) | +21.3 |
| Salto 50% | 41.1% (2) | 36.7% (1) | +4.4 |
| Salto 100% | 61.3% (2) | 56.7% (1) | +4.6 |
| Sin aumento | 11.5% (5) | 10.3% (4) | |

## Decisión

- Según §24, **M12 usa CUSUM**: en los dos crecimientos detecta 14.6 y 21.3
  puntos más que la regla semanal, sobre el mínimo de 5.
- Cuando la regla semanal detecta, suele avisar una semana antes, pero detecta
  menos casos.

## Alertas reales

| Periodo | Semanas | CUSUM | Regla semanal |
|---|---:|---:|---:|
| `fit` | 87 | 77 (3.8 al mes) | 63 (3.1 al mes) |
| Calibración | 12 | 6 (2.2 al mes) | 1 (0.4 al mes) |

La lista está en `alerts.csv`, con las palabras representativas de cada
patrón. En calibración, por ejemplo, el patrón 29 (robo de identidad) tuvo 222
reclamos la semana del 2024-12-09, cuando se esperaban 122, y el patrón 32
(pagos atrasados) tuvo 171 la del 2024-12-23, cuando se esperaban 64.

## Más alertas que el presupuesto

Con los umbrales del plan salen más alertas reales que 1 al mes. En las
ventanas sin aumento, CUSUM avisa en el 11.5% de los casos, frente a cerca de
4.6% si M9 estuviera bien calibrado. Un diagnóstico exploratorio en `fit` +
calibración, hecho en local con las mismas funciones, muestra dos causas:

- **Semanas extremas:** en `fit`, el 1.8% de las semanas supera el nivel que
  M9 da con probabilidad 0.6%.
- **Rachas:** la correlación entre los excesos de semanas consecutivas es 0.29,
  y M9 supone que son independientes. Parte de esas rachas pueden ser cambios
  reales.

CUSUM avisó en 35 de los 40 patrones, y los 6 con más alertas suman solo el 25%
de ellas. Ajustar el umbral a cerca de 1 alerta real al mes ($h = 6.11$)
bajaba la detección de los crecimientos a 27% y 53%.

Por decisión del usuario del 2026-10-01 se mantienen los umbrales del plan. El
equipo debe esperar unas 2 a 4 alertas al mes, y algunas serán cambios reales.

## Reporte de validación (2025-H1)

Se corrió una sola vez, después de registrar la decisión, con los umbrales y la
regla congelados. El CUSUM siguió desde su estado al cierre de calibración.

| Campo | Valor |
|---|---|
| Run | SLURM 53944: 29 s, 1.4 GB |
| Reporte | `reports/modeling/persistent_change/validation_results.json` y `validation_alerts.csv` |
| MLflow | `92db9743b24f479fae12dc96bc2c040c` |
| Comprobaciones | Las alertas de `fit` + calibración coinciden con las del diseño; M9 se reproduce en las 4960 filas (cola con diferencia máxima 0.037) |

| Periodo | Semanas | CUSUM | Regla semanal |
|---|---:|---:|---:|
| Validación | 25 | 42 (7.3 al mes) | 36 (6.3 al mes) |

Las alertas se concentran en un episodio:

- 25 de las 42 alertas de CUSUM (60%) caen en las 4 semanas del 2025-01-27 al
  2025-02-17; solo la semana del 2025-02-03 tiene 10.
- 39 de las 42 son de patrones cuyo producto principal es reportes de crédito.
- Fuera de esas 4 semanas quedan 17 alertas en 21 semanas, unas 3.5 al mes,
  parecido a `fit`.

Como lo esperado de M9 parte del total semanal, estas alertas indican que los
patrones de reportes de crédito ganaron participación en esas semanas. Es
coherente con la confirmación final, donde B2-R4 falló en semanas con cambios
grandes de composición. Nada se ajustó después de ver este reporte.

## Siguiente paso

M12 usa CUSUM con estos umbrales para las alertas de patrones emergentes. Las
alertas de un mismo episodio conviene mostrarlas juntas para que el equipo las
revise de una vez.
