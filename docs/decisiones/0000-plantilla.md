# 0000 · Título corto de la decisión

| | |
|---|---|
| **Fecha** | AAAA-MM-DD |
| **Etapa CRISP-DM** | 3 · Preparación |
| **Issue** | #NN |
| **Responsable** | @usuario (rol) |
| **Estado** | propuesta · aceptada · reemplazada por 00NN |

## Contexto

Qué observamos (con la cifra del EDA o del reporte que lo motiva) y por qué hay que decidir algo.

## Decisión

Qué se hace, en una o dos frases. Con la magnitud: «se excluyen las filas con monto negativo (13 % del total)».

## Alternativas consideradas

- Opción A — por qué no.
- Opción B — por qué no.

## Dónde vive

| | |
|---|---|
| **Parámetro** | `params.yaml` → `limpieza.atipicos.factor = 3.0` |
| **Código** | `etapa_3_preparacion/codigo/limpiar.py` |
| **Etapa** | `limpiar` (su `desc` enlaza esta entrada) |
| **Commit** | `abc1234` |

## Consecuencias

Qué cambia aguas abajo (filas, columnas, distribución del objetivo), qué riesgo queda y cómo se revisará.
