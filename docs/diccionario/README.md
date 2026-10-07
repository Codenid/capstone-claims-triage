# 📖 Diccionario de datos

`diccionario.csv` **lo genera el pipeline** (`etapa_3_preparacion/dvc.yaml` → etapa `documentar`): una fila por columna de
la tabla final con tipo, rol, nulos, cardinalidad y un ejemplo. A mano se desincroniza en la primera iteración.

Lo único que escribe el equipo es **la descripción de negocio de cada columna**, en `descripciones.yaml`:

```yaml
# descripciones.yaml · columna: descripción en lenguaje del negocio (unidad, origen, cómo se calcula)
monto: Importe de la transacción en soles, sin IGV, tal como lo registra el core bancario.
edad_meses: Antigüedad del cliente en meses a la fecha de la operación (derivada de fecha_alta).
```

`documentar` la lee y la vuelca en la columna `descripcion`; el informe de limpieza reporta la **cobertura**
(criterio de aceptación 5: 100 % de las columnas finales descritas).
