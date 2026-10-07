# Metodología del triaje de reclamos

Qué construimos, por qué lo hicimos así y qué muestran los resultados. Está
escrito para alguien que no siguió el proyecto día a día. Las cifras por
candidato están en el [catálogo de modelos](modelos.md); el detalle de cada
regla, en el [registro de decisiones](registro/models_plan.md).

## 1. El problema

Un banco recibe reclamos en texto libre y debe clasificarlos, atenderlos a
tiempo y notar cuándo algo nuevo está pasando. El prototipo apoya a la persona
que hace ese triaje con cinco salidas por reclamo o por semana:

1. El **motivo** más probable del reclamo (y los tres más probables).
2. La probabilidad de que termine con **alguna solución** y con **compensación
   monetaria**, según el historial.
3. Los **reclamos históricos parecidos**, aunque usen otras palabras.
4. El **patrón semántico** al que pertenece, y si es **novedoso**.
5. Cada semana, qué patrones reciben **más reclamos de lo esperado** de forma
   persistente.

No afirma detectar fraude ni medir el plazo interno del banco. La decisión
final es de la persona: el sistema entrega evidencia, no acciones automáticas.

## 2. Los datos

Usamos los reclamos públicos de la CFPB (EE. UU.): 3,837,184 reclamos con
narrativa, de 2015 a 2026, con 21 productos y 90 motivos canónicos. Modelamos
desde 2023, porque la taxonomía y el volumen cambian mucho antes.

Dos rasgos de estos datos condicionan todo el diseño:

- **Reutilización de texto.** 42.8% de las narrativas comparten texto
  normalizado con otra; muchas son plantillas de gestores de reclamos. Un
  modelo que memorice plantillas parece mejor de lo que es.
- **Deriva temporal.** Lo que se reclama y cómo responden las empresas cambia
  de un año a otro. Evaluar con datos del mismo periodo engaña.

Las etiquetas son aproximaciones: el motivo y las respuestas de la CFPB no
equivalen a las categorías internas de un banco peruano. El método sí se
traslada; las cifras no.

## 3. Cómo evitamos engañarnos

Estas decisiones se tomaron antes de entrenar y no se cambiaron después.

| Decisión | Por qué |
|---|---|
| Tres periodos por fecha: ajuste (2023-01 a 2024-09), calibración (2024-10 a 2024-12) y validación (2025-01 a 2025-06) | El sistema predecirá reclamos futuros, así que se evalúa hacia adelante en el tiempo |
| Los modelos se eligen solo en calibración; validación se abre una vez, con todo congelado | Si se elige mirando validación, el número final es optimista |
| La vista "sin texto compartido" decide: excluye narrativas ya vistas en el periodo de aprendizaje | Quita la ventaja artificial de recordar plantillas |
| Un modelo reemplaza a otro solo si mejora la métrica principal al menos 5% y un bootstrap por semanas (2,000 remuestreos) deja el intervalo 95% entero a su favor | Con 12 semanas de calibración, una mejora pequeña puede ser azar o una sola semana rara |
| Cada regla, umbral y métrica se escribe en el registro antes de correr | Evita ajustar el criterio después de ver el resultado |
| Los modelos bayesianos pasan por una revisión de priors y un piloto antes del run completo | Detecta priors absurdos y problemas de muestreo sin gastar horas de clúster |

Validación se abrió el 2026-10-01 y confirmó T1–T3, M9 y M10. Después se corrió
una ronda de alternativas (bloques A, B y C). Esa ronda también se decidió solo
con ajuste + calibración, pero sus cifras de validación se reportan como **ya
consultadas**: no son evidencia limpia, porque ya conocíamos ese periodo. No
queda otro periodo intacto: 2025-H2 se contaminó en un experimento anterior y
2026 es parcial.

## 4. Qué hace cada componente y por qué

### Motivo del reclamo (T1)

Clasificador de 90 motivos a partir del texto y del producto. El texto se
representa con embeddings BGE-large, un modelo de lenguaje que convierte cada
narrativa en un vector de 1,024 números donde textos de significado parecido
quedan cerca. Un clasificador lineal sobre BGE + producto dio Macro-F1 0.240 en
calibración, contra 0.214 de TF-IDF + producto y 0.076 de asignar el motivo
más frecuente del producto.

Después, **TabPFN-3.5**, un modelo fundacional tabular que aprende "en
contexto" a partir de 20,000 ejemplos sin entrenamiento propio, subió a 0.271
(+12.9%) usando las primeras 100 componentes del embedding más el producto. Es
el vigente, con una advertencia: sus pesos son de uso no comercial, así que
llevarlo al banco requiere licencia. Darle más columnas no ayuda: con las 256
componentes del PCA empata (0.268) y con el embedding completo pierde (0.192).

El top-3 es 95–96% en todos los modelos buenos: para una lista de sugerencias
casi todos sirven; la diferencia está en acertar a la primera.

### Solución y compensación (T2, T3)

Dos clasificadores binarios con TF-IDF del texto y regresión logística
calibrada. BGE no mejoró más del 5% aquí, y tampoco agregar el producto, así
que se quedó la opción más simple. El umbral se fijó en calibración
maximizando F1. En validación: AP 0.570 y 0.270, contra 0.451 y 0.127 de la
regla por producto.

### Respuesta fuera de plazo (T4): descartado

Ningún modelo superó a la regla por producto en validación. Además la
etiqueta de la CFPB no mide el plazo peruano. Lo reemplaza una regla sin
modelo: los días hábiles que faltan sobre 15 (o 45 con extensión), con los
feriados oficiales de gob.pe.

### Reclamos parecidos

Un índice FAISS exacto sobre los embeddings BGE de ajuste. Para un reclamo
nuevo devuelve los diez más cercanos por similitud coseno. El vecino más
cercano comparte producto 77–83% de las veces y motivo 37–48%: sirve para
mostrar precedentes, no para clasificar.

### Patrones semánticos

Agrupamos los reclamos de ajuste en 40 patrones: BGE → PCA a 256 dimensiones
(92% de la varianza) → UMAP a 15 dimensiones → k-means. Probamos HDBSCAN y
CURE sobre el mismo espacio: HDBSCAN dejaba 84% de los reclamos sin grupo y
CURE juntaba casi todo en uno. También probamos agrupar sin UMAP: k-means
sobre PCA es inestable (la partición cambia al cambiar la submuestra).

Un reclamo nuevo se asigna al patrón más cercano; si queda más lejos que el
99% de los reclamos de ajuste de ese patrón, se marca como **novedoso** (0.8%
de los casos). Cada patrón se describe con sus palabras más características.

Un hallazgo importante del análisis de sensibilidad: con otro espacio, o solo
con otra semilla de UMAP, salen particiones distintas de calidad parecida. Los
40 patrones son una herramienta operativa, no categorías naturales, y sus
nombres no deben presentarse como fijos.

### Conteo semanal por patrón (M9)

Para saber si un patrón recibe más reclamos de lo normal necesitamos cuántos
esperábamos. El modelo vigente predice el conteo de cada patrón como *total de
la semana × participación esperada*, con una distribución **binomial
negativa** (permite más variación que una Poisson, que daba intervalos
cubriendo solo 16% de los casos) y una dispersión por patrón.

La participación esperada usa una **memoria que decae**: cada semana pesa la
mitad que la siguiente más reciente, así que en la práctica mira las últimas
dos semanas. Ganó a la versión con 4 semanas de igual peso (WIS 30.3 vs 32.4
en calibración, 6.5% menos) y a dos alternativas más sofisticadas: un modelo de
espacio de estados y dos modelos fundacionales de series de tiempo (Chronos-2,
TimesFM 3.0) sin entrenamiento, que fueron peores.

WIS, *weighted interval score*, resume error de la mediana y calidad de los
intervalos: menor es mejor. La cobertura de los intervalos de 80% y 95% está
en 85% y 97%.

### Composición semanal (M10)

Un modelo Dirichlet-multinomial estima la proporción de cada patrón en la
semana, con concentración κ = 826 estimada en ajuste. Complementa a M9: M9
pregunta "¿este patrón trajo más de lo esperado?", M10 pregunta "¿la mezcla
de la semana es rara?". Mejoró el log score de la referencia multinomial de
−461 a −210.

### Alertas de aumento persistente (M11)

Cada semana, para cada patrón, calculamos cuánto se desvió el conteo de lo
esperado por M9, en una escala normal. Un **CUSUM** acumula esas desviaciones y
avisa cuando la suma supera un umbral; después reinicia. Se comparó con una
regla que avisa por una sola semana extrema, con el mismo presupuesto de una
falsa alarma al mes fijado por simulación. CUSUM detecta crecimientos
sostenidos de 10% y 20% semanal en 42% y 71% de los casos simulados, 10 y 17
puntos más que la regla semanal, y avisa sin aumento solo 6.5% de las veces.

## 5. Flujo de un reclamo nuevo

1. Normalizar el texto y calcular su embedding BGE.
2. Sugerir los tres motivos más probables (T1) y las probabilidades de
   solución y compensación (T2, T3).
3. Mostrar los diez reclamos históricos más parecidos.
4. Asignarlo a un patrón y marcar si es novedoso.
5. Calcular los días hábiles que faltan para el plazo.
6. Al cierre de la semana, actualizar los conteos por patrón y correr M9 y
   M11: las alertas van a revisión humana con los reclamos y palabras del
   patrón.

## 6. Resultados

| Componente | Calibración (decide) | Validación 2025-H1 |
|---|---|---|
| Motivo, TabPFN-3.5 vs regla por producto | Macro-F1 0.271 vs 0.076 | 0.270 vs 0.068 (ya consultado) |
| Solución (T2), TF-IDF vs regla | AP 0.609 vs 0.454 | 0.570 vs 0.451 |
| Compensación (T3), TF-IDF vs regla | AP 0.364 vs 0.174 | 0.270 vs 0.127 |
| Conteo semanal, NB memoria corta vs Poisson de 4 semanas | WIS 30.3 vs 43.1 | 129.6 vs 164.2 (ya consultado) |
| Composición, Dirichlet-multinomial vs multinomial | log score −210 vs −461 | −252 vs −3,338 |
| Alertas CUSUM | 1.4 alertas al mes en calibración | 30 alertas en 25 semanas (ya consultado) |

La validación de 2025-H1 es más difícil que la calibración en los conteos:
contiene una ráfaga del 13 de enero en la que un patrón recibió 58% de los
reclamos de la semana. Buena parte de las 30 alertas de M11 (16 de ellas entre
el 27 de enero y el 17 de febrero) son el **eco** de esa ráfaga: M9 esperó de
más para ese patrón y de menos para los otros en las semanas siguientes. Es
una limitación conocida; la memoria corta la atenúa pero no la corrige.

## 7. Límites y pendientes

- Las cifras de validación posteriores al 2026-10-01 están contaminadas por
  haberla abierto antes. 2026 se reserva, sin abrir, como test final de todo
  el sistema cuando su cobertura esté completa; su versión parcial está
  sesgada por el retraso de publicación de la CFPB.
- TabPFN-3.5 y TimesFM 3.0 tienen licencias no comerciales. Un experto debe
  validar esa lectura antes de proponerlos al banco.
- El eco de ráfagas en M9 y M11.
- El paso de las sugerencias a acciones de triaje (M12) está pendiente y
  requiere definiciones del banco: equipos, mapeo de motivos a equipos,
  presupuesto de alertas revisables al mes.
- Kumo Tabular y Kumo Relational (NVIDIA) no cupieron en la memoria del
  clúster; quedan como prueba futura.

## 8. Dónde está cada cosa

| Documento | Para qué |
|---|---|
| [`docs/metodologia.md`](metodologia.md) | Este documento: qué y por qué |
| [`docs/modelos.md`](modelos.md) | Catálogo: cada candidato, su métrica y su decisión |
| [`reports/modeling/README.md`](../reports/modeling/README.md) | Índice de los archivos de resultados |
| [`docs/registro/models_plan.md`](registro/models_plan.md) | Registro de reglas fijadas antes de correr; bitácora, no lectura |
| [`scripts/hpc/README.md`](../scripts/hpc/README.md) | Cómo se ejecuta cada etapa en el clúster |

Las corridas se registran en MLflow (DagsHub) con commit, hash de datos,
semilla y métricas; los artefactos grandes van por DVC.
