# 0007 · Patrones semánticos, conteos semanales, composición y alertas

| | |
|---|---|
| **Fecha** | 2026-10-07 (registro retrospectivo; decisiones del 2026-09-25 al 2026-10-06) |
| **Etapa CRISP-DM** | 4 · Modelado y 5 · Evaluación |
| **Issue** | #17 (PB-17), #18 (PB-18) |
| **Responsable** | @pipaber (DS) |
| **Estado** | aceptada |

## Contexto

El objetivo A1 pide detectar patrones semánticos cuyo volumen aumenta de forma
persistente. No existe etiqueta de «patrón» ni de «aumento»: hay que construir
los grupos, modelar su conteo semanal y fijar una regla de alerta con una tasa
de falsas alarmas declarada.

## Decisiones

- **Patrones (M6–M8).** Embedding BGE-large → PCA 256 (92.4 % de varianza,
  ajustado con 120,000 filas de ajuste) → UMAP 15 (semilla 42) → k-means
  k = 40. Novedad si la distancia al centro supera el percentil 99 del ajuste.
  La ronda §25.3 probó 9 espacios × 15 algoritmos (GMM, HDBSCAN, otras
  dimensiones) con el criterio de *lift* de vecinos: ningún candidato ganó y la
  partición se conserva. Son grupos útiles, no categorías del banco.
- **Conteo semanal por patrón (M9).** Binomial negativa jerárquica condicionada
  al total semanal. Trece candidatos; el vigente es la variante con memoria que
  decae (δ = 0.5, `nb_discounted_hierarchical_v1`), que ganó por la regla de
  [0005](0005-protocolo-de-evaluacion.md): WIS 30.30 contra 32.42 (−6.5 %,
  IC 95 % [−3.36, −0.84]). Espacio de estados, Chronos-2 y TimesFM 3.0 no
  ganaron. En 2025-H1: 129.6 contra 133.6, ya consultado.
- **Composición semanal (M10).** Dirichlet-multinomial con participaciones de
  4 semanas y concentración κ = 826; log score −210 contra −461 de la
  multinomial. Validado, hoy sin consumidor aguas abajo (pendiente decidir su
  rol: señal semanal de «mezcla rara»).
- **Alertas (M11).** Exceso semanal = score normal de la mid-PIT del conteo
  observado en la predictiva de M9. CUSUM (k = 0.5, h = 3.37) elegido frente a
  la regla semanal (z* = 2.53) con el mismo presupuesto de 1 falsa alarma al
  mes en los 40 patrones, porque detecta más crecimientos graduales (42 % y
  71 % contra 32 % y 54 % en los escenarios de +10 % y +20 % semanal).
  Calibración: 1.4 alertas al mes. 2025-H1: 30 alertas en 25 semanas, 16 de
  ellas eco de la ráfaga del 13 de enero.

## Alternativas consideradas

- Pronóstico autónomo de volumen — M9 y M10 reparten el total observado; no
  predicen cuántos reclamos llegarán.
- Alertar por una sola semana extrema — pierde los aumentos graduales, que
  son el caso de uso.
- Mezclas no gaussianas o procesos de Dirichlet — el bloque A mostró que el
  límite es el espacio, no la familia de mezcla.

## Dónde vive

| | |
|---|---|
| **Parámetro** | `configs/modeling.yaml` → `semantic_space`, `clustering`, `weekly_patterns`, `m9_challenge`, `persistent_change`; `configs/weekly_counts/*.yaml`; `configs/weekly_composition/*.yaml` |
| **Código** | `src/models/semantic_space.py`, `src/models/cluster_comparison.py`, `src/models/weekly_patterns.py`, `src/models/weekly_counts/`, `src/models/weekly_composition/`, `src/models/persistent_change.py` |
| **Resultados** | `reports/modeling/README.md` (índice), `reports/modeling/weekly_counts/challenges/`, `reports/modeling/persistent_change_c_a/` |

## Consecuencias

El eco de ráfagas es una limitación conocida: tras una semana con 58 % de los
reclamos en un patrón, M9 espera de más para ese patrón y de menos para los
otros. La memoria corta lo atenúa (16 alertas en el episodio contra 25) pero
no lo corrige.
