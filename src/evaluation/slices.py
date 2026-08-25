"""E11 — Sesgo por perfil del consumidor, controlando por producto.

La cifra cruda dice que un reclamo marcado ``Older American`` termina en
compensación monetaria mucho más seguido que uno sin etiqueta. Tomada así, esa
frase es falsa como afirmación causal y peligrosa como insumo de priorización:
los adultos mayores **no reclaman sobre los mismos productos** que el resto, y
la tasa de compensación depende sobre todo del producto. Sin estratificar, lo
que se mide es la mezcla de productos del grupo, no un trato distinto.

Este módulo calcula las dos cifras y las pone lado a lado:

- **tasa cruda**: la proporción observada dentro del grupo;
- **tasa ajustada**: estandarización directa — la tasa que tendría el grupo *si
  su mezcla de productos fuera la de la población*. Se calcula por estrato de
  producto y se reagrega con los pesos de la población de referencia.

La distancia entre ambas es exactamente la parte del efecto crudo que explica la
composición. Lo que sobrevive al ajuste es lo que hay que mirar en serio, y es
lo que alimenta la evaluación por cortes de la Fase 6.

Uso:
    uv run python -m src.evaluation.slices
"""

from __future__ import annotations

import pandas as pd

from src.data.agregados import REGIMENES, REGLA_RESPUESTA, vista
from src.data.load_raw import cargar
from src.paths import REPORTS, asegurar

ARTEFACTOS = REPORTS / "artefactos"

COL_FECHA = "Date received"
COL_PRODUCTO = "Product"
COL_TAGS = "Tags"
COL_ESTADO = "State"
COL_RESP = "Company response to consumer"
COL_PLAZO = "Timely response?"

SIN_TAG = "(sin etiqueta)"
TOP_ESTADOS = 12

# Estratos con menos filas que esto se descartan del ajuste: su tasa es ruido y
# la estandarización directa les daría el peso de la población, no el suyo. Lo
# que se pierde se reporta en la columna ``cobertura``.
MIN_FILAS_ESTRATO = 50

TARGETS = [
    ("relief", "T2 compensó"),
    ("monetario", "T3 monetario"),
    ("fuera_plazo", "T4 fuera de plazo"),
]


def _base() -> pd.DataFrame:
    """Producto, perfil del consumidor y los tres targets. Sin narrativa."""
    df = cargar([COL_FECHA, COL_PRODUCTO, COL_TAGS, COL_ESTADO, COL_RESP, COL_PLAZO])
    df = df.astype({c: "string" for c in df.columns})

    fecha = pd.to_datetime(df[COL_FECHA], format="%Y-%m-%d", errors="coerce")
    resp = df[COL_RESP]

    return pd.DataFrame(
        {
            "anio": fecha.dt.year,
            "producto": df[COL_PRODUCTO],
            "tags": df[COL_TAGS].fillna(SIN_TAG),
            "estado": df[COL_ESTADO],
            "relief": resp.map({k: v[0] for k, v in REGLA_RESPUESTA.items()}).astype(
                "Float64"
            ),
            "monetario": resp.map({k: v[1] for k, v in REGLA_RESPUESTA.items()}).astype(
                "Float64"
            ),
            "fuera_plazo": (df[COL_PLAZO] == "No").astype("Float64"),
        }
    )


def _ajuste(
    df: pd.DataFrame, col_grupo: str, target: str, pesos: pd.Series
) -> pd.DataFrame:
    """Tasa cruda y tasa estandarizada por producto, para cada grupo.

    ``pesos`` es la mezcla de productos de la población de referencia. La tasa
    ajustada de un grupo es el promedio de sus tasas por producto ponderado por
    esa mezcla, no por la suya: eso es lo que significa "controlando por
    producto".
    """
    v = df[df[target].notna()]
    estratos = (
        v.groupby([col_grupo, "producto"], dropna=True, observed=True)[target]
        .agg(n="size", tasa="mean")
        .reset_index()
    )
    estratos = estratos[estratos["n"] >= MIN_FILAS_ESTRATO]

    filas = []
    for grupo, sub in estratos.groupby(col_grupo, observed=True):
        w = pesos.reindex(sub["producto"]).to_numpy()
        cobertura = float(w.sum())
        ajustada = float((sub["tasa"].to_numpy() * w).sum() / cobertura) if cobertura else float("nan")
        crudo = v.loc[v[col_grupo] == grupo, target]
        filas.append(
            {
                "grupo": grupo,
                "n": int(len(crudo)),
                f"{target}_crudo": round(100 * float(crudo.mean()), 3),
                f"{target}_ajustado": round(100 * ajustada, 3),
                f"{target}_estratos": int(len(sub)),
                f"{target}_cobertura": round(100 * cobertura, 2),
            }
        )
    return pd.DataFrame(filas).set_index("grupo")


def _mezcla(df: pd.DataFrame, col_grupo: str) -> pd.DataFrame:
    """Un indicador de mezcla por grupo: cuánto pesa el producto dominante.

    Es la columna que hace visible *por qué* la tasa cruda y la ajustada no
    coinciden, sin obligar a mirar la tabla completa de producto x grupo.
    """
    dominante = df["producto"].value_counts().idxmax()
    g = df.groupby(col_grupo, observed=True)["producto"]
    return pd.DataFrame(
        {
            "pct_producto_dominante": (100 * g.apply(lambda s: (s == dominante).mean()))
            .round(2)
        }
    )


def _dimension(
    df: pd.DataFrame,
    col_grupo: str,
    etiqueta: str,
    regimen: str,
    grupos: pd.Index | None = None,
) -> pd.DataFrame:
    """Ensambla los tres targets y la mezcla para una dimensión de corte.

    ``grupos`` limita las filas que se emiten, pero **no** la población de
    referencia: los pesos de la estandarización y la fila ``(población)`` se
    calculan siempre sobre el corpus completo del régimen. Si se recortaran
    también, cada dimensión se compararía contra un patrón distinto y las dos
    tablas dejarían de ser leíbles juntas.
    """
    pesos = df["producto"].value_counts(normalize=True)
    if grupos is not None:
        df_grupos = df[df[col_grupo].isin(grupos)]
    else:
        df_grupos = df

    partes = [_ajuste(df_grupos, col_grupo, t, pesos) for t, _ in TARGETS]
    tabla = pd.concat(partes + [_mezcla(df_grupos, col_grupo)], axis=1)
    tabla = tabla.loc[:, ~tabla.columns.duplicated()]

    # Fila de referencia: la población entera, sin cortar. La tasa cruda y la
    # ajustada coinciden por construcción, y es la línea contra la que se lee
    # cualquier grupo.
    referencia = {"n": len(df), "pct_producto_dominante": round(
        100 * (df["producto"] == df["producto"].value_counts().idxmax()).mean(), 2
    )}
    for t, _ in TARGETS:
        tasa = round(100 * float(df[t].mean()), 3)
        referencia |= {
            f"{t}_crudo": tasa, f"{t}_ajustado": tasa,
            f"{t}_estratos": int(df["producto"].nunique()), f"{t}_cobertura": 100.0,
        }
    tabla.loc["(población)"] = pd.Series(referencia)

    tabla = tabla.reset_index(names="grupo")
    tabla.insert(0, "dimension", etiqueta)
    tabla.insert(0, "regimen", regimen)
    tabla["pct_del_total"] = (100 * tabla["n"] / len(df)).round(3)
    return tabla.sort_values("n", ascending=False)


def e11_sesgo(df: pd.DataFrame) -> pd.DataFrame:
    """E11 — Cortes por `Tags` y por `State`, crudos y ajustados por producto."""
    partes = []
    for regimen in REGIMENES:
        v = vista(df, regimen)
        partes.append(_dimension(v, "tags", "tags", regimen))

        top = v["estado"].value_counts().head(TOP_ESTADOS).index
        partes.append(_dimension(v, "estado", "estado", regimen, grupos=top))

    columnas = [
        "regimen", "dimension", "grupo", "n", "pct_del_total",
        "pct_producto_dominante",
    ]
    for t, _ in TARGETS:
        columnas += [f"{t}_crudo", f"{t}_ajustado", f"{t}_cobertura", f"{t}_estratos"]
    return pd.concat(partes, ignore_index=True)[columnas]


def construir() -> dict[str, pd.DataFrame]:
    asegurar(ARTEFACTOS)
    print("cargando 6 columnas del parquet crudo ...", flush=True)
    df = _base()
    print(f"  {len(df):,} filas\n", flush=True)

    salidas = {"e11_sesgo": e11_sesgo(df)}
    for nombre, tabla in salidas.items():
        destino = ARTEFACTOS / f"{nombre}.csv"
        tabla.to_csv(destino, index=False, encoding="utf-8")
        print(f"  {nombre:<26} {len(tabla):>5} filas -> {destino.name}")
    return salidas


if __name__ == "__main__":
    construir()
