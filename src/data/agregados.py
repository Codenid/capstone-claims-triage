"""Agregados pequeños para el informe (E2, E3, E4, E7, E8, E9 y la regla D1b).

El informe de Quarto **nunca** lee el parquet de 1.6 GB: lee los CSV que produce
este módulo. Cada etapa del pipeline deja un artefacto de unos pocos KB en
``reports/artefactos/``, y el informe se limita a graficarlos. Rendirlo cuesta
segundos, no minutos, y funciona en la máquina de cualquiera que tenga los
artefactos aunque no tenga el dato crudo.

Uso:
    uv run python -m src.data.agregados
"""

from __future__ import annotations

import pandas as pd

from src.paths import REPORTS, asegurar
from src.data.load_raw import cargar

ARTEFACTOS = REPORTS / "artefactos"

# Regla de etiquetado de la decisión D1b del plan. Los dos casos con ``None``
# se excluyen del entrenamiento: no son negativos, son ausencia de resultado.
REGLA_RESPUESTA: dict[str, tuple[int | None, int | None, str]] = {
    "Closed with explanation": (0, 0, "resuelto sin compensar"),
    "Closed with non-monetary relief": (1, 0, "compensación no económica"),
    "Closed with monetary relief": (1, 1, "compensación económica"),
    "Untimely response": (0, 0, "la empresa no respondió a tiempo"),
    "Closed": (0, 0, "categoría heredada, sin detalle"),
    "In progress": (None, None, "censura: aún sin resultado"),
}

COL_FECHA = "Date received"
COL_RESP = "Company response to consumer"
COL_PLAZO = "Timely response?"
COL_PRODUCTO = "Product"
COL_ISSUE = "Issue"
COL_EMPRESA = "Company"
COL_ZIP = "ZIP code"
COL_ESTADO = "State"

TOP_ISSUES = 15

# Ventana de entrenamiento de la decisión D3. 2026 queda fuera por D3b: no es
# un año incompleto, es otro proceso generador. Cada etapa que segmenta por
# régimen reporta las dos vistas — "historico" (todo el corpus, para poder
# reconciliar con cifras publicadas antes) y "2023_2025" (la que se entrena).
REGIMEN_INICIO = 2023
REGIMEN_FIN = 2025
REGIMENES = ("historico", "2023_2025")

# Cortes de la cola de empresas (E7): número de reclamos por empresa.
CORTES_COLA = [(0, 10), (10, 100), (100, 1_000), (1_000, 10_000), (10_000, None)]

TOP_EMPRESAS = 25
TOP_GEO = 20


def _base() -> pd.DataFrame:
    """Las 8 columnas necesarias, con fecha y etiquetas ya derivadas.

    Ninguna es la narrativa: eso la mantiene por debajo de 1 GB y permite que
    todas las etapas baratas (E2, E3, E4, E7, E8, E9) salgan de una sola
    lectura del parquet. La narrativa se procesa aparte, por lotes, en
    ``src/data/narrativa.py``.
    """
    df = cargar(
        [
            COL_FECHA, COL_PRODUCTO, COL_ISSUE, COL_RESP, COL_PLAZO,
            COL_EMPRESA, COL_ZIP, COL_ESTADO,
        ]
    )
    df = df.astype({c: "string" for c in df.columns})

    fecha = pd.to_datetime(df[COL_FECHA], format="%Y-%m-%d", errors="coerce")
    resp = df[COL_RESP]

    return pd.DataFrame(
        {
            "fecha": fecha,
            "anio": fecha.dt.year,
            "mes": fecha.dt.to_period("M").astype("string"),
            "producto": df[COL_PRODUCTO],
            "issue": df[COL_ISSUE],
            "empresa": df[COL_EMPRESA],
            "zip": df[COL_ZIP],
            "estado": df[COL_ESTADO],
            "respuesta": resp,
            "relief": resp.map({k: v[0] for k, v in REGLA_RESPUESTA.items()}).astype(
                "Float64"
            ),
            "monetario": resp.map({k: v[1] for k, v in REGLA_RESPUESTA.items()}).astype(
                "Float64"
            ),
            "fuera_plazo": (df[COL_PLAZO] == "No").astype("Int64"),
        }
    )


def vista(df: pd.DataFrame, regimen: str) -> pd.DataFrame:
    """Subconjunto de ``df`` correspondiente a un régimen de la decisión D3.

    ``historico`` es todo el corpus, 2026 incluido; ``2023_2025`` es la ventana
    que se entrena. Nunca se mezclan en la misma cifra sin decirlo.
    """
    if regimen == "historico":
        return df
    if regimen == "2023_2025":
        return df[df["anio"].between(REGIMEN_INICIO, REGIMEN_FIN)]
    raise ValueError(f"régimen desconocido: {regimen}")


def d1b_respuesta_empresa(df: pd.DataFrame) -> pd.DataFrame:
    """Los 6 valores de `Company response to consumer` y su etiqueta."""
    conteo = df["respuesta"].value_counts(dropna=False)
    filas = []
    for valor, n in conteo.items():
        etiqueta = valor if pd.notna(valor) else "(nulo)"
        t2, t3, motivo = REGLA_RESPUESTA.get(
            valor if pd.notna(valor) else "", (None, None, "sin etiqueta")
        )
        filas.append(
            {
                "respuesta": etiqueta,
                "n": int(n),
                "pct": round(100 * int(n) / len(df), 4),
                "t2_relief": "excluir" if t2 is None else t2,
                "t3_monetario": "excluir" if t3 is None else t3,
                "motivo": motivo,
            }
        )
    return pd.DataFrame(filas).sort_values("n", ascending=False).reset_index(drop=True)


def e02_volumen_mensual(df: pd.DataFrame) -> pd.DataFrame:
    """Reclamos por mes: cobertura temporal y quiebres de volumen."""
    v = df.groupby("mes", dropna=True).size().reset_index(name="n_reclamos")
    return v.sort_values("mes").reset_index(drop=True)


def e03_producto_anio(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva de taxonomía: conteo de `Product` crudo por año."""
    t = df.groupby(["producto", "anio"], dropna=True).size().reset_index(name="n")
    t["pct_del_anio"] = 100 * t["n"] / t.groupby("anio")["n"].transform("sum")
    return t.sort_values(["anio", "n"], ascending=[True, False]).reset_index(drop=True)


def e03_issue_anio(df: pd.DataFrame) -> pd.DataFrame:
    """Igual que el anterior, restringido a los `Issue` más frecuentes."""
    top = df["issue"].value_counts().head(TOP_ISSUES).index
    sub = df[df["issue"].isin(top)]
    t = sub.groupby(["issue", "anio"], dropna=True).size().reset_index(name="n")
    t["pct_del_anio"] = 100 * t["n"] / (
        df.groupby("anio").size().reindex(t["anio"]).to_numpy()
    )
    return t.sort_values(["anio", "n"], ascending=[True, False]).reset_index(drop=True)


def e09_tasas_anuales(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa base de T2/T3/T4 por año, con las filas censuradas contadas aparte."""
    g = df.groupby("anio", dropna=True)
    t = pd.DataFrame(
        {
            "n_reclamos": g.size(),
            "n_etiquetables": g["relief"].count(),
            "n_censurados": g.size() - g["relief"].count(),
            "tasa_relief": 100 * g["relief"].mean(),
            "tasa_monetario": 100 * g["monetario"].mean(),
            "tasa_fuera_plazo": 100 * g["fuera_plazo"].mean(),
        }
    ).reset_index()
    for c in ["tasa_relief", "tasa_monetario", "tasa_fuera_plazo"]:
        t[c] = t[c].astype(float).round(2)
    return t


def e09_tasas_producto(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa base por producto crudo, para la evaluación por cortes."""
    g = df.groupby("producto", dropna=True)
    t = pd.DataFrame(
        {
            "n_reclamos": g.size(),
            "tasa_relief": 100 * g["relief"].mean(),
            "tasa_monetario": 100 * g["monetario"].mean(),
            "tasa_fuera_plazo": 100 * g["fuera_plazo"].mean(),
        }
    ).reset_index()
    for c in ["tasa_relief", "tasa_monetario", "tasa_fuera_plazo"]:
        t[c] = t[c].astype(float).round(2)
    return t.sort_values("n_reclamos", ascending=False).reset_index(drop=True)


def _pareto(serie: pd.Series, nivel: str, regimen: str) -> pd.DataFrame:
    """Conteo ordenado de una categórica, con rango y porcentaje acumulado."""
    conteo = serie.dropna().value_counts()
    total = int(conteo.sum())
    return pd.DataFrame(
        {
            "nivel": nivel,
            "regimen": regimen,
            "rango": range(1, len(conteo) + 1),
            "etiqueta": conteo.index.astype("string"),
            "n": conteo.to_numpy(),
            "pct": (100 * conteo.to_numpy() / total).round(4),
            "pct_acum": (100 * conteo.to_numpy().cumsum() / total).round(4),
        }
    ).reset_index(drop=True)


def e04_pareto_issue(df: pd.DataFrame) -> pd.DataFrame:
    """E4 - Desbalance de clases en los dos niveles de la taxonomía declarada.

    Una fila por etiqueta cruda (``Product`` e ``Issue``) y por régimen, con
    rango y acumulado. De aquí salen las tres cifras que decide la Fase 3:
    cuánto pesa la clase mayoritaria, cuántas clases hacen falta para cubrir el
    80% del volumen, y cuántas etiquetas quedan por debajo del umbral de 1000
    filas con el que ninguna métrica por clase es estable.
    """
    partes = [
        _pareto(vista(df, r)[col], nivel, r)
        for r in REGIMENES
        for col, nivel in (("producto", "producto"), ("issue", "issue"))
    ]
    return pd.concat(partes, ignore_index=True)


def _bucket_cola(n: int) -> str:
    """Etiqueta del tramo de volumen al que pertenece una empresa."""
    for bajo, alto in CORTES_COLA:
        if alto is None:
            return f"{bajo:,}+".replace(",", " ")
        if bajo <= n < alto:
            return f"{bajo:,}-{alto - 1:,}".replace(",", " ")
    raise ValueError(n)


def _rangos_curva(n: int) -> list[int]:
    """Rangos espaciados en escala logarítmica.

    Permite dibujar la curva de concentración completa sin guardar las 6,831
    empresas en el artefacto.
    """
    fijos = [1, 2, 3, 5, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000]
    puntos = {r for r in fijos if r <= n}
    for i in range(121):
        puntos.add(max(1, min(n, round(n ** (i / 120)))))
    puntos.add(n)
    return sorted(puntos)


def e07_empresas(df: pd.DataFrame) -> pd.DataFrame:
    """E7 - Concentración del volumen por empresa reclamada.

    Tres bloques en un solo artefacto:

    - ``empresa``: las 25 con más reclamos, con nombre y acumulado;
    - ``curva``: el acumulado en rangos log-espaciados, que es la curva de
      concentración completa sin arrastrar miles de filas al CSV;
    - ``cola``: cuántas empresas y cuántos reclamos hay en cada tramo de
      volumen. Es la que decide el tamaño del bucket ``OTRA`` de la Fase 5.
    """
    filas = []
    for regimen in REGIMENES:
        conteo = vista(df, regimen)["empresa"].dropna().value_counts()
        total = int(conteo.sum())
        n_emp = len(conteo)
        acum = conteo.to_numpy().cumsum()

        for rango, (nombre, n) in enumerate(conteo.head(TOP_EMPRESAS).items(), 1):
            filas.append(
                {
                    "bloque": "empresa", "regimen": regimen, "rango": rango,
                    "empresa": nombre, "n": int(n),
                    "pct": round(100 * int(n) / total, 4),
                    "pct_acum": round(100 * float(acum[rango - 1]) / total, 4),
                    "n_empresas": 1,
                }
            )

        for rango in _rangos_curva(n_emp):
            filas.append(
                {
                    "bloque": "curva", "regimen": regimen, "rango": rango,
                    "empresa": pd.NA, "n": pd.NA, "pct": pd.NA,
                    "pct_acum": round(100 * float(acum[rango - 1]) / total, 4),
                    "n_empresas": rango,
                }
            )

        tramos = (
            conteo.rename("n_reclamos")
            .to_frame()
            .assign(tramo=lambda d: d["n_reclamos"].map(_bucket_cola))
            .groupby("tramo")
            .agg(n_empresas=("n_reclamos", "size"), n=("n_reclamos", "sum"))
            .reset_index()
        )
        orden = {_bucket_cola(bajo): i for i, (bajo, _) in enumerate(CORTES_COLA)}
        tramos = tramos.sort_values("tramo", key=lambda c: c.map(orden))
        for _, t in tramos.iterrows():
            filas.append(
                {
                    "bloque": "cola", "regimen": regimen,
                    "rango": orden[t["tramo"]] + 1, "empresa": t["tramo"],
                    "n": int(t["n"]), "pct": round(100 * int(t["n"]) / total, 4),
                    "pct_acum": pd.NA, "n_empresas": int(t["n_empresas"]),
                }
            )

    return pd.DataFrame(filas)


def _patron_zip(zips: pd.Series) -> pd.Series:
    """Clasifica cada ``ZIP code`` por su forma, no por su valor.

    El CFPB enmascara el código postal de dos maneras distintas y hay que
    contarlas por separado: ``NNNXX`` conserva el prefijo de 3 dígitos -que es
    la unidad geográfica utilizable- y ``XXXXX`` no conserva nada.
    """
    s = zips.astype("string")
    patron = pd.Series("otro", index=s.index, dtype="string")
    patron[s.isna()] = "(nulo)"
    valido = s.notna()
    patron[valido & s.str.fullmatch(r"\d{5}", na=False)] = "NNNNN"
    patron[valido & s.str.fullmatch(r"\d{3}XX", na=False)] = "NNNXX"
    patron[valido & s.str.fullmatch(r"X{5}", na=False)] = "XXXXX"
    return patron


def e08_geografia(df: pd.DataFrame) -> pd.DataFrame:
    """E8 - Qué queda del código postal después del enmascarado, y qué estados.

    Cuatro bloques: la forma de los 3.84 M códigos postales, los prefijos de 3
    dígitos más frecuentes, los estados más frecuentes, y un resumen con las
    cardinalidades. La conclusión operativa es cuántas filas admiten un ZIP-3
    utilizable como feature y cuántas se van a ``SIN_DATO`` con flag.
    """
    filas = []
    for regimen in REGIMENES:
        v = vista(df, regimen)
        total = len(v)
        patron = _patron_zip(v["zip"])

        for clave, n in patron.value_counts().items():
            filas.append(
                {
                    "bloque": "patron_zip", "regimen": regimen, "clave": clave,
                    "n": int(n), "pct": round(100 * int(n) / total, 4),
                    "n_unicos": pd.NA,
                }
            )

        derivable = patron.isin(["NNNNN", "NNNXX"])
        zip3 = v["zip"].where(derivable).str[:3]
        n_deriv = int(derivable.sum())
        for clave, n in zip3.dropna().value_counts().head(TOP_GEO).items():
            filas.append(
                {
                    "bloque": "zip3", "regimen": regimen, "clave": clave,
                    "n": int(n), "pct": round(100 * int(n) / total, 4),
                    "n_unicos": pd.NA,
                }
            )

        estado = v["estado"]
        for clave, n in estado.dropna().value_counts().head(TOP_GEO).items():
            filas.append(
                {
                    "bloque": "estado", "regimen": regimen, "clave": clave,
                    "n": int(n), "pct": round(100 * int(n) / total, 4),
                    "n_unicos": pd.NA,
                }
            )
        n_nulo_estado = int(estado.isna().sum())
        filas.append(
            {
                "bloque": "estado", "regimen": regimen, "clave": "(nulo)",
                "n": n_nulo_estado, "pct": round(100 * n_nulo_estado / total, 4),
                "n_unicos": pd.NA,
            }
        )

        resumen = {
            "filas": (total, pd.NA),
            "zip3_derivable": (n_deriv, int(zip3.dropna().nunique())),
            "zip_crudo": (int(v["zip"].notna().sum()), int(v["zip"].nunique())),
            "estado": (total - n_nulo_estado, int(estado.nunique())),
        }
        for clave, (n, unicos) in resumen.items():
            filas.append(
                {
                    "bloque": "resumen", "regimen": regimen, "clave": clave,
                    "n": n, "pct": round(100 * n / total, 4), "n_unicos": unicos,
                }
            )

    return pd.DataFrame(filas)


def construir() -> dict[str, pd.DataFrame]:
    asegurar(ARTEFACTOS)
    print("cargando 8 columnas del parquet crudo ...", flush=True)
    df = _base()
    print(f"  {len(df):,} filas\n", flush=True)

    salidas = {
        "d1b_respuesta_empresa": d1b_respuesta_empresa(df),
        "e02_volumen_mensual": e02_volumen_mensual(df),
        "e03_producto_anio": e03_producto_anio(df),
        "e03_issue_anio": e03_issue_anio(df),
        "e04_pareto_issue": e04_pareto_issue(df),
        "e07_empresas": e07_empresas(df),
        "e08_geografia": e08_geografia(df),
        "e09_tasas_anuales": e09_tasas_anuales(df),
        "e09_tasas_producto": e09_tasas_producto(df),
    }

    for nombre, tabla in salidas.items():
        destino = ARTEFACTOS / f"{nombre}.csv"
        tabla.to_csv(destino, index=False, encoding="utf-8")
        print(f"  {nombre:<26} {len(tabla):>5} filas -> {destino.name}")

    return salidas


if __name__ == "__main__":
    construir()
