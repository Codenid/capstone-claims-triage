"""E5 y E6 — Todo lo que exige leer la narrativa completa, en una sola pasada.

La narrativa son 3.77 GB en memoria: no cabe en un ``value_counts`` ni en un
``DataFrame`` junto al resto. Este módulo la recorre por lotes con
``iter_batches`` y nunca materializa la columna entera. De cada lote se guardan
solo vectores numéricos de una entrada por fila (hash, longitud, conteos), que
en total ocupan ~150 MB para los 3.8 M de reclamos.

Dos etapas del plan salen de la misma pasada porque ambas necesitan el texto:

- **E5, duplicados y plantillas.** El 39% de las filas comparte narrativa con
  otra fila. Eso obliga a que la división train/val/test sea *por grupo*: si la
  misma plantilla aparece en entrenamiento y en prueba, la métrica de prueba
  mide memorización, no generalización (decisión D4).
- **E6, contenido de la narrativa.** Longitud, densidad de enmascarado ``XXXX``
  y **PII residual**. Lo último es un requisito de la decisión D6: antes de
  mandar un solo texto a una API externa hay que saber cuánto dato personal
  sobrevivió al enmascarado del CFPB.

Una segunda pasada, mucho más barata, recupera el texto de ejemplo y el producto
de las plantillas más repetidas: solo lee las filas cuyo hash está en la lista
corta que dejó la primera pasada.

Uso:
    uv run python -m src.data.narrativa
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data.agregados import REGIMEN_FIN, REGIMEN_INICIO, REGIMENES
from src.paths import PARQUET_CRUDO, REPORTS, asegurar

ARTEFACTOS = REPORTS / "artefactos"

COL_TEXTO = "Consumer complaint narrative"
COL_FECHA = "Date received"
COL_PRODUCTO = "Product"

TAM_LOTE = 131_072
TOP_PLANTILLAS = 12
LARGO_MUESTRA = 150

# Tramos de repetición de una misma narrativa (tamaño del grupo).
CORTES_REPETICION = [
    ("1 (sin repetir)", 1, 2),
    ("2", 2, 3),
    ("3-5", 3, 6),
    ("6-10", 6, 11),
    ("11-50", 11, 51),
    ("51-500", 51, 501),
    ("501+", 501, None),
]

# Tramos de longitud en caracteres.
CORTES_LONGITUD = [
    ("<200", 0, 200),
    ("200-499", 200, 500),
    ("500-999", 500, 1_000),
    ("1 000-1 999", 1_000, 2_000),
    ("2 000-4 999", 2_000, 5_000),
    ("5 000+", 5_000, None),
]

# PII residual. El CFPB enmascara con ``XXXX`` antes de publicar, pero el
# enmascarado es imperfecto: estos patrones cuentan qué se le escapó. Se mide
# "textos que contienen al menos una coincidencia", no ocurrencias, porque la
# decisión que alimenta es binaria — si hay que volver a enmascarar o no.
PATRONES_PII = {
    "correo": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "url": r"(https?://|www\.)[A-Za-z0-9]",
    "telefono": r"\(?\d{3}\)?[ .\-]\d{3}[ .\-]\d{4}",
    "ssn": r"\d{3}-\d{2}-\d{4}",
    "digitos_6_o_mas": r"\d{6,}",
    "tarjeta_16_digitos": r"\d{4}[ \-]\d{4}[ \-]\d{4}[ \-]\d{4}",
}

# Marcadores de idioma. Es una heurística deliberadamente burda: sirve para
# decidir si hace falta un detector de idioma de verdad, no para reemplazarlo.
RE_ESPANOL = (
    r"(?i)\b(que|para|porque|pero|cuenta|dinero|tarjeta|banco|"
    r"pago|deuda|nunca|siempre|ellos)\b"
)
RE_ACENTOS = r"[áéíóúñÁÉÍÓÚÑ¿¡]"
MIN_MARCAS_ESPANOL = 4

# Convención del CFPB para los montos: ``{$14000.00}``. No es PII, es una señal
# de feature — cuenta cuántos importes menciona el reclamo.
RE_MONTO = r"\{\$"


def _normalizar(col):
    """Colapsa espacios, recorta y pasa a minúsculas. Todo dentro de Arrow."""
    return pc.utf8_lower(
        pc.utf8_trim_whitespace(pc.replace_substring_regex(col, r"\s+", " "))
    )


def _hash(col) -> np.ndarray:
    """Hash de 64 bits de una columna Arrow de texto.

    El módulo de deduplicación de la Fase 4 usará SHA-1 porque el hash tiene que
    ser estable entre corridas y entre máquinas. Aquí solo hace falta contar
    grupos, y el hash de pandas es un orden de magnitud más rápido. Con 2.6 M
    valores distintos la probabilidad de colisión es ~2e-7: irrelevante para un
    conteo, insuficiente para una clave.
    """
    valores = pd.Series(col.to_numpy(zero_copy_only=False), dtype="object")
    return pd.util.hash_array(valores.to_numpy(), categorize=True)


def _pasada_1() -> dict[str, np.ndarray]:
    """Recorre la narrativa por lotes y devuelve un vector por métrica."""
    archivo = pq.ParquetFile(PARQUET_CRUDO)
    columnas = [COL_TEXTO, COL_FECHA]
    acum: dict[str, list[np.ndarray]] = {}

    def guardar(nombre: str, valores) -> None:
        acum.setdefault(nombre, []).append(np.asarray(valores))

    filas = 0
    for lote in archivo.iter_batches(batch_size=TAM_LOTE, columns=columnas):
        texto = lote.column(COL_TEXTO)
        normalizado = _normalizar(texto)

        guardar("hash_crudo", _hash(texto))
        guardar("hash_norm", _hash(normalizado))
        guardar("longitud", pc.utf8_length(texto).to_numpy(zero_copy_only=False))
        guardar(
            "anio",
            pc.utf8_slice_codeunits(lote.column(COL_FECHA), 0, 4)
            .cast("int16")
            .to_numpy(zero_copy_only=False),
        )
        guardar(
            "n_xxxx",
            pc.count_substring_regex(texto, r"X{2,}").to_numpy(zero_copy_only=False),
        )
        guardar(
            "n_char_x",
            pc.count_substring(texto, "X").to_numpy(zero_copy_only=False),
        )
        guardar(
            "n_monto",
            pc.count_substring_regex(texto, RE_MONTO).to_numpy(zero_copy_only=False),
        )
        for nombre, patron in PATRONES_PII.items():
            guardar(
                f"pii_{nombre}",
                pc.match_substring_regex(texto, patron).to_numpy(zero_copy_only=False),
            )
        guardar(
            "marcas_es",
            pc.count_substring_regex(texto, RE_ESPANOL).to_numpy(zero_copy_only=False),
        )
        guardar(
            "acentos",
            pc.match_substring_regex(texto, RE_ACENTOS).to_numpy(zero_copy_only=False),
        )

        filas += lote.num_rows
        if filas % (TAM_LOTE * 8) == 0:
            print(f"    {filas:,} filas", flush=True)

    return {k: np.concatenate(v) for k, v in acum.items()}


def _mascara_regimen(anio: np.ndarray, regimen: str) -> np.ndarray:
    if regimen == "historico":
        return np.ones(anio.shape, dtype=bool)
    return (anio >= REGIMEN_INICIO) & (anio <= REGIMEN_FIN)


def _pasada_2(hashes_buscados: set[int]) -> dict[int, dict]:
    """Recupera texto de ejemplo y mezcla de producto de unas pocas plantillas."""
    archivo = pq.ParquetFile(PARQUET_CRUDO)
    encontrados: dict[int, dict] = {h: {"texto": None, "productos": []} for h in hashes_buscados}
    buscados = np.fromiter(hashes_buscados, dtype=np.uint64)

    for lote in archivo.iter_batches(
        batch_size=TAM_LOTE, columns=[COL_TEXTO, COL_PRODUCTO]
    ):
        texto = lote.column(COL_TEXTO)
        h = _hash(_normalizar(texto))
        selec = np.isin(h, buscados)
        if not selec.any():
            continue

        idx = np.flatnonzero(selec)
        textos = texto.take(idx).to_pylist()
        productos = lote.column(COL_PRODUCTO).take(idx).to_pylist()
        for i, pos in enumerate(idx):
            reg = encontrados[int(h[pos])]
            if reg["texto"] is None:
                reg["texto"] = textos[i]
            reg["productos"].append(productos[i])

    return encontrados


def _muestra(texto: str | None) -> str:
    """Recorte de una línea del texto, para que la tabla del informe sea legible."""
    if not texto:
        return ""
    plano = " ".join(texto.split())
    return plano[:LARGO_MUESTRA] + ("..." if len(plano) > LARGO_MUESTRA else "")


def e05_plantillas(datos: dict[str, np.ndarray]) -> pd.DataFrame:
    """E5 — Cuánto del corpus es texto repetido y cuáles son las plantillas.

    Tres bloques: el resumen de deduplicación, la distribución del tamaño de
    grupo, y las plantillas más repetidas con un extracto y su producto
    dominante. El tamaño de grupo se cuenta **dentro de cada régimen**: lo que
    condiciona la división de la decisión D4 es la repetición dentro de la
    ventana que se entrena, no la del corpus histórico.
    """
    filas = []
    tops: dict[str, list[tuple[int, int]]] = {}

    for regimen in REGIMENES:
        m = _mascara_regimen(datos["anio"], regimen)
        h_norm = datos["hash_norm"][m]
        n_filas = int(m.sum())

        valores, cuentas = np.unique(h_norm, return_counts=True)
        n_grupos = len(valores)
        n_repetidas = int(cuentas[cuentas > 1].sum())

        # El mismo conteo sobre el texto sin normalizar, para poder decir
        # cuánto agrupa de más la normalización (colapsar espacios y bajar a
        # minúsculas). La diferencia es la que separa la cifra de este informe
        # de la que aparece en versiones anteriores del plan.
        crudos, cuentas_crudo = np.unique(datos["hash_crudo"][m], return_counts=True)
        n_crudos = len(crudos)
        n_repetidas_crudo = int(cuentas_crudo[cuentas_crudo > 1].sum())

        orden = np.argsort(cuentas)[::-1][:TOP_PLANTILLAS]
        tops[regimen] = [(int(valores[i]), int(cuentas[i])) for i in orden]

        resumen = {
            "filas": (n_filas, 100.0),
            "textos_unicos_crudos": (n_crudos, 100 * n_crudos / n_filas),
            "textos_unicos_normalizados": (n_grupos, 100 * n_grupos / n_filas),
            "filas_en_grupo_repetido": (n_repetidas, 100 * n_repetidas / n_filas),
            "filas_en_grupo_repetido_crudo": (
                n_repetidas_crudo, 100 * n_repetidas_crudo / n_filas
            ),
            "grupo_mas_grande_crudo": (
                int(cuentas_crudo.max()), 100 * int(cuentas_crudo.max()) / n_filas
            ),
            "grupo_mas_grande": (int(cuentas.max()), 100 * int(cuentas.max()) / n_filas),
        }
        for clave, (n, pct) in resumen.items():
            filas.append(
                {
                    "bloque": "resumen", "regimen": regimen, "clave": clave,
                    "n_grupos": pd.NA, "n_filas": n, "pct_filas": round(pct, 4),
                    "producto": pd.NA, "pct_producto": pd.NA, "longitud": pd.NA,
                    "muestra": pd.NA,
                }
            )

        for etiqueta, bajo, alto in CORTES_REPETICION:
            sel = cuentas >= bajo if alto is None else (cuentas >= bajo) & (cuentas < alto)
            g = int(sel.sum())
            f = int(cuentas[sel].sum())
            filas.append(
                {
                    "bloque": "repeticiones", "regimen": regimen, "clave": etiqueta,
                    "n_grupos": g, "n_filas": f,
                    "pct_filas": round(100 * f / n_filas, 4),
                    "producto": pd.NA, "pct_producto": pd.NA, "longitud": pd.NA,
                    "muestra": pd.NA,
                }
            )

    buscados = {h for lista in tops.values() for h, _ in lista}
    print(f"  pasada 2: recuperando {len(buscados)} plantillas ...", flush=True)
    detalle = _pasada_2(buscados)

    for regimen in REGIMENES:
        m = _mascara_regimen(datos["anio"], regimen)
        n_filas = int(m.sum())
        for rango, (h, n) in enumerate(tops[regimen], 1):
            info = detalle[h]
            productos = pd.Series(info["productos"], dtype="string")
            modal = productos.value_counts()
            filas.append(
                {
                    "bloque": "plantilla", "regimen": regimen, "clave": f"#{rango}",
                    "n_grupos": 1, "n_filas": n,
                    "pct_filas": round(100 * n / n_filas, 4),
                    "producto": modal.index[0] if len(modal) else pd.NA,
                    "pct_producto": (
                        round(100 * int(modal.iloc[0]) / int(modal.sum()), 1)
                        if len(modal) else pd.NA
                    ),
                    "longitud": len(info["texto"] or ""),
                    "muestra": _muestra(info["texto"]),
                }
            )

    return pd.DataFrame(filas)


def e06_narrativa(datos: dict[str, np.ndarray]) -> pd.DataFrame:
    """E6 — Longitud, densidad de enmascarado, PII residual e idioma.

    El bloque ``pii`` es el que tiene consecuencia inmediata: mientras esas
    cifras no sean cero, ningún texto sale del proyecto sin un segundo
    enmascarado (regla 4 de la decisión D6).
    """
    filas = []

    def agregar(bloque: str, regimen: str, clave: str, n=pd.NA, pct=pd.NA, valor=pd.NA):
        filas.append(
            {
                "bloque": bloque, "regimen": regimen, "clave": clave,
                "n": n, "pct": pct, "valor": valor,
            }
        )

    for regimen in REGIMENES:
        m = _mascara_regimen(datos["anio"], regimen)
        n_filas = int(m.sum())
        largo = datos["longitud"][m]
        xxxx = datos["n_xxxx"][m]
        char_x = datos["n_char_x"][m]
        monto = datos["n_monto"][m]

        agregar("longitud", regimen, "media", valor=round(float(largo.mean()), 1))
        for p in (10, 25, 50, 75, 90, 99):
            agregar("longitud", regimen, f"p{p}", valor=float(np.percentile(largo, p)))
        agregar("longitud", regimen, "max", valor=float(largo.max()))
        agregar("longitud", regimen, "min", valor=float(largo.min()))

        for etiqueta, bajo, alto in CORTES_LONGITUD:
            sel = largo >= bajo if alto is None else (largo >= bajo) & (largo < alto)
            n = int(sel.sum())
            agregar(
                "longitud_bucket", regimen, etiqueta,
                n=n, pct=round(100 * n / n_filas, 4),
            )

        con_mascara = int((xxxx > 0).sum())
        agregar(
            "mascara", regimen, "textos_con_XXXX",
            n=con_mascara, pct=round(100 * con_mascara / n_filas, 4),
        )
        agregar("mascara", regimen, "media_bloques_XXXX", valor=round(float(xxxx.mean()), 2))
        agregar("mascara", regimen, "p50_bloques_XXXX", valor=float(np.percentile(xxxx, 50)))
        agregar("mascara", regimen, "p99_bloques_XXXX", valor=float(np.percentile(xxxx, 99)))
        agregar(
            "mascara", regimen, "pct_caracteres_enmascarados",
            valor=round(100 * float(char_x.sum()) / float(largo.sum()), 2),
        )
        con_monto = int((monto > 0).sum())
        agregar(
            "mascara", regimen, "textos_con_monto",
            n=con_monto, pct=round(100 * con_monto / n_filas, 4),
        )

        for nombre in PATRONES_PII:
            n = int(datos[f"pii_{nombre}"][m].sum())
            agregar("pii", regimen, nombre, n=n, pct=round(100 * n / n_filas, 4))

        es = int((datos["marcas_es"][m] >= MIN_MARCAS_ESPANOL).sum())
        agregar(
            "idioma", regimen, f"marcas_espanol_>={MIN_MARCAS_ESPANOL}",
            n=es, pct=round(100 * es / n_filas, 4),
        )
        ac = int(datos["acentos"][m].sum())
        agregar("idioma", regimen, "con_acentos", n=ac, pct=round(100 * ac / n_filas, 4))

    return pd.DataFrame(filas)


def construir() -> dict[str, pd.DataFrame]:
    asegurar(ARTEFACTOS)
    print("pasada 1: recorriendo la narrativa por lotes ...", flush=True)
    datos = _pasada_1()
    print(f"  {len(datos['longitud']):,} narrativas leidas\n", flush=True)

    salidas = {
        "e05_plantillas": e05_plantillas(datos),
        "e06_narrativa": e06_narrativa(datos),
    }
    for nombre, tabla in salidas.items():
        destino = ARTEFACTOS / f"{nombre}.csv"
        tabla.to_csv(destino, index=False, encoding="utf-8")
        print(f"  {nombre:<26} {len(tabla):>5} filas -> {destino.name}")
    return salidas


if __name__ == "__main__":
    construir()
