"""Utilidades del informe de Quarto.

El informe **no** lee el parquet crudo: lee los CSV pequeños que dejan las
etapas en ``reports/artefactos/``. Este módulo centraliza tres cosas:

- el catálogo de etapas y su estado (hecho / pendiente), que es lo que hace que
  el informe se llene solo a medida que el pipeline avanza;
- la carga de artefactos, tolerante a que todavía no existan;
- el estilo de las figuras, con la paleta validada para daltonismo.

Paleta: slots 1-3 de la paleta categórica de referencia. Los tres pasan los
gates de separación CVD y de visión normal con todos los pares en juego
(ΔE 9.2 CVD, 24.0 normal). El aqua queda por debajo de 3:1 contra el fondo, así
que cada figura que lo use lleva etiqueta directa visible y su tabla debajo.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.paths import REPORTS

ARTEFACTOS = REPORTS / "artefactos"

# --- Paleta ---------------------------------------------------------------
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_3 = "#8a8880"
REJILLA = "#e6e5e1"

SERIE_1 = "#2a78d6"  # azul
SERIE_2 = "#eb6834"  # naranja
SERIE_3 = "#1baf7a"  # aqua
CRITICO = "#d03b3b"  # status: solo para marcar anomalías, nunca como serie

# Rampa secuencial de un solo tono (azul, claro -> oscuro) para el heatmap.
RAMPA = LinearSegmentedColormap.from_list(
    "azul_secuencial",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def estilo() -> None:
    """Aplica el estilo del informe a matplotlib. Idempotente."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SUPERFICIE,
            "axes.facecolor": SUPERFICIE,
            "savefig.facecolor": SUPERFICIE,
            "figure.dpi": 130,
            "savefig.bbox": "tight",
            "font.size": 9.5,
            "text.color": TINTA,
            "axes.labelcolor": TINTA_2,
            "axes.edgecolor": REJILLA,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 10.5,
            "axes.titleweight": "semibold",
            "axes.titlecolor": TINTA,
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": REJILLA,
            "grid.linewidth": 0.8,
            "xtick.color": TINTA_3,
            "ytick.color": TINTA_3,
            "xtick.labelcolor": TINTA_2,
            "ytick.labelcolor": TINTA_2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 2.0,
            "lines.markersize": 4.5,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def sin_marco(ax: plt.Axes) -> plt.Axes:
    """Rejilla recesiva, sin marco superior/derecho, ticks discretos."""
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for lado in ("left",):
        ax.spines[lado].set_visible(False)
    return ax


# --- Catálogo de etapas ---------------------------------------------------


@dataclass(frozen=True)
class Etapa:
    clave: str
    fase: str
    descripcion: str
    artefacto: str
    comando: str

    @property
    def ruta(self):
        return ARTEFACTOS / self.artefacto

    @property
    def hecho(self) -> bool:
        return self.ruta.exists()


ETAPAS: list[Etapa] = [
    Etapa("E1", "EDA", "Perfil de tipos, nulos, cardinalidad y memoria",
          "e01_tabla_tipos.csv", "python -m src.data.profile"),
    Etapa("D1b", "Targets", "Regla de etiquetado de T2/T3 caso por caso",
          "d1b_respuesta_empresa.csv", "python -m src.data.agregados"),
    Etapa("E2", "EDA", "Volumen por mes y cobertura temporal",
          "e02_volumen_mensual.csv", "python -m src.data.agregados"),
    Etapa("E3", "EDA", "Deriva de taxonomía: Product / Issue por año",
          "e03_producto_anio.csv", "python -m src.data.agregados"),
    Etapa("E9", "EDA", "Tasa base de los targets por año y por producto",
          "e09_tasas_anuales.csv", "python -m src.data.agregados"),
    Etapa("E4", "EDA", "Desbalance de clases: Pareto de Product e Issue",
          "e04_pareto_issue.csv", "python -m src.data.agregados"),
    Etapa("E5", "EDA", "Duplicados y plantillas de narrativa",
          "e05_plantillas.csv", "python -m src.data.narrativa"),
    Etapa("E6", "EDA", "Narrativa: longitud, densidad de XXXX, PII residual",
          "e06_narrativa.csv", "python -m src.data.narrativa"),
    Etapa("E7", "EDA", "Concentración de empresas",
          "e07_empresas.csv", "python -m src.data.agregados"),
    Etapa("E8", "EDA", "Geografía: ZIP-3 y malformados",
          "e08_geografia.csv", "python -m src.data.agregados"),
    Etapa("E11", "EDA", "Sesgo por Tags y State, controlando por producto",
          "e11_sesgo.csv", "python -m src.evaluation.slices"),
    Etapa("E12", "EDA", "Piso de señal: TF-IDF + LogReg",
          "e12_piso_senal.csv", "python -m src.models.baseline"),
    Etapa("E13", "EDA", "¿Vale un embedding? TF-IDF vs embedding local",
          "e13_embeddings.csv", "python -m src.evaluation.compare_embeddings"),
    Etapa("E14", "EDA", "Estructura latente: UMAP + HDBSCAN",
          "e14_clusters.csv", "python -m src.data.cluster_evidence"),
    Etapa("F2", "Pipeline", "Tipado y contrato de esquema",
          "f02_tipado_audit.csv", "python -m src.data.typing"),
    Etapa("F3", "Pipeline", "Canonicalización de taxonomía",
          "f03_taxonomia.csv", "python -m src.data.canonicalize"),
    Etapa("F4", "Pipeline", "Repetidos y splits temporales operacional/purgado",
          "f4_splits.csv", "python -m src.data.split"),
    Etapa("F5", "Pipeline", "Matrices F0/R0/R1 y auditoría de fuga",
          "f5_features.csv", "python -m src.features.build"),
    Etapa("F6", "Modelos", "Métricas de T1-T4 y evaluación por cortes",
          "f6_metricas.csv", "python -m src.models.train"),
    Etapa("F7", "Negocio", "Simulación de la bandeja: FIFO vs riesgo",
          "f7_simulacion.csv", "python -m src.evaluation.simulation"),
]


def estado_pipeline() -> pd.DataFrame:
    """Tabla de estado de todas las etapas: lo que el informe reporta."""
    return pd.DataFrame(
        [
            {
                "Etapa": e.clave,
                "Fase": e.fase,
                "Qué produce": e.descripcion,
                "Estado": "listo" if e.hecho else "pendiente",
                "Artefacto": e.artefacto,
            }
            for e in ETAPAS
        ]
    )


def resumen_avance() -> tuple[int, int]:
    """(etapas listas, etapas totales)."""
    return sum(e.hecho for e in ETAPAS), len(ETAPAS)


# --- Carga de artefactos --------------------------------------------------


def artefacto(nombre: str) -> pd.DataFrame | None:
    """Lee un artefacto por nombre (con o sin .csv). ``None`` si no existe."""
    ruta = ARTEFACTOS / (nombre if nombre.endswith(".csv") else f"{nombre}.csv")
    if not ruta.exists():
        return None
    return pd.read_csv(ruta)


def pendiente(clave: str) -> str:
    """Bloque markdown para una etapa que aún no produjo su artefacto."""
    etapa = next((e for e in ETAPAS if e.clave == clave), None)
    if etapa is None:
        return f"::: {{.callout-warning}}\nEtapa `{clave}` no está en el catálogo.\n:::"
    return (
        f"::: {{.callout-warning appearance='simple'}}\n"
        f"## Pendiente — {etapa.clave}\n\n"
        f"{etapa.descripcion}. El artefacto `{etapa.artefacto}` todavía no existe; "
        f"esta sección se completa sola al ejecutar:\n\n"
        f"```bash\nuv run {etapa.comando}\n```\n"
        f":::"
    )


def miles(n: float) -> str:
    """Formato de miles con separador de coma, para el texto del informe."""
    return f"{n:,.0f}"
