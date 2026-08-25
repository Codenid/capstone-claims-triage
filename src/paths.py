"""Rutas del proyecto, resueltas desde la ubicación de este archivo.

Evita las rutas absolutas tipo ``C:\\Users\\...`` en notebooks: el mismo código
corre en la máquina de cualquier integrante del equipo y en CI.
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

CONFIGS = RAIZ / "configs"
DATA = RAIZ / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
DOCS = RAIZ / "docs"
REPORTS = RAIZ / "reports"
FIGURES = REPORTS / "figures"
MODELS = RAIZ / "models"

PARQUET_CRUDO = RAW / "cfpb_reclamos_narrativa.parquet"


def asegurar(*directorios: Path) -> None:
    """Crea los directorios de salida si no existen."""
    for d in directorios:
        d.mkdir(parents=True, exist_ok=True)
