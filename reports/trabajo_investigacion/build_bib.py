"""Pre-render: one bibliography for Pandoc and Typst.

Joins the two bibliographies of reports/informe (hand-written and generated
from Theorizer) with the references that only this document cites.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "informe"
PARTS = [SOURCE / "referencias.bib", SOURCE / "referencias_theorizer.bib", HERE / "referencias_trabajo.bib"]
OUTPUT = HERE / "bibliografia.bib"

OUTPUT.write_text(
    "\n".join(part.read_text(encoding="utf-8").rstrip("\n") for part in PARTS) + "\n",
    encoding="utf-8",
    newline="\n",
)
print(f"{OUTPUT.name}: {sum(1 for p in PARTS for _ in p.open(encoding='utf-8'))} lines")
