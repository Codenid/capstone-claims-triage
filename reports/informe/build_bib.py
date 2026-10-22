"""Pre-render step: one bibliography file for Pandoc and Typst.

Quarto's Typst template cannot take a list of bibliography files, so the
hand-written references and the generated Theorizer references are joined
into `bibliografia.bib`, which is not versioned.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS = ("referencias.bib", "referencias_theorizer.bib")
OUTPUT = HERE / "bibliografia.bib"

OUTPUT.write_text(
    "\n".join((HERE / part).read_text(encoding="utf-8").rstrip("\n") for part in PARTS)
    + "\n",
    encoding="utf-8",
    newline="\n",
)
print(f"{OUTPUT.name}: {sum(1 for p in PARTS for _ in (HERE / p).open(encoding='utf-8'))} lines")
