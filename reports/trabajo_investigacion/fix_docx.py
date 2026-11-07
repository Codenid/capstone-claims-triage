"""Post-render de Quarto: marca la primera fila de cada tabla de trabajo.docx como
fila de encabezado (w:tblHeader), para que Word la repita cuando la tabla
continúa en la página siguiente. Pandoc no lo hace por sí solo."""

import re
import shutil
import zipfile
from pathlib import Path

DOCX = Path("trabajo.docx")


def mark_header(table_xml: str) -> tuple[str, int]:
    first = re.search(r"<w:tr(?:\s[^>]*)?>", table_xml)
    if not first:
        return table_xml, 0
    head, rest = table_xml[: first.end()], table_xml[first.end() :]
    if rest.startswith("<w:trPr>"):
        if "<w:tblHeader" in rest[: rest.find("</w:trPr>")]:
            return table_xml, 0
        rest = "<w:trPr><w:tblHeader/>" + rest[len("<w:trPr>") :]
    else:
        rest = "<w:trPr><w:tblHeader/></w:trPr>" + rest
    return head + rest, 1


def main() -> None:
    if not DOCX.exists():
        return
    with zipfile.ZipFile(DOCX) as zin:
        items = [(info, zin.read(info.filename)) for info in zin.infolist()]
    marked = 0
    out = []
    for info, data in items:
        if info.filename == "word/document.xml":
            xml = data.decode("utf8")
            pieces = []
            for part in re.split(r"(<w:tbl>.*?</w:tbl>)", xml, flags=re.S):
                if part.startswith("<w:tbl>"):
                    part, k = mark_header(part)
                    marked += k
                pieces.append(part)
            data = "".join(pieces).encode("utf8")
        out.append((info, data))
    tmp = DOCX.with_suffix(".docx.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for info, data in out:
            zout.writestr(info, data)
    shutil.move(tmp, DOCX)
    print(f"fix_docx: {marked} tablas con fila de encabezado repetible")


if __name__ == "__main__":
    main()
