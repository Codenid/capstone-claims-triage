# Trabajo de investigación (plantilla de la Escuela de Posgrado)

Versión del informe con la estructura de la plantilla general tipo proyecto
de UTEC Posgrado (resumen, abstract, capítulos I–VI, conclusiones,
referencias y anexos) y citado IEEE, según los lineamientos de estilo de
citado del programa. El informe de cierre del curso, con la estructura y el
APA 7 que pide la rúbrica del P4, sigue en `reports/informe/`.

Toda cifra se lee de los archivos de resultados al renderizar, con el mismo
`facts.py` de `reports/informe/`. Las secciones técnicas (4.5, 5.3, 5.5 y
los anexos A–C) se derivan de las de `reports/informe/secciones/` con el
script del cierre (`derive_sections.py`, en el historial de la sesión); los
capítulos I–III, 4.1–4.4, 5.1–5.2, 5.4, VI, las conclusiones y los anexos
D–F son propios de este documento.

Lo marcado como «PENDIENTE (Product Lead)» o «SUPUESTO» en el texto no tiene
evidencia en el repositorio (one-pager, pitch, stakeholders, mercado,
viabilidad financiera): lo completan Gianmarco y Edgard antes de la entrega.

## Renderizar

```bash
cd reports/trabajo_investigacion
QUARTO_PYTHON=../../.venv/Scripts/python.exe quarto render          # trabajo.docx y trabajo.pdf
```

`trabajo.docx` usa `recursos/plantilla.docx` como documento de referencia
(estilos, encabezados y pies de la plantilla). Al abrirlo en Word, actualizar
los campos (F9) para que la tabla de contenido tome los números de página.
`trabajo.pdf` es la versión Typst, para lectura rápida; la entrega es el
Word.
