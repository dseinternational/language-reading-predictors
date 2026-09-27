# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Set the equation font of rendered Word documents to Noto Sans Math (#693).

Word draws every equation in the document's math font, recorded in
``word/settings.xml`` as ``<m:mathPr><m:mathFont m:val="..."/>``. Pandoc copies
only some settings from the reference document and not this one, so a rendered
report falls back to Word's Cambria Math whatever ``docs/template.docx`` says.
This script writes the font into the rendered file.

The report runs it as a Quarto ``post-render`` step, which lists the rendered
files in ``QUARTO_PROJECT_OUTPUT_FILES``; any ``.docx`` among them is updated.
It can also be run on named files::

    python scripts/set_docx_math_font.py output/report/language-reading-predictors-report.docx

It uses only the standard library, because Quarto may run it outside the
project environment. The XML is edited as text so that Word's namespace
declarations, including those that ``mc:Ignorable`` names, survive unchanged.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

MATH_FONT = "Noto Sans Math"
SETTINGS_PART = "word/settings.xml"
MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"

# Elements that follow m:mathPr in the w:settings schema sequence; the new
# element goes before the first of them that is present.
_AFTER_MATH_PR = (
    "w:attachedSchema",
    "w:themeFontLang",
    "w:clrSchemeMapping",
    "w:doNotIncludeSubdocsInStats",
    "w:doNotAutoCompressPictures",
    "w:forceUpgrade",
    "w:captions",
    "w:readModeInkLockDown",
    "w:smartTagType",
    "sl:schemaLibrary",
    "w:shapeDefaults",
    "w:doNotEmbedSmartTags",
    "w:decimalSymbol",
    "w:listSeparator",
)


def with_math_font(settings_xml: str, font: str = MATH_FONT) -> str:
    """Return ``settings.xml`` text whose document math font is ``font``."""
    value = font.replace("&", "&amp;").replace('"', "&quot;")
    font_tag = re.compile(r'<m:mathFont\b[^>]*?m:val="[^"]*"[^>]*?/>')
    if font_tag.search(settings_xml):
        return font_tag.sub(f'<m:mathFont m:val="{value}"/>', settings_xml, count=1)
    if "<m:mathPr/>" in settings_xml:
        return settings_xml.replace("<m:mathPr/>", f'<m:mathPr><m:mathFont m:val="{value}"/></m:mathPr>', 1)
    if "<m:mathPr>" in settings_xml:
        # m:mathFont is the first child of m:mathPr.
        return settings_xml.replace("<m:mathPr>", f'<m:mathPr><m:mathFont m:val="{value}"/>', 1)

    root = re.search(r"<w:settings\b[^>]*>", settings_xml)
    if root is None:
        raise ValueError("settings.xml has no w:settings element")
    if 'xmlns:m="' not in root.group(0):
        start = root.group(0)
        declared = start[:-1] + f' xmlns:m="{MATH_NS}">'
        settings_xml = settings_xml.replace(start, declared, 1)

    element = f'<m:mathPr><m:mathFont m:val="{value}"/></m:mathPr>'
    positions = [settings_xml.find(f"<{tag}") for tag in _AFTER_MATH_PR]
    positions = [p for p in positions if p >= 0]
    at = min(positions) if positions else settings_xml.rindex("</w:settings>")
    return settings_xml[:at] + element + settings_xml[at:]


def set_docx_math_font(path: Path, font: str = MATH_FONT) -> bool:
    """Write ``font`` as the math font of the Word document at ``path``.

    Returns ``False`` when the file already had it. Every other part is copied
    unchanged, in the same order and with the same compression.
    """
    with zipfile.ZipFile(path) as source:
        settings = source.read(SETTINGS_PART).decode("utf-8")
        updated = with_math_font(settings, font)
        if updated == settings:
            return False
        handle, temp_name = tempfile.mkstemp(suffix=".docx", dir=path.parent)
        os.close(handle)
        try:
            with zipfile.ZipFile(temp_name, "w") as target:
                for info in source.infolist():
                    data = updated.encode("utf-8") if info.filename == SETTINGS_PART else source.read(info)
                    target.writestr(info, data)
        except BaseException:
            os.unlink(temp_name)
            raise
    os.replace(temp_name, path)
    return True


def _targets(argv: list[str]) -> list[Path]:
    if argv:
        return [Path(arg) for arg in argv]
    listed = os.environ.get("QUARTO_PROJECT_OUTPUT_FILES", "")
    return [Path(line.strip()) for line in listed.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> None:
    for path in _targets(sys.argv[1:] if argv is None else argv):
        if path.suffix.lower() != ".docx" or not path.is_file():
            continue
        if set_docx_math_font(path):
            print(f"Set the equation font to {MATH_FONT} in {path}")
        else:
            print(f"The equation font is already {MATH_FONT} in {path}")


if __name__ == "__main__":
    main()
