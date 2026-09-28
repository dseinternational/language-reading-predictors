# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Word equation-font post-render step (issue #693)."""

from __future__ import annotations

import zipfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "set_docx_math_font.py"
_SPEC = spec_from_file_location("_lrp_set_docx_math_font", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load {_SCRIPT_PATH}")
script = module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)

_W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
_M = f'xmlns:m="{script.MATH_NS}"'

# The shape pandoc writes: m declared, no m:mathPr, themeFontLang after rsids.
PANDOC_SETTINGS = (
    f'<?xml version="1.0" encoding="UTF-8"?><w:settings {_M} {_W}>'
    '<w:zoom w:percent="130" /><w:rsids><w:rsidRoot w:val="00B061A4" /></w:rsids>'
    '<w:themeFontLang w:val="en-US" /><w:decimalSymbol w:val="." /></w:settings>'
)


def test_inserts_math_font_before_following_settings():
    out = script.with_math_font(PANDOC_SETTINGS)
    element = '<m:mathPr><m:mathFont m:val="Noto Sans Math"/></m:mathPr>'
    assert element in out
    assert out.index("</w:rsids>") < out.index(element) < out.index("<w:themeFontLang")


def test_replaces_an_existing_math_font():
    xml = f'<w:settings {_M} {_W}><m:mathPr><m:mathFont m:val="Cambria Math"/><m:dispDef/></m:mathPr></w:settings>'
    out = script.with_math_font(xml)
    assert 'm:val="Cambria Math"' not in out
    assert '<m:mathPr><m:mathFont m:val="Noto Sans Math"/><m:dispDef/></m:mathPr>' in out


def test_adds_font_to_math_properties_without_one():
    xml = f"<w:settings {_M} {_W}><m:mathPr><m:dispDef/></m:mathPr></w:settings>"
    assert '<m:mathPr><m:mathFont m:val="Noto Sans Math"/><m:dispDef/>' in script.with_math_font(xml)


def test_declares_the_math_namespace_when_missing():
    xml = f"<w:settings {_W}><w:zoom/></w:settings>"
    out = script.with_math_font(xml)
    assert out.startswith(f'<w:settings {_W} {_M}>')
    assert out.endswith('<m:mathPr><m:mathFont m:val="Noto Sans Math"/></m:mathPr></w:settings>')


def test_rejects_a_part_without_settings():
    with pytest.raises(ValueError):
        script.with_math_font("<w:document/>")


def _docx(path: Path, settings: str) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr(script.SETTINGS_PART, settings)
        z.writestr("word/document.xml", "<w:document/>")


def test_rewrites_only_the_settings_part(tmp_path):
    path = tmp_path / "report.docx"
    _docx(path, PANDOC_SETTINGS)
    assert script.set_docx_math_font(path) is True
    with zipfile.ZipFile(path) as z:
        assert z.namelist() == ["[Content_Types].xml", script.SETTINGS_PART, "word/document.xml"]
        assert all(info.compress_type == zipfile.ZIP_DEFLATED for info in z.infolist())
        assert z.read("word/document.xml") == b"<w:document/>"
        assert 'm:val="Noto Sans Math"' in z.read(script.SETTINGS_PART).decode()
    assert script.set_docx_math_font(path) is False
    assert [p.name for p in tmp_path.iterdir()] == ["report.docx"]


def test_main_reads_quarto_output_list_and_skips_other_formats(tmp_path, monkeypatch, capsys):
    docx = tmp_path / "report.docx"
    _docx(docx, PANDOC_SETTINGS)
    html = tmp_path / "index.html"
    html.write_text("<html/>")
    monkeypatch.setenv("QUARTO_PROJECT_OUTPUT_FILES", f"{html}\n{docx}\n")
    script.main([])
    assert "Set the equation font to Noto Sans Math" in capsys.readouterr().out
    assert html.read_text() == "<html/>"
