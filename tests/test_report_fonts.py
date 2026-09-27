# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reports use Noto Sans text and Noto Sans Math equations (issue #693)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from language_reading_predictors.models.base_pipeline import EstimatorPipeline

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "docs" / "models"
STYLESHEET = MODELS / "_partials" / "_fonts.scss"
TEMPLATES = sorted(MODELS.glob("*/index.qmd"))


def _front_matter(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), path
    return text.split("\n---\n", 1)[0]


def test_stylesheet_sets_both_fonts():
    scss = STYLESHEET.read_text(encoding="utf-8")
    assert '$font-family-sans-serif: "Noto Sans"' in scss
    assert "family=Noto+Sans:" in scss and "family=Noto+Sans+Math" in scss
    assert 'font-family: "Noto Sans Math", math;' in scss


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.parent.name)
def test_model_template_uses_house_fonts(template):
    head = _front_matter(template)
    assert "    theme: [cosmo, _partials/_fonts.scss]\n" in head
    assert "    html-math-method: mathml\n" in head


def test_every_model_template_was_checked():
    assert len(TEMPLATES) > 300


def test_gb_report_copies_the_stylesheet(tmp_path):
    config = SimpleNamespace(model_id="lrp-rli-gbg-001", variant_of=None)
    pipeline = SimpleNamespace(context=SimpleNamespace(config=config, output_dir=tmp_path))
    EstimatorPipeline.report(pipeline)  # type: ignore[arg-type]
    assert "_partials/_fonts.scss" in (tmp_path / "index.qmd").read_text(encoding="utf-8")
    assert (tmp_path / "_partials" / "_fonts.scss").read_bytes() == STYLESHEET.read_bytes()


def test_book_report_uses_house_fonts():
    config = (ROOT / "docs" / "report" / "_quarto.yml").read_text(encoding="utf-8")
    assert "theme: [default, ../models/_partials/_fonts.scss]" in config
    assert "html-math-method: mathml" in config
    assert 'mainfont: "Noto Sans"' in config
    assert 'sansfont: "Noto Sans"' in config
    assert 'mathfont: "Noto Sans Math"' in config
    assert "post-render: ../../scripts/set_docx_math_font.py" in config
    assert (ROOT / "scripts" / "set_docx_math_font.py").is_file()
