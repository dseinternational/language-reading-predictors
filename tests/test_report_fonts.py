# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reports consume the pinned DSE theme and shared fonts (issue #714)."""

from __future__ import annotations

import runpy
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from language_reading_predictors.models.base_pipeline import EstimatorPipeline
from language_reading_predictors.statistical_models.publication import copy_report_template

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "docs/models"
THEME = ROOT / "docs/theme"
TEMPLATES = sorted(MODELS.glob("*/index.qmd"))
check_pin = runpy.run_path(str(ROOT / "scripts/check_design_token_pin.py"))["check_pin"]


def _front_matter(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), path
    return text.split("\n---\n", 1)[0]


def test_shared_assets_match_their_pin():
    assert check_pin(ROOT) == []


@pytest.fixture
def copied_assets(tmp_path):
    shutil.copytree(ROOT / "docs/theme", tmp_path / "docs/theme")
    shutil.copytree(ROOT / "docs/fonts", tmp_path / "docs/fonts")
    return tmp_path


def test_pin_check_rejects_modified_theme(copied_assets):
    (copied_assets / "docs/theme/dse.scss").write_text("changed", encoding="utf-8")
    assert any("differs" in error and "dse.scss" in error for error in check_pin(copied_assets))


def test_pin_check_rejects_missing_font(copied_assets):
    (copied_assets / "docs/fonts/noto-sans/NotoSans-Regular.ttf").unlink()
    assert any("Missing asset" in error and "NotoSans-Regular.ttf" in error for error in check_pin(copied_assets))


def test_pin_check_rejects_unrecorded_font(copied_assets):
    (copied_assets / "docs/fonts/unrecorded.ttf").write_bytes(b"unrecorded")
    assert any("missing from pin" in error and "unrecorded.ttf" in error for error in check_pin(copied_assets))


def test_project_rules_only_keep_equations_within_the_page():
    scss = (THEME / "project.scss").read_text(encoding="utf-8")
    assert 'math[display="block"]' in scss
    assert "overflow-x: auto" in scss
    assert "font-family" not in scss
    assert "@import" not in scss
    assert not (MODELS / "_partials/_fonts.scss").exists()


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda p: p.parent.name)
def test_model_template_uses_shared_theme(template):
    head = _front_matter(template)
    assert "    theme: [theme/dse.scss, theme/project.scss]\n" in head
    assert "    html-math-method: mathml\n" in head


def test_every_model_template_was_checked():
    assert len(TEMPLATES) > 300


@pytest.mark.parametrize("family", ["gb", "statistical"])
def test_report_copies_the_theme(tmp_path, family):
    if family == "gb":
        config = SimpleNamespace(model_id="lrp-rli-gbg-001", variant_of=None)
        pipeline = SimpleNamespace(context=SimpleNamespace(config=config, output_dir=tmp_path))
        EstimatorPipeline.report(pipeline)  # type: ignore[arg-type]
    else:
        context = SimpleNamespace(spec=SimpleNamespace(model_id="lrp-rli-itt-001"), output_dir=str(tmp_path))
        copy_report_template(context)
    assert "theme/dse.scss" in (tmp_path / "index.qmd").read_text(encoding="utf-8")
    for filename in ("dse.scss", "project.scss"):
        assert (tmp_path / "theme" / filename).read_bytes() == (THEME / filename).read_bytes()


def test_book_report_uses_shared_theme_and_file_fonts():
    config = yaml.safe_load((ROOT / "docs/report/_quarto.yml").read_text(encoding="utf-8"))
    html = config["format"]["html"]
    assert html["theme"] == ["../theme/dse.scss", "../theme/project.scss"]
    assert html["html-math-method"] == "mathml"
    pdf = config["format"]["pdf"]
    assert pdf["pdf-engine"] == "lualatex"
    for role, filename, directory in (
        ("main", "NotoSans", "noto-sans"),
        ("sans", "NotoSans", "noto-sans"),
        ("mono", "NotoSansMono", "noto-sans-mono"),
        ("math", "NotoSansMath-Regular.ttf", "noto-sans-math"),
    ):
        assert pdf[f"{role}font"] == filename
        assert f"Path=../fonts/{directory}/" in pdf[f"{role}fontoptions"]
        family = ROOT / "docs/fonts" / directory
        faces = [filename] if filename.endswith(".ttf") else [
            f"{filename}-{face}.ttf"
            for face in (["Regular", "Bold"] if role == "mono" else ["Regular", "Bold", "Italic", "BoldItalic"])
        ]
        assert all((family / face).is_file() for face in faces)
        assert (family / "OFL.txt").is_file()
    assert not any("Scale=" in option for option in pdf["monofontoptions"])
    assert config["project"]["post-render"] == "../../scripts/set_docx_math_font.py"
