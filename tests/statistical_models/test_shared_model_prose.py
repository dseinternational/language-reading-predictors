# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check copied and rendered model prose without fitting study outcomes."""

from __future__ import annotations

import base64
import importlib
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from language_reading_predictors.statistical_models.publication import copy_report_template

from .test_mechanism_report_forms import QUARTO, _render

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "docs/models"
PROSE_INCLUDE = re.compile(r"\{\{< include (_partials/model/[^ ]+\.qmd) >\}\}")


def test_level_report_prose_matches_the_declared_waves():
    checked = 0
    for report in sorted(MODELS.glob("lrp-rli-lf-*/index.qmd")):
        text = report.read_text(encoding="utf-8")
        includes = PROSE_INCLUDE.findall(text)
        if not includes:
            continue
        model = importlib.import_module(
            "language_reading_predictors.statistical_models." + report.parent.name.replace("-", "_")
        )
        waves = model.SPEC.model_settings.waves
        expected = "level-arm-gap-window.qmd" if waves == ("t1", "t2") else "level-arm-gap.qmd"
        assert waves in (("t1", "t2"), ("t1", "t2", "t3", "t4"))
        assert includes == [f"_partials/model/{expected}", "_partials/model/level-intercepts.qmd"]
        checked += 1
    assert checked == 23


def test_all_shared_model_prose_paths_are_copied_from_the_partials_tree():
    reports = [p for p in MODELS.glob("*/index.qmd") if PROSE_INCLUDE.search(p.read_text(encoding="utf-8"))]
    assert len(reports) == 29
    for report in reports:
        for include in PROSE_INCLUDE.findall(report.read_text(encoding="utf-8")):
            assert (MODELS / include).is_file(), (report, include)


def _prose_fixture(tmp_path: Path, monkeypatch, model_id: str) -> Path:
    fit = tmp_path / model_id
    fit.mkdir()
    monkeypatch.setenv("LRP_OFFLINE_QUARTO", "1")
    copy_report_template(SimpleNamespace(spec=SimpleNamespace(model_id=model_id), output_dir=str(fit)))
    report = fit / "index.qmd"
    text = report.read_text(encoding="utf-8")

    # Keep the real title, model prose, equations, warnings and reading guide.
    # Fit-dependent results are outside this prose test and have their own tests.
    def replace_managed(match):
        name = match.group(1)
        return match.group(0) if name in ("_header", "_reading_guide") else ""

    text = re.sub(r"\{\{< include _partials/(_[a-z_]+)\.qmd >\}\}", replace_managed, text)
    report.write_text(text, encoding="utf-8")
    (fit / "model_graph.png").write_bytes(
        base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j8ioAAAAASUVORK5CYII=")
    )
    for include in PROSE_INCLUDE.findall(text):
        assert (fit / include).read_bytes() == (MODELS / include).read_bytes()
    return fit


@pytest.mark.skipif(QUARTO is None, reason="Quarto is not installed")
@pytest.mark.parametrize("model_id", ["lrp-rli-lf-001", "lrp-rli-lf-201", "lrp-rli-lf-106", "lrp-rli-ca-001"])
def test_shared_model_prose_renders_with_the_right_scope(tmp_path, monkeypatch, model_id):
    html = _render(_prose_fixture(tmp_path, monkeypatch, model_id))
    assert "Codex/GPT-6" in html
    assert "reporting conventions" in html
    assert "guarantees of simulation precision" in html
    assert "_partials/model/" not in html
    if model_id == "lrp-rli-ca-001":
        assert "Conditional coefficients can be biased in either direction" in html
        assert "most observed outcomes" in html
        assert "Arm-gap parameterisation" not in html
    else:
        assert "empirical Bayes" in html
        assert "Arm-gap parameterisation" in html
        if model_id == "lrp-rli-lf-201":
            assert "only t1 and t2" in html
            assert "This fit has no post-crossover waves" in html
            assert "The t3/t4 changes compare" not in html
        else:
            assert "The t3/t4 changes compare" in html
        if model_id == "lrp-rli-lf-106":
            assert "guessing-floor score mean" in html
            assert "neither is sufficient on its own" in html
