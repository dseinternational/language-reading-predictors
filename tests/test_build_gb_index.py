# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ``scripts/build_gb_index.py``.

Scripts aren't on the import path in this repo, so the module is loaded by file
path (matching ``tests/test_compare_horseshoe_vs_gb.py``).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_gb_index.py"

_CATALOGUE = """# Catalogue

## Layer 1 — Gradient-boosting discovery (`lrp-rli-gbg` / `lrp-rli-gbl`)

### Core outcomes (reading / language outcomes)

| Gain              | Level             | Outcome                     |
| ----------------- | ----------------- | --------------------------- |
| `lrp-rli-gbg-001` | `lrp-rli-gbl-001` | Word reading (`ewrswr`)     |
| —                 | `lrp-rli-gbl-002` | Language sample (`lsamto`)  |

## Layer 2 — Bayesian statistical models

| `lrp-rli-gbg-777` | `lrp-rli-gbl-777` | Must not be read |
"""


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("build_gb_index", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def _fit(
    models_dir: Path,
    model_id: str,
    *,
    r2: float,
    rendered: bool = True,
    complete: bool = True,
    ranking: list[dict] | None = None,
) -> None:
    fit_dir = models_dir / model_id
    fit_dir.mkdir(parents=True)
    (fit_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "cv_pooled_r2": r2,
                "cv_pooled_mae": 1.5,
                "n_observations": 157,
                "fit_complete": complete,
            }
        )
    )
    (fit_dir / "config.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "target_var": "ewrswr_gain",
                "run_config": "reporting",
                "provenance": {
                    "recorded_at_utc": "2026-09-22T09:14:27+00:00",
                    "source": {"commit": "494b93e0cb1b89d6", "dirty": False},
                },
            }
        )
    )
    if rendered:
        (fit_dir / "index.html").write_text("<html></html>")
    if ranking is not None:
        pd.DataFrame(ranking).to_csv(fit_dir / "predictor_ranking.csv", index=False)


def _row(member: str, imp: float, sign: str, same: bool = False, topk: float = 0.5) -> dict:
    return {
        "member": member,
        "perm_imp_mean": imp,
        "perm_imp_sd": 0.01,
        "mean_abs_shap": 0.2,
        "topk_freq": topk,
        "sign": sign,
        "same_skill_of_outcome": same,
    }


def test_parse_catalogue_reads_layer1_only(mod):
    sections = mod.parse_catalogue(_CATALOGUE)
    # Plain parentheticals stay; backticked model-ID ranges are dropped.
    assert [s.title for s in sections] == ["Core outcomes (reading / language outcomes)"]
    assert mod.parse_catalogue(
        "## Layer 1\n### Speech measures (`lrp-rli-gbg`/`lrp-rli-gbl` 017–028)\n"
    )[0].title == "Speech measures"
    rows = sections[0].rows
    assert rows[0] == mod.CatalogueRow("lrp-rli-gbg-001", "lrp-rli-gbl-001", "Word reading (`ewrswr`)")
    assert rows[1].gain is None and rows[1].level == "lrp-rli-gbl-002"


def test_build_page_links_flags_and_lists_uncatalogued(mod, tmp_path):
    models = tmp_path / "models"
    _fit(
        models,
        "lrp-rli-gbg-001",
        r2=-0.25,
        ranking=[
            _row("age", 0.15, "-", topk=0.83),
            _row("yarclet", 0.10, "+"),
            _row("<script>", 0.05, "0"),
            _row("trog", 0.01, "+"),  # fourth: not shown
            _row("zero", -0.01, "+"),
        ],
    )
    _fit(
        models,
        "lrp-rli-gbl-001",
        r2=0.58,
        rendered=False,
        complete=False,
        ranking=[_row("ewrswr_sib", 0.9, "+", same=True)],
    )
    _fit(models, "lrp-rli-gbl-099", r2=0.3)  # fitted, not catalogued, no ranking

    page = mod.build_page(models, _CATALOGUE, ["lrp-rli-gbg-001", "lrp-rli-gbl-050"], models)

    # Relative link for a rendered report; plain text for an unrendered one.
    assert '<a href="lrp-rli-gbg-001/index.html">lrp-rli-gbg-001</a>' in page
    assert "lrp-rli-gbl-001 <span class=\"muted\">(not rendered)</span>" in page
    assert "completion not recorded" in page
    # Negative R² is marked; the top three by importance appear, the fourth does not.
    assert '<b class="neg">-0.25</b>' in page
    assert "<code>age</code>" in page and "<code>yarclet</code>" in page
    assert "<code>trog</code>" not in page
    assert '<span class="arrow sn">↓</span><code>age</code>' in page
    assert '<span class="arrow sz">·</span>' in page
    assert '<span class="stab">83%</span>' in page
    # Data values are escaped, never injected as markup.
    assert "<code>&lt;script&gt;</code>" in page
    assert '<span class="flag">same skill</span>' in page
    # Level-only row; unfitted registered model; fitted but uncatalogued model.
    assert "No gain model for this measure." in page
    assert "<code>lrp-rli-gbl-002</code> has not been fitted here." in page
    assert "Not in the catalogue" in page
    assert "lrp-rli-gbl-050" in page and "lrp-rli-gbl-099" in page
    assert "Must not be read" not in page
    # Provenance summarises the fitted directories.
    assert "3 fitted models" in page and "494b93e0" in page and "clean source tree" in page


def test_links_are_relative_to_the_page(mod, tmp_path):
    models = tmp_path / "out" / "models"
    _fit(models, "lrp-rli-gbg-001", r2=0.2, ranking=[_row("age", 0.1, "+")])
    page = mod.build_page(models, _CATALOGUE, [], tmp_path / "site")
    assert 'href="../out/models/lrp-rli-gbg-001/index.html"' in page


def test_main_writes_index_under_output_root(mod, tmp_path, monkeypatch):
    _fit(tmp_path / "models", "lrp-rli-gbg-001", r2=0.2, ranking=[_row("age", 0.1, "+")])
    catalogue = tmp_path / "README.md"
    catalogue.write_text(_CATALOGUE)
    monkeypatch.setattr(mod, "_registered_model_ids", lambda: ["lrp-rli-gbg-001"])
    try:
        assert mod.main(["--output-dir", str(tmp_path), "--catalogue", str(catalogue)]) == 0
    finally:
        mod.paths.set_output_root(None)
    assert "lrp-rli-gbg-001/index.html" in (tmp_path / "models" / "index.html").read_text()
