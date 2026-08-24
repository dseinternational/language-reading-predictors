# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contract and rendered-HTML checks for the statistical-report restructure.

The include contract is the #373 order (the #352 findings-first scaffolding with
the result partial below the prior blocks). It is asserted against the real
`docs/models/*/index.qmd` set as well as synthetic fixtures — checking fixtures
alone is what let the validator drift from the repository until #607.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def test_transition_analysis_set_is_gate_visible_but_sensitivities_are_not():
    setup = (REPO / "docs/models/_partials/_setup.qmd").read_text(encoding="utf-8")
    results = (REPO / "docs/models/_partials/_results_adjusted.qmd").read_text(encoding="utf-8")
    assert '"analysis_set_by_transition.csv"' in setup
    assert '"common_horizon_sensitivity.csv"' not in setup
    assert '"transition_slope_sensitivity.csv"' not in setup
    assert "analysis_set_by_transition.csv" in results
    assert "common_horizon_sensitivity.csv" in results
    assert "transition_slope_sensitivity.csv" in results


QUARTO = shutil.which("quarto")

_REWRITER_SPEC = importlib.util.spec_from_file_location(
    "restructure_statistical_reports",
    REPO / "scripts/restructure_statistical_reports.py",
)
assert _REWRITER_SPEC is not None and _REWRITER_SPEC.loader is not None
_REWRITER = importlib.util.module_from_spec(_REWRITER_SPEC)
_REWRITER_SPEC.loader.exec_module(_REWRITER)
TemplateContractError = _REWRITER.TemplateContractError
is_statistical_template = _REWRITER.is_statistical_template
rewrite_template = _REWRITER.rewrite_template

_OLD_TEMPLATE = """---
title: Test report
---

{{< include _partials/_header.qmd >}}

{{< include _partials/_setup.qmd >}}

{{< include _partials/_convergence.qmd >}}

## Overview

Model-specific prose must stay byte-for-byte.

{{< include _partials/_priors.qmd >}}

{{< include _partials/_prior_predictive.qmd >}}

{{< include _partials/_diagnostics.qmd >}}

{{< include _partials/_results_itt.qmd >}}

{{< include _partials/_footer.qmd >}}
"""


def _prose_lines(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if line.strip() and "{{< include " not in line
    ]


def test_rewriter_is_conservative_idempotent_and_preserves_prose():
    assert is_statistical_template(_OLD_TEMPLATE)
    rewritten = rewrite_template(_OLD_TEMPLATE)
    assert _prose_lines(rewritten) == _prose_lines(_OLD_TEMPLATE)
    assert "_partials/_gate_badge.qmd" in rewritten
    assert "_partials/_key_findings.qmd" in rewritten
    assert "_partials/_reading_guide.qmd" in rewritten
    assert "_partials/_technical.qmd" in rewritten
    assert "_partials/_convergence.qmd" not in rewritten
    assert "_partials/_diagnostics.qmd" not in rewritten
    assert rewrite_template(rewritten) == rewritten


# The order every statistical template actually uses, established by #373
# ("priors-before-results reorder") on top of the #352 findings-first
# scaffolding. Until #607 the rewriter validated the superseded #352 order, so it
# rejected all 264 real templates while passing against synthetic fixtures alone.
_MANAGED_ORDER = (
    "_partials/_header.qmd",
    "_partials/_setup.qmd",
    "_partials/_gate_badge.qmd",
    "_partials/_key_findings.qmd",
    "_partials/_reading_guide.qmd",
    "_partials/_priors.qmd",
    "_partials/_prior_predictive.qmd",
    "<results>",
    "_partials/_technical.qmd",
    "_partials/_footer.qmd",
)


def _statistical_templates() -> list[Path]:
    return [
        path
        for path in sorted((REPO / "docs/models").glob("*/index.qmd"))
        if is_statistical_template(path.read_text(encoding="utf-8"))
    ]


def _managed_sequence(text: str) -> tuple[str, ...]:
    includes = [
        name
        for line in text.splitlines()
        if (name := _REWRITER._include(line.strip())) is not None
    ]
    return tuple(
        "<results>" if name.startswith("_partials/_results_") else name
        for name in includes
    )


def test_every_real_template_conforms_to_the_documented_order():
    """The contract must be asserted against the repository, not fixtures alone."""
    templates = _statistical_templates()
    assert len(templates) > 200, f"expected the full report set; found {len(templates)}"
    offenders = {
        path.relative_to(REPO): _managed_sequence(path.read_text(encoding="utf-8"))
        for path in templates
        if _managed_sequence(path.read_text(encoding="utf-8")) != _MANAGED_ORDER
    }
    assert not offenders, f"templates diverge from the documented order: {offenders}"


def test_rewriter_is_a_no_op_over_every_real_template():
    """A stale contract must fail here rather than 'fix' the whole report set."""
    templates = _statistical_templates()
    assert len(templates) > 200, f"expected the full report set; found {len(templates)}"
    rejected: list[str] = []
    rewritten: list[str] = []
    for path in templates:
        text = path.read_text(encoding="utf-8")
        try:
            updated = rewrite_template(text)
        except TemplateContractError as exc:
            rejected.append(f"{path.relative_to(REPO)}: {exc}")
            continue
        if updated != text:
            rewritten.append(str(path.relative_to(REPO)))
    assert not rejected, f"rewriter rejected real templates: {rejected[:5]}"
    assert not rewritten, f"rewriter would rewrite real templates: {rewritten[:5]}"


def test_migration_places_the_result_partial_below_the_prior_blocks():
    """The legacy migration path must target the #373 order, not the #352 one."""
    assert _managed_sequence(rewrite_template(_OLD_TEMPLATE)) == _MANAGED_ORDER


def test_rewriter_rejects_the_superseded_results_before_priors_order():
    """A #352-ordered template is non-conforming and must not pass silently."""
    superseded = "\n".join(
        [
            "---",
            "title: Superseded order",
            "---",
            "",
            "{{< include _partials/_header.qmd >}}",
            "{{< include _partials/_setup.qmd >}}",
            "{{< include _partials/_gate_badge.qmd >}}",
            "{{< include _partials/_key_findings.qmd >}}",
            "{{< include _partials/_reading_guide.qmd >}}",
            "",
            "## Overview",
            "",
            "{{< include _partials/_results_itt.qmd >}}",
            "{{< include _partials/_priors.qmd >}}",
            "{{< include _partials/_prior_predictive.qmd >}}",
            "{{< include _partials/_technical.qmd >}}",
            "{{< include _partials/_footer.qmd >}}",
            "",
        ]
    )
    with pytest.raises(TemplateContractError, match="expected order"):
        rewrite_template(superseded)


def test_rewriter_rejects_an_unrecognised_partial_contract():
    malformed = _OLD_TEMPLATE.replace(
        "{{< include _partials/_diagnostics.qmd >}}\n", ""
    )
    with pytest.raises(TemplateContractError, match="_diagnostics"):
        rewrite_template(malformed)


@pytest.mark.skipif(QUARTO is None, reason="Quarto is not installed")
def test_failed_gate_and_technical_fold_render_end_to_end(tmp_path):
    partials = tmp_path / "_partials"
    partials.mkdir()
    for name in ("_gate_badge.qmd", "_key_findings.qmd", "_technical.qmd"):
        shutil.copy(REPO / "docs/models/_partials" / name, partials / name)
    (partials / "_convergence.qmd").write_text(
        "## Full convergence detail\n\nFULL CONVERGENCE CONTENT\n"
    )
    (partials / "_diagnostics.qmd").write_text(
        "## Analyst diagnostic views\n\n"
        "```{python}\n"
        "# | echo: false\n"
        'print("ANALYST PPC CONTENT")\n'
        "```\n"
    )
    (tmp_path / "diagnostics_summary.json").write_text(
        json.dumps(
            {
                "passed": False,
                "checks": {
                    "rhat": False,
                    "ess": True,
                    "divergences": False,
                    "bfmi": True,
                },
                "divergences": 1,
                "max_rhat": 1.02,
                "min_ess": 1000.0,
                "bfmi_per_chain": [0.8, 0.9],
            }
        )
    )
    (tmp_path / "key_findings.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "sentences": [
                    {"kind": "decoy", "text": "SECRET FINDING MUST NOT RENDER"}
                ],
            }
        )
    )
    (tmp_path / "release_decision.json").write_text(
        json.dumps(
            {
                "status": "gate_failed",
                "publishable": False,
                "scientific_publication_eligible": False,
                "development_only": True,
                "sampling_preset": "dev",
                "publication_qualification": "dev is diagnostic-only",
            }
        )
    )
    (tmp_path / "index.qmd").write_text(
        "---\n"
        'title: "Failed-gate fixture"\n'
        "format: html\n"
        "---\n\n"
        "{{< include _partials/_gate_badge.qmd >}}\n\n"
        "{{< include _partials/_key_findings.qmd >}}\n\n"
        "{{< include _partials/_technical.qmd >}}\n"
    )
    env = {
        key: os.environ[key]
        for key in ("PATH", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")
        if key in os.environ
    }
    env["HOME"] = str(tmp_path)
    env["QUARTO_PYTHON"] = sys.executable
    env["XDG_CACHE_HOME"] = str(tmp_path / ".cache")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (str(REPO / "src"), str(REPO), env.get("PYTHONPATH")),
        )
    )
    subprocess.run(
        [QUARTO, "render", "index.qmd", "--to", "html"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "Sampling-quality gate: failed" in html
    assert "R-hat" in html
    assert "divergent transitions" in html
    assert "Findings withheld" in html
    assert "Development-only fit" in html
    assert "not eligible for scientific publication" in html
    assert "SECRET FINDING MUST NOT RENDER" not in html
    assert "FULL CONVERGENCE CONTENT" in html
    assert "ANALYST PPC CONTENT" in html
    assert 'aria-expanded="false"' in html
    assert "callout-collapse" in html


@pytest.mark.skipif(QUARTO is None, reason="Quarto is not installed")
def test_failed_gate_suppresses_scientific_tables_and_figures(tmp_path):
    import arviz as az
    import numpy as np

    partials = tmp_path / "_partials"
    partials.mkdir()
    for name in (
        "_setup.qmd",
        "_gate_badge.qmd",
        "_results_corr_factor.qmd",
        "_results_mechanism.qmd",
    ):
        shutil.copy(REPO / "docs/models/_partials" / name, partials / name)

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_id": "failed-result-fixture",
                "kind": "corr_factor",
                "outcome_symbol": "W",
                "title": "Failed result fixture",
                "extra": {
                    "mechanism_items": {
                        "caption": "SECRET CAPTION MUST NOT RENDER"
                    }
                },
            }
        )
    )
    (tmp_path / "diagnostics_summary.json").write_text(
        json.dumps(
            {
                "passed": False,
                "checks": {
                    "rhat": True,
                    "ess": True,
                    "divergences": False,
                    "bfmi": True,
                },
                "divergences": 1,
                "max_rhat": 1.001,
                "min_ess": 1000.0,
                "bfmi_per_chain": [0.8, 0.9],
            }
        )
    )
    az.from_dict({"posterior": {"theta": np.zeros((2, 4))}}).to_netcdf(
        tmp_path / "trace.nc"
    )
    (tmp_path / "loadings_summary.csv").write_text(
        "indicator,loading_median\nSECRET_LOADING,9\n"
    )
    (tmp_path / "factor_correlation.csv").write_text(
        ",SECRET_FACTOR\nSECRET_FACTOR,1\n"
    )
    (tmp_path / "diagnostics.csv").write_text(
        ",mean,sd,hdi_5.5%,hdi_94.5%,mcse_mean,mcse_sd,r_hat,ess_bulk,ess_tail\n"
        "theta,9876.54321,4321.09876,9870.12345,9880.67890,0.123,0.456,1.02,10,20\n"
    )
    (tmp_path / "diagnostics_deterministics.csv").write_text(
        ",mean,sd,hdi_5.5%,hdi_94.5%,mcse_mean,mcse_sd,r_hat,ess_bulk,ess_tail\n"
        "derived_theta,8765.43210,3210.98765,8760.12345,8770.67890,0.789,0.654,1.03,11,21\n"
    )
    (tmp_path / "structural_summary.csv").write_text(
        "coefficient,mean,lo50,hi50,lo,hi,prob_pos\n"
        "SECRET_SLOPE,9,8,10,7,11,1\n"
    )
    (tmp_path / "secret_result.png").write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
    )
    (tmp_path / "index.qmd").write_text(
        "---\n"
        'title: "Failed-result fixture"\n'
        "format: html\n"
        "---\n\n"
        "{{< include _partials/_setup.qmd >}}\n\n"
        "{{< include _partials/_gate_badge.qmd >}}\n\n"
        "{{< include _partials/_results_corr_factor.qmd >}}\n\n"
        "{{< include _partials/_results_mechanism.qmd >}}\n\n"
        "```{python}\n"
        "# | echo: false\n"
        "# | output: asis\n"
        'print(_img("secret_result.png", "SECRET RESULT FIGURE"))\n'
        'print(_csv("diagnostics.csv", index_col=0).to_html())\n'
        'print(_csv("diagnostics_deterministics.csv", index_col=0).to_html())\n'
        "```\n"
    )
    env = {
        key: os.environ[key]
        for key in ("PATH", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")
        if key in os.environ
    }
    env["HOME"] = str(tmp_path)
    env["QUARTO_PYTHON"] = sys.executable
    env["XDG_CACHE_HOME"] = str(tmp_path / ".cache")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (str(REPO / "src"), str(REPO), env.get("PYTHONPATH")),
        )
    )
    subprocess.run(
        [QUARTO, "render", "index.qmd", "--to", "html"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "Sampling-quality gate: failed" in html
    assert "No loadings summary" in html
    assert "No factor-correlation matrix" in html
    assert "SECRET_" not in html
    assert "SECRET RESULT FIGURE" not in html
    assert "SECRET CAPTION MUST NOT RENDER" not in html
    assert "9876.54321" not in html
    assert "4321.09876" not in html
    assert "9870.12345" not in html
    assert "9880.67890" not in html
    assert "8765.43210" not in html
    assert "3210.98765" not in html
    assert "8760.12345" not in html
    assert "8770.67890" not in html
    assert "r_hat" in html
    assert "ess_bulk" in html
    assert "mcse_mean" in html
    assert "mcse_sd" in html
    assert "0.123" in html
    assert "0.456" in html
    assert "0.789" in html
    assert "0.654" in html
