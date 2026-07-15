# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contract and rendered-HTML checks for the issue-321 report restructure."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
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
            }
        )
    )
    (tmp_path / "key_findings.json").write_text(
        json.dumps(
            {
                "status": "gate_failed",
                "failing_checks": ["R-hat", "divergent transitions"],
                "sentences": [
                    {"kind": "decoy", "text": "SECRET FINDING MUST NOT RENDER"}
                ],
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

    html = (tmp_path / "index.html").read_text()
    assert "Sampling-quality gate: failed" in html
    assert "R-hat" in html
    assert "divergent transitions" in html
    assert "Findings withheld" in html
    assert "SECRET FINDING MUST NOT RENDER" not in html
    assert "FULL CONVERGENCE CONTENT" in html
    assert "ANALYST PPC CONTENT" in html
    assert 'aria-expanded="false"' in html
    assert "callout-collapse" in html
