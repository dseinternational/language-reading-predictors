# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rendered-prose checks for the shared mechanism results partial (#586 finding 3).

The partial described *every* registered mechanism fit as an HSGP curve with an
``InverseGamma(5, 5)`` lengthscale. 21 of the 41 are linear and fit no GP at all,
and 16 of the 20 HSGP fits use ``InverseGamma(8, 8)`` — so 37 of 41 reports stated
a functional form, a prior or both that their own model never used, and every one
asserted the association's sign and existence were "robust" without consulting a
posterior or a sensitivity artefact.

These render the real partial against four synthetic fit directories — linear,
default HSGP, tight HSGP and continuous-covariate HSGP — with a stand-in for
``_setup.qmd`` so no ``trace.nc`` is needed. What is under test is the prose the
reader actually sees, not the code that decides it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
QUARTO = shutil.which("quarto")

# Stands in for _setup.qmd: the partial reads config, _csv, _img, _has, np and
# outcome_label from the shared kernel namespace, and nothing else.
_SETUP_STUB = """```{python}
# | echo: false
import json
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path(".")
with open(_here / "config.json") as _f:
    config = json.load(_f)
outcome_label = "Word reading (WR)"


def _has(name: str) -> bool:
    return (_here / name).exists()


def _csv(name: str):
    return pd.read_csv(_here / name) if _has(name) else None


def _img(name: str, alt: str = "") -> str:
    return f"![{alt}]({name})" if _has(name) else ""
```
"""


def _fixture_dir(tmp_path: Path, name: str, run_plan: dict) -> Path:
    import json

    fit = tmp_path / name
    (fit / "_partials").mkdir(parents=True)
    shutil.copy(
        REPO / "docs/models/_partials/_results_mechanism.qmd",
        fit / "_partials/_results_mechanism.qmd",
    )
    (fit / "config.json").write_text(
        json.dumps(
            {
                "model_id": f"lrp-rli-mech-{name}",
                "mechanism_symbol": run_plan["mechanism_symbol"],
                "resolved_run_plan": run_plan,
                "extra": {},
            }
        )
    )
    (fit / "index.qmd").write_text(
        "---\n"
        f'title: "{name} fixture"\n'
        "format: html\n"
        "---\n\n"
        + _SETUP_STUB
        + "\n{{< include _partials/_results_mechanism.qmd >}}\n"
    )
    return fit


def _render(fit: Path) -> str:
    env = {
        key: os.environ[key]
        for key in ("PATH", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT")
        if key in os.environ
    }
    env["HOME"] = str(fit)
    env["QUARTO_PYTHON"] = sys.executable
    env["XDG_CACHE_HOME"] = str(fit / ".cache")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(REPO / "src"), str(REPO), env.get("PYTHONPATH")))
    )
    subprocess.run(
        [QUARTO, "render", "index.qmd", "--to", "html"],
        cwd=fit,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return (fit / "index.html").read_text(encoding="utf-8")


def _plan(**overrides) -> dict:
    plan = {
        "mechanism_symbol": "L",
        "linear_mechanism": False,
        "mechanism_is_covariate": False,
        "mech_lengthscale_tight": False,
    }
    plan.update(overrides)
    return plan


@pytest.mark.skipif(QUARTO is None, reason="Quarto is not installed")
def test_linear_report_claims_no_curve_prior_or_steepest_interval(tmp_path):
    html = _render(_fixture_dir(tmp_path, "linear", _plan(linear_mechanism=True)))

    assert "single linear slope" in html
    assert "no GP lengthscale" in html
    # The three things a linear fit must never claim.
    assert "the HSGP curve here" not in html
    assert "InverseGamma" not in html
    assert "Where does the fitted curve rise fastest" not in html
    assert "no nonparametric curve to locate an interval in" in html


@pytest.mark.skipif(QUARTO is None, reason="Quarto is not installed")
def test_default_hsgp_report_states_its_own_lengthscale(tmp_path):
    html = _render(_fixture_dir(tmp_path, "default-hsgp", _plan()))

    assert "InverseGamma(5, 5)" in html
    assert "InverseGamma(8, 8)" not in html
    assert "Where does the fitted curve rise fastest" in html


@pytest.mark.skipif(QUARTO is None, reason="Quarto is not installed")
def test_tight_hsgp_report_states_its_own_lengthscale(tmp_path):
    html = _render(
        _fixture_dir(tmp_path, "tight-hsgp", _plan(mech_lengthscale_tight=True))
    )

    assert "InverseGamma(8, 8)" in html
    assert "InverseGamma(5, 5)" not in html


@pytest.mark.skipif(QUARTO is None, reason="Quarto is not installed")
def test_covariate_exposure_report_uses_raw_score_units(tmp_path):
    html = _render(
        _fixture_dir(
            tmp_path,
            "covariate-hsgp",
            _plan(mechanism_symbol="attend", mechanism_is_covariate=True),
        )
    )

    assert "raw score" in html
    # A continuous exposure is neither a bounded count nor letter sounds; the
    # readiness prose used to hard-code both into every report.
    assert "letter-sounds" not in html
    assert "out of its maximum" not in html
    assert "no logit transform applies" in html


@pytest.mark.skipif(QUARTO is None, reason="Quarto is not installed")
def test_no_report_asserts_the_association_is_robust(tmp_path):
    """The shared paragraph claimed robustness without consulting any artefact."""
    for name, plan in (
        ("robust-linear", _plan(linear_mechanism=True)),
        ("robust-hsgp", _plan(mech_lengthscale_tight=True)),
    ):
        html = _render(_fixture_dir(tmp_path, name, plan))
        assert "(robust)" not in html
        assert "sign and existence of the association (robust)" not in html
