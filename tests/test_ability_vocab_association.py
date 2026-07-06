# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for ``scripts/ability_vocab_association.py`` (issue #186, Q4).

``summarise_gamma`` (the logit stats + items-scale average marginal effect for the
block-design coefficient) is the new numeric logic in the read-out. Scripts aren't
on the import path in this repo, so the module is loaded by file path (matching
``tests/test_compare_horseshoe_vs_gb.py``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "ability_vocab_association.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("ability_vocab_association", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarise_gamma_logit_stats(mod):
    rng = np.random.default_rng(0)
    g = rng.normal(0.2, 0.1, 2000)
    out = mod.summarise_gamma(g, None, n_trials=30)

    assert out["gamma_logit_median"] == pytest.approx(float(np.median(g)))
    assert out["prob_positive"] == pytest.approx(float((g > 0).mean()))
    lo, hi = float(np.quantile(g, 0.05)), float(np.quantile(g, 0.95))
    assert out["gamma_logit_lo90"] == pytest.approx(lo)
    assert out["gamma_logit_hi90"] == pytest.approx(hi)
    # Nested equal-tailed intervals are ordered around the median.
    assert (
        out["gamma_logit_lo95"]
        <= out["gamma_logit_lo90"]
        <= out["gamma_logit_lo50"]
        <= out["gamma_logit_median"]
        <= out["gamma_logit_hi50"]
        <= out["gamma_logit_hi90"]
        <= out["gamma_logit_hi95"]
    )
    assert out["evidence_label"] == mod.evidence_label(out["prob_positive"])
    # No eta => no items-scale keys.
    assert "items_ame_median" not in out


def test_summarise_gamma_items_ame_matches_reference(mod):
    rng = np.random.default_rng(1)
    n_obs, n_draw = 12, 500
    g = rng.normal(0.3, 0.15, n_draw)
    eta = rng.normal(0.0, 1.0, (n_obs, n_draw))
    n_trials = 40

    out = mod.summarise_gamma(g, eta, n_trials=n_trials)

    # Reference AME of a +1 SD block-design shift (eta already carries g*z(blocks),
    # so the shift adds g to the linear predictor), per draw, then median.
    ame = (expit(eta + g[None, :]) - expit(eta)).mean(axis=0) * n_trials
    assert out["items_ame_median"] == pytest.approx(float(np.median(ame)))
    assert out["items_ame_lo90"] == pytest.approx(float(np.quantile(ame, 0.05)))
    assert out["items_ame_hi90"] == pytest.approx(float(np.quantile(ame, 0.95)))


def test_summarise_gamma_clearly_positive_direction(mod):
    out = mod.summarise_gamma(np.full(1000, 0.5), None, n_trials=10)
    assert out["prob_positive"] == 1.0
    assert out["favoured_direction"] == "positive"
