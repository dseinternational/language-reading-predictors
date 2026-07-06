# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for ``scripts/byrne_nonverbal_vocab_diagnostic.py`` (issue #186, Q4 Phase 3).

``partial_spearman`` (rank-residual partial correlation) is the new numeric logic.
Loaded by file path (scripts aren't importable), matching
``tests/test_compare_horseshoe_vs_gb.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "byrne_nonverbal_vocab_diagnostic.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location(
        "byrne_nonverbal_vocab_diagnostic", _SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_partial_spearman_removes_shared_driver(mod):
    # x and y correlate only through a shared driver z; partialling z removes it.
    rng = np.random.default_rng(0)
    z = rng.normal(size=300)
    x = z + rng.normal(scale=0.3, size=300)
    y = z + rng.normal(scale=0.3, size=300)
    marginal = spearmanr(x, y).statistic
    partial = mod.partial_spearman(x, y, z)
    assert marginal > 0.6  # strong marginal association (shared z)
    assert abs(partial) < 0.3  # mostly removed once z is partialled out


def test_partial_spearman_matches_marginal_when_control_irrelevant(mod):
    # z independent of x and y => partialling it barely changes the correlation.
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    y = x + rng.normal(scale=0.5, size=300)
    z = rng.normal(size=300)
    marginal = spearmanr(x, y).statistic
    partial = mod.partial_spearman(x, y, z)
    assert abs(partial - marginal) < 0.1
