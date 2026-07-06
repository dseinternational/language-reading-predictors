# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for ``scripts/blocks_vocab_gb_diagnostic.py`` (issue #186, Q4 Phase 2).

``blocks_rank`` (the permutation-importance rank extraction) is the new numeric
logic. Scripts aren't on the import path, so the module is loaded by file path
(matching ``tests/test_compare_horseshoe_vs_gb.py``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "blocks_vocab_gb_diagnostic.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("blocks_vocab_gb_diagnostic", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blocks_rank_found(mod):
    perm = pd.DataFrame(
        {
            "feature": ["trog", "blocks", "age", "spphon"],
            "importance_mean": [0.30, 0.20, 0.10, 0.05],
        }
    )
    rank, imp, n = mod.blocks_rank(perm)
    assert (rank, n) == (2, 4)
    assert imp == pytest.approx(0.20)


def test_blocks_rank_orders_by_importance_regardless_of_input_order(mod):
    perm = pd.DataFrame(
        {"feature": ["blocks", "age", "trog"], "importance_mean": [0.05, 0.40, 0.20]}
    )
    rank, imp, n = mod.blocks_rank(perm)
    assert rank == 3
    assert n == 3
    assert imp == pytest.approx(0.05)


def test_blocks_rank_absent(mod):
    perm = pd.DataFrame({"feature": ["trog", "age"], "importance_mean": [0.3, 0.1]})
    rank, imp, n = mod.blocks_rank(perm)
    assert rank is None
    assert imp is None
    assert n == 2
