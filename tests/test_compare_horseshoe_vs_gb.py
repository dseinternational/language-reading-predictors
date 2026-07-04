# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for ``scripts/compare_horseshoe_vs_gb.py`` (#116 Phase E).

The construct-level alignment between the horseshoe ranking and the gradient-
boosting ranking is the one piece of new numeric logic in the comparison script.
Scripts aren't on the import path in this repo, so the module is loaded by file
path (matching ``tests/test_rank_predictors.py``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "compare_horseshoe_vs_gb.py"


@pytest.fixture(scope="module")
def cmp_mod():
    spec = importlib.util.spec_from_file_location("compare_horseshoe_vs_gb", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_member_to_symbol_token_based(cmp_mod):
    col2sym = {"rowpvt": "R", "yarclet": "L", "age": "age"}
    # Exact column match.
    assert cmp_mod.member_to_symbol("yarclet", col2sym) == "L"
    assert cmp_mod.member_to_symbol("rowpvt", col2sym) == "R"
    # Wave/baseline suffix or prefix resolves as a token.
    assert cmp_mod.member_to_symbol("yarclet_t1", col2sym) == "L"
    assert cmp_mod.member_to_symbol("t1_yarclet", col2sym) == "L"
    # Covariate construct (exact + tokenised).
    assert cmp_mod.member_to_symbol("age", col2sym) == "age"
    assert cmp_mod.member_to_symbol("age_t2", col2sym) == "age"
    # Substring-only collisions must NOT map to `age` (the #160 review bug):
    # `agespeak` / `agebooks` contain "age" but are not the age construct.
    assert cmp_mod.member_to_symbol("agespeak", col2sym) is None
    assert cmp_mod.member_to_symbol("agebooks", col2sym) is None
    # Unmapped demographic column -> None.
    assert cmp_mod.member_to_symbol("gender", col2sym) is None


def test_gb_construct_ranking_takes_max_per_construct(cmp_mod):
    col2sym = {"yarclet": "L", "rowpvt": "R", "eowpvt": "E", "gender": None}
    gb = pd.DataFrame(
        {
            "member": ["yarclet", "yarclet_t1", "rowpvt", "eowpvt", "gender"],
            "perm_imp_mean": [0.10, 0.25, 0.05, 0.40, 0.99],
        }
    )
    out = cmp_mod.gb_construct_ranking(gb, {k: v for k, v in col2sym.items() if v})
    ranks = dict(zip(out["symbol"], out["gb_rank"], strict=True))
    imps = dict(zip(out["symbol"], out["gb_perm_imp"], strict=True))
    # Best (max) per construct: L keeps 0.25 (not 0.10); demographic dropped.
    assert imps["L"] == 0.25
    assert set(out["symbol"]) == {"L", "R", "E"}
    # E (0.40) > L (0.25) > R (0.05).
    assert ranks["E"] == 1 and ranks["L"] == 2 and ranks["R"] == 3


def test_compare_rankings_agreement(cmp_mod):
    # Perfectly concordant orderings -> rho = +1, full top-k overlap.
    hs = pd.DataFrame(
        {
            "predictor": ["L", "E", "T", "R"],
            "p_abs_gt_delta": [0.9, 0.8, 0.5, 0.2],
        }
    )
    gb = pd.DataFrame(
        {
            "member": ["yarclet", "eowpvt", "trog", "rowpvt"],
            "perm_imp_mean": [0.4, 0.3, 0.2, 0.1],
        }
    )
    col2sym = {"yarclet": "L", "eowpvt": "E", "trog": "T", "rowpvt": "R"}
    merged, summary = cmp_mod.compare_rankings(hs, gb, col2sym=col2sym, topk=3)

    assert summary["shared_constructs"] == 4
    assert summary["spearman_rho"] == pytest.approx(1.0)
    assert summary["topk_overlap"] == 3
    # hs_rank derived from p_abs_gt_delta when no explicit rank column is present.
    assert list(merged.sort_values("hs_rank")["predictor"]) == ["L", "E", "T", "R"]


def test_compare_rankings_discordant_and_partial_overlap(cmp_mod):
    # Reversed GB order -> rho = -1; only the middle construct is shared top-k.
    hs = pd.DataFrame(
        {"predictor": ["L", "E", "T"], "p_abs_gt_delta": [0.9, 0.8, 0.7]}
    )
    gb = pd.DataFrame(
        {"member": ["yarclet", "eowpvt", "trog"], "perm_imp_mean": [0.1, 0.2, 0.3]}
    )
    col2sym = {"yarclet": "L", "eowpvt": "E", "trog": "T"}
    _merged, summary = cmp_mod.compare_rankings(hs, gb, col2sym=col2sym, topk=2)
    assert summary["spearman_rho"] == pytest.approx(-1.0)
    # hs top-2 {L,E}; gb top-2 {T,E} -> overlap {E}.
    assert summary["topk_overlap_symbols"] == ["E"]


def test_compare_rankings_few_shared_gives_nan_rho(cmp_mod):
    hs = pd.DataFrame({"predictor": ["L", "E"], "p_abs_gt_delta": [0.9, 0.8]})
    gb = pd.DataFrame({"member": ["yarclet", "gender"], "perm_imp_mean": [0.4, 0.1]})
    col2sym = {"yarclet": "L"}  # only L is shared
    _merged, summary = cmp_mod.compare_rankings(hs, gb, col2sym=col2sym, topk=3)
    assert summary["shared_constructs"] == 1
    assert np.isnan(summary["spearman_rho"])


def test_real_column_map_has_core_measures(cmp_mod):
    """The lazily-built map covers the core measure constructs + covariates."""
    m = cmp_mod.column_to_symbol_map()
    assert m.get("yarclet") == "L"
    assert m.get("eowpvt") == "E"
    assert m.get("rowpvt") == "R"
    assert m.get("age") == "age"
