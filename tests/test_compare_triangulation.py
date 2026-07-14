# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the cross-design triangulation in ``compare_statistical_models.py``.

The new numeric logic is ``build_triangulation``: per outcome, read the logit-scale
treatment effect from up to three designs (ITT ``tau``, DiD ``delta``, gain-factor
``beta_trt``) and decide direction agreement + interval overlap. Scripts aren't on the
import path, so the module is loaded by file path (matching
``tests/test_compare_horseshoe_vs_gb.py``). The three summary CSVs + the convergence
gate JSON are fabricated under a temp output root so no model fit is needed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "compare_statistical_models.py"
)


@pytest.fixture(scope="module")
def cmp_mod():
    spec = importlib.util.spec_from_file_location("compare_statistical_models", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_dir(cmp_mod, model_id: str, config: str = "dev") -> Path:
    return Path(cmp_mod._run_dir(model_id, config))


def _write_gate(d: Path, passed: bool) -> None:
    (d / "diagnostics_summary.json").write_text(json.dumps({"passed": passed}))


def _write_itt(cmp_mod, model_id, *, median, lo, hi, prob, passed=True):
    d = _run_dir(cmp_mod, model_id)
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{
            "tau_logit_median": median, "tau_logit_lo": lo, "tau_logit_hi": hi,
            "prob_tau_pos": prob,
        }]
    ).to_csv(d / "tau_summary.csv", index=False)
    _write_gate(d, passed)


def _write_did(cmp_mod, model_id, *, median, lo, hi, prob, passed=True):
    d = _run_dir(cmp_mod, model_id)
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"delta_median": median, "delta_lo": lo, "delta_hi": hi, "prob_delta_pos": prob}]
    ).to_csv(d / "did_summary.csv", index=False)
    _write_gate(d, passed)


def _write_gf(cmp_mod, model_id, *, median, lo, hi, prob, passed=True):
    d = _run_dir(cmp_mod, model_id)
    d.mkdir(parents=True, exist_ok=True)
    # factor_summary.csv is one row per coefficient; the triangulation reads beta_trt.
    pd.DataFrame(
        [
            {"term": "alpha", "median": 0.0, "lo": -1.0, "hi": 1.0, "prob_positive": 0.5},
            {"term": "beta_trt", "median": median, "lo": lo, "hi": hi, "prob_positive": prob},
        ]
    ).to_csv(d / "factor_summary.csv", index=False)
    _write_gate(d, passed)


@pytest.fixture
def out_root(cmp_mod, tmp_path):
    cmp_mod._paths.set_output_root(str(tmp_path))
    yield tmp_path
    cmp_mod._paths.set_output_root(None)


def test_triangulation_consistent_and_disagreeing(cmp_mod, out_root):
    # W: all three designs positive with mutually overlapping intervals -> consistent.
    _write_itt(cmp_mod, "lrp-rli-itt-010", median=0.30, lo=0.05, hi=0.55, prob=0.98)
    _write_did(cmp_mod, "lrp-rli-did-001", median=0.25, lo=0.02, hi=0.50, prob=0.96)
    _write_gf(cmp_mod, "lrp-rli-gf-001", median=0.35, lo=0.10, hi=0.60, prob=0.99)
    # L: DiD points the other way -> direction disagreement (not consistent).
    _write_itt(cmp_mod, "lrp-rli-itt-007", median=0.20, lo=-0.10, hi=0.50, prob=0.85)
    _write_did(cmp_mod, "lrp-rli-did-002", median=-0.30, lo=-0.60, hi=0.05, prob=0.10)
    _write_gf(cmp_mod, "lrp-rli-gf-004", median=0.15, lo=-0.05, hi=0.40, prob=0.90)

    df = cmp_mod.build_triangulation("dev")
    assert df is not None
    by = {r["outcome"]: r for _, r in df.iterrows()}

    w = by["W"]
    assert w["n_designs"] == 3 and w["all_converged"]
    assert bool(w["direction_agree"]) and bool(w["intervals_overlap"])
    assert bool(w["consistent"]) is True

    lrow = by["L"]
    assert bool(lrow["direction_agree"]) is False
    assert bool(lrow["consistent"]) is False


def test_overlapping_check_catches_disjoint_intervals(cmp_mod, out_root):
    # All positive (direction agrees) but the ITT and GF intervals are disjoint.
    _write_itt(cmp_mod, "lrp-rli-itt-010", median=0.10, lo=0.02, hi=0.18, prob=0.99)
    _write_did(cmp_mod, "lrp-rli-did-001", median=0.30, lo=0.10, hi=0.55, prob=0.98)
    _write_gf(cmp_mod, "lrp-rli-gf-001", median=0.60, lo=0.40, hi=0.80, prob=0.999)
    df = cmp_mod.build_triangulation("dev")
    w = {r["outcome"]: r for _, r in df.iterrows()}["W"]
    assert bool(w["direction_agree"]) is True
    assert bool(w["intervals_overlap"]) is False  # max(lo)=0.40 > min(hi)=0.18
    assert bool(w["consistent"]) is False


def test_single_design_outcome_is_skipped(cmp_mod, out_root):
    # Only the ITT present for W -> fewer than two designs -> W not emitted.
    _write_itt(cmp_mod, "lrp-rli-itt-010", median=0.30, lo=0.05, hi=0.55, prob=0.98)
    df = cmp_mod.build_triangulation("dev")
    assert df is None or "W" not in set(df["outcome"])


def test_verdict_is_na_when_fewer_than_two_converged(cmp_mod, out_root):
    # Two designs present but only one passed its gate -> verdict not assessable (NA),
    # yet the row is still emitted with the per-design estimates.
    _write_itt(cmp_mod, "lrp-rli-itt-006", median=0.20, lo=0.01, hi=0.40, prob=0.97, passed=True)
    _write_did(cmp_mod, "lrp-rli-did-009", median=0.18, lo=0.00, hi=0.36, prob=0.95, passed=False)
    df = cmp_mod.build_triangulation("dev")
    e = {r["outcome"]: r for _, r in df.iterrows()}["E"]
    assert e["n_designs"] == 2 and e["n_converged"] == 1
    assert pd.isna(e["consistent"]) and pd.isna(e["direction_agree"])


def test_all_negative_overlapping_is_consistent(cmp_mod, out_root):
    # All three designs negative (prob_pos < 0.5) with overlapping intervals -> consistent
    # via the all(p <= 0.5) branch (the positive case is covered above; #295 review).
    _write_itt(cmp_mod, "lrp-rli-itt-007", median=-0.30, lo=-0.55, hi=-0.05, prob=0.02)
    _write_did(cmp_mod, "lrp-rli-did-002", median=-0.25, lo=-0.50, hi=-0.02, prob=0.04)
    _write_gf(cmp_mod, "lrp-rli-gf-004", median=-0.35, lo=-0.60, hi=-0.10, prob=0.01)
    df = cmp_mod.build_triangulation("dev")
    lrow = {r["outcome"]: r for _, r in df.iterrows()}["L"]
    assert bool(lrow["direction_agree"]) is True
    assert bool(lrow["intervals_overlap"]) is True  # max(lo)=-0.50 <= min(hi)=-0.10
    assert bool(lrow["consistent"]) is True
