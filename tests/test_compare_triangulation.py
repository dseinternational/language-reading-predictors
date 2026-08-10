# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the cross-design triangulation in ``compare_statistical_models.py``.

The numeric logic is ``build_triangulation``: per outcome, read each family's
**canonical items-scale AME** (#391 finding 5 — ITT t2 AME, crossover t2 arm-gap
items, gain-factor period-1 marginal from ``treatment_marginal.csv``) and decide
direction agreement + interval overlap on that scale, carrying the raw logit
coefficients only as appendix columns. Scripts aren't on the import path, so the
module is loaded by file path (matching ``tests/test_compare_horseshoe_vs_gb.py``).
The summary CSVs + the convergence gate JSON are fabricated under a temp output
root so no model fit is needed.
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
    (d / "diagnostics_summary.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "checks": {
                    "rhat": True,
                    "ess": True,
                    "divergences": passed,
                    "bfmi": True,
                },
                "divergences": 0 if passed else 1,
                "max_rhat": 1.001,
                "min_ess": 1000.0,
                "bfmi_per_chain": [0.8, 0.9],
            }
        )
    )


def _write_itt(
    cmp_mod, model_id, *, outcome, items_median, items_lo, items_hi, prob, passed=True
):
    """ITT ``tau_summary.csv``: the reader multiplies ``tau_prob_*`` by the
    measure's item count, so the fixture divides the requested items values back
    onto the probability scale. The logit block is a fixed distinct shape so
    appendix assertions cannot be satisfied by the items columns."""
    d = _run_dir(cmp_mod, model_id)
    d.mkdir(parents=True, exist_ok=True)
    n = cmp_mod.MEASURES[outcome].n_trials
    pd.DataFrame(
        [{
            "tau_logit_median": 0.30, "tau_logit_lo": 0.05, "tau_logit_hi": 0.55,
            "prob_tau_logit_pos": prob, "prob_tau_pos": prob,
            "tau_prob_median": items_median / n,
            "tau_prob_lo": items_lo / n,
            "tau_prob_hi": items_hi / n,
            "prob_ame_pos": prob,
        }]
    ).to_csv(d / "tau_summary.csv", index=False)
    _write_gate(d, passed)


def _write_did_legacy(cmp_mod, model_id, *, median, lo, hi, prob, passed=True):
    """A pre-redesign DiD artefact: logit ``delta`` only, no items pushforward —
    readable, but with nothing on the estimand scale for the verdict."""
    d = _run_dir(cmp_mod, model_id)
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"delta_median": median, "delta_lo": lo, "delta_hi": hi, "prob_delta_pos": prob}]
    ).to_csv(d / "did_summary.csv", index=False)
    _write_gate(d, passed)


def _write_did_arm_wave(
    cmp_mod, model_id, *, items_median, items_lo, items_hi, prob, passed=True
):
    d = _run_dir(cmp_mod, model_id)
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "tau_t2_median": 0.25,
                "tau_t2_lo": 0.02,
                "tau_t2_hi": 0.50,
                "prob_tau_t2_pos": prob,
                "tau_t2_items_median": items_median,
                "tau_t2_items_lo": items_lo,
                "tau_t2_items_hi": items_hi,
                # A deliberately different post-crossover quantity: triangulation
                # must select the clean randomised t2 contrast, not catch-up.
                "delta_crossover_median": -9.0,
            }
        ]
    ).to_csv(d / "did_summary.csv", index=False)
    _write_gate(d, passed)


def _write_gf(
    cmp_mod,
    model_id,
    *,
    items_median,
    items_lo,
    items_hi,
    prob,
    beta_trt_median=0.35,
    beta_trt_lo=0.10,
    beta_trt_hi=0.60,
    beta_trt_prob=0.99,
    with_marginal=True,
    passed=True,
):
    """Gain-factor artefacts: the canonical ``treatment_marginal.csv`` (consumed
    for the verdict, #391 finding 5) plus ``factor_summary.csv`` whose
    ``beta_trt`` feeds only the logit appendix columns."""
    d = _run_dir(cmp_mod, model_id)
    d.mkdir(parents=True, exist_ok=True)
    if with_marginal:
        pd.DataFrame(
            [{
                "trt_items_median": items_median,
                "trt_items_lo": items_lo,
                "trt_items_hi": items_hi,
                "prob_trt_pos": prob,
            }]
        ).to_csv(d / "treatment_marginal.csv", index=False)
    pd.DataFrame(
        [
            {"term": "alpha", "median": 0.0, "lo": -1.0, "hi": 1.0, "prob_positive": 0.5},
            {
                "term": "beta_trt",
                "median": beta_trt_median,
                "lo": beta_trt_lo,
                "hi": beta_trt_hi,
                "prob_positive": beta_trt_prob,
            },
        ]
    ).to_csv(d / "factor_summary.csv", index=False)
    _write_gate(d, passed)


@pytest.fixture
def out_root(cmp_mod, tmp_path):
    cmp_mod._paths.set_output_root(str(tmp_path))
    yield tmp_path
    cmp_mod._paths.set_output_root(None)


def test_triangulation_consistent_and_disagreeing(cmp_mod, out_root):
    # W: all three designs positive with mutually overlapping items intervals -> consistent.
    _write_itt(cmp_mod, "lrp-rli-itt-010", outcome="W", items_median=3.0, items_lo=0.5, items_hi=5.5, prob=0.98)
    _write_did_arm_wave(cmp_mod, "lrp-rli-did-001", items_median=2.5, items_lo=0.2, items_hi=5.0, prob=0.96)
    _write_gf(cmp_mod, "lrp-rli-gf-001", items_median=3.5, items_lo=1.0, items_hi=6.0, prob=0.99)
    # L: DiD points the other way -> direction disagreement (not consistent).
    _write_itt(cmp_mod, "lrp-rli-itt-007", outcome="L", items_median=2.0, items_lo=-1.0, items_hi=5.0, prob=0.85)
    _write_did_arm_wave(cmp_mod, "lrp-rli-did-002", items_median=-3.0, items_lo=-6.0, items_hi=0.5, prob=0.10)
    _write_gf(cmp_mod, "lrp-rli-gf-004", items_median=1.5, items_lo=-0.5, items_hi=4.0, prob=0.90)

    df = cmp_mod.build_triangulation("dev")
    assert df is not None
    by = {r["outcome"]: r for _, r in df.iterrows()}

    w = by["W"]
    assert w["n_designs"] == 3 and w["all_converged"]
    assert w["n_ame_verdict_pool"] == 3
    assert "B excluded" in w["response_link_scope"]
    assert bool(w["direction_agree"]) and bool(w["intervals_overlap"])
    assert bool(w["consistent"]) is True

    lrow = by["L"]
    assert bool(lrow["direction_agree"]) is False
    assert bool(lrow["consistent"]) is False


def test_overlapping_check_catches_disjoint_intervals(cmp_mod, out_root):
    # All positive (direction agrees) but the ITT and GF items intervals are disjoint.
    _write_itt(cmp_mod, "lrp-rli-itt-010", outcome="W", items_median=1.0, items_lo=0.2, items_hi=1.8, prob=0.99)
    _write_did_arm_wave(cmp_mod, "lrp-rli-did-001", items_median=3.0, items_lo=1.0, items_hi=5.5, prob=0.98)
    _write_gf(cmp_mod, "lrp-rli-gf-001", items_median=6.0, items_lo=4.0, items_hi=8.0, prob=0.999)
    df = cmp_mod.build_triangulation("dev")
    w = {r["outcome"]: r for _, r in df.iterrows()}["W"]
    assert bool(w["direction_agree"]) is True
    assert bool(w["intervals_overlap"]) is False  # max(lo)=4.0 > min(hi)=1.8
    assert bool(w["consistent"]) is False


def test_single_design_outcome_is_skipped(cmp_mod, out_root):
    # Only the ITT present for W -> fewer than two designs -> W not emitted.
    _write_itt(cmp_mod, "lrp-rli-itt-010", outcome="W", items_median=3.0, items_lo=0.5, items_hi=5.5, prob=0.98)
    df = cmp_mod.build_triangulation("dev")
    assert df is None or "W" not in set(df["outcome"])


def test_verdict_is_na_when_fewer_than_two_converged(cmp_mod, out_root):
    # Two designs present but only one passed its gate -> verdict not assessable (NA),
    # yet the row is still emitted with the per-design estimates.
    _write_itt(cmp_mod, "lrp-rli-itt-006", outcome="E", items_median=2.0, items_lo=0.1, items_hi=4.0, prob=0.97, passed=True)
    _write_did_arm_wave(cmp_mod, "lrp-rli-did-009", items_median=1.8, items_lo=0.0, items_hi=3.6, prob=0.95, passed=False)
    df = cmp_mod.build_triangulation("dev")
    e = {r["outcome"]: r for _, r in df.iterrows()}["E"]
    assert e["n_designs"] == 2 and e["n_converged"] == 1
    assert pd.isna(e["consistent"]) and pd.isna(e["direction_agree"])


def test_all_negative_overlapping_is_consistent(cmp_mod, out_root):
    # All three designs negative (prob_pos < 0.5) with overlapping items intervals ->
    # consistent via the all(p <= 0.5) branch (the positive case is covered above).
    _write_itt(cmp_mod, "lrp-rli-itt-007", outcome="L", items_median=-3.0, items_lo=-5.5, items_hi=-0.5, prob=0.02)
    _write_did_arm_wave(cmp_mod, "lrp-rli-did-002", items_median=-2.5, items_lo=-5.0, items_hi=-0.2, prob=0.04)
    _write_gf(cmp_mod, "lrp-rli-gf-004", items_median=-3.5, items_lo=-6.0, items_hi=-1.0, prob=0.01,
              beta_trt_median=-0.35, beta_trt_lo=-0.60, beta_trt_hi=-0.10, beta_trt_prob=0.01)
    df = cmp_mod.build_triangulation("dev")
    lrow = {r["outcome"]: r for _, r in df.iterrows()}["L"]
    assert bool(lrow["direction_agree"]) is True
    assert bool(lrow["intervals_overlap"]) is True  # max(lo)=-5.0 <= min(hi)=-1.0
    assert bool(lrow["consistent"]) is True


def test_registry_catalogue_includes_every_complete_graded_suite(cmp_mod):
    """Complete ordinary-logit suites remain; link-sensitive B is deliberately out."""
    by_outcome = {
        outcome: (itt_id, did_id, gf_id)
        for outcome, itt_id, did_id, gf_id in cmp_mod.TRIANGULATION_OUTCOMES
    }
    assert by_outcome == {
        "W": ("lrp-rli-itt-010", "lrp-rli-did-001", "lrp-rli-gf-001"),
        "R": ("lrp-rli-itt-005", "lrp-rli-did-005", "lrp-rli-gf-002"),
        "E": ("lrp-rli-itt-006", "lrp-rli-did-009", "lrp-rli-gf-003"),
        "L": ("lrp-rli-itt-007", "lrp-rli-did-002", "lrp-rli-gf-004"),
        "TR": ("lrp-rli-itt-001", "lrp-rli-did-008", "lrp-rli-gf-009"),
        "TE": ("lrp-rli-itt-002", "lrp-rli-did-004", "lrp-rli-gf-010"),
        "F": ("lrp-rli-itt-025", "lrp-rli-did-010", "lrp-rli-gf-007"),
    }


def test_itt_vs_joint_excludes_response_link_sensitive_blending(cmp_mod, out_root):
    for model_id, outcome in cmp_mod.ITT_IDS:
        _write_itt(cmp_mod, model_id, outcome=outcome, items_median=3.0, items_lo=0.5, items_hi=5.5, prob=0.98)

    joint_dir = _run_dir(cmp_mod, cmp_mod.JOINT_ID)
    joint_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"outcome": outcome, "tau_median": 0.3, "tau_lo": 0.1, "tau_hi": 0.5}
            for outcome in ["W", "R", "E", "L", "B"]
        ]
    ).to_csv(joint_dir / "tau_summary.csv", index=False)
    _write_gate(joint_dir, True)

    df = cmp_mod.build_itt_vs_joint("dev")

    assert df is not None
    assert set(df["outcome"]) == {"W", "R", "E", "L"}
    assert "B excluded" in df["response_link_scope"].iat[0]


def test_triangulation_uses_randomised_t2_contrast_from_redesigned_did(
    cmp_mod, out_root
):
    _write_itt(cmp_mod, "lrp-rli-itt-010", outcome="W", items_median=3.0, items_lo=0.5, items_hi=5.5, prob=0.98)
    _write_did_arm_wave(
        cmp_mod,
        "lrp-rli-did-001",
        items_median=2.5,
        items_lo=0.2,
        items_hi=5.0,
        prob=0.96,
    )
    df = cmp_mod.build_triangulation("dev")
    w = {row["outcome"]: row for _, row in df.iterrows()}["W"]
    assert w["did_logit_term"] == "tau_t2"
    assert w["did_items_median"] == pytest.approx(2.5)
    assert w["did_logit_median"] == pytest.approx(0.25)


def test_gf_verdict_follows_the_canonical_marginal_not_beta_trt(cmp_mod, out_root):
    """#391 finding 5 acceptance: the triangulation consumes the gain-factor
    period-1 items-scale AME from ``treatment_marginal.csv``, never the raw
    conditional ``beta_trt``. The fixture makes the two disagree in sign: the
    verdict must follow the marginal (direction disagreement with the two
    positive designs), while ``beta_trt`` still appears — positive — in the
    clearly-labelled logit appendix columns."""
    _write_itt(cmp_mod, "lrp-rli-itt-010", outcome="W", items_median=3.0, items_lo=0.5, items_hi=5.5, prob=0.98)
    _write_did_arm_wave(cmp_mod, "lrp-rli-did-001", items_median=2.5, items_lo=0.2, items_hi=5.0, prob=0.96)
    _write_gf(
        cmp_mod,
        "lrp-rli-gf-001",
        items_median=-1.0,
        items_lo=-3.0,
        items_hi=0.8,
        prob=0.20,
        beta_trt_median=0.35,
        beta_trt_prob=0.99,
    )
    df = cmp_mod.build_triangulation("dev")
    w = {r["outcome"]: r for _, r in df.iterrows()}["W"]
    assert w["gf_estimand"] == "gf_period1_ame_items"
    assert w["gf_items_median"] == pytest.approx(-1.0)
    assert w["gf_prob_pos"] == pytest.approx(0.20)
    assert bool(w["direction_agree"]) is False
    assert w["gf_logit_term"] == "beta_trt"
    assert w["gf_logit_median"] == pytest.approx(0.35)


def test_scale_and_population_columns_are_explicit(cmp_mod, out_root):
    """#391 finding 5 acceptance: every design row states its scale and its
    averaging population — the marginals share the items scale but not the
    averaging population, and the CSV must say so rather than imply a common
    estimand."""
    _write_itt(cmp_mod, "lrp-rli-itt-010", outcome="W", items_median=3.0, items_lo=0.5, items_hi=5.5, prob=0.98)
    _write_did_arm_wave(cmp_mod, "lrp-rli-did-001", items_median=2.5, items_lo=0.2, items_hi=5.0, prob=0.96)
    _write_gf(cmp_mod, "lrp-rli-gf-001", items_median=3.5, items_lo=1.0, items_hi=6.0, prob=0.99)
    df = cmp_mod.build_triangulation("dev")
    w = {r["outcome"]: r for _, r in df.iterrows()}["W"]
    assert w["itt_scale"] == "items" and w["did_scale"] == "items" and w["gf_scale"] == "items"
    assert w["itt_population"] == (
        "t2 available-case modified ITT analysis rows (both arms)"
    )
    assert w["did_population"] == "t2 wave rows (both arms), arm-gap pushforward"
    assert w["gf_population"] == "period-1 randomised transition rows (both arms)"


def test_logit_only_legacy_design_is_excluded_from_the_ame_verdict(cmp_mod, out_root):
    """A legacy artefact with no items emission (old DiD ``delta``, or a
    gain-factor fit predating ``treatment_marginal.csv``) stays readable in the
    appendix columns but cannot enter the AME-scale verdict — with only one
    items-scale design left, the verdict is NA rather than a logit/items
    hybrid."""
    _write_itt(cmp_mod, "lrp-rli-itt-010", outcome="W", items_median=3.0, items_lo=0.5, items_hi=5.5, prob=0.98)
    _write_did_legacy(cmp_mod, "lrp-rli-did-001", median=0.25, lo=0.02, hi=0.50, prob=0.96)
    _write_gf(cmp_mod, "lrp-rli-gf-001", items_median=3.5, items_lo=1.0, items_hi=6.0, prob=0.99, with_marginal=False)
    df = cmp_mod.build_triangulation("dev")
    w = {r["outcome"]: r for _, r in df.iterrows()}["W"]
    assert w["n_designs"] == 3 and w["n_converged"] == 3
    assert w["n_ame_verdict_pool"] == 1
    assert pd.isna(w["consistent"]) and pd.isna(w["direction_agree"])
    assert w["did_logit_term"] == "legacy_delta_constrained_to_t2"
    assert pd.isna(w["did_items_median"]) and pd.isna(w["gf_items_median"])
