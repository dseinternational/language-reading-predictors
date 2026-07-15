# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the key-findings generator (issue #320).

Golden-sentence tests per core archetype from synthetic CSV rows, the
convergence-gate interlock, missing-CSV degradation, the no-``nan`` guard and
the five-sentence cap. The partial content guards at the bottom follow the
``test_concurrent_pipeline`` read-the-qmd idiom.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from language_reading_predictors.statistical_models.reporting import (
    KEY_FINDINGS_FILENAME,
    KEY_FINDINGS_MAX_SENTENCES,
    generate_key_findings,
)

REPO = Path(__file__).resolve().parents[2]


def _write_json(d: Path, name: str, payload: dict) -> None:
    with open(d / name, "w") as f:
        json.dump(payload, f)


def _write_csv(d: Path, name: str, row: dict) -> None:
    pd.DataFrame([row]).to_csv(d / name, index=False)


def _passing_gate() -> dict:
    return {
        "passed": True,
        "checks": {"rhat": True, "ess": True, "divergences": True, "bfmi": True},
    }


def _config(kind: str, **overrides) -> dict:
    cfg = {
        "model_id": f"lrp-test-{kind}",
        "kind": kind,
        "outcome_symbol": "W",
        "title": "Test model",
        "extra": {},
    }
    cfg.update(overrides)
    return cfg


def _rope_row(**overrides) -> dict:
    row = {
        "items_median": 2.4,
        "items_lo": -0.3,
        "items_hi": 5.9,
        "delta_items": 1.0,
        "pd": 0.94,
        "prob_benefit_ge_delta": 0.81,
        "prob_in_rope": 0.17,
        "prob_harm_ge_delta": 0.01,
        "direction_label": "moderate",
        "benefit_label": "moderate",
        "favoured_direction": "positive",
        "favoured_direction_prob": 0.94,
        "favoured_direction_label": "moderate",
    }
    row.update(overrides)
    return row


def _setup_dir(tmp_path: Path, kind: str, *, config: dict | None = None) -> Path:
    d = tmp_path / f"{kind}-dev"
    d.mkdir()
    _write_json(d, "config.json", config or _config(kind))
    _write_json(d, "diagnostics_summary.json", _passing_gate())
    return d


def _texts(payload: dict) -> str:
    return " ".join(s["text"] for s in payload["sentences"])


# --- gate interlock and degradation -------------------------------------------


def test_gate_failed_withholds_findings(tmp_path):
    d = _setup_dir(tmp_path, "itt")
    _write_json(
        d,
        "diagnostics_summary.json",
        {
            "passed": False,
            "checks": {"rhat": False, "ess": True, "divergences": False, "bfmi": True},
        },
    )
    _write_csv(d, "rope_summary.csv", _rope_row())
    payload = generate_key_findings(d)
    assert payload["status"] == "gate_failed"
    assert payload["sentences"] == []
    assert "R-hat" in payload["failing_checks"]
    assert "divergent transitions" in payload["failing_checks"]
    assert (d / KEY_FINDINGS_FILENAME).exists()


def test_missing_diagnostics_summary_degrades(tmp_path):
    d = tmp_path / "no-gate"
    d.mkdir()
    _write_json(d, "config.json", _config("itt"))
    payload = generate_key_findings(d)
    assert payload["status"] == "not_available"
    assert "convergence gate" in payload["reason"]


def test_missing_csvs_degrade_to_not_available(tmp_path):
    d = _setup_dir(tmp_path, "itt")
    payload = generate_key_findings(d)
    assert payload["status"] == "not_available"
    assert "rope_summary.csv" in payload["reason"]
    # The payload must still be valid JSON on disk (the partial renders it).
    with open(d / KEY_FINDINGS_FILENAME) as f:
        assert json.load(f)["status"] == "not_available"


def test_missing_config_degrades(tmp_path):
    d = tmp_path / "no-config"
    d.mkdir()
    _write_json(d, "diagnostics_summary.json", _passing_gate())
    payload = generate_key_findings(d)
    assert payload["status"] == "not_available"
    assert "config.json" in payload["reason"]


def test_nan_in_headline_degrades_not_emits(tmp_path):
    d = _setup_dir(tmp_path, "itt")
    _write_csv(d, "rope_summary.csv", _rope_row(items_median=float("nan")))
    payload = generate_key_findings(d)
    assert payload["status"] == "not_available"
    assert "nan" not in _texts(payload).lower()


# --- core-four golden sentences ------------------------------------------------


def test_itt_golden_sentences(tmp_path):
    d = _setup_dir(tmp_path, "itt")
    _write_csv(d, "rope_summary.csv", _rope_row())
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    kinds = [s["kind"] for s in payload["sentences"]]
    assert kinds == ["headline", "confidence", "rope", "causal"]
    texts = [s["text"] for s in payload["sentences"]]
    assert texts[0] == (
        "Best estimate: the intervention changed Word reading (EWRSWR) by "
        "**+2.4 items** over the trial period "
        "(95% credible range -0.3 to +5.9)."
    )
    assert texts[1] == (
        "There is a 94% probability that the true effect is positive — moderate "
        "evidence that the intervention helps."
    )
    assert texts[2] == (
        "A change of at least 1 item was pre-specified as the smallest "
        "difference that would matter in practice: the probability the benefit "
        "reaches that size is 81%, and the probability the effect is too small "
        "to matter either way is 17%."
    )
    assert "randomly assigned" in texts[3]
    assert "cause-and-effect" in texts[3]


def test_itt_floored_risk_difference_wording(tmp_path):
    d = _setup_dir(tmp_path, "itt", config=_config("itt", outcome_symbol="P"))
    _write_csv(
        d,
        "rope_summary.csv",
        _rope_row(
            items_median=0.18,
            items_lo=0.02,
            items_hi=0.35,
            delta_items=0.10,
            delta_scale="risk_difference",
            provisional_delta=False,
        ),
    )
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    headline = payload["sentences"][0]["text"]
    assert "percentage points" in headline
    assert "+18" in headline
    assert "scoring above zero" in headline
    rope = payload["sentences"][2]["text"]
    assert "10 percentage points" in rope


def test_itt_without_rope_falls_back_to_tau_summary(tmp_path):
    d = _setup_dir(tmp_path, "itt", config=_config("itt", outcome_symbol="F"))
    _write_csv(
        d,
        "tau_summary.csv",
        {
            "tau_prob_median": 0.02,
            "tau_prob_lo": -0.01,
            "tau_prob_hi": 0.05,
            "prob_tau_pos": 0.9,
        },
    )
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    kinds = [s["kind"] for s in payload["sentences"]]
    assert "note" in kinds  # the no-agreed-delta note replaces the ROPE verdict
    assert "headline" in kinds  # F has a known measure, so items translate
    texts = _texts(payload)
    assert "No minimally-important difference" in texts


def test_gain_factors_golden_sentences(tmp_path):
    d = _setup_dir(tmp_path, "gain_factors")
    _write_csv(d, "rope_summary.csv", _rope_row())
    _write_csv(d, "treatment_marginal.csv", {"trt_items_median": 2.0})
    pd.DataFrame(
        [
            {
                "term": "beta_trt",
                "role": "causal",
                "median": 0.4,
                "prob_positive": 0.95,
            },
            {
                "term": "gamma_own",
                "role": "association",
                "median": 0.8,
                "prob_positive": 1.0,  # every draw agreed: must not display 100%
            },
            {
                "term": "gamma_A",
                "role": "association",
                "median": -0.1,
                "prob_positive": 0.35,
            },
        ]
    ).to_csv(d / "factor_summary.csv", index=False)
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    kinds = [s["kind"] for s in payload["sentences"]]
    assert kinds == ["headline", "confidence", "rope", "causal", "highlight"]
    texts = [s["text"] for s in payload["sentences"]]
    assert "during the randomised first period" in texts[0]
    assert "only cause-and-effect estimate" in texts[3]
    assert "the child's own starting point on this measure" in texts[4]
    assert "99.9%" in texts[4]  # a certainty of 1.0 in finite draws caps at 99.9%
    assert "100%" not in texts[4]
    assert "not a cause" in texts[4]


def test_gain_factors_treated_only_has_no_causal_headline(tmp_path):
    cfg = _config("gain_factors", extra={"treated_only": True})
    d = _setup_dir(tmp_path, "gain_factors", config=cfg)
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    texts = _texts(payload)
    assert "no treatment effect" in texts
    assert all(s["kind"] != "headline" for s in payload["sentences"])


def test_gain_factors_falls_back_to_treatment_marginal(tmp_path):
    d = _setup_dir(tmp_path, "gain_factors", config=_config("gain_factors", outcome_symbol="F"))
    _write_csv(
        d,
        "treatment_marginal.csv",
        {
            "trt_items_median": 0.6,
            "trt_items_lo": -0.2,
            "trt_items_hi": 1.4,
            "prob_trt_pos": 0.88,
        },
    )
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    assert "+0.6 items" in payload["sentences"][0]["text"]


def test_level_factors_golden_sentences(tmp_path):
    d = _setup_dir(tmp_path, "level_factors")
    _write_csv(d, "rope_summary.csv", _rope_row())
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    texts = [s["text"] for s in payload["sentences"]]
    assert "at the end of the randomised period (t2)" in texts[0]
    assert "Only this t2 comparison is randomised" in texts[3]
    assert "crossed over" in texts[3]


def test_did_golden_sentences(tmp_path):
    d = _setup_dir(tmp_path, "did")
    _write_csv(
        d,
        "did_summary.csv",
        {
            "tau_t2_items_median": 3.1,
            "tau_t2_items_lo": 0.4,
            "tau_t2_items_hi": 6.0,
            "prob_tau_t2_pos": 0.985,
            "off_floor": False,
            "delta_crossover_items_available": True,
            "delta_crossover_items_median": 1.2,
        },
    )
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    kinds = [s["kind"] for s in payload["sentences"]]
    assert kinds == ["headline", "confidence", "causal", "highlight"]
    texts = [s["text"] for s in payload["sentences"]]
    assert "3.1 items higher" in texts[0]
    assert "randomised comparison" in texts[0]
    assert "98% probability" in texts[1]
    assert "descriptive associations" in texts[2]
    assert "narrowed by about 1.2 items" in texts[3]
    assert "not a second randomised effect" in texts[3]


def test_did_off_floor_uses_percentage_points(tmp_path):
    d = _setup_dir(tmp_path, "did", config=_config("did", outcome_symbol="P"))
    _write_csv(
        d,
        "did_summary.csv",
        {
            "tau_t2_items_median": 0.22,
            "tau_t2_items_lo": 0.05,
            "tau_t2_items_hi": 0.40,
            "prob_tau_t2_pos": 0.99,
            "off_floor": True,
            "delta_crossover_items_available": False,
        },
    )
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    headline = payload["sentences"][0]["text"]
    assert "+22 percentage points" in headline
    assert "scoring above zero" in headline


def test_did_dose_companion_degrades_honestly(tmp_path):
    d = _setup_dir(tmp_path, "did")
    _write_csv(d, "did_summary.csv", {"beta_dose_median": 0.1, "delta_median": 0.2})
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    texts = _texts(payload)
    assert "observational association" in texts
    assert all(s["kind"] != "headline" for s in payload["sentences"])


# --- fallback and global properties ---------------------------------------------


@pytest.mark.parametrize("kind", ["joint", "mechanism", "mediation", "aligned"])
def test_fallback_families_get_honest_placeholder(tmp_path, kind):
    d = _setup_dir(tmp_path, kind)
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    texts = _texts(payload)
    assert "has not yet been written" in texts
    assert kind in texts
    assert "adjusted associations or descriptive quantities" in texts


def test_sentence_cap_and_no_nan_everywhere(tmp_path):
    """Every builder respects the hard cap and never emits ``nan`` text."""
    cases = []
    for kind in ("itt", "gain_factors", "level_factors"):
        d = _setup_dir(tmp_path, kind)
        _write_csv(d, "rope_summary.csv", _rope_row())
        cases.append(d)
    d = _setup_dir(tmp_path, "did")
    _write_csv(
        d,
        "did_summary.csv",
        {
            "tau_t2_items_median": 3.1,
            "tau_t2_items_lo": 0.4,
            "tau_t2_items_hi": 6.0,
            "prob_tau_t2_pos": 0.985,
            "off_floor": False,
            "delta_crossover_items_available": True,
            "delta_crossover_items_median": float("nan"),  # optional field: skipped
        },
    )
    cases.append(d)
    cases.append(_setup_dir(tmp_path, "joint"))
    for d in cases:
        payload = generate_key_findings(d)
        assert len(payload["sentences"]) <= KEY_FINDINGS_MAX_SENTENCES
        assert "nan" not in _texts(payload).lower()
        for s in payload["sentences"]:
            assert s["text"].strip()
            assert s["kind"]


# --- partial and pilot-include guards --------------------------------------------


def test_key_findings_partial_is_a_dumb_renderer():
    text = (REPO / "docs/models/_partials/_key_findings.qmd").read_text()
    assert "key_findings.json" in text
    assert "gate_failed" in text
    assert "not available" in text
    assert "callout-important" in text  # the red withheld-findings warning
    # Self-contained: must not depend on _setup.qmd helpers so #321 can move it.
    assert "_csv(" not in text
    assert "_json(" not in text


def test_reading_guide_is_a_collapsed_callout():
    text = (REPO / "docs/models/_partials/_reading_guide.qmd").read_text()
    assert 'collapse="true"' in text
    assert "How to read this report" in text
    for term in (
        "Posterior distribution",
        "Credible interval",
        "prediction interval",
        "ROPE",
        "Causal vs association",
        "logit",
        "Beta-Binomial",
        "convergence gate",
    ):
        assert term in text, term
    assert "METHODS.md" in text


def test_pilot_reports_include_the_new_partials():
    pilots = ("lrp-rli-itt-010", "lrp-rli-gf-001", "lrp-rli-lf-001", "lrp-rli-did-001")
    for model_id in pilots:
        text = (REPO / f"docs/models/{model_id}/index.qmd").read_text()
        assert "_partials/_key_findings.qmd" in text, model_id
        # Gate first: the box must render after the convergence banner.
        assert text.index("_partials/_convergence.qmd") < text.index(
            "_partials/_key_findings.qmd"
        ), model_id
    itt = (REPO / "docs/models/lrp-rli-itt-010/index.qmd").read_text()
    assert "_partials/_reading_guide.qmd" in itt
