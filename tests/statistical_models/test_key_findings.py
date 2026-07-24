# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the key-findings generator (issue #320).

Golden-sentence tests cover every registered family from synthetic CSV rows,
along with the convergence-gate interlock, missing-CSV degradation, the
no-``nan`` guard and the five-sentence cap. The partial content guards at the
bottom follow the ``test_concurrent_pipeline`` read-the-qmd idiom.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from language_reading_predictors.statistical_models.definitions import KINDS
from language_reading_predictors.statistical_models.reporting import (
    KEY_FINDINGS_FILENAME,
    KEY_FINDINGS_MAX_SENTENCES,
    _KF_BUILDERS,
    convergence_gate_badge_markdown,
    generate_key_findings,
)

REPO = Path(__file__).resolve().parents[2]


def _write_json(d: Path, name: str, payload: dict) -> None:
    with open(d / name, "w") as f:
        json.dump(payload, f)


def _write_csv(d: Path, name: str, row: dict) -> None:
    pd.DataFrame([row]).to_csv(d / name, index=False)


def _write_rows(d: Path, name: str, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(d / name, index=False)


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


def test_malformed_config_degrades(tmp_path):
    d = tmp_path / "bad-config"
    d.mkdir()
    (d / "config.json").write_text("{not json")
    _write_json(d, "diagnostics_summary.json", _passing_gate())
    payload = generate_key_findings(d)
    assert payload["status"] == "not_available"
    assert "could not be parsed" in payload["reason"]


def test_malformed_diagnostics_summary_degrades(tmp_path):
    d = tmp_path / "bad-diag"
    d.mkdir()
    _write_json(d, "config.json", _config("itt"))
    (d / "diagnostics_summary.json").write_text("{not json")
    payload = generate_key_findings(d)
    assert payload["status"] == "not_available"
    assert "could not be parsed" in payload["reason"]


def test_convergence_gate_badge_passes_compactly():
    markdown = convergence_gate_badge_markdown(_passing_gate())
    assert "callout-tip" in markdown
    assert "Sampling-quality gate: passed" in markdown
    assert "All sampling-quality checks passed" in markdown
    assert "Technical checks" in markdown


def test_convergence_gate_badge_fails_closed_and_names_checks():
    markdown = convergence_gate_badge_markdown(
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
    assert "callout-important" in markdown
    assert "Sampling-quality gate: failed" in markdown
    assert "R-hat" in markdown
    assert "divergent transitions" in markdown
    assert "Findings are withheld" in markdown

    unavailable = convergence_gate_badge_markdown(None)
    assert "callout-important" in unavailable
    assert "convergence summary incomplete" in unavailable


def test_non_boolean_gate_verdict_fails_closed_everywhere(tmp_path):
    d = _setup_dir(tmp_path, "itt")
    inconsistent = {
        "passed": "yes",
        "checks": {"rhat": True, "ess": True, "divergences": True, "bfmi": True},
    }
    _write_json(d, "diagnostics_summary.json", inconsistent)
    _write_csv(d, "rope_summary.csv", _rope_row())

    payload = generate_key_findings(d)
    assert payload["status"] == "gate_failed"
    assert payload["sentences"] == []
    assert payload["failing_checks"] == ["convergence summary incomplete"]
    assert "callout-important" in convergence_gate_badge_markdown(inconsistent)


def test_gate_outranks_malformed_config(tmp_path):
    d = tmp_path / "bad-config-failed-gate"
    d.mkdir()
    (d / "config.json").write_text("{not json")
    _write_json(
        d,
        "diagnostics_summary.json",
        {"passed": False, "checks": {"rhat": False, "ess": True, "divergences": True, "bfmi": True}},
    )
    payload = generate_key_findings(d)
    assert payload["status"] == "gate_failed"


def test_negative_effect_reads_as_evidence_of_harm(tmp_path):
    d = _setup_dir(tmp_path, "itt")
    _write_csv(
        d,
        "rope_summary.csv",
        _rope_row(
            items_median=-2.4,
            items_lo=-5.9,
            items_hi=0.3,
            pd=0.03,
            favoured_direction="negative",
            favoured_direction_prob=0.97,
            favoured_direction_label="strong",
        ),
    )
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    confidence = payload["sentences"][1]["text"]
    # Harm-aware (#179): the number and the label qualify the SAME claim.
    assert confidence == (
        "There is a 97% probability that the true effect is negative — strong "
        "evidence that the intervention is harmful."
    )


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
        "Best estimate: the intervention changed Word reading (WR) by "
        "**+2.4 items** over the trial period "
        "(89% credible range -0.3 to +5.9)."
    )
    assert texts[1] == (
        "There is a 94% probability that the true effect is positive — moderate "
        "evidence that the intervention helps."
    )
    assert texts[2] == (
        "The project agreed after its initial results review that a change of at "
        "least 1 item would be the smallest difference that matters in practice. "
        "The probability the benefit reaches that size is 81%, and the probability "
        "the effect is too small to matter either way is 17%; because the threshold "
        "is post-hoc, read this beside the threshold-sensitivity analysis."
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


# --- remaining family archetypes ------------------------------------------------


def _remaining_family_case(tmp_path: Path, kind: str) -> tuple[Path, str]:
    """Synthetic fit artefacts plus one family-specific expected phrase."""
    d = _setup_dir(tmp_path, kind)
    if kind == "joint":
        _write_rows(
            d,
            "joint_treatment_marginal.csv",
            [
                {
                    "outcome": "W",
                    "items_median": 2.4,
                    "items_lo": -0.3,
                    "items_hi": 5.9,
                    "prob_pos": 0.94,
                    "delta_items": 1.0,
                    "prob_benefit_ge_delta": 0.81,
                    "prob_in_rope": 0.17,
                },
                {
                    "outcome": "L",
                    "items_median": 1.2,
                    "items_lo": -1.0,
                    "items_hi": 3.4,
                    "prob_pos": 0.79,
                    "delta_items": 1.0,
                    "prob_benefit_ge_delta": 0.55,
                    "prob_in_rope": 0.31,
                },
            ],
        )
        return d, "Across the 2 outcomes"
    if kind == "mechanism":
        _write_csv(
            d,
            "mechanism_summary.csv",
            {
                "exposure_low": 0,
                "exposure_high": 30,
                "exposure_unit": "L items",
                "items_median": 3.2,
                "items_lo": 0.4,
                "items_hi": 6.8,
                "prob_pos": 0.98,
            },
        )
        return d, "fitted exposure range"
    if kind in {"mediation", "mediation_multi"}:
        _write_rows(
            d,
            "mediation_summary.csv",
            [
                {
                    "quantity": "total",
                    "words_median": 2.5,
                    "words_lo": -0.2,
                    "words_hi": 5.4,
                    "prob_pos": 0.95,
                    "off_floor": False,
                },
                {
                    "quantity": "NIE_joint" if kind == "mediation_multi" else "NIE",
                    "words_median": 0.8,
                    "words_lo": -0.4,
                    "words_hi": 2.1,
                    "prob_pos": 0.87,
                    "off_floor": False,
                },
            ],
        )
        return d, "g-formula decomposition"
    if kind == "aligned":
        _write_csv(
            d,
            "cohort_marginal.csv",
            {
                "trt_items_median": 1.8,
                "trt_items_lo": -0.7,
                "trt_items_hi": 4.5,
                "prob_trt_pos": 0.91,
            },
        )
        return d, "per-protocol cohort association"
    if kind == "adjusted":
        _write_csv(
            d,
            "predicted_gain_words.csv",
            {
                "predictor": "L",
                "label": "Letter sounds",
                "delta_words_mean": 1.9,
                "delta_words_lo": 0.1,
                "delta_words_hi": 3.8,
                "prob_pos": 0.97,
            },
        )
        return d, "clearest adjusted predictor"
    if kind == "corr_factor":
        _write_csv(
            d,
            "factor_correlation_summary.csv",
            {
                "domain_i": "vocabulary",
                "domain_j": "code",
                "median": 0.62,
                "mean": 0.62,
                "lo": 0.18,
                "hi": 0.88,
                "prob_pos": 0.99,
            },
        )
        return d, "latent-domain correlation"
    if kind == "dose_response":
        _write_csv(
            d,
            "dose_marginal_summary.csv",
            {
                "items_median": 0.9,
                "items_lo": -0.3,
                "items_hi": 2.2,
                "prob_pos": 0.89,
            },
        )
        return d, "1-SD increase in sessions"
    if kind == "lcsm":
        _write_csv(
            d,
            "coupling_summary.csv",
            {
                "coefficient": "g_L (prior L -> W change)",
                "median": 0.31,
                "mean": 0.31,
                "lo": 0.02,
                "hi": 0.61,
                "prob_pos": 0.98,
            },
        )
        return d, "longitudinal coupling"
    if kind == "horseshoe":
        _write_csv(
            d,
            "predictor_ranking.csv",
            {
                "rank": 1,
                "predictor": "letter_sounds",
                "p_abs_gt_delta": 0.93,
                "beta_median": 0.42,
                "beta_mean": 0.42,
                "beta_hdi_lo": 0.04,
                "beta_hdi_hi": 0.80,
            },
        )
        return d, "top-ranked predictor"
    if kind == "growth":
        _write_csv(
            d,
            "growth_association_summary.csv",
            {
                "coefficient": "gamma",
                "outcome": "W",
                "median": 0.18,
                "lo89": -0.03,
                "hi89": 0.40,
                "prob_positive": 0.94,
            },
        )
        return d, "baseline non-verbal ability"
    if kind == "historical_joint":
        _write_rows(
            d,
            "measure_correlation_summary.csv",
            [
                {
                    "measure_i": "basread",
                    "measure_j": "bpvs",
                    "label_i": "BAS word reading",
                    "label_j": "BPVS receptive vocabulary",
                    "median": 0.62,
                    "mean": 0.62,
                    "lo": 0.31,
                    "hi": 0.83,
                    "prob_pos": 0.999,
                }
            ],
        )
        return d, "clearest between-child coupling"
    if kind == "historical_growth":
        _write_rows(
            d,
            "posterior_growth_summary.csv",
            [
                {
                    "quantity": "growth_1_3_items",
                    "label": "Wave 1 to wave 3",
                    "readgrp_label": "readers",
                    "mean": 7.2,
                    "q_lo": 4.0,
                    "q50": 7.2,
                    "q_hi": 10.5,
                    "p_gt_0": 0.999,
                }
            ],
        )
        return d, "descriptive natural-history growth"
    if kind == "survival":
        _write_rows(
            d,
            "survival_summary.csv",
            [
                {
                    "term": "tau (log hazard shift, treated)",
                    "median": 0.41,
                    "ci_low": -0.08,
                    "ci_high": 0.91,
                    "hazard_ratio": 1.51,
                    "P(>0)": 0.95,
                },
                {
                    "term": "baseline off-floor prob [t1-t2]",
                    "median": 0.20,
                    "ci_low": 0.08,
                    "ci_high": 0.36,
                    "hazard_ratio": float("nan"),
                    "P(>0)": float("nan"),
                },
            ],
        )
        return d, "hazard ratio"
    if kind == "block_exposure":
        _write_csv(
            d,
            "block_exposure_summary.csv",
            {
                "delta_items_median": 2.1,
                "delta_items_lo": -0.4,
                "delta_items_hi": 4.8,
                "prob_delta_pos": 0.94,
            },
        )
        return d, "parallel-trends association"
    if kind == "concurrent":
        _write_csv(
            d,
            "concurrent_marginals.csv",
            {
                "timepoint": 3,
                "adjustment": "adjusted",
                "term": "L",
                "label": "Letter sounds",
                "role": "association",
                "scale": "+1 SD",
                "items_median": 2.3,
                "items_lo": 0.2,
                "items_hi": 4.5,
                "prob_pos": 0.98,
                "converged": True,
            },
        )
        return d, "same-wave predictor"
    if kind == "long_corr_factor":
        _write_csv(
            d,
            "latent_items_slopes.csv",
            {
                "wave": 2,
                "predictor_indicator": "R",
                "target_indicator": "L",
                "items_per_item_mean": 0.24,
                "items_per_item_lo": 0.05,
                "items_per_item_hi": 0.44,
                "prob_pos": 0.99,
            },
        )
        return d, "translated latent coupling"
    raise AssertionError(f"No synthetic case for {kind}")


@pytest.mark.parametrize(
    "kind",
    sorted(
        KINDS
        - {
            "itt",
            "did",
            "gain_factors",
            "level_factors",
        }
    ),
)
def test_every_remaining_family_has_bespoke_findings(tmp_path, kind):
    d, expected = _remaining_family_case(tmp_path, kind)
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    assert expected in _texts(payload)
    assert "has not yet been written" not in _texts(payload)
    assert 3 <= len(payload["sentences"]) <= KEY_FINDINGS_MAX_SENTENCES


def test_joint_findings_identify_smallest_difference_as_post_hoc(tmp_path):
    d, _ = _remaining_family_case(tmp_path, "joint")

    payload = generate_key_findings(d)

    assert "post-hoc, project-agreed smallest-important difference" in _texts(payload)


def test_horseshoe_findings_do_not_claim_threshold_was_pre_specified(tmp_path):
    d, _ = _remaining_family_case(tmp_path, "horseshoe")

    payload = generate_key_findings(d)

    assert "model's worth-noticing coefficient threshold" in _texts(payload)
    assert "pre-specified" not in _texts(payload)


def test_mechanism_findings_headline_interaction_when_present(tmp_path):
    """#404 review: a moderated mechanism fit headlines gamma_int (median, 50%/89%
    intervals, tail probability) ahead of the unmoderated curve contrast."""
    d = _setup_dir(tmp_path, "mechanism")
    _write_csv(
        d,
        "mechanism_summary.csv",
        {
            "exposure_low": 0,
            "exposure_high": 30,
            "exposure_unit": "L items",
            "items_median": 3.2,
            "items_lo": 0.4,
            "items_hi": 6.8,
            "prob_pos": 0.98,
        },
    )
    _write_csv(
        d,
        "interaction_summary.csv",
        {
            "gamma_int_median": -0.33,
            "gamma_int_mean": -0.33,
            "gamma_int_lo": -0.57,
            "gamma_int_hi": -0.09,
            "gamma_int_lo50": -0.42,
            "gamma_int_hi50": -0.24,
            "prob_gamma_int_pos": 0.06,
            "gamma_mod_median": 0.1,
            "gamma_mod_mean": 0.1,
            "gamma_mod_lo": -0.2,
            "gamma_mod_hi": 0.4,
            "gamma_mod_lo50": -0.05,
            "gamma_mod_hi50": 0.25,
            "prob_gamma_mod_pos": 0.7,
        },
    )
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    first = payload["sentences"][0]["text"]
    # gamma_int leads, on the logit scale, with median + both intervals + tail prob.
    assert "moderation coefficient" in first
    assert "logit" in first
    assert "-0.33" in first and "-0.42" in first and "-0.57" in first
    assert "P(> 0) = 0.06" in first
    # The unmoderated curve contrast is retained as supporting context.
    assert "fitted exposure range" in _texts(payload)


def test_mechanism_findings_without_interaction_are_unchanged(tmp_path):
    """A non-moderated mechanism fit (no interaction_summary.csv) still leads with
    the curve contrast — the interaction headline is strictly conditional."""
    d, expected = _remaining_family_case(tmp_path, "mechanism")
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    assert "moderation coefficient" not in _texts(payload)
    assert expected in payload["sentences"][0]["text"]


def test_builder_registry_covers_every_declared_family():
    assert KINDS <= _KF_BUILDERS.keys()


def test_corr_factor_structural_only_keeps_three_sentence_contract(tmp_path):
    d = _setup_dir(tmp_path, "corr_factor")
    _write_csv(
        d,
        "structural_summary.csv",
        {
            "coefficient": "beta_code_to_reading",
            "median": 0.37,
            "mean": 0.37,
            "lo": -0.05,
            "hi": 0.79,
            "prob_pos": 0.95,
        },
    )

    payload = generate_key_findings(d)

    assert payload["status"] == "ok"
    assert len(payload["sentences"]) == 3
    assert "clearest structural slope" in _texts(payload)


def test_unknown_future_family_keeps_honest_fallback(tmp_path):
    d = _setup_dir(tmp_path, "future_family")
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    assert "has not yet been written" in _texts(payload)


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


def test_all_statistical_reports_use_the_findings_first_order():
    expected = (
        "_partials/_header.qmd",
        "_partials/_setup.qmd",
        "_partials/_gate_badge.qmd",
        "_partials/_key_findings.qmd",
        "_partials/_reading_guide.qmd",
        "_partials/_priors.qmd",
        "_partials/_prior_predictive.qmd",
        "_partials/_results_",
        "_partials/_technical.qmd",
        "_partials/_footer.qmd",
    )
    statistical_reports = []
    for path in sorted((REPO / "docs/models").glob("*/index.qmd")):
        text = path.read_text()
        if "_partials/_setup.qmd" not in text:
            continue
        statistical_reports.append(path)
        missing = [name for name in expected if name not in text]
        assert not missing, (
            f"{path.parent.name}: missing expected partials: {', '.join(missing)}"
        )
        positions = [text.index(name) for name in expected]
        assert positions == sorted(positions), path.parent.name
        assert "_partials/_convergence.qmd" not in text, path.parent.name
        assert "_partials/_diagnostics.qmd" not in text, path.parent.name
    assert statistical_reports


def test_technical_partial_keeps_full_checks_inside_the_fold():
    text = (REPO / "docs/models/_partials/_technical.qmd").read_text()
    assert 'collapse="true"' in text
    assert 'title="Technical checks"' in text
    assert text.count("_partials/_convergence.qmd") == 1
    assert text.count("_partials/_diagnostics.qmd") == 1
