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

from language_reading_predictors.statistical_models import release as _release
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
    return _diag(rhat=True, ess=True, divergences=True, bfmi=True)


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


def _write_core_inventory(d: Path) -> None:
    """Place the stored-path core inventory and a matching manifest.

    The release decision's artefacts stage fails closed without a readable,
    non-empty ``artifact_manifest.json`` and without the core outputs every
    family writes (2026-08-22 ITT audit, finding 2). A fixture standing in for a
    *stored* fit needs both before the key-findings assertions below can be about
    key findings. Taken from the release module's own contract so widening the
    floor cannot silently leave these fixtures behind.
    """
    from language_reading_predictors.statistical_models import release as _release

    for name in _release._CORE_ARTIFACTS_BASE:
        path = d / name
        if not path.exists():
            path.write_bytes(b"fixture")
    _write_json(
        d,
        "artifact_manifest.json",
        {
            "artifacts": [
                {"filename": name, "status": "written", "required": True}
                for name in _release._CORE_ARTIFACTS_BASE
            ]
        },
    )


def _setup_dir(
    tmp_path: Path,
    kind: str,
    *,
    config: dict | None = None,
    directory_name: str | None = None,
) -> Path:
    d = tmp_path / (directory_name or f"{kind}-dev")
    d.mkdir(parents=True)
    _write_json(d, "config.json", config or _config(kind))
    _write_json(d, "diagnostics_summary.json", _passing_gate())
    _write_core_inventory(d)
    if kind == "itt":
        _write_rows(
            d,
            "analysis_set.csv",
            [
                {
                    "arm": "intervention",
                    "G": 1,
                    "randomised_n": 29,
                    "lost_to_follow_up_n": 1,
                    "analysed_archive_n": 28,
                    "discontinued_but_followed_n": 2,
                    "available_t1_n": 28,
                    "fitted_n": 28,
                    "absent_from_archive_n": 1,
                    "not_in_fitted_analysis_n": 1,
                    "excluded_after_archive_n": 0,
                },
                {
                    "arm": "control",
                    "G": 0,
                    "randomised_n": 28,
                    "lost_to_follow_up_n": 2,
                    "analysed_archive_n": 26,
                    "discontinued_but_followed_n": 2,
                    "available_t1_n": 26,
                    "fitted_n": 26,
                    "absent_from_archive_n": 2,
                    "not_in_fitted_analysis_n": 2,
                    "excluded_after_archive_n": 0,
                },
            ],
        )
    if _release.gate_applies(_read_json(d, "config.json")):
        _write_psense(d, term=_release.causal_term_for(_read_json(d, "config.json")))
    return d


def _read_json(d: Path, name: str) -> dict:
    return json.loads((d / name).read_text())


def _write_psense(
    d: Path, *, prior: float = 0.01, likelihood: float = 0.02, term: str = "tau"
) -> None:
    """Write a power-scaling row for the #392 robustness release gate.

    Every gated fixture needs one because the gate is deliberately fail-closed: a fit
    with no ``psense_summary.csv`` has not been measured clean, it has not been
    measured, and that withholds. Defaults are below the 0.05 flag threshold, so a
    fixture releases unless it overrides them. ArviZ writes a tick for an unflagged
    parameter, which is reproduced here so the fixture matches the real artefact.

    ``term`` follows the family, since the gate reads ``beta_trt`` for gain factors,
    ``b_grp_time[1]`` for level factors and ``tau_t2`` (or a dose slope) for DiD.
    """
    frame = pd.DataFrame(
        [{"prior": prior, "likelihood": likelihood, "diagnosis": "✓"}],
        index=[term],
    )
    frame.to_csv(d / "psense_summary.csv")


def _texts(payload: dict) -> str:
    return " ".join(s["text"] for s in payload["sentences"])


# --- gate interlock and degradation -------------------------------------------


def test_gate_failed_withholds_findings(tmp_path):
    d = _setup_dir(tmp_path, "itt")
    _write_json(
        d,
        "diagnostics_summary.json",
        _diag(rhat=False, ess=True, divergences=False, bfmi=True),
    )
    _write_csv(d, "rope_summary.csv", _rope_row())
    payload = generate_key_findings(d)
    assert payload["status"] == "gate_failed"
    assert payload["sentences"] == []
    assert "R-hat" in payload["failing_checks"]
    assert "divergent transitions" in payload["failing_checks"]
    assert (d / KEY_FINDINGS_FILENAME).exists()


def _diag(**checks: bool) -> dict:
    """A diagnostics-gate payload; ``passed`` is True iff every check is True."""
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "divergences": 0 if checks["divergences"] else 1,
        "max_rhat": 1.001 if checks["rhat"] else 1.02,
        "min_ess": 1000.0 if checks["ess"] else 100.0,
        "bfmi_per_chain": [0.8, 0.9] if checks["bfmi"] else [0.2, 0.9],
    }


def _diverged() -> dict:
    return _diag(rhat=True, ess=True, divergences=False, bfmi=True)


def test_legacy_model_spec_gate_exception_does_not_override_failure(tmp_path):
    # The 2026-08-02 policy retired permanent model-spec waivers. A qualification must
    # be trace-bound and separately reviewed, so an old config dictionary fails closed.
    legacy_exception = {
        "checks": ["divergences"],
        "reason": "legacy model-spec waiver",
        "issue": 409,
        "signed_off": "2026-07-24",
    }
    d = _setup_dir(
        tmp_path,
        "itt",
        config=_config("itt", spec_extra={"gate_exception": legacy_exception}),
    )
    _write_json(d, "diagnostics_summary.json", _diverged())
    _write_csv(d, "rope_summary.csv", _rope_row())
    payload = generate_key_findings(d)
    assert payload["status"] == "gate_failed"
    assert "divergent transitions" in payload["failing_checks"]
    badge = convergence_gate_badge_markdown(_diverged(), legacy_exception)
    assert "callout-important" in badge
    assert "recorded exception" not in badge


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


def test_itt_missing_analysis_set_withholds_causal_findings(tmp_path):
    d = _setup_dir(tmp_path, "itt")
    (d / "analysis_set.csv").unlink()
    _write_csv(d, "rope_summary.csv", _rope_row())
    payload = generate_key_findings(d)
    assert payload["status"] == "not_available"
    assert "causal population" in payload["reason"]
    assert payload["sentences"] == []


def test_itt_inconsistent_analysis_set_withholds_causal_findings(tmp_path):
    d = _setup_dir(tmp_path, "itt")
    audit = pd.read_csv(d / "analysis_set.csv")
    audit.loc[audit["G"] == 1, "not_in_fitted_analysis_n"] = 99
    audit.to_csv(d / "analysis_set.csv", index=False)
    _write_csv(d, "rope_summary.csv", _rope_row())
    payload = generate_key_findings(d)
    assert payload["status"] == "not_available"
    assert "arithmetic" in payload["reason"]
    assert payload["sentences"] == []


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
        _diag(rhat=False, ess=True, divergences=False, bfmi=True)
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
    inconsistent = _passing_gate()
    inconsistent["passed"] = "yes"
    _write_json(d, "diagnostics_summary.json", inconsistent)
    _write_csv(d, "rope_summary.csv", _rope_row())

    payload = generate_key_findings(d)
    assert payload["status"] == "gate_failed"
    assert payload["sentences"] == []
    assert payload["failing_checks"] == ["convergence summary incomplete"]
    assert "callout-important" in convergence_gate_badge_markdown(inconsistent)


def test_true_gate_verdict_cannot_override_failed_divergence_check(tmp_path):
    d = _setup_dir(tmp_path, "itt")
    inconsistent = _diag(rhat=True, ess=True, divergences=False, bfmi=True)
    inconsistent["passed"] = True
    _write_json(d, "diagnostics_summary.json", inconsistent)
    _write_csv(d, "rope_summary.csv", _rope_row())

    payload = generate_key_findings(d)
    assert payload["status"] == "gate_failed"
    assert payload["failing_checks"] == ["divergent transitions"]
    assert payload["sentences"] == []
    assert "callout-important" in convergence_gate_badge_markdown(inconsistent)


def test_true_gate_verdict_cannot_override_missing_required_check(tmp_path):
    d = _setup_dir(tmp_path, "itt")
    incomplete = _passing_gate()
    del incomplete["checks"]["divergences"]
    _write_json(d, "diagnostics_summary.json", incomplete)
    _write_csv(d, "rope_summary.csv", _rope_row())

    payload = generate_key_findings(d)
    assert payload["status"] == "gate_failed"
    assert payload["failing_checks"] == ["convergence summary incomplete"]
    assert payload["sentences"] == []
    assert "callout-important" in convergence_gate_badge_markdown(incomplete)


def test_stored_pass_cannot_override_failing_raw_measurements(tmp_path):
    d = _setup_dir(tmp_path, "itt")
    inconsistent = _passing_gate()
    inconsistent.update(
        {
            "divergences": 17,
            "max_rhat": 1.25,
            "min_ess": 10,
            "bfmi_per_chain": [0.1, 0.8],
        }
    )
    _write_json(d, "diagnostics_summary.json", inconsistent)
    _write_csv(d, "rope_summary.csv", _rope_row())

    payload = generate_key_findings(d)
    assert payload["status"] == "gate_failed"
    assert payload["failing_checks"] == [
        "R-hat",
        "effective sample size",
        "divergent transitions",
        "sampling energy (BFMI)",
    ]
    assert payload["sentences"] == []
    assert "callout-important" in convergence_gate_badge_markdown(inconsistent)


def test_gate_outranks_malformed_config(tmp_path):
    d = tmp_path / "bad-config-failed-gate"
    d.mkdir()
    (d / "config.json").write_text("{not json")
    _write_json(
        d,
        "diagnostics_summary.json",
        _diag(rhat=False, ess=True, divergences=True, bfmi=True),
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
        "Best estimate: the model-estimated intervention-minus-comparison contrast "
        "for Word reading (WR) was **+2.4 items** over the trial period in the "
        "available-case modified ITT analysis (89% credible range -0.3 to +5.9)."
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
    assert "54 fitted children" in texts[3]
    assert "29" not in texts[3]  # fitted arm counts, not published allocation
    assert "28 immediate-intervention" in texts[3]
    assert "26 waiting-list" in texts[3]
    assert "available-case modified ITT estimate" in texts[3]
    assert "depend jointly on assigned arm and potential outcomes" in texts[3]
    assert "not the effect for all 57 randomised children" in texts[3]


def _bounds_row(**overrides) -> dict:
    row = {
        "outcome": "L",
        "scale": "proportion_correct",
        "observed_intervention_n": 28,
        "observed_control_n": 26,
        "missing_intervention_n": 1,
        "missing_control_n": 2,
        "worst_case_items_lower": 4.053,
        "worst_case_items_upper": 7.442,
        "n_trials": 32,
    }
    row.update(overrides)
    return row


def test_itt_attrition_bounds_are_quoted_in_the_causal_sentence(tmp_path):
    """Every graded itt fit writes model-free extreme-case attrition bounds; the
    causal sentence quotes them (2026-08-19) and says whether the direction
    survives any completion of the missing children. They ride on the causal
    sentence — never dropped by the five-sentence cap — rather than displacing
    the size-of-benefit statement as a sixth sentence."""
    d = _setup_dir(tmp_path, "itt", config=_config("itt", outcome_symbol="L"))
    _write_csv(d, "rope_summary.csv", _rope_row())
    _write_csv(d, "attrition_bounds.csv", _bounds_row())
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    kinds = [s["kind"] for s in payload["sentences"]]
    assert kinds == ["headline", "confidence", "rope", "causal"]
    text = payload["sentences"][3]["text"]
    assert "not the effect for all 57 randomised children." in text
    assert "Completing the 3 randomised children with no timepoint-2 score" in text
    assert "(1 intervention, 2 control)" in text
    assert "between +4.1 and +7.4 items" in text
    assert "does not depend on how those outcomes are completed" in text
    assert "fitted for word reading only" in text


def test_itt_attrition_bounds_clause_flags_a_straddling_bound(tmp_path):
    d = _setup_dir(tmp_path, "itt", config=_config("itt", outcome_symbol="EG"))
    _write_csv(d, "rope_summary.csv", _rope_row())
    _write_csv(
        d,
        "attrition_bounds.csv",
        _bounds_row(
            outcome="EG",
            missing_control_n=3,
            worst_case_items_lower=-1.36,
            worst_case_items_upper=3.88,
            n_trials=37,
        ),
    )
    text = generate_key_findings(d)["sentences"][3]["text"]
    assert "Completing the 4 randomised children" in text
    assert "between -1.4 and +3.9 marks" in text
    assert "could reverse direction" in text


def test_itt_attrition_bounds_clause_is_optional_and_skips_floor_rule_fits(tmp_path):
    # No table -> no clause, findings unchanged (the golden test above).
    d = _setup_dir(tmp_path, "itt")
    _write_csv(d, "rope_summary.csv", _rope_row())
    assert "Completing the" not in generate_key_findings(d)["sentences"][3]["text"]
    # Floor-rule fits: the raw post-score contrast is not their estimand.
    cfg = _config("itt", outcome_symbol="P")
    cfg["resolved_run_plan"] = {"floor_rule": True}
    d2 = _setup_dir(tmp_path, "itt", config=cfg, directory_name="itt-floor")
    _write_csv(d2, "rope_summary.csv", _rope_row())
    _write_csv(
        d2,
        "attrition_bounds.csv",
        _bounds_row(outcome="P", worst_case_items_lower=-14.4, worst_case_items_upper=-1.4),
    )
    texts = [s["text"] for s in generate_key_findings(d2)["sentences"]]
    assert not any("Completing the" in t for t in texts)


def test_word_reading_key_findings_label_full_57_as_missing_data_sensitivity(
    tmp_path,
):
    from language_reading_predictors.statistical_models.itt_missingness import (
        DEFAULT_DELTA_ITEMS,
        MISSINGNESS_SUMMARY_FILENAME,
        sha256_file,
    )

    config = _config(
        "itt",
        model_id="lrp-rli-itt-010",
        resolved_run_plan={"score_mean_link": "logit"},
    )
    d = _setup_dir(tmp_path, "itt", config=config)
    analysis = pd.read_csv(d / "analysis_set.csv")
    control = analysis["G"].eq(0)
    analysis.loc[control, "fitted_n"] = 25
    analysis.loc[control, "not_in_fitted_analysis_n"] = 3
    analysis.loc[control, "excluded_after_archive_n"] = 1
    analysis.to_csv(d / "analysis_set.csv", index=False)
    _write_csv(d, "rope_summary.csv", _rope_row())
    rows = [
        {
            "scenario": "screening_model_observed_profiles",
            "scenario_class": "bridge",
            "estimand_class": "common_profile_standardisation",
            "target_population": "53 common observed profiles",
            "effect_items_median": 2.1,
            "effect_items_lo89": -0.4,
            "effect_items_hi89": 4.7,
        },
        {
            "scenario": "mar_all_57",
            "scenario_class": "missing_at_random",
            "estimand_class": "common_profile_standardisation",
            "target_population": "all 57 common profiles",
            "effect_items_median": 1.9,
            "effect_items_lo89": -0.7,
            "effect_items_hi89": 4.5,
        },
        {
            "scenario": "jump_to_reference_intervention_nonstarter",
            "scenario_class": "reference_based",
            "estimand_class": "randomised_arm_factual_completion",
            "target_population": "29 intervention versus 28 control profiles",
            "effect_items_median": 1.6,
            "effect_items_lo89": -0.9,
            "effect_items_hi89": 4.2,
        },
    ]
    for delta_i in DEFAULT_DELTA_ITEMS:
        for delta_c in DEFAULT_DELTA_ITEMS:
            rows.append(
                {
                    "scenario": f"delta_i_{delta_i:+g}_c_{delta_c:+g}",
                    "scenario_class": "arm_specific_delta_grid",
                    "estimand_class": "randomised_arm_factual_completion",
                    "target_population": "29 intervention versus 28 control profiles",
                    "delta_intervention_items": delta_i,
                    "delta_control_items": delta_c,
                    "clipped_intervention_fraction": 1.0 if delta_i == -8 else 0.0,
                    "clipped_control_fraction": 0.67 if delta_c == -8 else 0.0,
                    "effect_items_median": 1.9 + delta_i / 29 - 3 * delta_c / 28,
                    "effect_items_lo89": -1.0,
                    "effect_items_hi89": 5.0,
                }
            )
    _write_rows(d, MISSINGNESS_SUMMARY_FILENAME, rows)
    _write_rows(
        d,
        "attrition_bounds.csv",
        [
            {
                "outcome": "W",
                "worst_case_items_lower": -6.29,
                "worst_case_items_upper": 4.90,
            }
        ],
    )
    decision = _release.ReleaseEvaluation(
        status="ok",
        stage="robustness",
        config=config,
    )

    payload = generate_key_findings(d, decision=decision)

    assert [sentence["kind"] for sentence in payload["sentences"]] == [
        "headline",
        "confidence",
        "rope",
        "sensitivity",
        "causal",
    ]
    sensitivity = payload["sentences"][3]["text"]
    assert sensitivity.startswith("Missing-outcome sensitivity:")
    assert "same 53 observed outcomes" in sensitivity
    assert "all 57 under MAR" in sensitivity
    assert "factual randomised arms" in sensitivity
    assert "one intervention non-starter" in sensitivity
    assert "100%" in sensitivity and "67%" in sensitivity
    assert "-6.3 to +4.9 items" in sensitivity
    assert "unrestricted missing outcomes can reverse direction" in sensitivity
    assert "assumption-dependent secondary estimates" in sensitivity
    assert "available-case modified ITT analysis" in payload["sentences"][0]["text"]
    assert payload["sentences"][1]["text"].startswith(
        "For the 53-outcome available-case modified ITT model of record"
    )
    assert payload["itt_missingness_sensitivity_sha256"] == sha256_file(
        d / MISSINGNESS_SUMMARY_FILENAME
    )


def _blending_config(model_id: str, link: str) -> dict:
    return _config(
        "itt",
        model_id=model_id,
        outcome_symbol="B",
        model_settings={"score_mean_link": link},
        resolved_run_plan={"score_mean_link": link},
    )


def _write_blending_link_summary(d: Path) -> Path:
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        BLENDING_RENDERED_SCIENTIFIC_ARTIFACTS,
        BLENDING_SENSITIVITY_SCHEMA_VERSION,
    )
    from language_reading_predictors.statistical_models.sensitivity import sha256_file

    (d / "trace.nc").write_text("primary B trace")
    (d / "pareto_k.csv").write_text("primary B row map")
    companion_dir = d.parent / "lrp-rli-itt-108-dev"
    companion_dir.mkdir()
    _write_json(
        companion_dir,
        "config.json",
        _blending_config(
            "lrp-rli-itt-108",
            "three_choice_guessing_floor",
        ),
    )
    (companion_dir / "trace.nc").write_text("companion B trace")
    (companion_dir / "pareto_k.csv").write_text("companion B row map")
    (companion_dir / "diagnostics_summary.json").write_bytes(
        (d / "diagnostics_summary.json").read_bytes()
    )
    (companion_dir / "analysis_set.csv").write_bytes(
        (d / "analysis_set.csv").read_bytes()
    )
    # Built by hand rather than through ``_setup_dir``, so it needs the stored
    # path's core inventory explicitly.
    _write_core_inventory(companion_dir)
    _write_psense(companion_dir)

    def _artifact_hashes(directory: Path, label: str) -> str:
        for name in BLENDING_RENDERED_SCIENTIFIC_ARTIFACTS:
            path = directory / name
            if not path.is_file():
                path.write_bytes(f"{label}:{name}".encode())
        return json.dumps(
            {
                name: sha256_file(directory / name)
                for name in BLENDING_RENDERED_SCIENTIFIC_ARTIFACTS
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    primary_artifacts = _artifact_hashes(d, "primary")
    companion_artifacts = _artifact_hashes(companion_dir, "companion")

    primary_trace_sha = sha256_file(d / "trace.nc")
    companion_trace_sha = sha256_file(companion_dir / "trace.nc")
    primary_row_sha = sha256_file(d / "pareto_k.csv")
    companion_row_sha = sha256_file(companion_dir / "pareto_k.csv")
    shared = {
        "schema_version": BLENDING_SENSITIVITY_SCHEMA_VERSION,
        "config": "dev",
        "outcome": "B",
        "sensitivity_of": "lrp-rli-itt-008",
        "data_sha256": "a" * 64,
        "environment_lock_sha256": "b" * 64,
        "source_commit": "c" * 40,
        "n": 54,
        "n_intervention": 28,
        "n_control": 26,
        "subject_order_sha256": "d" * 64,
        "treatment_order_sha256": "e" * 64,
        "sampling_draws": 100,
        "sampling_tune": 100,
        "sampling_chains": 2,
        "sampling_target_accept": 0.95,
        "sampling_random_seed": 47,
        "ci_prob": 0.89,
        "converged": True,
        "max_rhat": 1.0,
        "min_ess": 800.0,
        "min_bfmi": 0.8,
        "n_divergences": 0,
        "loo_elpd": -10.0,
        "loo_p": 2.0,
        "pareto_k_max": 0.4,
        "good_k_threshold": 0.7,
        "loo_reliable": True,
        "prob_meaningful_benefit": 0.4,
        "prob_practically_negligible": 0.5,
        "prior_effect_items_median": 0.0,
        "prior_effect_items_lo": -1.3,
        "prior_effect_items_hi": 1.3,
        "guessing_floor_minus_logit_elpd": 1.0,
        "guessing_floor_minus_logit_elpd_se": 0.5,
    }
    _write_rows(
        d,
        "blending_link_sensitivity.csv",
        [
            {
                **shared,
                "model_id": "lrp-rli-itt-008",
                "score_mean_link": "logit",
                "config_sha256": sha256_file(d / "config.json"),
                "trace_sha256": primary_trace_sha,
                "trace_file": f"lrp-rli-itt-008-{primary_trace_sha[:16]}.nc",
                "row_map_sha256": primary_row_sha,
                "row_map_file": f"lrp-rli-itt-008-rows-{primary_row_sha[:16]}.csv",
                "scientific_artifacts_sha256": primary_artifacts,
                "effect_items_median": 1.0,
                "effect_items_lo": 0.2,
                "effect_items_hi": 1.8,
                "prob_effect_positive": 0.95,
            },
            {
                **shared,
                "model_id": "lrp-rli-itt-108",
                "score_mean_link": "three_choice_guessing_floor",
                "config_sha256": sha256_file(companion_dir / "config.json"),
                "trace_sha256": companion_trace_sha,
                "trace_file": f"lrp-rli-itt-108-{companion_trace_sha[:16]}.nc",
                "row_map_sha256": companion_row_sha,
                "row_map_file": f"lrp-rli-itt-108-rows-{companion_row_sha[:16]}.csv",
                "scientific_artifacts_sha256": companion_artifacts,
                "effect_items_median": 0.5,
                "effect_items_lo": -0.2,
                "effect_items_hi": 1.1,
                "prob_effect_positive": 0.85,
                "loo_elpd": -9.0,
            },
        ],
    )
    (companion_dir / "blending_link_sensitivity.csv").write_bytes(
        (d / "blending_link_sensitivity.csv").read_bytes()
    )
    # The local evaluator byte-binds the installed copies to the central archive
    # manifest (finding 1, notes/202608201205-itt-code-review-findings.md); the
    # calling tests therefore nest the fit dirs one level down (``models/<fit>``)
    # so this per-test archive lands inside tmp_path, mirroring production.
    archive = d.parent.parent / "blending_link_sensitivity"
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "blending_link_sensitivity.csv").write_bytes(
        (d / "blending_link_sensitivity.csv").read_bytes()
    )
    return companion_dir


def test_blending_headline_is_withheld_without_trace_backed_link_pair(tmp_path):
    d = _setup_dir(
        tmp_path,
        "itt",
        config=_blending_config("lrp-rli-itt-008", "logit"),
        directory_name="lrp-rli-itt-008-dev",
    )
    _write_csv(d, "rope_summary.csv", _rope_row(items_median=1.0))
    payload = generate_key_findings(d)
    # Since the 2026-08-20 review the missing pair is caught by the release
    # decision's robustness stage (finding 1), before the key-findings builder.
    assert payload["status"] == "robustness_unresolved"
    assert "phoneme-blending link pair" in payload["reason"]
    assert "B link sensitivity is missing" in payload["reason"]
    assert payload["sentences"] == []


def test_blending_key_findings_show_both_current_links(tmp_path):
    from language_reading_predictors.statistical_models.sensitivity import sha256_file

    d = _setup_dir(
        tmp_path,
        "itt",
        config=_blending_config("lrp-rli-itt-008", "logit"),
        directory_name="models/lrp-rli-itt-008-dev",
    )
    # Deliberately conflict with the bundle. B headlines and direction must come
    # from its trace-recomputed paired row, never this separately stored table.
    _write_csv(d, "rope_summary.csv", _rope_row(items_median=-9.0, pd=0.01))
    companion_dir = _write_blending_link_summary(d)
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    kinds = [sentence["kind"] for sentence in payload["sentences"]]
    assert kinds == ["headline", "sensitivity", "confidence", "causal"]
    texts = _texts(payload)
    assert "available-case modified ITT estimate" in texts
    assert "not a full-randomised-cohort ITT estimate" in texts
    assert "Under the ordinary-logit model" in texts
    assert "ordinary logit model gives +1.0 items" in texts
    assert "one-in-three guessing-floor model gives +0.5 items" in texts
    assert "Read neither link in isolation" in texts
    assert "95% probability" in texts
    assert "-9.0 items" not in texts
    assert payload["blending_link_sensitivity_sha256"] == sha256_file(
        d / "blending_link_sensitivity.csv"
    )

    companion_payload = generate_key_findings(companion_dir)
    assert companion_payload["status"] == "ok", companion_payload.get("reason")
    companion_texts = _texts(companion_payload)
    assert "Under the one-in-three guessing-floor model" in companion_texts
    assert "**+0.5 items**" in companion_texts
    assert "85% probability" in companion_texts


def test_non_itt_blending_outcome_does_not_require_the_paired_bundle(tmp_path):
    """A ``B`` outcome outside the registered pair must still finalise (#466 scope).

    ``blending_sensitivity`` builds its two-trace bundle for ``lrp-rli-itt-008`` and
    ``lrp-rli-itt-108`` only, but nine further models across the aligned, concurrent,
    did, dose_response, gain_factors, level_factors and mediation families share the
    ``B`` outcome symbol. Stamping the bundle hash on outcome symbol alone raised
    ``FileNotFoundError`` inside ``runtime.finalize_report`` — after sampling, discarding the
    whole fit — because those families' builders never reach the catchable
    ``_KeyFindingsUnavailable`` that ``_kf_build_itt`` raises.
    """
    d, _ = _remaining_family_case(tmp_path, "aligned")
    config = json.loads((d / "config.json").read_text())
    config["model_id"] = "lrp-rli-al-006"
    config["outcome_symbol"] = "B"
    _write_json(d, "config.json", config)

    assert not (d / "blending_link_sensitivity.csv").exists()
    payload = generate_key_findings(d)

    assert payload["status"] == "ok", payload.get("reason")
    assert "blending_link_sensitivity_sha256" not in payload
    assert payload["sentences"]


def test_aligned_off_floor_uses_resolved_plan_and_percentage_points(tmp_path):
    """The aligned pipeline stores its likelihood in the resolved run plan."""
    config = _config(
        "aligned",
        outcome_symbol="P",
        resolved_run_plan={
            "likelihood": "bernoulli_offfloor",
            "off_floor": True,
        },
    )
    d = _setup_dir(tmp_path, "aligned", config=config)
    _write_csv(
        d,
        "cohort_marginal.csv",
        {
            "trt_items_median": 0.032,
            "trt_items_lo": -0.072,
            "trt_items_hi": 0.132,
            "prob_trt_pos": 0.70,
        },
    )

    payload = generate_key_findings(d)

    headline = payload["sentences"][0]["text"]
    assert "+3.2 percentage points" in headline
    assert "items" not in headline


def test_blending_link_summary_stale_for_current_config_withholds(tmp_path):
    d = _setup_dir(
        tmp_path,
        "itt",
        config=_blending_config("lrp-rli-itt-008", "logit"),
        directory_name="models/lrp-rli-itt-008-dev",
    )
    _write_csv(d, "rope_summary.csv", _rope_row(items_median=1.0))
    _write_blending_link_summary(d)
    config = json.loads((d / "config.json").read_text())
    config["title"] = "changed after sensitivity installation"
    _write_json(d, "config.json", config)
    payload = generate_key_findings(d)
    # The release decision's robustness stage now catches the stale pair first
    # (finding 1, 2026-08-20 review); the reason still names the changed file.
    assert payload["status"] == "robustness_unresolved"
    assert "config has changed" in payload["reason"]


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
    assert "available-case modified ITT analysis" in headline
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
    assert "only potentially cause-and-effect estimate" in texts[3]
    assert "fitted available-case rows" in texts[3]
    assert "the child's own starting point on this measure" in texts[4]
    assert "99.9%" in texts[4]  # a certainty of 1.0 in finite draws caps at 99.9%
    assert "100%" not in texts[4]
    assert "not a cause" in texts[4]


def test_gain_factors_highlight_can_pick_taught_vocabulary_baseline(tmp_path):
    # Gain-factors code review 2026-08-20, finding 3: gamma_TR / gamma_TE / gamma_N
    # were absent from the highlight's label map, so the "most clearly resolved
    # link" sentence silently fell back to a weaker labelled association (live in
    # the stored gf-008 fit, where gamma_TR at P = 0.9995 lost to gamma_own at
    # P = 0.9988). With the labels present the strongest association must win.
    d = _setup_dir(tmp_path, "gain_factors")
    _write_csv(d, "rope_summary.csv", _rope_row())
    pd.DataFrame(
        [
            {"term": "beta_trt", "role": "causal", "median": 0.4, "prob_positive": 0.95},
            {"term": "gamma_own", "role": "association", "median": 0.27, "prob_positive": 0.9988},
            {"term": "gamma_TR", "role": "association", "median": 0.23, "prob_positive": 0.9995},
        ]
    ).to_csv(d / "factor_summary.csv", index=False)
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    highlight = next(s for s in payload["sentences"] if s["kind"] == "highlight")
    assert "taught receptive vocabulary at the start of the period" in highlight["text"]


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
    # No psense_summary.csv → no caution bullet (base case is four sentences).
    assert [s["kind"] for s in payload["sentences"]] == [
        "headline", "confidence", "rope", "causal"
    ]


def test_level_factors_caveats_the_headline_as_at_mean_ability(tmp_path):
    # #389 finding 1 / #271 item 5: the t2 items-scale headline nets out the *full*
    # group contribution and adds back only b_grp_time[1], excluding the
    # time-invariant gamma_grp_ability term (identified mostly from the three
    # non-randomised timepoints). The ability-dependent part of the benefit is
    # therefore held at *mean ability* while every other feature of each fitted t2
    # row is averaged over, and the box has to describe that average rather than
    # claiming a prediction for one typical child (#584 finding 5).
    d = _setup_dir(tmp_path, "level_factors")
    _write_csv(d, "rope_summary.csv", _rope_row())
    _write_rows(
        d,
        "factor_summary.csv",
        [
            {"term": "b_grp_time[1]", "role": "causal", "prob_positive": 0.94},
            {"term": "gamma_grp_ability", "role": "association", "prob_positive": 0.31},
        ],
    )
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    causal = next(s for s in payload["sentences"] if s["kind"] == "causal")
    assert "average across the children in this comparison" in causal["text"]
    assert "not randomised" in causal["text"]
    # Carried on the causal sentence, not as a sixth: the box truncates at five.
    assert [s["kind"] for s in payload["sentences"]] == [
        "headline", "confidence", "rope", "causal"
    ]


def test_level_factors_t1_referenced_plan_names_the_change_in_the_causal_sentence(tmp_path):
    # #552: under the t1-referenced arm-gap parameterisation the persisted plan
    # records arm_gap_reference="t1" and focal_term="d_grp_time[t2]", and the box
    # tells the reader the randomised quantity is the change in the arm difference
    # from t1 to t2 (a difference-in-differences), not the raw t2 gap. The
    # mean-ability caveat still rides on the same causal sentence.
    cfg = _config(
        "level_factors",
        resolved_run_plan={
            "arm_gap_reference": "t1",
            "group_by_time": True,
            "focal_term": "d_grp_time[t2]",
        },
    )
    d = _setup_dir(tmp_path, "level_factors", config=cfg)
    _write_csv(d, "rope_summary.csv", _rope_row())
    _write_rows(
        d,
        "factor_summary.csv",
        [
            {"term": "arm_gap_t1", "role": "balance", "prob_positive": 0.12},
            {"term": "d_grp_time[t2]", "role": "causal", "prob_positive": 0.94},
            {"term": "b_grp_time[1]", "role": "levels_view", "prob_positive": 0.80},
            {"term": "gamma_grp_ability", "role": "association", "prob_positive": 0.31},
        ],
    )
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    causal = next(s for s in payload["sentences"] if s["kind"] == "causal")
    assert "difference-in-differences" in causal["text"]
    assert "chance difference between the arms at t1" in causal["text"]
    assert "average across the children in this comparison" in causal["text"]
    assert [s["kind"] for s in payload["sentences"]] == [
        "headline", "confidence", "rope", "causal"
    ]


def test_level_factors_free_plan_keeps_the_plain_causal_sentence(tmp_path):
    cfg = _config(
        "level_factors",
        resolved_run_plan={
            "arm_gap_reference": "free",
            "group_by_time": True,
            "focal_term": "b_grp_time[1]",
        },
    )
    d = _setup_dir(tmp_path, "level_factors", config=cfg)
    _write_csv(d, "rope_summary.csv", _rope_row())
    _write_rows(
        d,
        "factor_summary.csv",
        [{"term": "b_grp_time[1]", "role": "causal", "prob_positive": 0.94}],
    )
    payload = generate_key_findings(d)
    causal = next(s for s in payload["sentences"] if s["kind"] == "causal")
    assert "difference-in-differences" not in causal["text"]
    assert "Only this t2 comparison is randomised" in causal["text"]


def test_level_factors_omits_the_ability_caveat_without_the_term(tmp_path):
    # A level model fitted without group x ability must not carry the caveat — it
    # would describe a coefficient the reader cannot find in the summary table.
    d = _setup_dir(tmp_path, "level_factors")
    _write_csv(d, "rope_summary.csv", _rope_row())
    _write_rows(
        d,
        "factor_summary.csv",
        [{"term": "b_grp_time[1]", "role": "causal", "prob_positive": 0.94}],
    )
    payload = generate_key_findings(d)
    causal = next(s for s in payload["sentences"] if s["kind"] == "causal")
    assert "average across the children in this comparison" not in causal["text"]


def test_level_factors_keeps_the_causal_sentence_when_psense_also_flags(tmp_path):
    # The caveat rides on the causal sentence precisely so this case still fits: a
    # psense warning plus a sixth sentence would push the causal one past the cap.
    d = _setup_dir(tmp_path, "level_factors")
    _write_csv(d, "rope_summary.csv", _rope_row())
    _write_rows(
        d,
        "factor_summary.csv",
        [{"term": "gamma_grp_ability", "role": "association", "prob_positive": 0.31}],
    )
    pd.DataFrame(
        {
            "prior": [0.083],
            "likelihood": [0.053],
            "diagnosis": ["potential prior-data conflict"],
        },
        index=["b_grp_time[1]"],
    ).to_csv(d / "psense_summary.csv")
    payload = generate_key_findings(d)
    kinds = [s["kind"] for s in payload["sentences"]]
    assert kinds == ["headline", "confidence", "rope", "robustness", "causal"]
    assert len(kinds) <= KEY_FINDINGS_MAX_SENTENCES
    assert (
        "average across the children in this comparison"
        in payload["sentences"][-1]["text"]
    )


def test_results_factors_partial_gates_the_ability_caveat_on_the_term():
    # The report prose and the key-findings prose must make the same claim, and the
    # partial must print it only for a fit that actually has the term (#389 finding 1).
    text = (REPO / "docs/models/_partials/_results_factors.qmd").read_text(encoding="utf-8")
    assert 'factor_summary.term == "gamma_grp_ability"' in text
    # The moderation convention is stated to the reader ("centred ability" is the
    # standardised scale's "mean ability", the wording the docstring uses too).
    assert "held at centred ability" in text
    assert "notes/202606261230-gain-level-factors-design.md" in text
    # #584 decision 1: the partial states the arm-free standardisation, and says why
    # removing the balance term is what makes the two arms comparable.
    assert "arm-free standardised" in text
    assert "notes/202608231800-level-factors-584-decisions.md" in text
    # #552: the partial reads the focal term from the persisted plan rather than
    # hard-coding b_grp_time[1], and fences the balance / levels-view roles off
    # from the adjusted-associations table.
    assert '_plan.get("focal_term")' in text
    assert '["causal", "balance", "levels_view"]' in text


def test_results_factors_partial_renders_off_floor_marginal_in_percentage_points():
    # Gain-factors code review 2026-08-20, finding 1: the off-floor fits store the
    # treatment marginal as a risk difference (n_trials = 1), and the partial's
    # items-scale sentence rendered it as "-0.0 items" in the published gf-005
    # report. The block must branch on the plan's off_floor flag and render
    # percentage points (whole numbers) instead — in BOTH the primary and the
    # moderation-variant sentences, which share the unit machinery.
    text = (REPO / "docs/models/_partials/_results_factors.qmd").read_text(encoding="utf-8")
    assert '_gf_off = bool(_plan.get("off_floor", False))' in text
    assert "percentage points** on the chance of being off the floor" in text
    # The shared value formatter and unit string are used by both branches — the
    # raw one-decimal items f-string must not survive anywhere in the block.
    assert text.count("_tm_value(_t.trt_items_median)") == 2
    assert "{_t.trt_items_median:+.1f}" not in text


def test_design_note_records_the_group_ability_exclusion():
    # The rationale used to live only in a comment on level_t2_marginal_effect, which
    # is why #389 re-derived it from the outside as an open question (#271 item 5).
    text = (REPO / "notes/202606261230-gain-level-factors-design.md").read_text(encoding="utf-8")
    assert "Decision 4" in text
    assert "gamma_grp_ability" in text
    assert "at mean ability" in text


def test_level_factors_surfaces_t2_psense_warning(tmp_path):
    # #389 finding 3: when power-scaling flags the t2 term (b_grp_time[1]), a caution
    # is surfaced beside the headline instead of being hidden in a collapsed prior
    # section. Carried by the release gate since it was extended to this family, so
    # the caution is the gate's ``robustness`` note rather than a level-factor one —
    # one canonical statement instead of two saying the same thing.
    d = _setup_dir(tmp_path, "level_factors")
    _write_csv(d, "rope_summary.csv", _rope_row())
    pd.DataFrame(
        {
            "prior": [0.083],
            "likelihood": [0.053],
            "diagnosis": ["potential prior-data conflict"],
        },
        index=["b_grp_time[1]"],
    ).to_csv(d / "psense_summary.csv")
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    assert payload["release"]["tau_class"] == "prior_data_conflict"
    kinds = [s["kind"] for s in payload["sentences"]]
    assert "robustness" in kinds
    assert kinds.index("robustness") < kinds.index("causal")
    warn = payload["sentences"][kinds.index("robustness")]["text"]
    assert "lower bound" in warn


@pytest.mark.parametrize("clear_marker", ["✓", "-", ""])
def test_level_factors_no_warning_when_t2_psense_clear(tmp_path, clear_marker):
    # A clear diagnosis for the t2 term produces no caution bullet.
    #
    # "✓" is the case that matters: it is what arviz_stats actually writes for an
    # unflagged parameter and the most common value in the stored suite, while "-"
    # never appears in it. Treating the tick as a diagnosis published a
    # "prior-sensitive" caution on six of the eleven level-factor reporting fits
    # whose t2 term is in fact clear.
    d = _setup_dir(tmp_path, "level_factors")
    _write_csv(d, "rope_summary.csv", _rope_row())
    pd.DataFrame(
        {"prior": [0.01], "likelihood": [0.02], "diagnosis": [clear_marker]},
        index=["b_grp_time[1]"],
    ).to_csv(d / "psense_summary.csv")
    payload = generate_key_findings(d)
    assert "warning" not in [s["kind"] for s in payload["sentences"]]


def test_level_factors_withhold_beats_an_unrecognised_psense_marker(tmp_path):
    # The gate classifies on the prior and likelihood statistics rather than on the
    # marker string, so an unknown verdict cannot slip past: prior 0.9 against
    # likelihood 0.01 is prior-dominant, and the headline is withheld outright —
    # a stronger response than the caution this previously produced.
    d = _setup_dir(tmp_path, "level_factors")
    _write_csv(d, "rope_summary.csv", _rope_row())
    pd.DataFrame(
        {"prior": [0.9], "likelihood": [0.01], "diagnosis": ["some future verdict"]},
        index=["b_grp_time[1]"],
    ).to_csv(d / "psense_summary.csv")
    payload = generate_key_findings(d)
    assert payload["status"] == "robustness_unresolved"
    assert payload["release"]["tau_class"] == "prior_dominant"
    assert payload["sentences"] == []


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
    assert "**+22 percentage-point** contrast" in headline
    assert "scoring above zero" in headline


def test_did_dose_companion_degrades_honestly(tmp_path):
    d = _setup_dir(tmp_path, "did")
    _write_csv(d, "did_summary.csv", {"beta_dose_median": 0.1, "delta_median": 0.2})
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    texts = _texts(payload)
    assert "observational association" in texts
    assert all(s["kind"] != "headline" for s in payload["sentences"])


def test_did_period_varying_dose_companion_is_recognised(tmp_path):
    """#390: the period-varying dose fit (LRPDID07) has no ``beta_dose`` column
    at all — its did_summary carries the family's ``dose_interpretation``
    marker — and must take the honest dose wording, not the stale-schema
    unavailable path (which also dropped its release decision)."""
    d = _setup_dir(tmp_path, "did")
    _write_csv(
        d,
        "did_summary.csv",
        {
            "beta_period_median": 0.3,
            "theta_treated_median": 0.2,
            "dose_interpretation": (
                "beta_dose is an observational intensive-margin association; "
                "theta_treated is the model's treatment-presence term"
            ),
        },
    )
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
                # The headline quotes the median (house standard); the mean is
                # deliberately different so a regression to it would be visible.
                "delta_words_median": 1.9,
                "delta_words_mean": 2.3,
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
    if kind == "joint_mechanism":
        # The family's switches live under ``extra``: the top-level ``design`` key is
        # the human-readable study-design string, so a builder reading it would branch
        # on the wrong value and never emit the levels caveat.
        _write_json(
            d,
            "config.json",
            {
                **_config(kind),
                "design": "per-wave cross-sectional bivariate levels",
                "extra": {"design": "levels", "contrast": ["N", "W"]},
            },
        )
        # Two waves, so the builder's per-wave path (range + clearest-wave lead) is
        # exercised, plus the levels-only conditional-slope / share-retained rows.
        def _jm(wave, term, median, lo, hi, prob_pos):
            span = (hi - lo) / 4.0
            return {
                "wave": wave,
                "term": term,
                "label": term,
                "median": median,
                "mean": median,
                "lo50": median - span,
                "hi50": median + span,
                "lo": lo,
                "hi": hi,
                "prob_pos": prob_pos,
                "converged": True,
            }

        _write_rows(
            d,
            "joint_mechanism_slopes.csv",
            [
                _jm("t3", "beta_mech[W]", 0.24, 0.14, 0.34, 1.0),
                _jm("t3", "beta_mech[N]", 1.02, 0.72, 1.33, 1.0),
                _jm("t3", "delta_ls_decoding", 0.79, 0.47, 1.09, 0.999),
                _jm("t3", "rho_outcome", 0.41, 0.12, 0.66, 0.99),
                _jm("t3", "beta_mech_focal_given_held", 0.17, 0.06, 0.28, 0.99),
                _jm("t3", "share_retained", 0.71, 0.42, 0.95, 1.0),
                _jm("t4", "beta_mech[W]", 0.29, 0.18, 0.40, 1.0),
                _jm("t4", "beta_mech[N]", 0.95, 0.61, 1.29, 1.0),
                _jm("t4", "delta_ls_decoding", 0.66, 0.31, 1.01, 0.996),
                _jm("t4", "rho_outcome", 0.38, 0.08, 0.64, 0.98),
                _jm("t4", "beta_mech_focal_given_held", 0.19, 0.07, 0.31, 0.99),
                _jm("t4", "share_retained", 0.66, 0.38, 0.92, 1.0),
            ],
        )
        return d, "decoding-use signature"
    if kind == "pooled_levels":
        # ``d`` already exists from the shared setup above; the family reads its
        # symbols from the resolved plan, so only config.json needs replacing.
        _write_json(
            d,
            "config.json",
            {
                "kind": "pooled_levels",
                "outcome_symbol": "W",
                "mechanism_symbol": "L",
                "resolved_run_plan": {
                    "outcome_symbol": "W",
                    "mechanism_symbol": "L",
                    "decompose_between_within": True,
                    "waves": [1, 2, 3, 4],
                    "use_wave_intercepts": True,
                },
            },
        )
        # The family's whole point is the split, so the synthetic case carries a
        # large between-child coefficient beside a near-null within-child one.
        _write_rows(
            d,
            "pooled_levels_summary.csv",
            [
                {"term": "beta_between", "role": "association", "median": 1.61,
                 "lo": 1.34, "hi": 1.87, "prob_positive": 1.0},
                {"term": "beta_within", "role": "association", "median": 0.04,
                 "lo": -0.06, "hi": 0.14, "prob_positive": 0.742},
            ],
        )
        return d, "Between children"
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


def test_concurrent_findings_qualify_missingness_indicators_as_nuisance(tmp_path):
    d, _ = _remaining_family_case(tmp_path, "concurrent")
    payload = generate_key_findings(d)
    assert "nuisance subgroup offsets, not skill effects" in _texts(payload)


# --- adjusted family, 2026-08-22 review ----------------------------------------

#: A non-RLI fit must carry its fit-time input snapshot or the release decision
#: withholds at the inputs stage (fail-closed); this is the "all confirmed" form.
_RLM_READY_CONTRACT = {"study_id": "rlm", "publication_ready": True, "blockers": []}


def _adjusted_words_rows() -> list[dict]:
    return [
        {
            "predictor": "age",
            "label": "Age (months)",
            "delta_words_median": -1.1,
            "delta_words_mean": -1.1,
            "delta_words_lo": -2.3,
            "delta_words_hi": 0.2,
            "prob_pos": 0.08,
        },
        {
            "predictor": "bassim",
            "label": "BAS similarities/verbal reasoning",
            "delta_words_median": 0.6,
            "delta_words_mean": 0.6,
            "delta_words_lo": -0.8,
            "delta_words_hi": 2.0,
            "prob_pos": 0.77,
        },
    ]


def test_adjusted_rlm_headline_names_the_byrne_measure_not_the_title(tmp_path):
    """Finding 2: the outcome label resolves through the study catalogue, as
    ``_setup.qmd`` does — the stored Byrne boxes read "... items of difference in
    Byrne wave-1 predictors of verbal-memory gain, waves 1-3 (...)"."""
    d = _setup_dir(
        tmp_path,
        "adjusted",
        config=_config(
            "adjusted",
            outcome_symbol="basdig",
            study_id="rlm",
            title="Byrne wave-1 predictors of verbal-memory gain, waves 1-3",
            publication_input_contract=_RLM_READY_CONTRACT,
        ),
    )
    _write_rows(d, "predicted_gain_words.csv", _adjusted_words_rows())
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    headline = payload["sentences"][0]["text"]
    assert "BAS recall of digits" in headline
    assert "Byrne wave-1 predictors" not in headline
    assert "between-child adjusted association" in _texts(payload)


def test_adjusted_transition_design_gets_a_repeated_transition_causal_sentence(
    tmp_path,
):
    """Finding 6: the stacked Byrne transition model is not between-child."""
    d = _setup_dir(
        tmp_path,
        "adjusted",
        config=_config(
            "adjusted",
            outcome_symbol="basread",
            study_id="rlm",
            resolved_run_plan={"transition_waves": [1, 2, 3, 4, 5]},
            publication_input_contract=_RLM_READY_CONTRACT,
        ),
    )
    _write_rows(d, "predicted_gain_words.csv", _adjusted_words_rows())
    payload = generate_key_findings(d)
    texts = _texts(payload)
    assert "repeated-transition adjusted association" in texts
    assert "between-child" not in texts
    assert "BAS word reading" in payload["sentences"][0]["text"]


def test_adjusted_headline_never_ranks_a_missing_data_indicator(tmp_path):
    """Finding 3: a stored pre-fix natural-scale table still carries the
    ``*_missing`` nuisance rows; the most resolved of them must not be headlined."""
    rows = _adjusted_words_rows()
    rows.append(
        {
            "predictor": "deapp_c_missing",
            "label": "Speech missing (indicator)",
            "delta_words_median": -2.0,
            "delta_words_mean": -2.0,
            "delta_words_lo": -4.0,
            "delta_words_hi": -0.1,
            "prob_pos": 0.01,
        }
    )
    d = _setup_dir(tmp_path, "adjusted")
    _write_rows(d, "predicted_gain_words.csv", rows)
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    assert "Speech missing" not in _texts(payload)
    assert "Age (months)" in payload["sentences"][0]["text"]


def _concurrent_row(timepoint, term, items, prob, **overrides) -> dict:
    row = {
        "timepoint": timepoint,
        "adjustment": "adjusted",
        "term": term,
        "label": term,
        "role": "association",
        "scale": "+1 SD",
        "items_median": items,
        "items_lo": items - 2.0,
        "items_hi": items + 2.0,
        "prob_pos": prob,
        "converged": True,
    }
    row.update(overrides)
    return row


def test_concurrent_headline_tie_break_is_primary_wave_then_larger_contrast(tmp_path):
    """2026-08-22 review (extension): rows at the resolution ceiling are tied at the
    reported precision, and ties go to the primary (first) wave, then the larger
    items-scale contrast — not to whichever row's Monte-Carlo noise happens to
    round higher (``rlm-ca-001``'s headline had flipped t1 → t2 between refits)."""
    d = _setup_dir(tmp_path, "concurrent")
    _write_rows(
        d,
        "concurrent_marginals.csv",
        [
            # t2 is "more resolved" only below the reported precision.
            _concurrent_row(2, "bassim", 8.7, 1.0),
            _concurrent_row(1, "bassim", 7.5, 0.99997),
            # Same wave, same resolution, smaller contrast: loses the tie.
            _concurrent_row(1, "trog", 3.1, 0.99998),
            # A later wave that is genuinely less resolved never wins.
            _concurrent_row(3, "basdig", 9.9, 0.93),
        ],
    )
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    headline = payload["sentences"][0]["text"]
    assert headline.startswith("At t1, ")
    assert "bassim" in headline and "+7.5" in headline
    # A genuinely more resolved later wave still wins over the primary wave.
    _write_rows(
        d,
        "concurrent_marginals.csv",
        [
            _concurrent_row(1, "bassim", 7.5, 0.97),
            _concurrent_row(2, "trog", 1.2, 0.999),
        ],
    )
    payload = generate_key_findings(d)
    assert payload["sentences"][0]["text"].startswith("At t2, ")
    assert "trog" in payload["sentences"][0]["text"]


def test_mediation_findings_use_generic_causal_qualification(tmp_path):
    d, _ = _remaining_family_case(tmp_path, "mediation")

    payload = generate_key_findings(d)

    assert "not an identified causal mediation effect" in _texts(payload)
    assert "not an identified natural mediation effect" not in _texts(payload)


def test_joint_findings_identify_smallest_difference_as_post_hoc(tmp_path):
    d, _ = _remaining_family_case(tmp_path, "joint")

    payload = generate_key_findings(d)

    assert "available-case modified ITT" in _texts(payload)
    assert "post-hoc, project-agreed smallest-important difference" in _texts(payload)


def test_joint_mechanism_findings_exclude_non_converged_waves(tmp_path):
    """A non-converged wave sub-fit is published flagged in the slopes CSV, but it
    must not lead — or range into — the gate-interlocked findings box: the
    clearest-wave selection previously ran over every row while the fit-level
    release gate covers only the anchor wave (2026-08-21 joint-mechanism review,
    finding 4)."""
    d, _ = _remaining_family_case(tmp_path, "joint_mechanism")

    def _jm(wave, term, median, prob_pos, converged):
        return {
            "wave": wave, "term": term, "label": term, "median": median,
            "mean": median, "lo50": median - 0.05, "hi50": median + 0.05,
            "lo": median - 0.2, "hi": median + 0.2, "prob_pos": prob_pos,
            "converged": converged,
        }

    _write_rows(
        d,
        "joint_mechanism_slopes.csv",
        [
            _jm("t3", "beta_mech[W]", 0.24, 1.0, True),
            _jm("t3", "beta_mech[N]", 1.02, 1.0, True),
            _jm("t3", "delta_ls_decoding", 0.79, 0.90, True),
            _jm("t3", "share_retained", 0.71, 1.0, True),
            # The clearest delta sits on the NON-converged wave: the old
            # selection would have headlined it.
            _jm("t4", "beta_mech[W]", 0.29, 1.0, False),
            _jm("t4", "beta_mech[N]", 0.95, 1.0, False),
            _jm("t4", "delta_ls_decoding", 0.66, 0.999, False),
            _jm("t4", "share_retained", 0.20, 1.0, False),
        ],
    )
    payload = generate_key_findings(d)
    text = _texts(payload)
    assert payload["status"] == "ok"
    assert "at t4" not in text
    assert "t4 0.20" not in text
    assert "Wave(s) t4 did not meet the convergence gate" in text


# --- joint contrast-first box (2026-08-21 joint review, findings 2 + 4) ---------


def _joint_contrast_case(tmp_path: Path) -> Path:
    """A two-outcome joint contrast fit (the lrp-rli-itt-015 shape)."""
    d = _setup_dir(tmp_path, "joint")
    _write_rows(
        d,
        "joint_treatment_marginal.csv",
        [
            {
                "outcome": "TE",
                "items_median": 1.5,
                "items_lo": 0.4,
                "items_hi": 2.7,
                "prob_pos": 0.98,
                "delta_items": 1.0,
                "prob_benefit_ge_delta": 0.79,
                "prob_in_rope": 0.21,
            },
            {
                "outcome": "UE",
                "items_median": 0.4,
                "items_lo": -0.4,
                "items_hi": 1.2,
                "prob_pos": 0.79,
                "delta_items": 1.0,
                "prob_benefit_ge_delta": 0.08,
                "prob_in_rope": 0.66,
            },
        ],
    )
    _write_rows(
        d,
        "tau_summary.csv",
        [
            {
                "outcome": "TE",
                "ame_prob_median": 0.0646,
                "ame_prob_lo": 0.0172,
                "ame_prob_hi": 0.1112,
                "prob_ame_pos": 0.98,
            },
            {
                "outcome": "UE",
                "ame_prob_median": 0.0304,
                "ame_prob_lo": -0.0300,
                "ame_prob_hi": 0.0908,
                "prob_ame_pos": 0.79,
            },
        ],
    )
    _write_csv(
        d,
        "tau_difference.csv",
        {
            "contrast": "TE_minus_UE",
            "headline_scale": "proportion_correct_risk_difference",
            "diff_prob_median": 0.0342,
            "diff_prob_lo": -0.0429,
            "diff_prob_hi": 0.1109,
            "diff_prob_lo50": 0.0017,
            "diff_prob_hi50": 0.0666,
            "prob_diff_pos": 0.76,
            "contrast_kind": "generalisation",
            "contrast_label": "Expressive taught versus not-taught vocabulary",
            "positive_interpretation": (
                "A positive contrast means the intervention increased the "
                "proportion correct more for taught expressive words than for "
                "not-taught expressive words."
            ),
            "negative_interpretation": (
                "A negative contrast means the opposite ordering."
            ),
            "transfer_outcome": "UE",
            "transfer_interpretation": (
                "Assess whether expressive generalisation is small from the "
                "marginal UE average marginal effect against a substantively "
                "defined negligible-effect threshold."
            ),
            "dependence_note": "Factorised; see the registered companion.",
        },
    )
    return d


def test_joint_contrast_fit_headlines_the_declared_contrast(tmp_path):
    """Finding 2: a contrast model's box must state its declared estimand — the
    between-outcome difference — not only the two better-resolved marginals."""
    d = _joint_contrast_case(tmp_path)
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    headline = payload["sentences"][0]
    assert headline["kind"] == "headline"
    assert "generalisation contrast" in headline["text"]
    assert "Expressive taught versus not-taught vocabulary" in headline["text"]
    assert "+3.4" in headline["text"]
    assert "percentage points" in headline["text"]
    confidence = payload["sentences"][1]
    assert confidence["kind"] == "confidence"
    assert "76" in confidence["text"]
    assert "increased the proportion correct more for taught" in confidence["text"]
    kinds = [s["kind"] for s in payload["sentences"]]
    assert "transfer" in kinds
    assert kinds[-1] == "causal"
    assert "negligible-effect threshold" in _texts(payload)


def test_joint_contrast_release_note_drops_the_marginals_not_the_transfer(
    tmp_path,
):
    """With five sentences and a robustness note to insert, the droppable
    marginals context makes room; the transfer read and causal caveat stay."""
    d = _joint_contrast_case(tmp_path)
    _write_psense(d, prior=0.09, likelihood=0.22)  # prior-data conflict -> note
    payload = generate_key_findings(d)
    kinds = [s["kind"] for s in payload["sentences"]]
    assert "robustness" in kinds
    assert "transfer" in kinds
    assert "note" not in kinds
    assert kinds[-1] == "causal"


def test_joint_range_uses_percentage_points_and_flags_p_and_b(tmp_path):
    """Finding 4: the cross-outcome range must not pool item units across tests
    with different denominators, and P / B carry their standing qualifications."""
    d = _setup_dir(tmp_path, "joint")
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
                "outcome": "P",
                "items_median": 0.4,
                "items_lo": -0.5,
                "items_hi": 1.3,
                "prob_pos": 0.71,
            },
            {
                "outcome": "B",
                "items_median": 0.6,
                "items_lo": -0.6,
                "items_hi": 1.9,
                "prob_pos": 0.77,
            },
        ],
    )
    _write_rows(
        d,
        "tau_summary.csv",
        [
            {
                "outcome": "W",
                "ame_prob_median": 0.043,
                "ame_prob_lo": -0.005,
                "ame_prob_hi": 0.104,
                "prob_ame_pos": 0.94,
            },
            {
                "outcome": "P",
                "ame_prob_median": 0.010,
                "ame_prob_lo": -0.012,
                "ame_prob_hi": 0.033,
                "prob_ame_pos": 0.71,
            },
            {
                "outcome": "B",
                "ame_prob_median": 0.030,
                "ame_prob_lo": -0.030,
                "ame_prob_hi": 0.095,
                "prob_ame_pos": 0.77,
            },
        ],
    )
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    headline = payload["sentences"][0]["text"]
    assert "percentage points" in headline
    assert "items**" not in headline
    texts = _texts(payload)
    assert "floor rule" in texts
    assert "response-link" in texts
    assert "1 was more likely than not" in texts


def test_horseshoe_findings_do_not_claim_threshold_was_pre_specified(tmp_path):
    d, _ = _remaining_family_case(tmp_path, "horseshoe")

    payload = generate_key_findings(d)

    assert "model's worth-noticing coefficient threshold" in _texts(payload)
    assert "pre-specified" not in _texts(payload)


def test_historical_joint_within_companion_headlines_dynamic_correlation(tmp_path):
    d = _setup_dir(tmp_path, "historical_joint")
    _write_csv(
        d,
        "within_measure_correlation_summary.csv",
        {
            "measure_i": "basread",
            "measure_j": "bpvs",
            "label_i": "BAS word reading",
            "label_j": "BPVS receptive vocabulary",
            "median": 0.48,
            "mean": 0.47,
            "lo50": 0.35,
            "hi50": 0.60,
            "lo": 0.12,
            "hi": 0.76,
            "prob_pos": 0.98,
        },
    )
    _write_csv(
        d,
        "between_within_correlation_comparison.csv",
        {
            "measure_i": "basread",
            "measure_j": "bpvs",
            "within_minus_between_median": -0.18,
            "within_minus_between_lo": -0.51,
            "within_minus_between_hi": 0.13,
            "prob_within_gt_between": 0.17,
        },
    )

    payload = generate_key_findings(d)

    assert payload["status"] == "ok"
    assert "clearest within-child coupling" in payload["sentences"][0]["text"]
    assert "inner 50% range" in payload["sentences"][0]["text"]
    assert "within-minus-between correlation" in _texts(payload)
    assert "does not identify direction" in _texts(payload)


def test_historical_joint_within_companion_withholds_unresolved_correlations(tmp_path):
    d = _setup_dir(tmp_path, "historical_joint")
    _write_csv(
        d,
        "within_measure_correlation_summary.csv",
        {
            "measure_i": "basread",
            "measure_j": "bpvs",
            "label_i": "BAS word reading",
            "label_j": "BPVS receptive vocabulary",
            "median": 0.48,
            "lo50": 0.35,
            "hi50": 0.60,
            "lo": 0.12,
            "hi": 0.76,
            "prob_pos": 0.98,
            "pair_resolvable": False,
        },
    )
    _write_rows(
        d,
        "within_scale_summary.csv",
        [
            {
                "measure": "basread",
                "label": "BAS word reading",
                "median": 0.31,
                "lo50": 0.29,
                "hi50": 0.34,
                "lo": 0.26,
                "hi": 0.38,
                "minimum_resolvable_sd": 0.05,
                "prob_above_minimum": 1.0,
                "resolvable": True,
            },
            {
                "measure": "bpvs",
                "label": "BPVS receptive vocabulary",
                "median": 0.04,
                "lo50": 0.02,
                "hi50": 0.06,
                "lo": 0.00,
                "hi": 0.10,
                "minimum_resolvable_sd": 0.05,
                "prob_above_minimum": 0.35,
                "resolvable": False,
            },
        ],
    )

    payload = generate_key_findings(d)

    assert payload["status"] == "ok"
    assert "did not resolve a within-child correlation" in _texts(payload)
    assert "not substantively identified" in _texts(payload)
    assert "clearest within-child coupling" not in _texts(payload)


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


def _moderation_items_rows(**overrides) -> list[dict]:
    """A ``moderation_items.csv`` as ``pipelines.mechanism.write_moderation_items``
    writes it (the mech-061 reporting values, rounded)."""
    common = {
        "exposure_low": 17.0,
        "exposure_high": 28.0,
        "moderator_low": 4.0,
        "moderator_high": 8.0,
        "exposure_symbol": "L",
        "exposure_unit": "L items",
        "moderator_symbol": "B",
        "moderator_unit": "B items",
        "outcome_symbol": "W",
        "outcome_unit": "W items",
        "n_obs": 156,
        "ci_prob": 0.89,
        "scale": "items",
    }
    rows = [
        {"quantity": "increment_at_moderator_low", "median": 2.896, "lo": 0.927, "hi": 4.771, "prob_pos": 0.99},
        {"quantity": "increment_at_moderator_high", "median": 1.745, "lo": -0.451, "hi": 3.788, "prob_pos": 0.90},
        {"quantity": "interaction", "median": -1.147, "lo": -2.327, "hi": -0.049, "prob_pos": 0.047},
        {"quantity": "interaction_if_logit_additive", "median": 0.179, "lo": 0.011, "hi": 0.430, "prob_pos": 0.958},
        {"quantity": "interaction_logit", "median": -0.159, "lo": -0.298, "hi": -0.024, "prob_pos": 0.029, "scale": "logit"},
    ]
    out = []
    for r in rows:
        row = {**common, **r}
        row.update(overrides.get(r["quantity"], {}))
        out.append(row)
    return out


def _moderated_mechanism_dir(tmp_path, *, prob_gamma_int_pos=0.03):
    d = _setup_dir(
        tmp_path,
        "mechanism",
        config=_config(
            "mechanism",
            mechanism_symbol="L",
            extra={"moderator_symbol": "B"},
        ),
    )
    _write_csv(
        d,
        "mechanism_summary.csv",
        {
            "exposure_low": 2,
            "exposure_high": 32,
            "exposure_unit": "L items",
            "items_median": 6.8,
            "items_lo": 2.2,
            "items_hi": 12.1,
            "prob_pos": 0.995,
        },
    )
    _write_csv(
        d,
        "interaction_summary.csv",
        {
            "gamma_int_median": -0.11,
            "gamma_int_mean": -0.11,
            "gamma_int_lo": -0.21,
            "gamma_int_hi": -0.02,
            "gamma_int_lo50": -0.15,
            "gamma_int_hi50": -0.07,
            "prob_gamma_int_pos": prob_gamma_int_pos,
            "gamma_mod_median": 0.16,
            "gamma_mod_mean": 0.16,
            "gamma_mod_lo": 0.06,
            "gamma_mod_hi": 0.25,
            "gamma_mod_lo50": 0.11,
            "gamma_mod_hi50": 0.20,
            "prob_gamma_mod_pos": 0.995,
        },
    )
    return d


def test_mechanism_moderation_names_its_skills_and_never_claims_items_additivity(tmp_path):
    """2026-08-19: the focal sentence names the fitted exposure, outcome and
    moderator (mech-072 is L -> N, not word reading), and the logit-scale direction
    claim no longer carries the ", not additivity in word counts" tail, which read
    as a finding against items-scale additivity the fit never tested."""
    d = _moderated_mechanism_dir(tmp_path)
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    first = payload["sentences"][0]["text"]
    assert "slope of word reading (WR) on letter-sound knowledge (LS)" in first
    assert "per +1 SD of phoneme blending (PA)" in first
    assert "not additivity" not in _texts(payload)
    assert "substitution on the logit scale" in payload["sentences"][1]["text"]
    # Without moderation_items.csv the box keeps its previous five-sentence shape.
    assert [s["kind"] for s in payload["sentences"]] == [
        "headline",
        "confidence",
        "headline",
        "confidence",
        "causal",
    ]


def test_mechanism_moderation_items_sentence_reads_the_items_scale_table(tmp_path):
    """With moderation_items.csv present the box carries the items-scale
    re-expression — the interquartile increments at the low and high moderator
    cell, their difference, the logit-additive benchmark and a ladder verdict —
    and the unmoderated curve folds into one droppable context sentence so the
    causal sentence stays inside the cap (#464)."""
    d = _moderated_mechanism_dir(tmp_path)
    _write_rows(d, "moderation_items.csv", _moderation_items_rows())
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    kinds = [s["kind"] for s in payload["sentences"]]
    assert kinds == ["headline", "confidence", "scale", "note", "causal"]
    items = payload["sentences"][2]["text"]
    assert "In word reading (WR) items" in items
    assert "(17 to 28 LS items)" in items
    assert "**+2.9 items** when phoneme blending (PA) is 4 PA items and +1.7 when it is 8" in items
    assert "a difference of -1.1 items (89% -2.3 to -0.0; P(negative) = 95%)" in items
    assert "would have shown +0.2" in items
    assert "moderate evidence that the substitution holds in items too" in items
    assert "not an artefact of the bounded scale" in items
    context = payload["sentences"][3]["text"]
    assert context.startswith("For context, across the fitted exposure range (2 to 32 LS items)")
    assert "+6.8 items" in context and "P(positive) = 99.5%" in context
    assert payload["sentences"][-1]["kind"] == "causal"


def test_mechanism_moderation_items_verdicts_follow_the_evidence_ladder(tmp_path):
    """An items-scale direction below the moderate rung is reported as unsettled;
    when the logit scale is itself inconclusive the sentence says so for both."""
    d = _moderated_mechanism_dir(tmp_path)
    _write_rows(
        d,
        "moderation_items.csv",
        _moderation_items_rows(
            interaction={"median": -0.7, "lo": -2.2, "hi": 0.7, "prob_pos": 0.21}
        ),
    )
    text = generate_key_findings(d)["sentences"][2]["text"]
    assert "P(negative) = 79%" in text
    assert "on the items scale the direction is suggestive" in text
    assert "should not be read as a finding about items" in text

    (tmp_path / "b").mkdir()
    d2 = _moderated_mechanism_dir(tmp_path / "b", prob_gamma_int_pos=0.45)
    _write_rows(
        d2,
        "moderation_items.csv",
        _moderation_items_rows(
            interaction={"median": -0.2, "lo": -1.8, "hi": 1.4, "prob_pos": 0.42}
        ),
    )
    text2 = generate_key_findings(d2)["sentences"][2]["text"]
    assert "the direction is inconclusive on both scales" in text2


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


def test_key_findings_partial_is_a_self_contained_renderer():
    text = (REPO / "docs/models/_partials/_key_findings.qmd").read_text(encoding="utf-8")
    assert "key_findings.json" in text
    assert "convergence_gate_clean_passed" in text
    assert "gate_failed" in text
    assert "gate_exception" in text
    assert "retired model-spec gate exception" in text
    assert "not available" in text
    assert "callout-important" in text  # the red withheld-findings warning
    assert "evaluate_local_blending_link_sensitivity" in text
    assert "blending_link_sensitivity_sha256" in text
    assert "response-link" in text
    assert "summary changed" in text
    assert "itt_missingness_sensitivity_sha256" in text
    assert "evaluate_publication" in text
    assert "full-cohort" in text
    assert "sensitivity is not current" in text
    # Self-contained: must not depend on _setup.qmd helpers so #321 can move it.
    assert "_csv(" not in text
    assert "_json(" not in text


def test_itt_results_require_the_trace_backed_blending_link_pair():
    text = (REPO / "docs/models/_partials/_results_itt.qmd").read_text(encoding="utf-8")
    assert "evaluate_local_blending_link_sensitivity" in text
    assert "Phoneme-blending findings withheld" in text
    assert "lrp-rli-itt-008" in text
    assert "lrp-rli-itt-108" in text
    assert 'if _symbol == "B" and _scientific_results_released' in text
    assert "_scientific_results_released = False" in text
    assert "neither row should be selected" in text
    assert "isolation" in text


def test_reading_guide_is_a_collapsed_callout():
    text = (REPO / "docs/models/_partials/_reading_guide.qmd").read_text(encoding="utf-8")
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
        text = path.read_text(encoding="utf-8")
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
    text = (REPO / "docs/models/_partials/_technical.qmd").read_text(encoding="utf-8")
    assert 'collapse="true"' in text
    assert 'title="Technical checks"' in text
    assert text.count("_partials/_convergence.qmd") == 1
    assert text.count("_partials/_diagnostics.qmd") == 1


def test_integrated_report_uses_the_same_fail_closed_gate():
    text = (REPO / "docs/report/_report_data.qmd").read_text(encoding="utf-8")
    assert "convergence_gate_clean_passed" in text
    assert 'diag.get("passed") is True' not in text


# --- #392 P1: the robustness release gate -------------------------------------


def _floor_config(**overrides) -> dict:
    """An ITT config for a P/N floor-rule primary."""
    cfg = _config("itt", outcome_symbol="P")
    cfg["resolved_run_plan"] = {
        "floor_rule": True,
        "floor_rule_provenance": "post_hoc_data_adaptive_t2_zero_rate",
        "floor_estimand_role": "exploratory_headline",
    }
    cfg.update(overrides)
    return cfg


def test_clean_power_scaling_releases_itt_findings_unchanged(tmp_path):
    """The gate must be invisible when the evidence is clean."""
    d = _setup_dir(tmp_path, "itt")
    _write_csv(d, "rope_summary.csv", _rope_row())
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    assert payload["release"]["status"] == "release"
    assert payload["release"]["tau_class"] == "clear"
    assert [s["kind"] for s in payload["sentences"]] == [
        "headline",
        "confidence",
        "rope",
        "causal",
    ]


def test_prior_data_conflict_releases_with_an_attenuation_note(tmp_path):
    """A conservative zero-centred prior attenuates a real effect rather than
    inventing one, so a conflict where the data still move the posterior is released
    with a note that the size is a lower bound — not withheld."""
    d = _setup_dir(tmp_path, "itt")
    _write_psense(d, prior=0.09, likelihood=0.22)
    _write_csv(d, "rope_summary.csv", _rope_row())
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    assert payload["release"]["status"] == "release"
    assert payload["release"]["tau_class"] == "prior_data_conflict"
    kinds = [s["kind"] for s in payload["sentences"]]
    assert "robustness" in kinds
    # The note must not displace the causal sentence (#464): it goes before it.
    assert kinds.index("robustness") < kinds.index("causal")
    assert "lower bound" in _texts(payload)


def test_prior_dominant_tau_withholds_the_causal_headline(tmp_path):
    """The case worth gating: the prior out-works the data on the causal term."""
    d = _setup_dir(tmp_path, "itt")
    _write_psense(d, prior=0.41, likelihood=0.02)
    _write_csv(d, "rope_summary.csv", _rope_row())
    payload = generate_key_findings(d)
    assert payload["status"] == "robustness_unresolved"
    assert payload["sentences"] == []
    assert "responds to the prior" in payload["reason"]
    assert payload["release"]["tau_class"] == "prior_dominant"
    assert "tau_prior_sensitivity.csv" in payload["release"]["evidence"]


def _write_tau_sweep(d: Path, *, rows: list[dict] | None = None, outcome: str = "W"):
    """A minimally valid attached treatment-prior sweep, bound to this fit.

    Carries the standard sweep's full column set (so a hand-rolled CSV of the same
    name cannot pass), two prior scales, converged cells, matching primary hashes,
    a stable effect sign, and — per the level/did installer contract (#489) —
    a real digest-matching cell-trace file beside the fit for every basename
    ``trace_file``. Individual tests break exactly one clause.
    """
    from language_reading_predictors.statistical_models.sensitivity import (
        _STANDARD_REQUIRED_COLUMNS,
        sha256_file,
    )

    (d / "trace.nc").write_text("fit trace")
    base = dict.fromkeys(_STANDARD_REQUIRED_COLUMNS, 1)
    base.update(
        {
            "outcome": outcome,
            "converged": True,
            "primary_config_sha256": sha256_file(d / "config.json"),
            "primary_trace_sha256": sha256_file(d / "trace.nc"),
        }
    )
    if rows is None:
        rows = [
            {"tau_sigma": 0.25, "tau_logit_mean": 0.31},
            {"tau_sigma": 0.5, "tau_logit_mean": 0.44},
        ]
    merged_rows = []
    for index, row in enumerate(rows):
        merged = {**base, **row}
        name = f"trace_{outcome}_tau-{index}.nc"
        (d / name).write_bytes(f"installed cell trace {index}".encode())
        merged["trace_file"] = name
        merged["trace_sha256"] = sha256_file(d / name)
        merged_rows.append(merged)
    pd.DataFrame(merged_rows).to_csv(d / "tau_prior_sensitivity.csv", index=False)


def _prior_dominant_dir(tmp_path: Path) -> Path:
    d = _setup_dir(tmp_path, "itt")
    _write_psense(d, prior=0.41, likelihood=0.02)
    _write_csv(d, "rope_summary.csv", _rope_row())
    return d


def test_an_attached_tau_sweep_turns_a_withhold_into_a_qualified_release(tmp_path):
    """Evidence-bound: a valid co-located treatment-prior sweep lifts the withhold,
    and the finding then ships labelled prior-informed rather than unqualified."""
    d = _prior_dominant_dir(tmp_path)
    _write_tau_sweep(d)
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    assert payload["release"]["status"] == "qualify"
    assert "prior-informed and exploratory" in _texts(payload)


def test_an_empty_sweep_file_does_not_lift_the_withhold(tmp_path):
    """Presence is not evidence. A zero-byte or header-only file passing the gate
    would make the policy evidence-*named* rather than evidence-bound."""
    d = _prior_dominant_dir(tmp_path)
    (d / "tau_prior_sensitivity.csv").write_text("")
    payload = generate_key_findings(d)
    assert payload["status"] == "robustness_unresolved"
    assert "empty or unreadable" in payload["reason"]


def test_a_csv_of_the_right_name_but_wrong_shape_does_not_lift_the_withhold(tmp_path):
    d = _prior_dominant_dir(tmp_path)
    _write_csv(d, "tau_prior_sensitivity.csv", {"outcome": "W", "tau_sigma": 0.5})
    payload = generate_key_findings(d)
    assert payload["status"] == "robustness_unresolved"
    assert "not a standard treatment-prior sweep" in payload["reason"]


def test_a_single_prior_scale_is_not_a_sweep(tmp_path):
    d = _prior_dominant_dir(tmp_path)
    _write_tau_sweep(
        d,
        rows=[
            {"tau_sigma": 0.5, "tau_logit_mean": 0.31},
            {"tau_sigma": 0.5, "tau_logit_mean": 0.44},
        ],
    )
    payload = generate_key_findings(d)
    assert payload["status"] == "robustness_unresolved"
    assert "fewer than two scales" in payload["reason"]


def test_an_unconverged_sweep_cell_is_not_evidence(tmp_path):
    d = _prior_dominant_dir(tmp_path)
    _write_tau_sweep(
        d,
        rows=[
            {"tau_sigma": 0.25, "tau_logit_mean": 0.31},
            {"tau_sigma": 0.5, "tau_logit_mean": 0.44, "converged": False},
        ],
    )
    payload = generate_key_findings(d)
    assert payload["status"] == "robustness_unresolved"
    assert "did not converge" in payload["reason"]


def test_a_deleted_installed_trace_un_lifts_the_withhold(tmp_path):
    """The level/did installers write basename ``trace_file`` entries beside the
    fit; if such a trace later disappears (or is swapped), the manifest merely
    *names* evidence and must stop lifting the gate (#489 review)."""
    d = _prior_dominant_dir(tmp_path)
    _write_tau_sweep(d)
    (d / "trace_W_tau-0.nc").unlink()
    payload = generate_key_findings(d)
    assert payload["status"] == "robustness_unresolved"
    assert "no longer trace-backed" in payload["reason"]


def test_a_swapped_installed_trace_un_lifts_the_withhold(tmp_path):
    d = _prior_dominant_dir(tmp_path)
    _write_tau_sweep(d)
    (d / "trace_W_tau-0.nc").write_bytes(b"different bytes")
    payload = generate_key_findings(d)
    assert payload["status"] == "robustness_unresolved"
    assert "recorded digest" in payload["reason"]


def test_a_sweep_bound_to_a_different_fit_does_not_lift_the_withhold(tmp_path):
    """"Computed from the same trace and commit as the posterior" is the stated bar,
    so a sweep carrying another fit's primary hashes is not this fit's evidence."""
    d = _prior_dominant_dir(tmp_path)
    _write_tau_sweep(d)
    frame = pd.read_csv(d / "tau_prior_sensitivity.csv")
    frame["primary_trace_sha256"] = "0" * 64
    frame.to_csv(d / "tau_prior_sensitivity.csv", index=False)
    payload = generate_key_findings(d)
    assert payload["status"] == "robustness_unresolved"
    assert "different trace.nc" in payload["reason"]


def test_a_sweep_whose_effect_changes_sign_does_not_lift_the_withhold(tmp_path):
    """Sign stability is the bar, not interval width: a conservative prior is
    expected to move the magnitude, so only a direction flip disqualifies."""
    d = _prior_dominant_dir(tmp_path)
    _write_tau_sweep(
        d,
        rows=[
            {"tau_sigma": 0.25, "tau_logit_mean": 0.31},
            {"tau_sigma": 0.5, "tau_logit_mean": -0.12},
        ],
    )
    payload = generate_key_findings(d)
    assert payload["status"] == "robustness_unresolved"
    assert "changes sign" in payload["reason"]


def test_unmeasured_power_scaling_withholds_rather_than_passing_silently(tmp_path):
    """#381's meta-finding, enforced: no psense means not measured, not measured
    clean. Repairable by regenerate_psense + regenerate_key_findings, no refit."""
    d = _setup_dir(tmp_path, "itt")
    (d / "psense_summary.csv").unlink()
    _write_csv(d, "rope_summary.csv", _rope_row())
    payload = generate_key_findings(d)
    assert payload["status"] == "robustness_unresolved"
    assert "unmeasured rather than measured clean" in payload["reason"]
    assert payload["release"]["tau_class"] == "unavailable"


def test_duplicate_tau_rows_are_ambiguous_not_first_wins(tmp_path):
    """A gate that silently picks one of two disagreeing diagnoses is worse than
    one that reports it cannot tell."""
    d = _setup_dir(tmp_path, "itt")
    pd.DataFrame(
        [
            {"prior": 0.01, "likelihood": 0.02, "diagnosis": "✓"},
            {"prior": 0.40, "likelihood": 0.01, "diagnosis": "potential strong prior"},
        ],
        index=["tau", "tau"],
    ).to_csv(d / "psense_summary.csv")
    _write_csv(d, "rope_summary.csv", _rope_row())
    payload = generate_key_findings(d)
    assert payload["status"] == "robustness_unresolved"
    assert payload["release"]["tau_class"] == "unavailable"


def test_floored_primary_withholds_without_its_required_grid(tmp_path):
    """#392 P1a: the P/N floor models were emitting an unqualified headline before
    the floor-sensitivity gate further down the report was ever reached."""
    d = _setup_dir(tmp_path, "itt", config=_floor_config())
    # The floor gate keys off ArviZ's diagnosis string (as `_results_floored.qmd`
    # does), not off this module's finer numeric class, so the two fire together.
    pd.DataFrame(
        [
            {
                "prior": 0.12,
                "likelihood": 0.30,
                "diagnosis": "potential prior-data conflict",
            }
        ],
        index=["tau"],
    ).to_csv(d / "psense_summary.csv")
    _write_csv(d, "rope_summary.csv", _rope_row(delta_scale="risk_difference"))
    payload = generate_key_findings(d)
    assert payload["status"] == "robustness_unresolved"
    assert payload["sentences"] == []
    assert "floor_tau_prior_sensitivity.csv" in payload["reason"]
    assert payload["release"]["floor_rule"] is True
    assert payload["release"]["floor_grid_required"] is True
    assert payload["release"]["floor_grid_ready"] is False


def test_floored_primary_releases_when_no_grid_is_required(tmp_path):
    """A clean `tau` diagnosis does not require the grid — the gate mirrors
    `_results_floored.qmd`'s condition exactly so the two cannot disagree."""
    d = _setup_dir(tmp_path, "itt", config=_floor_config())
    _write_csv(d, "rope_summary.csv", _rope_row(delta_scale="risk_difference"))
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    assert payload["release"]["floor_grid_required"] is False


def test_released_floored_findings_name_the_post_hoc_subgroup(tmp_path):
    """#392 P1a second half: when a floor model *is* released, its causal sentence
    must identify the post-hoc baseline-floor subgroup and the missingness
    assumption, not read as an unqualified full-cohort ITT."""
    d = _setup_dir(tmp_path, "itt", config=_floor_config())
    _write_csv(d, "rope_summary.csv", _rope_row(delta_scale="risk_difference"))
    payload = generate_key_findings(d)
    causal = next(s for s in payload["sentences"] if s["kind"] == "causal")
    assert "available-case modified ITT estimate" in causal["text"]
    assert "scored at the floor of this measure at baseline" in causal["text"]
    assert "chosen after the data were seen" in causal["text"]
    assert "already off the floor" in causal["text"]


def test_nonfloor_available_case_modified_itt_keeps_its_selection_wording(tmp_path):
    d = _setup_dir(tmp_path, "itt")
    _write_csv(d, "rope_summary.csv", _rope_row())
    causal = next(
        s for s in generate_key_findings(d)["sentences"] if s["kind"] == "causal"
    )
    assert "at the floor of this measure" not in causal["text"]
    assert "available-case modified ITT estimate" in causal["text"]


def test_release_gate_does_not_touch_observational_families(tmp_path):
    """Scope is the randomisation-anchored families. An observational one has no
    causal headline to gate, so it must release with no release block at all — the
    extension must not silently gate the whole suite."""
    d = _setup_dir(tmp_path, "mechanism")
    payload = generate_key_findings(d)
    assert "release" not in payload


def test_gate_covers_the_randomised_families_and_fails_closed_unmeasured(tmp_path):
    """The #392 rule mirrored onto did / gain_factors / level_factors.

    Unmeasured withholds here for the same reason it does in ITT: a fit with no
    psense has not been measured clean, it has not been measured.
    """
    for kind in ("did", "gain_factors", "level_factors"):
        root = tmp_path / kind
        root.mkdir()
        d = _setup_dir(root, kind)
        (d / "psense_summary.csv").unlink()
        payload = generate_key_findings(d)
        assert payload["status"] == "robustness_unresolved", kind
        assert payload["release"]["tau_class"] == "unavailable", kind


def test_treated_only_gain_factor_companions_are_out_of_scope(tmp_path):
    """A treated-only companion has no randomised term to gate.

    Every row is on intervention, so the treatment indicator is constant, the factory
    drops ``beta_trt``, and the resolved plan says the fit is associational. Without
    this the fail-closed rule reads a structurally absent term as an unmeasured one
    and withholds all eight companions — "not measured" and "not present" are the
    same absence to a lookup and opposite things to a reader.
    """
    d = _setup_dir(tmp_path, "gain_factors")
    config = json.loads((d / "config.json").read_text())
    config["resolved_run_plan"] = {**config.get("resolved_run_plan", {}), "treated_only": True}
    (d / "config.json").write_text(json.dumps(config))
    payload = generate_key_findings(d)
    assert "release" not in payload
    assert payload["status"] != "robustness_unresolved"


def test_moderation_variant_gain_factors_are_out_of_scope(tmp_path):
    """A moderation variant's ``beta_trt`` exists but is never the causal headline.

    By the #391 finding 3 decision its interaction-aware marginal is model-dependent
    (the trt interactions are estimated partly on post-crossover rows) and the
    randomised headline lives in the interaction-free primary — which is gated.
    Gating the variant would demand treatment-prior sweep evidence for a number the
    family never releases as causal.
    """
    d = _setup_dir(tmp_path, "gain_factors")
    config = json.loads((d / "config.json").read_text())
    config["resolved_run_plan"] = {
        **config.get("resolved_run_plan", {}),
        "moderation_variant": True,
    }
    (d / "config.json").write_text(json.dumps(config))
    pd.DataFrame(
        [
            {
                "term": "gamma_int_trt_ability",
                "role": "association",
                "median": 0.30,
                "lo": -0.09,
                "hi": 0.65,
                "prob_positive": 0.88,
            }
        ]
    ).to_csv(d / "factor_summary.csv", index=False)
    payload = generate_key_findings(d)
    assert "release" not in payload
    assert payload["status"] != "robustness_unresolved"
    texts = [s["text"] for s in payload.get("sentences", [])]
    assert any("model-dependent adjusted association" in t for t in texts)
    assert any(t.startswith("Moderation by general cognitive ability") for t in texts)
    assert not any("only potentially cause-and-effect" in t for t in texts)


def test_gain_factors_off_floor_direction_words_state_status_not_transition(tmp_path):
    """#490 review: the gain-family off-floor Bernoulli outcome is post-period
    STATUS (post > 0) — pooling moving off, staying above and returning to the
    floor — so the confidence sentence must not describe it as "coming off the
    floor" (that phrasing belongs to the ITT floored primaries, whose estimand IS
    a transition among children observed at the baseline floor)."""
    d = _setup_dir(tmp_path, "gain_factors")
    _write_csv(
        d,
        "rope_summary.csv",
        _rope_row(delta_items=0.10, delta_scale="risk_difference"),
    )
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    texts = _texts(payload)
    assert "being off the floor at the period end" in texts
    assert "coming off the floor" not in texts


def test_level_factors_off_floor_direction_words_state_status_not_transition(tmp_path):
    """#490 review follow-up: the level-family off-floor Bernoulli outcome is
    off-floor STATUS at each wave (score > 0) — per-wave prevalence, pooling
    moving off, staying above and returning to the floor — so the t2 confidence
    sentence must not describe it as "coming off the floor" (that phrasing
    belongs to the ITT floored primaries, whose estimand IS a transition among
    children observed at the baseline floor)."""
    d = _setup_dir(tmp_path, "level_factors")
    _write_csv(
        d,
        "rope_summary.csv",
        _rope_row(delta_items=0.10, delta_scale="risk_difference"),
    )
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    texts = _texts(payload)
    assert "being off the floor at t2" in texts
    assert "coming off the floor" not in texts


def test_did_off_floor_direction_words_state_status_not_transition(tmp_path):
    """The same rule for the arm-by-wave family: the off-floor DiD models fit
    off-floor prevalence at each wave — their own report prose insists the
    contrasts are "differences in *being* off floor" — so the tau_t2 confidence
    sentence names the t2 status, not a floor-exit transition."""
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
    texts = _texts(payload)
    assert "being off the floor at t2" in texts
    assert "coming off the floor" not in texts


def test_each_family_reads_its_own_causal_term():
    """The gate must name the term the headline actually rests on.

    ``level_factors`` fits one ``b_grp_time`` per timepoint and only t2 is randomised;
    reading the bare vector name returns "unavailable" for all eleven fits. The DiD
    dose models have no ``tau_t2`` at all — the choice mirrors ``DiDRunPlan.effect_term``
    and is read from the persisted plan so the decision and the fit's own psense
    emission cannot disagree.
    """
    from language_reading_predictors.statistical_models.release import causal_term_for

    assert causal_term_for({"kind": "itt"}) == "tau"
    assert causal_term_for({"kind": "joint"}) == "tau"
    assert causal_term_for({"kind": "gain_factors"}) == "beta_trt"
    # A stored pre-#552 level fit (no focal_term in its plan) was fitted with the
    # free per-timepoint vector, so the fallback is its raw t2 gap; a fit under
    # the t1-referenced parameterisation names the t2 change (#552).
    assert causal_term_for({"kind": "level_factors"}) == "b_grp_time[1]"
    assert (
        causal_term_for(
            {"kind": "level_factors", "resolved_run_plan": {"focal_term": "d_grp_time[t2]"}}
        )
        == "d_grp_time[t2]"
    )
    assert (
        causal_term_for(
            {"kind": "level_factors", "resolved_run_plan": {"focal_term": "b_grp_time[1]"}}
        )
        == "b_grp_time[1]"
    )
    assert causal_term_for({"kind": "did"}) == "tau_t2"
    assert (
        causal_term_for({"kind": "did", "resolved_run_plan": {"dose": True}})
        == "beta_dose"
    )
    assert (
        causal_term_for(
            {"kind": "did", "resolved_run_plan": {"dose": True, "period_varying": True}}
        )
        == "mu_dose"
    )


def test_a_release_gate_that_cannot_be_evaluated_fails_closed(tmp_path, monkeypatch):
    """A gate that cannot be evaluated must withhold, not silently ungate — the
    alternative reinstates the defect precisely when something unexpected is wrong."""
    from language_reading_predictors.statistical_models import release as release_mod

    d = _setup_dir(tmp_path, "itt")
    _write_csv(d, "rope_summary.csv", _rope_row())

    def _boom(*_args, **_kwargs):
        raise RuntimeError("evidence store unreachable")

    monkeypatch.setattr(release_mod, "evaluate_release", _boom)
    payload = generate_key_findings(d)
    assert payload["status"] == "robustness_unresolved"
    assert "could not be evaluated" in payload["reason"]
    assert "evidence store unreachable" in payload["reason"]


def test_release_classes_reproduce_arviz_psense_diagnoses_exactly():
    """The release class must agree with the psense table printed in the report.

    Rather than assert a rule of our own, this drives ArviZ's own ``_diagnose``
    predicate over a grid and checks the two agree on every cell. It also pins the
    case an intuitive rule gets backwards: a posterior sensitive to the *likelihood*
    and insensitive to the prior is the ideal, not a conflict — the data are driving
    the result and the prior is doing nothing.
    """
    from language_reading_predictors.statistical_models.release import (
        PSENSE_THRESHOLD,
        classify_tau_sensitivity,
    )

    expected_for = {
        "potential prior-data conflict": "prior_data_conflict",
        "potential strong prior / weak likelihood": "prior_dominant",
        "✓": "clear",
    }

    def arviz_diagnose(prior: float, likelihood: float) -> str:
        # arviz_stats.psense_summary._diagnose, reproduced comparison for comparison.
        if prior >= PSENSE_THRESHOLD and likelihood >= PSENSE_THRESHOLD:
            return "potential prior-data conflict"
        if prior > PSENSE_THRESHOLD > likelihood:
            return "potential strong prior / weak likelihood"
        return "✓"

    grid = (0.0, 0.01, 0.049, PSENSE_THRESHOLD, 0.051, 0.2, 0.9)
    seen = set()
    for prior in grid:
        for likelihood in grid:
            frame = pd.DataFrame(
                [{"prior": prior, "likelihood": likelihood}], index=["tau"]
            )
            got, _, _, _ = classify_tau_sensitivity(frame)
            diagnosis = arviz_diagnose(prior, likelihood)
            assert got == expected_for[diagnosis], (prior, likelihood, diagnosis, got)
            seen.add(got)
    assert seen == {"clear", "prior_data_conflict", "prior_dominant"}

    # The healthy case, stated explicitly: data-sensitive, prior-insensitive.
    frame = pd.DataFrame([{"prior": 0.015, "likelihood": 0.092}], index=["tau"])
    assert classify_tau_sensitivity(frame)[0] == "clear"


def test_joint_tau_aggregates_worst_first_over_outcomes(tmp_path):
    """``joint`` fits one randomised ``tau`` per outcome, and the box speaks for all.

    There is no single element to classify, so the decision takes the worst class
    present and names the element that drove it. A per-fit decision that reported
    anything better than its worst constituent would have the box overstating what the
    fit supports — and a bare ``tau`` lookup finds no row at all, which under the
    fail-closed rule would withhold every joint fit for a diagnosis that is measured.
    """
    from language_reading_predictors.statistical_models.release import (
        classify_tau_sensitivity,
    )

    frame = pd.DataFrame(
        [
            {"prior": 0.01, "likelihood": 0.18, "diagnosis": "✓"},
            {"prior": 0.09, "likelihood": 0.22, "diagnosis": "potential prior-data conflict"},
            {"prior": 0.01, "likelihood": 0.16, "diagnosis": "✓"},
        ],
        index=["tau[W]", "tau[L]", "tau[R]"],
    )
    cls, prior, _lik, diagnosis = classify_tau_sensitivity(frame, term="tau")
    assert cls == "prior_data_conflict"
    assert prior == 0.09
    assert "tau[L]" in diagnosis

    dominant = frame.copy()
    dominant.loc["tau[R]"] = {"prior": 0.30, "likelihood": 0.01, "diagnosis": "x"}
    cls, _prior, _lik, diagnosis = classify_tau_sensitivity(dominant, term="tau")
    assert cls == "prior_dominant"
    assert "tau[R]" in diagnosis

    clean = frame.copy()
    clean.loc["tau[L]"] = {"prior": 0.01, "likelihood": 0.2, "diagnosis": "✓"}
    assert classify_tau_sensitivity(clean, term="tau")[0] == "clear"


def test_a_scalar_term_is_unaffected_by_the_vector_path(tmp_path):
    """The element aggregation must not change how a scalar term is classified."""
    from language_reading_predictors.statistical_models.release import (
        classify_tau_sensitivity,
    )

    scalar = pd.DataFrame(
        [{"prior": 0.01, "likelihood": 0.02, "diagnosis": "✓"}], index=["tau"]
    )
    assert classify_tau_sensitivity(scalar, term="tau") == ("clear", 0.01, 0.02, "✓")
    assert classify_tau_sensitivity(scalar, term="tau_t2")[0] == "unavailable"


def test_pooled_levels_covariate_exposure_and_skills_are_named(tmp_path):
    """#553: a raw-score covariate exposure is read in its own units (the fit's
    recorded raw-units SD) and the same-wave skill adjusters are named in the
    causal sentence, so the box never calls a raw score a logit or hides what the
    model held fixed."""
    d = _setup_dir(
        tmp_path,
        "pooled_levels",
        config={
            "kind": "pooled_levels",
            "outcome_symbol": "W",
            "mechanism_symbol": "erbto",
            "resolved_run_plan": {
                "outcome_symbol": "W",
                "mechanism_symbol": "erbto",
                "mechanism_is_covariate": True,
                "exposure_kind": "raw_covariate",
                "skill_symbols": ["TR"],
                "decompose_between_within": True,
                "waves": [1, 2, 3, 4],
                "use_wave_intercepts": True,
            },
            "extra": {"mechanism_exposure_sd_raw": 9.47},
        },
    )
    _write_rows(
        d,
        "pooled_levels_summary.csv",
        [
            {"term": "beta_between", "role": "association", "median": 0.91,
             "lo": 0.61, "hi": 1.25, "prob_positive": 1.0},
            {"term": "beta_within", "role": "association", "median": 0.14,
             "lo": -0.04, "hi": 0.31, "prob_positive": 0.886},
        ],
    )
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"
    texts = " ".join(s["text"] for s in payload["sentences"])
    assert "phonological memory (word/nonword repetition; 1 SD ≈ 9.5 raw points)" in texts
    assert "holds fixed the same-wave levels of Taught receptive vocabulary" in texts or (
        "holds fixed the same-wave levels of" in texts
    )
    assert "erbto" not in texts


# --- 2026-08-21 review fixes (finding 2: winner-picking candidate sets) -------


def test_lcsm_coupling_headline_excludes_covariate_rows(tmp_path):
    """The age precision slope out-resolves both couplings but must not win the
    "clearest longitudinal coupling" headline (live in the released 067 box)."""
    d = _setup_dir(tmp_path, "lcsm")
    _write_rows(
        d,
        "coupling_summary.csv",
        [
            {"coefficient": "g_L (prior L -> W change)", "median": 0.31,
             "mean": 0.31, "lo": 0.02, "hi": 0.61, "prob_pos": 0.98},
            {"coefficient": "d_age[W] (age -> W change)", "median": -0.15,
             "mean": -0.15, "lo": -0.21, "hi": -0.09, "prob_pos": 0.0001},
            {"coefficient": "b_hs (hs -> W change)", "median": 0.4,
             "mean": 0.4, "lo": 0.1, "hi": 0.7, "prob_pos": 0.9999},
        ],
    )
    payload = generate_key_findings(d)
    texts = _texts(payload)
    assert payload["status"] == "ok"
    assert "prior L -> W change" in texts
    assert "age" not in texts.split("causal")[0].split("check")[0]


def test_lcsm_lagged_coupling_confidence_uses_change_wording(tmp_path):
    d = _setup_dir(tmp_path, "lcsm")
    _write_rows(
        d,
        "coupling_summary.csv",
        [
            {"coefficient": "h_L (prior L change -> W change)", "median": 0.4,
             "mean": 0.4, "lo": 0.1, "hi": 0.7, "prob_pos": 0.99},
            {"coefficient": "g_L (prior L -> W change)", "median": 0.1,
             "mean": 0.1, "lo": -0.2, "hi": 0.4, "prob_pos": 0.7},
        ],
    )
    payload = generate_key_findings(d)
    texts = _texts(payload)
    assert "greater earlier change accompanies" in texts
    assert "higher earlier level" not in texts


def test_lcsm_window1_highlight_names_the_focal_outcome(tmp_path):
    """081 previously quoted the word-reading contrast, unnamed, under a
    taught-vocabulary model; the sentence must quote and name the focal row."""
    d = _setup_dir(
        tmp_path, "lcsm", config=_config("lcsm", outcome_symbol="TE")
    )
    _write_rows(
        d,
        "coupling_summary.csv",
        [
            {"coefficient": "g_W_TE (prior W -> TE change)", "median": 0.3,
             "mean": 0.3, "lo": 0.0, "hi": 0.6, "prob_pos": 0.95},
        ],
    )
    _write_rows(
        d,
        "itt_window1_contrast.csv",
        [
            {"coefficient": "itt_w1[W] (immediate - waitlist, window-1 latent change)",
             "median": 0.42, "lo": 0.09, "hi": 0.75, "prob_pos": 0.98},
            {"coefficient": "itt_w1[TE] (immediate - waitlist, window-1 latent change)",
             "median": 0.29, "lo": -0.02, "hi": 0.60, "prob_pos": 0.95},
        ],
    )
    payload = generate_key_findings(d)
    highlight = next(
        s["text"] for s in payload["sentences"] if s["kind"] == "highlight"
    )
    assert "+0.29" in highlight
    assert "+0.42" not in highlight


def test_corr_factor_structural_slope_prefers_plan_factors_over_covariates(tmp_path):
    """beta_age out-resolves the errors-in-variables focal slope but is an
    adjustment covariate, not a structural factor slope (live in mm-002/102)."""
    d = _setup_dir(
        tmp_path,
        "corr_factor",
        config=_config(
            "corr_factor",
            resolved_run_plan={"structural_factors": ["code"]},
        ),
    )
    _write_rows(
        d,
        "structural_summary.csv",
        [
            {"coefficient": "beta_code", "median": 0.35, "mean": 0.35,
             "lo": 0.10, "hi": 0.61, "prob_pos": 0.986},
            {"coefficient": "beta_age", "median": -0.35, "mean": -0.35,
             "lo": -0.53, "hi": -0.17, "prob_pos": 0.0013},
        ],
    )
    payload = generate_key_findings(d)
    texts = _texts(payload)
    assert "beta code" in texts
    assert "beta age" not in texts


def test_growth_interaction_plan_headlines_gamma_int(tmp_path):
    """Finding 1: gc-085's box previously answered gc-069's question; with the
    interaction declared, gamma_int is the headline and a summary without its
    rows degrades instead of publishing the wrong estimand."""
    cfg = _config("growth", resolved_run_plan={"age_ability_interaction": True})
    d = _setup_dir(tmp_path, "growth", config=cfg)
    _write_rows(
        d,
        "growth_association_summary.csv",
        [
            {"coefficient": "gamma", "outcome": "RG", "median": 0.15,
             "lo89": 0.06, "hi89": 0.25, "prob_positive": 0.99},
            {"coefficient": "gamma_int", "outcome": "RG", "median": 0.08,
             "lo89": -0.02, "hi89": 0.18, "prob_positive": 0.91},
        ],
    )
    payload = generate_key_findings(d)
    texts = _texts(payload)
    assert payload["status"] == "ok"
    assert "interaction" in texts
    assert "+0.08" in texts
    # The gamma main effect stays visible as context, not as the headline.
    headline = next(
        s["text"] for s in payload["sentences"] if s["kind"] == "headline"
    )
    assert "+0.08" in headline

    stale = _setup_dir(
        tmp_path, "growth", config=cfg, directory_name="growth-stale"
    )
    _write_rows(
        stale,
        "growth_association_summary.csv",
        [
            {"coefficient": "gamma", "outcome": "RG", "median": 0.15,
             "lo89": 0.06, "hi89": 0.25, "prob_positive": 0.99},
        ],
    )
    payload = generate_key_findings(stale)
    assert payload["status"] != "ok" or "gamma_int" not in _texts(payload)


# ---------------------------------------------------------------------------
# 2026-08-22 ITT audit regressions (issue #577, finding 9)
# ---------------------------------------------------------------------------


def test_analysis_set_reads_without_the_deprecated_t1_alias(tmp_path):
    """New fits stop writing ``available_t1_n``; the reader must not require it.

    It was a duplicate of ``analysed_archive_n`` under a name that claimed
    outcome-specific t1 availability, which is measure-specific: 50 children have
    a t1 nonword score against 53 for word reading.
    """
    d = _setup_dir(tmp_path, "itt")
    audit = pd.read_csv(d / "analysis_set.csv").drop(columns=["available_t1_n"])
    audit.to_csv(d / "analysis_set.csv", index=False)
    _write_csv(d, "rope_summary.csv", _rope_row())
    payload = generate_key_findings(d)
    assert payload["status"] == "ok"


def test_a_stored_bundle_keeping_the_alias_is_still_cross_checked(tmp_path):
    """Old bundles keep the column, and the equality that made it redundant."""
    d = _setup_dir(tmp_path, "itt")
    audit = pd.read_csv(d / "analysis_set.csv")
    assert "available_t1_n" in audit.columns
    audit["available_t1_n"] = audit["available_t1_n"] + 1
    audit.to_csv(d / "analysis_set.csv", index=False)
    _write_csv(d, "rope_summary.csv", _rope_row())
    payload = generate_key_findings(d)
    assert payload["status"] == "not_available"
    assert "arithmetic" in payload["reason"]


def test_results_factors_partial_branches_on_a_plan_with_no_focal_term():
    """#584 lower-severity 6: a supported pooled level plan has no randomised
    element, so the partial must not open by promising one causal coefficient."""
    text = (REPO / "docs/models/_partials/_results_factors.qmd").read_text(encoding="utf-8")
    assert '_no_focal = "focal_term" in _plan_intro and not _plan_intro.get("focal_term")' in text
    assert "This model reports **no causal coefficient**" in text
    # The promise of one causal coefficient is now reachable only when a focal
    # term is resolved.
    intro = text.split("if _no_focal and not _mv:", 1)[1]
    assert intro.index("The factor model reports **one** causal coefficient") > intro.index("elif _mv:")


def test_results_factors_partial_gates_the_blending_link_pair():
    """#584 decision 2: a level B report withholds its scientific results when the
    response-link pair is not ready, and shows both cards when it is."""
    text = (REPO / "docs/models/_partials/_results_factors.qmd").read_text(encoding="utf-8")
    assert "evaluate_level_blending_link_pair" in text
    assert "_scientific_results_released = False" in text
    assert "one-in-three guessing floor" in text

