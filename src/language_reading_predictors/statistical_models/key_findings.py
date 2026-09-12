# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Key findings calculations and summaries."""

from __future__ import annotations

from language_reading_predictors.statistical_models.family_registry import FAMILIES

from language_reading_predictors.statistical_models.findings.common import (
    _KF_FACTOR_LABELS as _KF_FACTOR_LABELS,
    _kf_float as _kf_float,
    _kf_pct as _kf_pct,
    _kf_csv_row as _kf_csv_row,
    _PSENSE_CLEAR_MARKERS as _PSENSE_CLEAR_MARKERS,
    _kf_psense_diagnosis as _kf_psense_diagnosis,
    _kf_csv as _kf_csv,
    _kf_most_resolved_row as _kf_most_resolved_row,
    _kf_plain_label as _kf_plain_label,
    _kf_dag_unit as _kf_dag_unit,
    _kf_measure_label as _kf_measure_label,
    _kf_association_direction as _kf_association_direction,
    _kf_outcome_label as _kf_outcome_label,
    _kf_direction_words as _kf_direction_words,
    _kf_headline_from_rope as _kf_headline_from_rope,
    _kf_rope_sentence as _kf_rope_sentence,
    _kf_has_factor_term as _kf_has_factor_term,
    _kf_strongest_factor as _kf_strongest_factor,
    _KF_MODERATION_LABELS as _KF_MODERATION_LABELS,
    _KF_COVARIATE_MODERATOR_LABELS as _KF_COVARIATE_MODERATOR_LABELS,
)
from language_reading_predictors.statistical_models.findings.dose_response import (
    _kf_dose_companion_location as _kf_dose_companion_location,
    _kf_dose_blending_link_sentence as _kf_dose_blending_link_sentence,
)
from language_reading_predictors.statistical_models.findings.gain_factors import (
    _kf_moderation_sentences as _kf_moderation_sentences,
)
from language_reading_predictors.statistical_models.findings.growth import (
    _kf_growth_interaction_sentences as _kf_growth_interaction_sentences,
)
from language_reading_predictors.statistical_models.findings.historical_joint import (
    _kf_pair_selection_note as _kf_pair_selection_note,
)
from language_reading_predictors.statistical_models.findings.itt import (
    _kf_itt_analysis_population as _kf_itt_analysis_population,
    _kf_itt_causal_sentence as _kf_itt_causal_sentence,
    _kf_blending_link_evidence as _kf_blending_link_evidence,
    _kf_itt_missingness_sentence as _kf_itt_missingness_sentence,
    _kf_itt_attrition_bounds_clause as _kf_itt_attrition_bounds_clause,
)
from language_reading_predictors.statistical_models.findings.joint import (
    _kf_joint_pp as _kf_joint_pp,
    _kf_joint_optional_text as _kf_joint_optional_text,
    _kf_joint_marginal_phrase as _kf_joint_marginal_phrase,
)
from language_reading_predictors.statistical_models.findings.joint_mechanism import (
    _kf_jm_interval as _kf_jm_interval,
    _kf_jm_wave_series as _kf_jm_wave_series,
    _kf_jm_psense_flags as _kf_jm_psense_flags,
)
from language_reading_predictors.statistical_models.findings.level_factors import (
    _kf_level_blending_link_sentence as _kf_level_blending_link_sentence,
)
from language_reading_predictors.statistical_models.findings.mechanism import (
    _kf_mechanism_shape_caveat as _kf_mechanism_shape_caveat,
    _kf_mechanism_slope_sentences as _kf_mechanism_slope_sentences,
    _kf_lower_first as _kf_lower_first,
    _kf_moderator_label as _kf_moderator_label,
    _kf_mechanism_curve_context as _kf_mechanism_curve_context,
    _kf_moderation_items_sentence as _kf_moderation_items_sentence,
)
import json
import os
from collections.abc import Mapping
from language_reading_predictors.statistical_models.findings.adjusted import (
    _kf_build_adjusted as _kf_build_adjusted,
)
from language_reading_predictors.statistical_models.findings.aligned import (
    _kf_build_aligned as _kf_build_aligned,
)
from language_reading_predictors.statistical_models.findings.block_exposure import (
    _kf_build_block_exposure as _kf_build_block_exposure,
)
from language_reading_predictors.statistical_models.findings.common import (
    _KeyFindingsUnavailable,
    _kf_sentence,
)
from language_reading_predictors.statistical_models.findings.concurrent import (
    _kf_build_concurrent as _kf_build_concurrent,
)
from language_reading_predictors.statistical_models.findings.corr_factor import (
    _kf_build_corr_factor as _kf_build_corr_factor,
)
from language_reading_predictors.statistical_models.findings.did import (
    _kf_build_did as _kf_build_did,
)
from language_reading_predictors.statistical_models.findings.dose_response import (
    _kf_build_dose_response as _kf_build_dose_response,
)
from language_reading_predictors.statistical_models.findings.gain_factors import (
    _kf_build_gain_factors as _kf_build_gain_factors,
)
from language_reading_predictors.statistical_models.findings.growth import (
    _kf_build_growth as _kf_build_growth,
)
from language_reading_predictors.statistical_models.findings.historical_growth import (
    _kf_build_historical_growth as _kf_build_historical_growth,
)
from language_reading_predictors.statistical_models.findings.historical_joint import (
    _kf_build_historical_joint as _kf_build_historical_joint,
)
from language_reading_predictors.statistical_models.findings.horseshoe import (
    _kf_build_horseshoe as _kf_build_horseshoe,
)
from language_reading_predictors.statistical_models.findings.itt import (
    _kf_build_itt as _kf_build_itt,
)
from language_reading_predictors.statistical_models.findings.joint import (
    _kf_build_joint as _kf_build_joint,
)
from language_reading_predictors.statistical_models.findings.joint_mechanism import (
    _kf_build_joint_mechanism as _kf_build_joint_mechanism,
)
from language_reading_predictors.statistical_models.findings.lcsm import (
    _kf_build_lcsm as _kf_build_lcsm,
)
from language_reading_predictors.statistical_models.findings.level_factors import (
    _kf_build_level_factors as _kf_build_level_factors,
)
from language_reading_predictors.statistical_models.findings.long_corr_factor import (
    _kf_build_long_corr_factor as _kf_build_long_corr_factor,
)
from language_reading_predictors.statistical_models.findings.mechanism import (
    _kf_build_mechanism as _kf_build_mechanism,
    mechanism_headline_estimand,
)
from language_reading_predictors.statistical_models.findings.mediation import (
    _kf_build_mediation as _kf_build_mediation,
)
from language_reading_predictors.statistical_models.findings.pooled_levels import (
    _kf_build_pooled_levels as _kf_build_pooled_levels,
)
from language_reading_predictors.statistical_models.findings.survival import (
    _kf_build_survival as _kf_build_survival,
)


KEY_FINDINGS_FILENAME = "key_findings.json"


KEY_FINDINGS_SCHEMA_VERSION = 1


KEY_FINDINGS_MAX_SENTENCES = 5


def _kf_build_fallback(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Unknown future family: an honest placeholder, never a wrong summary."""
    kind = config.get("kind") or "this"
    return [
        _kf_sentence(
            f"A plain-language key-findings summary has not yet been written for "
            f"the {kind} model family.",
            "note",
        ),
        _kf_sentence(
            "Unless a term is explicitly flagged as randomised in the results "
            "below, the estimates in this report are adjusted associations or "
            "descriptive quantities, not causal effects.",
            "causal",
        ),
        _kf_sentence(
            "See the results section below for the full estimates with their "
            "uncertainty.",
            "note",
        ),
    ]


_KF_BUILDERS = {
    kind: globals()[descriptor.key_findings_builder]
    for kind, descriptor in FAMILIES.items()
    if descriptor.key_findings_builder is not None
}


#: Roles that may be dropped to make room for a release note. The causal sentence is
#: never droppable: #464 recorded that silently losing it is exactly what happens when
#: a sixth sentence is appended past the cap, and it is the sentence carrying the
#: study's central qualification.
_KF_DROPPABLE_ROLES = ("rope", "note")


def _kf_with_release_note(
    sentences: list[dict[str, str]], note: str
) -> list[dict[str, str]]:
    """Insert a robustness note before the causal sentence, within the cap.

    The box truncates at :data:`KEY_FINDINGS_MAX_SENTENCES`, and #464 recorded the
    failure mode: appending a sixth sentence silently drops the causal one, because
    truncation takes the first five. So the note goes *before* the causal sentence,
    and if that would overflow, a droppable sentence makes room. If nothing is
    droppable the note is omitted rather than displacing anything — a missing note is
    a smaller loss than a missing qualification, and the note is also recorded
    verbatim under ``release`` in the payload either way.
    """
    result = list(sentences)
    causal_at = next(
        (i for i, s in enumerate(result) if s.get("kind") == "causal"), len(result)
    )
    if len(result) >= KEY_FINDINGS_MAX_SENTENCES:
        droppable = [
            i for i, s in enumerate(result) if s.get("kind") in _KF_DROPPABLE_ROLES
        ]
        if not droppable:
            return result
        removed = droppable[-1]
        del result[removed]
        if removed < causal_at:
            causal_at -= 1
    result.insert(causal_at, _kf_sentence(note, "robustness"))
    return result


def generate_key_findings(output_dir, *, decision=None) -> dict:
    """Build and write ``key_findings.json`` for a fit output directory (#320).

    Reads only artefacts already in ``output_dir`` (``config.json``,
    ``diagnostics_summary.json`` and the family CSVs), so it can be re-run over
    an existing fit without refitting. Missing artefacts degrade to a
    ``not_available`` payload with a reason, never an exception; sentences are
    capped at :data:`KEY_FINDINGS_MAX_SENTENCES` and can never contain a
    non-finite number (:func:`_kf_float` raises, and the builder's whole payload
    then degrades). Returns the payload it wrote.

    ``decision`` is the fit's :class:`release.ReleaseEvaluation` — whether it may
    publish findings at all, and why. Report finalisation computes it and passes
    it in (#394 design point 3); when it is omitted, as by the regeneration
    scripts, this function evaluates it over the stored directory. Either way the
    ordering it encodes holds: inputs, then the sampling-quality gate, then
    required artefacts, then robustness. Nothing here re-decides any of that —
    what remains below is building the sentences.
    """
    out = str(output_dir)
    from language_reading_predictors.statistical_models.release import (
        evaluate_publication,
    )

    if decision is None:
        decision = evaluate_publication(out)
    config = decision.config if decision.config is not None else None

    payload: dict = {
        "schema_version": KEY_FINDINGS_SCHEMA_VERSION,
        "model_id": (config or {}).get("model_id"),
        "kind": (config or {}).get("kind"),
        "sentences": [],
    }

    if decision.status == "gate_failed":
        payload["status"] = "gate_failed"
        payload["failing_checks"] = list(decision.failing_checks)
        return _write_key_findings(out, payload)

    if decision.status == "robustness_unresolved":
        payload["status"] = "robustness_unresolved"
        payload["reason"] = decision.reason
        if decision.robustness is not None:
            payload["release"] = decision.robustness.as_dict()
        return _write_key_findings(out, payload)

    if not decision.publishable:
        # Unreadable/unresolved inputs and incomplete required artefacts.
        payload["status"] = decision.status
        payload["reason"] = decision.reason
        if decision.input_failures:
            payload["input_failures"] = list(decision.input_failures)
        if decision.missing_artifacts:
            payload["missing_artifacts"] = list(decision.missing_artifacts)
        return _write_key_findings(out, payload)

    release = decision.robustness
    builder = _KF_BUILDERS.get(config.get("kind"), _kf_build_fallback)
    try:
        sentences = builder(out, config)
    except _KeyFindingsUnavailable as exc:
        payload["status"] = "not_available"
        payload["reason"] = str(exc)
        return _write_key_findings(out, payload)
    except (KeyError, ValueError, OSError) as exc:
        # A malformed CSV must degrade to an explicit note, never break a fit
        # or a render (#320 acceptance criteria).
        payload["status"] = "not_available"
        payload["reason"] = f"key-findings builder failed: {exc}"
        return _write_key_findings(out, payload)

    if config.get("kind") == "mechanism":
        # #602: the published headline number carries its estimand id, reference
        # population and exposure interval, so a reader never has to infer which of
        # the family's two natural-scale contrasts a number came from.
        estimand = mechanism_headline_estimand(out)
        if estimand is not None:
            payload["headline_estimand"] = estimand

    if release is not None:
        payload["release"] = release.as_dict()
        if release.note:
            sentences = _kf_with_release_note(sentences, release.note)

    payload["status"] = "ok"
    payload["sentences"] = sentences[:KEY_FINDINGS_MAX_SENTENCES]
    if str(config.get("outcome_symbol")) == "B":
        # The #466 provenance stamp belongs to the two *registered* paired-link fits
        # that build the bundle, not to every ``B`` outcome. Nine further models
        # (aligned, concurrent, did, dose_response, gain_factors, level_factors and
        # mediation) share the outcome symbol but never write the CSV, and their
        # family builders never reach the catchable ``_KeyFindingsUnavailable`` that
        # ``_kf_build_itt`` raises — so hashing unconditionally killed those fits here
        # in ``runtime.finalize_report``, *after* sampling, discarding the staging directory.
        # Imports stay function-local: ``blending_sensitivity`` imports this module.
        from language_reading_predictors.statistical_models.blending_sensitivity import (
            BLENDING_LINK_MODELS,
            BLENDING_SENSITIVITY_FILENAME,
        )

        if str(config.get("model_id")) in {mid for mid, _ in BLENDING_LINK_MODELS}:
            from language_reading_predictors.statistical_models.sensitivity import (
                sha256_file,
            )

            payload["blending_link_sensitivity_sha256"] = sha256_file(
                os.path.join(out, BLENDING_SENSITIVITY_FILENAME)
            )
    if str(config.get("model_id")) == "lrp-rli-itt-010":
        from language_reading_predictors.statistical_models.itt_missingness import (
            MISSINGNESS_SUMMARY_FILENAME,
            sha256_file,
        )

        payload["itt_missingness_sensitivity_sha256"] = sha256_file(
            os.path.join(out, MISSINGNESS_SUMMARY_FILENAME)
        )
    return _write_key_findings(out, payload)


def _write_key_findings(output_dir: str, payload: dict) -> dict:
    with open(os.path.join(output_dir, KEY_FINDINGS_FILENAME), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return payload
