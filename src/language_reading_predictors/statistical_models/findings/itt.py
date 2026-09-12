# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the itt family."""

from __future__ import annotations

from pathlib import Path
import os
from collections.abc import Mapping
import numpy as np
import pandas as pd
from language_reading_predictors.statistical_models.findings.common import (
    _KeyFindingsUnavailable,
    _kf_csv,
    _kf_csv_row,
    _kf_direction_words,
    _kf_float,
    _kf_headline_from_rope,
    _kf_outcome_label,
    _kf_rope_sentence,
    _kf_sentence,
)


def _kf_itt_analysis_population(output_dir: str | Path) -> dict[str, int]:
    """Validate and summarise the two-arm available-case audit for an ITT fit.

    The causal sentence is not allowed to infer its population from a model title or
    a generic config label.  ``analysis_set.csv`` is generated from the actual fitted
    rows and carries the published randomised allocation, archived cohort and fitted
    arm counts.  Missing or incoherent arithmetic withholds the key findings rather
    than publishing an unqualified randomisation claim.
    """

    path = os.path.join(str(output_dir), "analysis_set.csv")
    if not os.path.exists(path):
        raise _KeyFindingsUnavailable("analysis_set.csv is missing, so the fitted causal population cannot be verified")
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise _KeyFindingsUnavailable("analysis_set.csv is not readable") from exc
    required = {
        "arm",
        "G",
        "randomised_n",
        "lost_to_follow_up_n",
        "analysed_archive_n",
        "discontinued_but_followed_n",
        "fitted_n",
        "absent_from_archive_n",
        "not_in_fitted_analysis_n",
        "excluded_after_archive_n",
    }
    if len(frame) != 2 or not required.issubset(frame.columns):
        raise _KeyFindingsUnavailable("analysis_set.csv does not contain exactly the two required arm rows")
    # ``available_t1_n`` is the deprecated duplicate of ``analysed_archive_n``
    # (2026-08-22 ITT audit, finding 9): it never held outcome-specific t1
    # availability — which is measure-specific, 50 for N against 53 for W — and
    # fits from this commit on stop writing it. Stored bundles still carry it and
    # are still checked for the equality that made it redundant.
    optional = {"available_t1_n"} & set(frame.columns)
    numeric = frame[list((required | optional) - {"arm"})].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise _KeyFindingsUnavailable("analysis_set.csv contains non-numeric counts")
    if not np.equal(numeric.to_numpy(), np.floor(numeric.to_numpy())).all():
        raise _KeyFindingsUnavailable("analysis_set.csv counts must be integers")
    work = frame.copy()
    for column in numeric:
        work[column] = numeric[column].astype(int)
    if set(work["G"]) != {0, 1} or work["G"].duplicated().any():
        raise _KeyFindingsUnavailable("analysis_set.csv does not identify both arms")
    if not (
        (work["randomised_n"] - work["analysed_archive_n"]).eq(work["lost_to_follow_up_n"]).all()
        and (not optional or work["analysed_archive_n"].eq(work["available_t1_n"]).all())
        and work["lost_to_follow_up_n"].eq(work["absent_from_archive_n"]).all()
        and (work["randomised_n"] - work["fitted_n"]).eq(work["not_in_fitted_analysis_n"]).all()
        and (work["analysed_archive_n"] - work["fitted_n"]).eq(work["excluded_after_archive_n"]).all()
        and (work["randomised_n"] >= work["analysed_archive_n"]).all()
        and (work["analysed_archive_n"] >= work["fitted_n"]).all()
        and (work["fitted_n"] > 0).all()
    ):
        raise _KeyFindingsUnavailable("analysis_set.csv arm-count arithmetic is inconsistent")
    indexed = work.set_index("G")
    return {
        "randomised": int(work["randomised_n"].sum()),
        "archived": int(work["analysed_archive_n"].sum()),
        "lost_to_follow_up": int(work["lost_to_follow_up_n"].sum()),
        "discontinued_but_followed": int(work["discontinued_but_followed_n"].sum()),
        "fitted": int(work["fitted_n"].sum()),
        "fitted_intervention": int(_kf_float(indexed.loc[1, "fitted_n"])),
        "fitted_control": int(_kf_float(indexed.loc[0, "fitted_n"])),
    }


def _kf_itt_causal_sentence(population: Mapping[str, int], *, floor_rule: bool = False) -> str:
    """Selected-population causal wording shared by every single-outcome ITT.

    ``floor_rule`` names the extra qualification the P/N off-floor primaries carry
    (#392): they are a *post-hoc*, data-adaptive contrast within the subgroup observed
    at the floor at baseline, so the population is narrower than the trial's and the
    subgroup was chosen after seeing the data. The review found these emitting the
    ordinary ITT sentence, which understates both.
    """

    label = (
        "This is a post-hoc subgroup available-case modified ITT estimate, not a full-randomised-cohort ITT estimate. "
        if floor_rule
        else ("This is an available-case modified ITT estimate, not a full-randomised-cohort ITT estimate. ")
    )
    scope = (
        (
            "Random assignment supports a cause-and-effect reading only within the "
            "subgroup of children who scored at the floor of this measure at "
            "baseline — a group chosen after the data were seen, so this is an "
            "exploratory analysis rather than a planned one — and only under the "
            "available-case assumption: for the "
        )
        if floor_rule
        else (
            "Random assignment supports a cause-and-effect reading only under the available-case assumption: for the "
        )
    )
    tail = (
        (
            " Without further missing-data assumptions, this is neither the effect "
            f"for all {population['randomised']} randomised children nor the effect "
            "for children who were already off the floor."
        )
        if floor_rule
        else (
            " Without further missing-data assumptions, this is not the effect for "
            f"all {population['randomised']} randomised children."
        )
    )
    return (
        label + scope + f"{population['fitted']} fitted children "
        f"({population['fitted_intervention']} immediate-intervention and "
        f"{population['fitted_control']} waiting-list), archive inclusion, outcome "
        "observation and any complete-case restriction must not depend jointly on "
        "assigned arm and potential outcomes. The "
        f"{population['lost_to_follow_up']} children absent from the analysed archive "
        "were lost to follow-up; this is distinct from the "
        f"{population['discontinued_but_followed']} children who stopped intervention "
        "but were followed and retained by assignment." + tail
    )


def _kf_blending_link_evidence(
    output_dir: str | Path,
    config: Mapping,
) -> tuple[Mapping, str] | None:
    """Return the current trace-recomputed B row and its paired-link sentence."""

    if str(config.get("outcome_symbol")) != "B":
        return None
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        BLENDING_COMPANION_MODEL_ID,
        BLENDING_PRIMARY_MODEL_ID,
        evaluate_local_blending_link_sensitivity,
    )

    status = evaluate_local_blending_link_sensitivity(output_dir, config=config)
    if not status.get("required"):
        # A B-outcome fit outside the registered 008/108 pair has no bundle to
        # quote (the evaluator returns no ``summary``); the release gate withholds
        # such a fit separately (2026-08-20 ITT review), so degrade gracefully
        # here rather than KeyError on the absent key.
        return None
    if not status.get("ready"):
        raise _KeyFindingsUnavailable(str(status.get("reason") or "B link sensitivity is not ready"))
    summary = status["summary"].set_index("model_id")
    ordinary = summary.loc[BLENDING_PRIMARY_MODEL_ID]
    guessing = summary.loc[BLENDING_COMPANION_MODEL_ID]
    current_model_id = str(config.get("model_id"))
    if current_model_id not in summary.index:
        raise _KeyFindingsUnavailable("current B model is not one of the validated paired-link fits")
    current = summary.loc[current_model_id]

    def _effect(row: Mapping | pd.Series) -> str:
        return (
            f"{_kf_float(row['effect_items_median']):+.1f} items "
            f"(89% credible range {_kf_float(row['effect_items_lo']):+.1f} to "
            f"{_kf_float(row['effect_items_hi']):+.1f})"
        )

    sentence = (
        "The phoneme-blending conclusion is response-link sensitive: the ordinary "
        f"logit model gives {_effect(ordinary)}, whereas the mechanically motivated "
        f"one-in-three guessing-floor model gives {_effect(guessing)}. Read neither "
        "link in isolation; the pair, not the more favourable estimate, is the "
        "robustness result. The shared latent-scale priors also map to different "
        "items-scale priors under the two links, as shown in the paired report."
    )
    return current, sentence


def _kf_itt_missingness_sentence(output_dir: str | Path, config: Mapping) -> str | None:
    """The mandatory full-57 word-reading sensitivity, when registered."""

    if str(config.get("model_id")) != "lrp-rli-itt-010":
        return None
    frame = _kf_csv(output_dir, "itt_missingness_sensitivity.csv")
    if frame is None or "scenario" not in frame.columns:
        raise _KeyFindingsUnavailable("itt_missingness_sensitivity.csv is absent or malformed")
    indexed = frame.set_index(frame["scenario"].astype(str), drop=False)
    required = (
        "screening_model_observed_profiles",
        "mar_all_57",
        "jump_to_reference_intervention_nonstarter",
    )
    if any(scenario not in indexed.index for scenario in required):
        raise _KeyFindingsUnavailable("the bridge, MAR or jump-to-reference missingness row is absent")
    bridge = indexed.loc[required[0]].to_dict()
    mar = indexed.loc[required[1]].to_dict()
    j2r = indexed.loc[required[2]].to_dict()
    grid = frame.loc[frame["scenario_class"].astype(str) == "arm_specific_delta_grid"]
    if len(grid) != 25:
        raise _KeyFindingsUnavailable("the 25-row missingness delta grid is incomplete")
    grid_medians = pd.to_numeric(grid["effect_items_median"], errors="coerce")
    if not np.isfinite(grid_medians.to_numpy(dtype=float)).all():
        raise _KeyFindingsUnavailable("the missingness delta grid is non-numeric")
    delta_i = pd.to_numeric(grid["delta_intervention_items"], errors="coerce")
    delta_c = pd.to_numeric(grid["delta_control_items"], errors="coerce")
    factual_mar_rows = grid.loc[delta_i.eq(0.0) & delta_c.eq(0.0)]
    if len(factual_mar_rows) != 1:
        raise _KeyFindingsUnavailable("the factual-arm zero-delta MAR completion is absent")
    factual_mar = factual_mar_rows.iloc[0]
    clipping = grid[["clipped_intervention_fraction", "clipped_control_fraction"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(clipping.to_numpy(dtype=float)).all():
        raise _KeyFindingsUnavailable("the missingness clipping audit is non-numeric")
    bounds = _kf_csv(output_dir, "attrition_bounds.csv")
    needed_bounds = {"outcome", "worst_case_items_lower", "worst_case_items_upper"}
    if bounds is None or not needed_bounds.issubset(bounds.columns):
        raise _KeyFindingsUnavailable("the word-reading sharp attrition bounds are absent")
    word_bounds = bounds.loc[bounds["outcome"].astype(str).eq("W")]
    if len(word_bounds) != 1:
        raise _KeyFindingsUnavailable("the word-reading sharp-bound row is not unique")
    sharp_lo = _kf_float(word_bounds.iloc[0]["worst_case_items_lower"])
    sharp_hi = _kf_float(word_bounds.iloc[0]["worst_case_items_upper"])

    def _effect(row: Mapping | pd.Series) -> str:
        return (
            f"{_kf_float(row['effect_items_median']):+.1f} items "
            f"(89% credible range {_kf_float(row['effect_items_lo89']):+.1f} to "
            f"{_kf_float(row['effect_items_hi89']):+.1f})"
        )

    return (
        "Missing-outcome sensitivity: refitting the same 53 observed outcomes with "
        f"screening word reading and age gave {_effect(bridge)} over common observed "
        f"profiles; common-profile standardisation over all 57 under MAR gave "
        f"{_effect(mar)}. MAR here assumes outcome observation is independent of the "
        "unseen t2 score conditional on assigned arm, screening word reading and "
        "screening age, together with the fitted outcome model and covariate overlap; "
        "the observed data cannot test that assumption. The reference and delta "
        "analyses instead complete the factual randomised arms (29 intervention "
        f"versus 28 control): their zero-delta MAR anchor was {_effect(factual_mar)}, "
        "and giving the one intervention non-starter the control mean surface gave "
        f"{_effect(j2r)}. Across the fixed arm-specific delta grid, posterior "
        f"medians ranged from {float(grid_medians.min()):+.1f} to "
        f"{float(grid_medians.max()):+.1f} items; up to "
        f"{100 * float(clipping['clipped_intervention_fraction'].max()):.0f}% of "
        "missing-intervention and "
        f"{100 * float(clipping['clipped_control_fraction'].max()):.0f}% of "
        "missing-control profile predictions reached a physical test bound. The "
        f"model-free extreme-case benchmark spans {sharp_lo:+.1f} to {sharp_hi:+.1f} "
        "items, so unrestricted missing outcomes can reverse direction. These are "
        "assumption-dependent secondary estimates, not recovered outcomes; the "
        "mean-surface no-benefit restriction is not distributional reference-based "
        "multiple imputation."
    )


def _kf_itt_attrition_bounds_clause(output_dir: str | Path, config: Mapping) -> str | None:
    """A clause for the causal sentence quoting the model-free attrition bounds.

    ``attrition_bounds.csv`` (``itt.write_itt_analysis_set``) completes the
    randomised children with no timepoint-2 outcome at the test floor or ceiling
    in the least and most favourable ways and bounds the *raw* timepoint-2 arm
    difference; every ``itt`` fit writes it, but until 2026-08-19 only word
    reading's key findings quoted it (inside the mandatory missingness sentence).
    The bound belongs with the available-case qualification it quantifies, so it
    is appended to the causal sentence — which the five-sentence cap never drops —
    rather than added as a sixth sentence that would displace the size-of-benefit
    statement. Word reading is skipped (already covered); floor-rule fits are
    skipped because their headline estimand is an off-floor risk difference among
    baseline-floor children, which the raw post-score contrast does not describe
    (for phonetic spelling that contrast is dominated by the baseline arm
    imbalance). Optional: an absent or malformed table yields ``None`` rather than
    withholding the findings (``notes/202608182200-findings-by-question.md``,
    question 8).
    """

    if str(config.get("model_id")) == "lrp-rli-itt-010":
        return None
    plan = config.get("resolved_run_plan") or {}
    if bool(plan.get("floor_rule", False)):
        return None
    bounds = _kf_csv(output_dir, "attrition_bounds.csv")
    needed = {
        "outcome",
        "missing_intervention_n",
        "missing_control_n",
        "worst_case_items_lower",
        "worst_case_items_upper",
    }
    if bounds is None or len(bounds) != 1 or not needed.issubset(bounds.columns):
        return None
    row = bounds.iloc[0]
    try:
        missing_i = int(_kf_float(row["missing_intervention_n"]))
        missing_c = int(_kf_float(row["missing_control_n"]))
        lo = _kf_float(row["worst_case_items_lower"])
        hi = _kf_float(row["worst_case_items_upper"])
    except TypeError, ValueError:
        return None
    if not (np.isfinite(lo) and np.isfinite(hi)) or missing_i + missing_c <= 0:
        return None
    symbol = str(row["outcome"])
    units = (
        "half-marks on the doubled information scale"
        if symbol == "EI"
        else "marks"
        if symbol in {"EG", "EI40"}
        else "items"
    )
    if lo > 0 or hi < 0:
        verdict = "so the direction does not depend on how those outcomes are completed"
    else:
        verdict = "so unrestricted missing outcomes could reverse direction"
    n_missing = missing_i + missing_c
    return (
        f" Completing the {n_missing} randomised "
        f"{'child' if n_missing == 1 else 'children'} with no timepoint-2 score on "
        f"this measure ({missing_i} intervention, {missing_c} control) at the test "
        "floor or ceiling in the least and most favourable ways bounds the raw "
        f"timepoint-2 arm difference between {lo:+.1f} and {hi:+.1f} {units}, "
        f"{verdict}; that bounds the unadjusted post-score contrast, not the "
        "covariate-adjusted estimate above, and the model-based missing-data "
        "envelope (MAR, reference-based and delta scenarios) has been fitted for "
        "word reading only."
    )


def _kf_build_itt(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
    """Available-case modified ITT suite: rope card first, then tau fallback."""
    outcome_label = _kf_outcome_label(config)
    population = _kf_itt_analysis_population(output_dir)
    blending_evidence = _kf_blending_link_evidence(output_dir, config)
    score_mean_link = str((config.get("resolved_run_plan") or {}).get("score_mean_link", "logit"))
    sentences: list[dict[str, str]] = []
    if blending_evidence is not None:
        blending_row, blending_sentence = blending_evidence
        link_prefix = (
            "Under the ordinary-logit model, "
            if score_mean_link == "logit"
            else "Under the one-in-three guessing-floor model, "
        )
        sentences.append(
            _kf_sentence(
                f"{link_prefix}available-case modified ITT estimate: the model-estimated "
                f"intervention-minus-comparison contrast for {outcome_label} was "
                f"**{_kf_float(blending_row['effect_items_median']):+.1f} items** "
                "over the trial period "
                f"(89% credible range "
                f"{_kf_float(blending_row['effect_items_lo']):+.1f} to "
                f"{_kf_float(blending_row['effect_items_hi']):+.1f}).",
                "headline",
            )
        )
        sentences.append(_kf_sentence(blending_sentence, "sensitivity"))
        direction_sentence = _kf_direction_words(blending_row["prob_effect_positive"], is_rd=False)
        sentences.append(
            _kf_sentence(
                f"{link_prefix}{direction_sentence[0].lower()}{direction_sentence[1:]}",
                "confidence",
            )
        )
    else:
        rope = _kf_csv_row(output_dir, "rope_summary.csv")
        if rope is not None:
            headline, is_rd = _kf_headline_from_rope(
                rope,
                outcome_label,
                "over the trial period in the available-case modified ITT analysis",
            )
            sentences.append(_kf_sentence(headline, "headline"))
            direction = _kf_direction_words(rope["pd"], is_rd=is_rd)
            if str(config.get("model_id")) == "lrp-rli-itt-010":
                direction = (
                    "For the 53-outcome available-case modified ITT model of record, "
                    f"{direction[0].lower()}{direction[1:]}"
                )
            sentences.append(_kf_sentence(direction, "confidence"))
            sentences.append(_kf_sentence(_kf_rope_sentence(rope, is_rd=is_rd), "rope"))
        else:
            tau = _kf_csv_row(output_dir, "tau_summary.csv")
            if tau is None:
                raise _KeyFindingsUnavailable("neither rope_summary.csv nor tau_summary.csv is present")
            from language_reading_predictors.statistical_models.measures import MEASURES

            measure = MEASURES.get(str(config.get("outcome_symbol", "")))
            if measure is not None:
                n = measure.n_trials
                med = _kf_float(tau["tau_prob_median"]) * n
                lo = _kf_float(tau["tau_prob_lo"]) * n
                hi = _kf_float(tau["tau_prob_hi"]) * n
                sentences.append(
                    _kf_sentence(
                        "Available-case modified ITT estimate: the model-estimated "
                        f"intervention-minus-comparison contrast for {outcome_label} "
                        f"was **{med:+.1f} items** over the trial period "
                        f"(89% credible range {lo:+.1f} to {hi:+.1f}).",
                        "headline",
                    )
                )
            sentences.append(
                _kf_sentence(
                    _kf_direction_words(tau["prob_tau_pos"], is_rd=False),
                    "confidence",
                )
            )
            sentences.append(
                _kf_sentence(
                    "No minimally-important difference has been agreed for this "
                    "outcome, so no is-it-big-enough-to-matter verdict is reported.",
                    "note",
                )
            )
    missingness_sentence = _kf_itt_missingness_sentence(output_dir, config)
    if missingness_sentence is not None:
        sentences.append(_kf_sentence(missingness_sentence, "sensitivity"))
    causal_sentence = _kf_itt_causal_sentence(
        population,
        floor_rule=bool((config.get("resolved_run_plan") or {}).get("floor_rule", False)),
    )
    attrition_clause = _kf_itt_attrition_bounds_clause(output_dir, config)
    if attrition_clause is not None:
        causal_sentence += attrition_clause
    sentences.append(_kf_sentence(causal_sentence, "causal"))
    return sentences
