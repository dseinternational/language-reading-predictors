# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the level factors family."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from language_reading_predictors.statistical_models.findings.common import (
    _KeyFindingsUnavailable,
    _kf_csv_row,
    _kf_direction_words,
    _kf_float,
    _kf_has_factor_term,
    _kf_headline_from_rope,
    _kf_outcome_label,
    _kf_rope_sentence,
    _kf_sentence,
)


def _kf_level_blending_link_sentence(output_dir: str | Path, config: Mapping) -> str | None:
    """The level family's paired-link sentence, or ``None`` when not a B fit.

    Mirrors :func:`_kf_blending_link_evidence` for ``level_factors`` (#584
    decision 2). Phoneme blending's ten items are three-alternative forced choice,
    so the ordinary inverse-logit mean can predict below-chance scores and the
    guessing-floor companion cannot; the two estimates are one piece of evidence,
    and a key-findings box that showed either alone would overstate what the fit
    establishes. Fails closed: an unready pair raises, which withholds the box
    rather than publishing a single-link headline.
    """
    if str(config.get("kind")) != "level_factors" or str(config.get("outcome_symbol")) != "B":
        return None
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        evaluate_level_blending_link_pair,
    )

    status = evaluate_level_blending_link_pair(output_dir, config=config)
    if not status.get("required"):
        return None
    if not status.get("ready"):
        raise _KeyFindingsUnavailable(str(status.get("reason") or "the B link pair is not ready"))
    cards = status["cards"]
    this_id = str(config.get("model_id"))
    other_id = next(k for k in cards if k != this_id)
    ordinary, floored = (
        (cards[k] for k in (this_id, other_id))
        if cards[this_id]["score_mean_link"] == "logit"
        else (cards[other_id], cards[this_id])
    )
    return (
        "Phoneme blending is scored from ten three-choice items, so a child "
        "answering at random scores about 3 out of 10. Two models are reported "
        "together because that floor matters: the ordinary model, which does not "
        f"know about it, puts the timepoint-2 effect at "
        f"**{_kf_float(ordinary['items_median']):+.1f} items** (89% credible range "
        f"{_kf_float(ordinary['items_lo']):+.1f} to "
        f"{_kf_float(ordinary['items_hi']):+.1f}), and the model that holds the "
        f"score at or above chance puts it at "
        f"**{_kf_float(floored['items_median']):+.1f} items** "
        f"({_kf_float(floored['items_lo']):+.1f} to "
        f"{_kf_float(floored['items_hi']):+.1f}). Neither number is the answer on "
        "its own."
    )


def _kf_build_level_factors(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
    """Level family: only the t2 group contrast is the randomised
    treated-versus-untreated effect; the later timepoints are randomised
    early-start-versus-delayed-start schedule contrasts (#631 finding 13)."""
    outcome_label = _kf_outcome_label(config)
    rope = _kf_csv_row(output_dir, "rope_summary.csv")
    if rope is None:
        raise _KeyFindingsUnavailable("rope_summary.csv (the t2 items-scale contrast) is not present")
    sentences: list[dict[str, str]] = []
    headline, is_rd = _kf_headline_from_rope(rope, outcome_label, "at the end of the randomised period (t2)")
    sentences.append(_kf_sentence(headline, "headline"))
    # Phoneme blending: the paired-link sentence rides immediately behind the
    # headline, because the headline alone is a single-link number (#584 decision 2).
    link_sentence = _kf_level_blending_link_sentence(output_dir, config)
    if link_sentence is not None:
        sentences.append(_kf_sentence(link_sentence, "sensitivity"))
    # #389 finding 3 — surfacing the t2 power-scaling verdict beside the headline —
    # is now the release gate's job rather than this builder's. The gate covers the
    # plan's focal t2 term (``d_grp_time[t2]``, or ``b_grp_time[1]`` under the free
    # comparator / on stored pre-#552 fits) for every level-factor fit, and it
    # classifies on the prior and
    # likelihood statistics rather than on the marker string, which is the better rule
    # (see ``_kf_psense_diagnosis``: an unrecognised marker on a clean estimate should
    # not publish a caution). Keeping a family-specific warning as well would say the
    # same thing twice and cost the reader the ROPE sentence, since the box caps at
    # five and ``rope`` is droppable — a size claim traded for a duplicated caution.
    sentences.append(
        _kf_sentence(
            # The level-family off-floor outcome is off-floor STATUS at each
            # wave (score > 0) — prevalence, not a floor-exit transition — so
            # the t2 sentence names the status estimand (#490 review follow-up).
            _kf_direction_words(rope["pd"], is_rd=is_rd, rd_event="being off the floor at t2"),
            "confidence",
        )
    )
    sentences.append(_kf_sentence(_kf_rope_sentence(rope, is_rd=is_rd), "rope"))
    plan = config.get("resolved_run_plan") or {}
    t1_referenced = str(plan.get("arm_gap_reference", "free")) == "t1" and bool(plan.get("group_by_time", True))
    causal = (
        "Only this t2 comparison compares being taught with not yet being taught. "
        + (
            "It is the **change** in the arm difference from the pre-randomisation "
            "baseline (t1) to t2 — a difference-in-differences of adjusted levels — "
            "so the model estimates the chance difference between the arms at t1 "
            "and subtracts it rather than carrying it into the estimate. That t1 "
            "difference is itself estimated under a cautious prior, so in a sample "
            "this small the subtraction is partial rather than exact. "
            if t1_referenced
            else ""
        )
        + "A cause-and-effect reading is "
        "limited to the fitted available-case t2 population and assumes outcome "
        "and required-covariate observation do not depend jointly on arm and "
        "potential outcomes. Group differences at later timepoints — after the "
        "waiting-list children had crossed over to the intervention — are still "
        "set by the original random assignment, but they compare an earlier with "
        "a later start of the same teaching, both groups having been taught by "
        "then: they are not treated-versus-untreated effects, and the model "
        "cannot say why any difference arises (longer teaching, carryover, "
        "maturation and test ceilings are inseparable). Ability and background "
        "terms remain adjusted associations."
    )
    # The headline nets out the WHOLE group contribution — balance term, focal
    # contrast and moderation increment — and adds back only the focal contrast
    # (#584 decision 1, the arm-free standardisation), so the ability-dependent part
    # of the benefit is held at mean ability and both arms are read from the same
    # starting point, while every other feature of each fitted t2 row (its own age,
    # ability main effect, adjusters and fitted child intercept) is retained and
    # averaged over (#271 item 5; design note Decision 4). It is therefore an average
    # across the fitted children, NOT a prediction for one typical child, which is
    # what this sentence used to say (#584 finding 5).
    # Appended to the causal sentence rather than added as a sixth: the box truncates
    # at KEY_FINDINGS_MAX_SENTENCES, and on a psense-flagged fit a sixth sentence
    # would silently drop this causal one — the least droppable of the set.
    if _kf_has_factor_term(output_dir, "gamma_grp_ability"):
        causal += (
            " That headline is an **average across the children in this "
            "comparison** — each kept at their own age, ability and background, and "
            "each read from the same starting point once the chance difference "
            "between the arms is taken out — with the part of the benefit that "
            "depends on ability held at the average: the model does let the benefit "
            "differ by ability, but that part is estimated partly from the "
            "timepoints that are not randomised, so it is reported on its own below "
            "rather than folded into the cause-and-effect figure."
        )
    sentences.append(_kf_sentence(causal, "causal"))
    return sentences
