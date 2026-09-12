# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the gain factors family."""

from __future__ import annotations

from pathlib import Path
import os
from collections.abc import Mapping
import pandas as pd
from dse_research_utils.statistics.evidence import (
    favoured_direction,
)
from language_reading_predictors.statistical_models.findings.common import (
    _KF_MODERATION_LABELS,
    _KeyFindingsUnavailable,
    _kf_csv_row,
    _kf_direction_words,
    _kf_float,
    _kf_headline_from_rope,
    _kf_outcome_label,
    _kf_pct,
    _kf_rope_sentence,
    _kf_sentence,
    _kf_strongest_factor,
)


def _kf_moderation_sentences(output_dir: str | Path) -> list[str]:
    """One sentence per fitted treatment-moderation coefficient (#391 finding 3).

    Reads ``factor_summary.csv`` for the ``gamma_int_trt_*`` rows a moderation
    variant fits (at most two: ability and own baseline). Logit-scale medians with
    the 89% interval — the coefficients sit on the interaction-product scale, so no
    items translation is attempted."""
    path = os.path.join(str(output_dir), "factor_summary.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    needed = {"term", "median", "lo", "hi", "prob_positive"}
    if df.empty or not needed.issubset(df.columns):
        return []
    out: list[str] = []
    for _, row in df.iterrows():
        term = str(row["term"])
        label = _KF_MODERATION_LABELS.get(term)
        if label is None:
            continue
        try:
            med = _kf_float(row["median"])
            lo = _kf_float(row["lo"])
            hi = _kf_float(row["hi"])
            p = _kf_float(row["prob_positive"])
        except _KeyFindingsUnavailable:
            continue
        fav = favoured_direction(p)
        stronger = (
            "a larger on-intervention association"
            if fav["favoured_direction"] == "positive"
            else "a smaller on-intervention association"
        )
        out.append(
            f"Moderation by {label}: {med:+.2f} logits (89% credible range "
            f"{lo:+.2f} to {hi:+.2f}) — a "
            f"{_kf_pct(fav['favoured_direction_prob'])}% probability that children "
            f"higher on it saw {stronger}, read as a model-dependent adjusted "
            f"association."
        )
    return out


def _kf_build_gain_factors(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
    """Gain (ANCOVA) family: the randomised on-intervention term is the only
    causal coefficient, averaged over the period-1 transition; treated-only
    companions have no causal term at all, and a moderation variant (#391
    finding 3) presents every number — its netted treatment marginal included —
    as a model-dependent adjusted association."""
    outcome_label = _kf_outcome_label(config)
    plan = config.get("resolved_run_plan") or {}
    extra = config.get("extra") or {}
    treated_only = bool(plan.get("treated_only", extra.get("treated_only", False)))
    moderation_variant = bool(
        plan.get("moderation_variant", extra.get("moderation_variant", False))
    )
    sentences: list[dict[str, str]] = []
    if moderation_variant:
        sentences.append(
            _kf_sentence(
                f"This companion model asks whether the on-intervention association "
                f"with {outcome_label} varies with children's starting point and "
                f"general cognitive ability. Those moderation terms are estimated "
                f"across all study periods — including after the comparison group "
                f"had crossed over — so every number here is a model-dependent "
                f"adjusted association, not a cause; the randomised headline lives "
                f"in the interaction-free primary model.",
                "causal",
            )
        )
        for text in _kf_moderation_sentences(output_dir):
            sentences.append(_kf_sentence(text, "moderation"))
        tm = _kf_csv_row(output_dir, "treatment_marginal.csv")
        if tm is not None:
            is_rd = bool(
                plan.get("off_floor", extra.get("likelihood") == "bernoulli_offfloor")
            )
            try:
                scale = 100.0 if is_rd else 1.0
                med = _kf_float(tm["trt_items_median"]) * scale
                lo = _kf_float(tm["trt_items_lo"]) * scale
                hi = _kf_float(tm["trt_items_hi"]) * scale
            except _KeyFindingsUnavailable:
                pass
            else:
                # ``or 0.0`` maps a rounded -0.0 back to +0.0 so a hair-negative
                # median never renders as "-0".
                nd = 0 if is_rd else 1
                med, lo, hi = (round(v, nd) or 0.0 for v in (med, lo, hi))
                unit = (
                    f"**{med:+.0f} percentage points** on the chance of being "
                    f"off the floor at the period end (89% credible range "
                    f"{lo:+.0f} to {hi:+.0f})"
                    if is_rd
                    else f"**{med:+.1f} items** (89% credible range {lo:+.1f} "
                    f"to {hi:+.1f})"
                )
                sentences.append(
                    _kf_sentence(
                        f"For context, netting those moderation terms out gives a "
                        f"model-dependent on-intervention contrast of {unit} "
                        f"during the randomised first period.",
                        "headline",
                    )
                )
        highlight = _kf_strongest_factor(output_dir)
        if highlight:
            sentences.append(_kf_sentence(highlight, "highlight"))
        return sentences
    if treated_only:
        sentences.append(
            _kf_sentence(
                f"This companion model looks only at children while they were "
                f"receiving the intervention, so it estimates no treatment effect "
                f"on {outcome_label} — every result in it is an adjusted "
                f"association, not a cause.",
                "causal",
            )
        )
        highlight = _kf_strongest_factor(output_dir)
        if highlight:
            sentences.append(_kf_sentence(highlight, "highlight"))
        return sentences
    rope = _kf_csv_row(output_dir, "rope_summary.csv")
    scope = "during the randomised first period"
    if rope is not None:
        headline, is_rd = _kf_headline_from_rope(rope, outcome_label, scope)
        sentences.append(_kf_sentence(headline, "headline"))
        sentences.append(
            _kf_sentence(
                # The gain-family off-floor outcome is post-period STATUS
                # (post > 0), not an off-floor transition — say so (#391 review).
                _kf_direction_words(
                    rope["pd"],
                    is_rd=is_rd,
                    rd_event="being off the floor at the period end",
                ),
                "confidence",
            )
        )
        sentences.append(_kf_sentence(_kf_rope_sentence(rope, is_rd=is_rd), "rope"))
    else:
        tm = _kf_csv_row(output_dir, "treatment_marginal.csv")
        if tm is None:
            raise _KeyFindingsUnavailable(
                "neither rope_summary.csv nor treatment_marginal.csv is present"
            )
        med = _kf_float(tm["trt_items_median"])
        lo = _kf_float(tm["trt_items_lo"])
        hi = _kf_float(tm["trt_items_hi"])
        sentences.append(
                _kf_sentence(
                    f"Best estimate: the model-estimated on-intervention contrast "
                    f"for {outcome_label} was **{med:+.1f} items** {scope} "
                f"(89% credible range {lo:+.1f} to {hi:+.1f}).",
                "headline",
            )
        )
        sentences.append(
            _kf_sentence(_kf_direction_words(tm["prob_trt_pos"], is_rd=False), "confidence")
        )
    sentences.append(
        _kf_sentence(
            "The on-intervention effect is the only potentially cause-and-effect "
            "estimate in this report because it rests on the randomised first "
            "period. That reading is limited to the fitted available-case rows and "
            "assumes outcome and required-covariate observation do not depend "
            "jointly on treatment and potential outcomes; every other factor is an "
            "adjusted association.",
            "causal",
        )
    )
    highlight = _kf_strongest_factor(output_dir)
    if highlight:
        sentences.append(_kf_sentence(highlight, "highlight"))
    return sentences
