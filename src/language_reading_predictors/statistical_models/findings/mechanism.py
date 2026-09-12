# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the mechanism family."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import numpy as np
from dse_research_utils.statistics.evidence import (
    favoured_direction,
)
from language_reading_predictors.statistical_models.findings.common import (
    _KF_COVARIATE_MODERATOR_LABELS,
    _KeyFindingsUnavailable,
    _kf_association_direction,
    _kf_csv,
    _kf_csv_row,
    _kf_dag_unit,
    _kf_float,
    _kf_measure_label,
    _kf_outcome_label,
    _kf_pct,
    _kf_sentence,
)


def _kf_mechanism_shape_caveat(output_dir: str | Path, config: Mapping) -> dict[str, str] | None:
    """Qualify a nonlinear-shape or threshold claim the fit cannot support (#586).

    Two separate reasons a shape reading may not survive, both of which the key
    findings previously passed over in silence:

    * the located steepest interval fails its qualification checks, so it is a
      description of the fitted curve rather than a threshold (finding 1);
    * power scaling flags a focal GP hyperparameter, so the shape is leaning on the
      regularisation rather than the likelihood (finding 12). Mechanism fits are
      exempt from the treatment-effect robustness gate because they are
      observational — but that is about identification, not robustness.

    Returns ``None`` for a linear fit (no shape is claimed) and for an HSGP fit that
    is both qualified and unflagged.
    """
    plan = config.get("resolved_run_plan") or {}
    if plan.get("linear_mechanism"):
        return None
    reasons: list[str] = []
    readiness = _kf_csv_row(output_dir, "readiness_threshold.csv")
    if readiness is not None:
        if "knee_well_defined" not in readiness:
            reasons.append(
                "the steepest-interval summary predates the curvature and boundary "
                "checks, so it is not evidence of a threshold"
            )
        elif not bool(readiness["knee_well_defined"]):
            if bool(readiness.get("boundary_pinned")):
                reasons.append(
                    "the curve is steepest at the edge of the observed exposure "
                    "range, so no threshold is located within the data"
                )
            else:
                reasons.append(
                    "the fitted curve does not bend clearly enough to locate a "
                    "threshold"
                )
    psense = _kf_csv(output_dir, "psense_summary.csv")
    if psense is not None and "diagnosis" in psense.columns and len(psense.columns):
        names = psense[psense.columns[0]].astype(str)
        # "✓" is this column's *clear* marker, so anything else is the flag.
        flagged = psense[
            names.str.startswith("f_mech__")
            & ~psense["diagnosis"].astype(str).str.strip().isin(["✓", "", "nan"])
        ]
        if len(flagged):
            reasons.append(
                "power scaling flags the curve's own prior "
                f"({', '.join(sorted(set(flagged[psense.columns[0]].astype(str))))}), "
                "so its shape depends on the regularisation as well as the data"
            )
    if not reasons:
        return None
    return _kf_sentence(
        "Read the strength and direction of this association, not its shape: "
        + "; ".join(reasons)
        + ".",
        "note",
    )


def mechanism_headline_estimand(output_dir: str | Path) -> dict | None:
    """The declared headline contrast of a stored mechanism fit, machine-readably.

    Reads the first row of ``mechanism_summary.csv``, which is the headline by
    construction (:func:`mechanism_items.mechanism_summary_table` writes it first).
    Recorded in ``key_findings.json`` so the published headline number carries its
    estimand id, reference population and exposure interval rather than leaving a
    reader to infer which of the family's two contrasts a number came from (#602).
    """
    row = _kf_csv_row(output_dir, "mechanism_summary.csv")
    if row is None or "estimand" not in row:
        return None
    keys = (
        "estimand",
        "contrast",
        "reference_population",
        "child_intercept",
        "exposure_unit",
        "exposure_quantile_low",
        "exposure_quantile_high",
        "exposure_low",
        "exposure_high",
        "items_median",
        "items_lo",
        "items_hi",
        "prob_pos",
    )
    record: dict = {"source": "mechanism_summary.csv"}
    for key in keys:
        if key in row and not (
            isinstance(row[key], float) and not np.isfinite(row[key])
        ):
            value = row[key]
            record[key] = (
                float(value) if isinstance(value, (int, float, np.floating)) else str(value)
            )
    return record


def _kf_mechanism_slope_sentences(output_dir: str | Path) -> list[dict[str, str]]:
    """One sentence per exposure-slope question, for the #603 / #604 sensitivities.

    Reads ``mechanism_slope_summary.csv``, which a pooled fit never writes, so this
    is empty for every registered primary. At most two sentences (one per
    sensitivity), because the key-findings box caps at
    :data:`KEY_FINDINGS_MAX_SENTENCES` and the causal sentence must survive.
    """
    table = _kf_csv(output_dir, "mechanism_slope_summary.csv")
    if table is None or "component" not in table.columns:
        return []
    by = {str(r["component"]): r for _, r in table.iterrows()}
    out: list[dict[str, str]] = []

    if "between" in by and "within" in by:
        b, w = by["between"], by["within"]
        out.append(
            _kf_sentence(
                "Splitting the exposure into its between-child and within-child "
                f"parts: **between** children, {_kf_float(b['median']):+.2f} "
                f"(89% {_kf_float(b['lo']):+.2f} to {_kf_float(b['hi']):+.2f}) — do "
                "children with a generally higher exposure score generally higher? "
                f"**Within** a child, {_kf_float(w['median']):+.2f} "
                f"(89% {_kf_float(w['lo']):+.2f} to {_kf_float(w['hi']):+.2f}) — "
                "when a child's own exposure moves, does their outcome move with "
                "it? Both are per 1 SD of the exposure on the model's scale, both "
                "are adjusted associations, and a single pooled coefficient would "
                "have been a precision-weighted blend of the two.",
                "detail",
            )
        )

    # ``by`` collapses repeated components, so the per-period rows are taken from
    # the table itself; only the singleton components are looked up by key.
    slopes = [
        r for _, r in table.iterrows() if str(r.get("component")) == "phase_slope"
    ]
    scale = by.get("phase_scale")
    if slopes:
        listed = "; ".join(
            f"{str(r['period'])} {_kf_float(r['median']):+.2f} "
            f"({_kf_float(r['lo']):+.2f} to {_kf_float(r['hi']):+.2f})"
            for r in slopes
        )
        spread = (
            f" The between-period spread is {_kf_float(scale['median']):.2f} "
            f"(89% {_kf_float(scale['lo']):.2f} to {_kf_float(scale['hi']):.2f})."
            if scale is not None
            else ""
        )
        out.append(
            _kf_sentence(
                f"Letting the slope vary by period: {listed}.{spread} A difference "
                "between periods is evidence against pooling, not evidence that the "
                "relationship changed over time — only the first transition is "
                "randomised-arm-clean, and the periods also differ in age, "
                "treatment history and measurement position.",
                "detail",
            )
        )
    return out


def _kf_build_mechanism(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
    """Adjusted mechanism association, on the family's declared headline estimand."""
    outcome_label = _kf_outcome_label(config)
    summary = _kf_csv_row(output_dir, "mechanism_summary.csv")
    sentences: list[dict[str, str]] = []
    if summary is not None:
        med = _kf_float(summary["items_median"])
        lo = _kf_float(summary["items_lo"])
        hi = _kf_float(summary["items_hi"])
        low = _kf_float(summary["exposure_low"])
        high = _kf_float(summary["exposure_high"])
        unit = _kf_dag_unit(summary.get("exposure_unit", "predictor units"))
        # The declared headline is the interquartile contrast standardised over the
        # fitted rows (#602). Older fits carry the pre-#602 single-row summary, whose
        # interval was the observed minimum and maximum; they have no quantile
        # columns, so say "fitted exposure range" rather than claim percentiles the
        # number was not computed at.
        q_lo = summary.get("exposure_quantile_low")
        q_hi = summary.get("exposure_quantile_high")
        has_quantiles = (
            q_lo is not None
            and q_hi is not None
            and isinstance(q_lo, (int, float))
            and isinstance(q_hi, (int, float))
            and np.isfinite(q_lo)
            and np.isfinite(q_hi)
        )
        interval = (
            f"between the {int(round(100 * _kf_float(q_hi)))}th and "
            f"{int(round(100 * _kf_float(q_lo)))}th percentile of the fitted exposure "
            f"({high:g} against {low:g} {unit})"
            if has_quantiles
            else f"across the fitted exposure range ({low:g} to {high:g} {unit})"
        )
        sentences.append(
            _kf_sentence(
                f"Comparing children {interval}, {outcome_label} differed by "
                f"**{med:+.1f} items** (89% credible range {lo:+.1f} to {hi:+.1f}), "
                "averaged over the children analysed with every other term — "
                "period, covariates, baseline and each child's own fitted "
                "intercept — held at its fitted value.",
                "headline",
            )
        )
        sentences.append(
            _kf_sentence(
                _kf_association_direction(
                    summary["prob_pos"],
                    positive_claim="higher exposure accompanies a higher outcome",
                    negative_claim="higher exposure accompanies a lower outcome",
                ),
                "confidence",
            )
        )
        sentences.extend(_kf_mechanism_slope_sentences(output_dir))
    else:
        curve = _kf_csv(output_dir, "mechanism_curve.csv")
        if curve is None:
            raise _KeyFindingsUnavailable(
                "neither mechanism_summary.csv nor mechanism_curve.csv is present"
            )
        x_col = "mech_x" if "mech_x" in curve.columns else "mech_logit"
        required = {x_col, "f_mean", "f_lo", "f_hi"}
        if not required.issubset(curve.columns):
            raise _KeyFindingsUnavailable(
                "mechanism_curve.csv does not have the expected columns"
            )
        ordered = curve.sort_values(x_col)
        low_row, high_row = ordered.iloc[0], ordered.iloc[-1]
        sentences.append(
            _kf_sentence(
                f"Across the fitted predictor range, its model contribution changed "
                f"from {_kf_float(low_row['f_mean']):+.2f} logit units "
                f"(89% range {_kf_float(low_row['f_lo']):+.2f} to "
                f"{_kf_float(low_row['f_hi']):+.2f}) to "
                f"{_kf_float(high_row['f_mean']):+.2f} "
                f"({_kf_float(high_row['f_lo']):+.2f} to "
                f"{_kf_float(high_row['f_hi']):+.2f}).",
                "headline",
            )
        )
        sentences.append(
            _kf_sentence(
                "This older fit has pointwise curve intervals but no saved "
                "posterior end-to-end contrast, so a single direction probability "
                "is not available until it is refitted.",
                "note",
            )
        )
    sentences.append(
        _kf_sentence(
            "The curve is an adjusted association between measured skills measured at "
            "the same wave, not evidence that changing one skill would cause the other "
            "to change; the child random intercept is not a control for general "
            "ability.",
            "causal",
        )
    )
    shape_caveat = _kf_mechanism_shape_caveat(output_dir, config)
    if shape_caveat is not None:
        sentences.append(shape_caveat)
    # Moderated (joint-readiness) mechanism models ask about gamma_int, not the
    # unmoderated curve, so headline it when present (#404 review): its median,
    # 50%/89% intervals and tail probability, on the logit scale. The unmoderated
    # curve sentences above then read as supporting context.
    interaction = _kf_csv_row(output_dir, "interaction_summary.csv")
    if interaction is not None and "gamma_int_median" in interaction:
        med = _kf_float(interaction["gamma_int_median"])
        lo = _kf_float(interaction["gamma_int_lo"])
        hi = _kf_float(interaction["gamma_int_hi"])
        lo50 = _kf_float(interaction["gamma_int_lo50"])
        hi50 = _kf_float(interaction["gamma_int_hi50"])
        p = _kf_float(interaction["prob_gamma_int_pos"])
        exposure_label = _kf_lower_first(
            _kf_measure_label(config.get("mechanism_symbol") or "the exposure")
        )
        outcome_mid = _kf_lower_first(outcome_label)
        moderator_label = _kf_moderator_label(config)
        focal = _kf_sentence(
            f"The moderation coefficient — how the slope of {outcome_mid} on "
            f"{exposure_label} changes per +1 SD of {moderator_label}, on the latent "
            f"logit scale — is **{med:+.2f}** (50% interval {lo50:+.2f} to "
            f"{hi50:+.2f}; 89% {lo:+.2f} to {hi:+.2f}), with P(> 0) = {p:.2f}.",
            "headline",
        )
        # The claim is about the logit scale only. On a bounded outcome the sign
        # of a product term is not a statement about items — below the midpoint
        # of the scale two positive effects that are additive in items show a
        # negative logit product — so the items-scale reading is a separate
        # sentence from moderation_items.csv, never implied here (2026-08-19).
        direction = _kf_sentence(
            _kf_association_direction(
                interaction["prob_gamma_int_pos"],
                positive_claim=(
                    f"the {exposure_label} slope tends to be steeper where "
                    f"{moderator_label} is higher (synergy on the logit scale)"
                ),
                negative_claim=(
                    f"the {exposure_label} slope tends to be shallower where "
                    f"{moderator_label} is higher (substitution on the logit scale)"
                ),
            ),
            "confidence",
        )
        items_sentence = _kf_moderation_items_sentence(
            output_dir, config, prob_gamma_int_pos=p
        )
        if items_sentence is None:
            sentences = [focal, direction, *sentences]
        else:
            # The items-scale sentence needs a slot under the cap. The unmoderated
            # curve is supporting context on a moderated fit, so its two
            # sentences fold into one droppable context sentence rather than
            # the causal sentence falling off the end (#464).
            causal = [s_ for s_ in sentences if s_.get("kind") == "causal"]
            context = _kf_mechanism_curve_context(summary, outcome_label)
            # The shape caveat qualifies exactly the curve this context sentence
            # reports, so it is folded into it rather than added as a sixth sentence
            # that truncation would silently drop (#464 / #586).
            if context is not None and shape_caveat is not None:
                context = _kf_sentence(
                    context["text"].rstrip(".") + ". " + shape_caveat["text"], "note"
                )
            elif context is None and shape_caveat is not None:
                context = shape_caveat
            sentences = [
                focal,
                direction,
                items_sentence,
                *([context] if context is not None else []),
                *causal,
            ]
    return sentences


def _kf_lower_first(text: str) -> str:
    """Lower-case the first character for mid-sentence use of a display label."""
    return text[:1].lower() + text[1:] if text else text


def _kf_moderator_label(config: Mapping) -> str:
    """Display label for a moderated mechanism fit's moderator, mid-sentence."""
    extra = config.get("extra") or {}
    symbol = (
        config.get("moderator_symbol")
        or extra.get("moderator_symbol")
        or (config.get("resolved_run_plan") or {}).get("moderator_symbol")
    )
    if not symbol:
        return "the moderator"
    symbol = str(symbol)
    if symbol in _KF_COVARIATE_MODERATOR_LABELS:
        return _KF_COVARIATE_MODERATOR_LABELS[symbol]
    return _kf_lower_first(_kf_measure_label(symbol))


def _kf_mechanism_curve_context(
    summary: Mapping | None, outcome_label: str
) -> dict[str, str] | None:
    """The unmoderated curve's end-to-end contrast as one context sentence.

    Used on moderated fits once the items-scale moderation sentence is present:
    the two curve sentences (size, direction) fold into one so the box stays
    under the cap with the causal sentence intact. Marked ``note`` — the one
    droppable role here — so a release note can still displace it rather than
    the interaction or causal sentences.
    """
    if summary is None:
        return None
    med = _kf_float(summary["items_median"])
    lo = _kf_float(summary["items_lo"])
    hi = _kf_float(summary["items_hi"])
    low = _kf_float(summary["exposure_low"])
    high = _kf_float(summary["exposure_high"])
    unit = _kf_dag_unit(summary.get("exposure_unit", "predictor units"))
    fav = favoured_direction(_kf_float(summary["prob_pos"]))
    return _kf_sentence(
        f"For context, across the fitted exposure range ({low:g} to {high:g} "
        f"{unit}) {_kf_lower_first(outcome_label)} differed by **{med:+.1f} "
        f"items** on average (89% credible range {lo:+.1f} to {hi:+.1f}; "
        f"P({fav['favoured_direction']}) = {_kf_pct(fav['favoured_direction_prob'])}%) "
        "— the unmoderated curve.",
        "note",
    )


def _kf_moderation_items_sentence(
    output_dir: str | Path, config: Mapping, *, prob_gamma_int_pos: float
) -> dict[str, str] | None:
    """The moderated fit's interaction re-expressed in outcome items (2026-08-19).

    Reads ``moderation_items.csv`` (``pipelines.mechanism.write_moderation_items``):
    the interquartile exposure increment in items at the low and at the high
    moderator cell, their difference — the items-scale interaction — and the
    same difference under logit-additivity (``gamma_int = 0``), the bounded-scale
    benchmark. The verdict clause compares the items-scale direction with the
    logit-scale one on the house evidence ladder: at least moderate evidence in
    the same direction means the logit-scale pattern is not an artefact of the
    bounded scale; at least moderate evidence the other way means it is; anything
    weaker says the items-scale direction is not settled. Returns ``None`` when
    the table is absent, so older fits keep their previous box.
    """
    table = _kf_csv(output_dir, "moderation_items.csv")
    if table is None or "quantity" not in table.columns:
        return None
    by = {str(r["quantity"]): r for _, r in table.iterrows()}
    needed = (
        "increment_at_moderator_low",
        "increment_at_moderator_high",
        "interaction",
        "interaction_if_logit_additive",
    )
    if any(q not in by for q in needed):
        return None
    inter = by["interaction"]
    inc_lo = _kf_float(by["increment_at_moderator_low"]["median"])
    inc_hi = _kf_float(by["increment_at_moderator_high"]["median"])
    dd = _kf_float(inter["median"])
    lo = _kf_float(inter["lo"])
    hi = _kf_float(inter["hi"])
    bench = _kf_float(by["interaction_if_logit_additive"]["median"])
    x_lo = _kf_float(inter["exposure_low"])
    x_hi = _kf_float(inter["exposure_high"])
    m_lo = _kf_float(inter["moderator_low"])
    m_hi = _kf_float(inter["moderator_high"])
    exposure_unit = _kf_dag_unit(inter.get("exposure_unit", "items"))
    moderator_unit = _kf_dag_unit(inter.get("moderator_unit", ""))
    moderator_label = _kf_moderator_label(config)
    exposure_label = _kf_lower_first(
        _kf_measure_label(config.get("mechanism_symbol") or "the exposure")
    )
    outcome_label = _kf_lower_first(_kf_outcome_label(config))
    fav_items = favoured_direction(_kf_float(inter["prob_pos"]))
    fav_logit = favoured_direction(_kf_float(prob_gamma_int_pos))
    items_dir = fav_items["favoured_direction"]
    label = fav_items["favoured_direction_label"]
    settled = label in ("moderate", "strong", "very strong")
    # The items-scale result can only *corroborate or overturn the interpretation of*
    # a fitted logit interaction — it cannot supply one. Checking the items evidence
    # first let a settled items direction confirm a logit interaction whose own sign
    # was undecided ("strong evidence that the synergy holds ... not an artefact" off
    # P(gamma_int > 0) = 0.55), so the fitted coefficient is now the gate (#586
    # finding 7). Latent when found — no stored fit paired an inconclusive gamma_int
    # with settled items evidence — but the ordering was wrong either way.
    logit_settled = fav_logit["favoured_direction_label"] != "inconclusive"
    pattern = "synergy" if fav_logit["favoured_direction"] == "positive" else "substitution"
    if not logit_settled:
        verdict = (
            "the fitted logit-scale interaction is itself directionally inconclusive, "
            f"so neither scale supports a {pattern} reading — the items figures "
            "describe the fitted surface, they do not settle the interaction"
        )
    elif settled and items_dir == fav_logit["favoured_direction"]:
        verdict = (
            f"{label} evidence that the {pattern} holds in items too, so it is not "
            "an artefact of the bounded scale"
        )
    elif settled:
        verdict = (
            f"{label} evidence that the pattern reverses in items, so the "
            f"logit-scale {pattern} is the bounded scale at work"
        )
    else:
        verdict = (
            f"on the items scale the direction is {label}, so the logit-scale "
            f"{pattern} should not be read as a finding about items"
        )
    return _kf_sentence(
        f"In {outcome_label} items, the interquartile {exposure_label} increment "
        f"({x_lo:g} to {x_hi:g} {exposure_unit}) is worth **{inc_lo:+.1f} items** "
        f"when {moderator_label} is {m_lo:g} {moderator_unit} and {inc_hi:+.1f} when "
        f"it is {m_hi:g}: a difference of {dd:+.1f} items (89% {lo:+.1f} to "
        f"{hi:+.1f}; P({items_dir}) = {_kf_pct(fav_items['favoured_direction_prob'])}%), "
        f"where additivity on the logit scale would have shown {bench:+.1f} — "
        f"{verdict}.",
        "scale",
    )
