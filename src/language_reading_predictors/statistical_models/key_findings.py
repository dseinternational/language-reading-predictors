# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The plain-language key-findings box, per family (#320).

One builder per ``ModelSpec.kind``, each reading that family's own stored CSVs
rather than a posterior, so ``key_findings.json`` can be regenerated without a
refit. Gate-interlocked: a fit whose release decision withholds publishes no
sentences. The convergence-gate readers live here because the box is the first
consumer of their verdict.
"""


from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from dse_research_utils.statistics.evidence import (
    favoured_direction,
)


KEY_FINDINGS_FILENAME = "key_findings.json"


KEY_FINDINGS_SCHEMA_VERSION = 1


KEY_FINDINGS_MAX_SENTENCES = 5


# Plain-language labels for the factor-model coefficients (the family-highlight
# sentence). Terms not listed here are skipped rather than surfaced raw — a
# key-findings box must never ask the reader to decode a coefficient name.
_KF_FACTOR_LABELS: dict[str, str] = {
    "gamma_own": "the child's own starting point on this measure",
    "gamma_own_offfloor": "starting the period already off the floor on this measure",
    "gamma_A": "the child's age",
    "gamma_ability": "general cognitive ability (block design)",
    "gamma_R": "receptive vocabulary at the start of the period",
    "gamma_E": "expressive vocabulary at the start of the period",
    "gamma_TR": "taught receptive vocabulary at the start of the period",
    "gamma_TE": "taught expressive vocabulary at the start of the period",
    "gamma_L": "letter-sound knowledge at the start of the period",
    "gamma_W": "word reading at the start of the period",
    "gamma_N": "nonword reading at the start of the period",
    "gamma_B": "sound blending at the start of the period",
    "gamma_hs": "hearing",
    "gamma_deapp_c": "speech accuracy",
    "gamma_erbto": "phonological memory (nonword repetition)",
}


class _KeyFindingsUnavailable(Exception):
    """Raised by a builder when the CSVs it needs are missing or unusable."""


def _kf_float(value) -> float:
    """Return ``value`` as a finite float, else raise (the no-``nan`` guard)."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise _KeyFindingsUnavailable(f"non-numeric value {value!r}") from exc
    if not np.isfinite(v):
        raise _KeyFindingsUnavailable(f"non-finite value {value!r}")
    return v


def _kf_pct(prob) -> str:
    """A probability as a plain percentage string, never rounding to a false
    certainty (``0.998`` renders as ``99.8``, not ``100``)."""
    p = _kf_float(prob)
    if not 0.0 <= p <= 1.0:
        raise _KeyFindingsUnavailable(f"probability out of range: {p!r}")
    v = 100.0 * p
    # Never display a false certainty: an empirical posterior probability of 1
    # (or 0) just means every retained draw agreed, so cap the display at 99.9
    # (or floor it at 0.1) rather than claiming 100% / 0%.
    if round(v) >= 100:
        return f"{min(v, 99.9):.1f}"
    if round(v) <= 0:
        return f"{max(v, 0.1):.1f}"
    return f"{v:.0f}"


def _kf_sentence(text: str, kind: str) -> dict[str, str]:
    return {"text": text, "kind": kind}


def _kf_csv_row(output_dir, name: str) -> dict | None:
    """First row of ``{output_dir}/{name}`` as a plain dict, or None if absent."""
    path = os.path.join(str(output_dir), name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df.iloc[0].to_dict()


# Values in psense_summary.csv's ``diagnosis`` column that mean "not flagged".
# "✓" is what arviz_stats actually writes for a clear parameter; the wording
# variants match sensitivity.tau_psense_status, which is the established
# convention for reading this column. The rest are defensive placeholders.
_PSENSE_CLEAR_MARKERS = frozenset(
    {
        "✓",
        "-",
        "nan",
        "none",
        "ok",
        "no concern",
        "no conflict",
        "no prior-data conflict",
    }
)


def _kf_psense_diagnosis(
    output_dir, term: str, *, filename: str = "psense_summary.csv"
) -> str | None:
    """Power-scaling diagnosis for ``term`` from ``psense_summary.csv`` (#389 finding 3).

    Returns the ``diagnosis`` string (e.g. "potential prior-data conflict") when the
    parameter is flagged, or ``None`` when the file or row is absent or the parameter
    is clear — so a caller can surface a warning beside the headline without breaking
    fits that never ran power-scaling.

    ``arviz_stats`` writes a **tick** for an unflagged parameter, not a blank, and that
    is the single most common value in the stored suite (1117 of 2648 rows). Treating
    it as a diagnosis would publish a "prior-sensitive" caution on a *clean* estimate —
    so the clear markers are matched explicitly. Anything unrecognised is deliberately
    treated as a flag: an unknown marker should over-warn, not go silent.

    ``filename`` names a per-fit power-scaling table where a family publishes several
    posteriors from one fit — the joint-mechanism levels design writes one per wave —
    so each published result can be read against its own diagnosis rather than the
    artefact-hosting fit's (2026-08-23 joint-mechanism follow-up review, finding 6)."""
    path = os.path.join(str(output_dir), filename)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception:
        return None
    if "diagnosis" not in df.columns or term not in df.index:
        return None
    diag = str(df.loc[term, "diagnosis"]).strip()
    if not diag or diag.lower() in _PSENSE_CLEAR_MARKERS:
        return None
    return diag


def _kf_csv(output_dir, name: str) -> pd.DataFrame | None:
    """Read one fit CSV, returning ``None`` when it is absent or empty."""
    path = os.path.join(str(output_dir), name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return None if df.empty else df


def _kf_most_resolved_row(
    df: pd.DataFrame,
    *,
    prob_col: str,
    resolution_decimals: int | None = None,
    tie_breakers: Sequence[tuple[str, bool]] = (),
) -> dict:
    """Return the row whose direction is clearest, never the largest estimate.

    The ranking is distance of ``P(positive)`` from 0.5.  This avoids presenting
    differently-scaled coefficients as though their raw magnitudes were
    comparable, and it keeps the selection rule tied to uncertainty.

    ``resolution_decimals`` and ``tie_breakers`` are opt-in (the default keeps
    every existing builder's behaviour): a builder whose rows can all sit at the
    resolution ceiling passes the number of decimals at which two probabilities
    count as tied — chosen well above the Monte-Carlo noise in ``P`` (the
    concurrent family uses 2, i.e. ties to the nearest 1 %) — and a sequence of
    ``(column, ascending)`` secondary keys that decide among tied rows on a
    stated, data-meaningful basis (the concurrent family's primary wave first,
    then the larger items-scale contrast — 2026-08-22 adjusted-family review,
    extension follow-up: ``rlm-ca-001``'s headline wave had flipped t1 → t2
    between two refits on a 1e-4 difference in ``P``).
    """
    if prob_col not in df.columns:
        raise _KeyFindingsUnavailable(f"{prob_col} is missing")
    probabilities = pd.to_numeric(df[prob_col], errors="coerce")
    usable = df[np.isfinite(probabilities)].copy()
    if usable.empty:
        raise _KeyFindingsUnavailable(f"{prob_col} has no finite values")
    resolution = (pd.to_numeric(usable[prob_col], errors="coerce") - 0.5).abs()
    if resolution_decimals is not None:
        resolution = resolution.round(int(resolution_decimals))
    usable["_kf_resolution"] = resolution
    for column, _ascending in tie_breakers:
        if column not in usable.columns:
            raise _KeyFindingsUnavailable(f"tie-break column {column} is missing")
    # ``kind="stable"``: the metric saturates at 0.5 once several rows reach
    # P(positive) = 1, and pandas' default quicksort leaves the winner among
    # tied rows dependent on pandas' internals rather than on the data — so the
    # published headline could change without the fit changing (2026-08-21
    # historical-families review, finding 3, where one fit had eight tied rows).
    # A stable sort makes the choice "the first tied row in the artefact's own
    # order" (after any declared tie-breakers), which is reproducible and
    # inspectable.
    by = ["_kf_resolution", *(column for column, _ in tie_breakers)]
    ascending = [False, *(bool(flag) for _, flag in tie_breakers)]
    return (
        usable.sort_values(by, ascending=ascending, kind="stable")
        .iloc[0]
        .to_dict()
    )


def _kf_plain_label(value) -> str:
    """Make an artefact identifier readable without inventing a construct name."""
    return str(value).replace("_", " ").strip()


def _kf_dag_unit(value) -> str:
    """Readable exposure unit with a leading measure symbol mapped to its DAG
    symbol (#374): e.g. ``'L items'`` -> ``'LS items'``. Leaves units that do not
    begin with a mapped modelling symbol unchanged."""
    from language_reading_predictors.statistical_models.measures import DAG_SYMBOL

    text = _kf_plain_label(value)
    head, sep, tail = text.partition(" ")
    return f"{DAG_SYMBOL.get(head, head)}{sep}{tail}"


def _kf_measure_label(symbol) -> str:
    """Display label for a registered measure symbol, a documented raw-score
    covariate (the pooled-levels covariate exposures, #553), else the symbol."""
    from language_reading_predictors.statistical_models.measures import MEASURES
    from language_reading_predictors.statistical_models.pooled_levels import (
        COVARIATE_EXPOSURE_LABELS,
    )

    measure = MEASURES.get(str(symbol))
    if measure is not None:
        return measure.label
    return COVARIATE_EXPOSURE_LABELS.get(str(symbol), _kf_plain_label(symbol))


def _kf_association_direction(
    prob_pos,
    *,
    positive_claim: str,
    negative_claim: str,
) -> str:
    """Harm-aware direction/strength sentence for a non-causal quantity."""
    p = _kf_float(prob_pos)
    fav = favoured_direction(p)
    positive = fav["favoured_direction"] == "positive"
    sign = "positive" if positive else "negative"
    claim = positive_claim if positive else negative_claim
    return (
        f"The posterior probability of a {sign} association is "
        f"{_kf_pct(fav['favoured_direction_prob'])}% — "
        f"{fav['favoured_direction_label']} evidence that {claim}."
    )


def _kf_outcome_label(config: Mapping) -> str:
    """Outcome display label, mirroring the ``_setup.qmd`` derivation.

    The RLI ``MEASURES`` map first; for any other study the registered dataset
    catalogue (``datasets.resolve_dataset``), exactly as ``_setup.qmd`` does.
    Without that second step every Byrne (RLM) key-findings headline fell through
    to the model *title* and read "… items of difference in Byrne wave-1 predictors
    of receptive-vocabulary gain, waves 1-3 (confirmed-input, mutually adjusted)"
    where it should have named BPVS receptive vocabulary (2026-08-22 adjusted-family
    review, finding 2).
    """
    from language_reading_predictors.statistical_models.measures import MEASURES

    symbol = config.get("outcome_symbol")
    measure = MEASURES.get(symbol) if symbol else None
    study_id = config.get("study_id")
    if measure is None and symbol and study_id and study_id != "rli":
        try:
            from language_reading_predictors.statistical_models.datasets import (
                resolve_dataset,
            )

            _dataset, study_measures = resolve_dataset(study_id)
            measure = study_measures.get(symbol)
        except (KeyError, TypeError):
            measure = None
    if measure is not None:
        return measure.label
    return config.get("title") or symbol or "the outcome"


def _kf_direction_words(
    prob_pos, *, is_rd: bool, rd_event: str = "coming off the floor"
) -> str:
    """The harm-aware confidence sentence body (#179): evidence for the
    *favoured* direction, so a clearly negative effect reads as evidence of harm
    rather than 'inconclusive'.

    ``rd_event`` names the risk-difference event so each family states its own
    estimand: the default suits the ITT floored primaries (a genuine off-floor
    *transition* among children observed at the baseline floor), while the
    gain-family off-floor models pass "being off the floor at the period end" —
    their Bernoulli outcome is post-period *status* (``post > 0``), pooling
    moving off, staying above and returning to the floor (#391 review). The
    level- and DiD-family off-floor models pass "being off the floor at t2"
    for the same reason: they model per-wave off-floor *prevalence*, with the
    randomised contrast read at the t2 wave (#490 review follow-up)."""
    p = _kf_float(prob_pos)
    fav = favoured_direction(p)
    label = fav["favoured_direction_label"]
    if fav["favoured_direction"] == "positive":
        sign_word = "positive"
        claim = (
            f"the intervention raises the chance of {rd_event}"
            if is_rd
            else "the intervention helps"
        )
    else:
        sign_word = "negative"
        claim = (
            f"the intervention lowers the chance of {rd_event}"
            if is_rd
            else "the intervention is harmful"
        )
    # State the probability for the FAVOURED direction so the number and the
    # evidence label qualify the same claim (harm-aware, #179): a clearly
    # negative effect reads "97% probability ... negative — strong evidence of
    # harm", not "3% probability ... positive — strong evidence of harm".
    return (
        f"There is a {_kf_pct(fav['favoured_direction_prob'])}% probability "
        f"that the true effect is {sign_word} — {label} evidence that {claim}."
    )


def _kf_headline_from_rope(rope: Mapping, outcome_label: str, scope: str) -> tuple[str, bool]:
    """Headline sentence from a ``rope_summary.csv`` row.

    Returns ``(sentence, is_risk_difference)``. ``scope`` is a clause naming the
    comparison (e.g. 'over the trial period'), so each family can state exactly
    which contrast the number is."""
    is_rd = str(rope.get("delta_scale", "")) == "risk_difference"
    scale = 100.0 if is_rd else 1.0
    med = _kf_float(rope["items_median"]) * scale
    lo = _kf_float(rope["items_lo"]) * scale
    hi = _kf_float(rope["items_hi"]) * scale
    if is_rd:
        text = (
            f"Best estimate: the model-estimated intervention-minus-comparison "
            f"contrast in the chance of scoring above zero on {outcome_label} was "
            f"**{med:+.0f} percentage points** {scope} "
            f"(89% credible range {lo:+.0f} to {hi:+.0f})."
        )
    else:
        text = (
            f"Best estimate: the model-estimated intervention-minus-comparison "
            f"contrast for {outcome_label} was **{med:+.1f} items** {scope} "
            f"(89% credible range {lo:+.1f} to {hi:+.1f})."
        )
    return text, is_rd


def _kf_itt_analysis_population(output_dir) -> dict[str, int]:
    """Validate and summarise the two-arm available-case audit for an ITT fit.

    The causal sentence is not allowed to infer its population from a model title or
    a generic config label.  ``analysis_set.csv`` is generated from the actual fitted
    rows and carries the published randomised allocation, archived cohort and fitted
    arm counts.  Missing or incoherent arithmetic withholds the key findings rather
    than publishing an unqualified randomisation claim.
    """

    path = os.path.join(str(output_dir), "analysis_set.csv")
    if not os.path.exists(path):
        raise _KeyFindingsUnavailable(
            "analysis_set.csv is missing, so the fitted causal population cannot be verified"
        )
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
        raise _KeyFindingsUnavailable(
            "analysis_set.csv does not contain exactly the two required arm rows"
        )
    # ``available_t1_n`` is the deprecated duplicate of ``analysed_archive_n``
    # (2026-08-22 ITT audit, finding 9): it never held outcome-specific t1
    # availability — which is measure-specific, 50 for N against 53 for W — and
    # fits from this commit on stop writing it. Stored bundles still carry it and
    # are still checked for the equality that made it redundant.
    optional = {"available_t1_n"} & set(frame.columns)
    numeric = frame[list((required | optional) - {"arm"})].apply(
        pd.to_numeric, errors="coerce"
    )
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
        (work["randomised_n"] - work["analysed_archive_n"])
        .eq(work["lost_to_follow_up_n"])
        .all()
        and (
            not optional
            or work["analysed_archive_n"].eq(work["available_t1_n"]).all()
        )
        and work["lost_to_follow_up_n"].eq(work["absent_from_archive_n"]).all()
        and (work["randomised_n"] - work["fitted_n"])
        .eq(work["not_in_fitted_analysis_n"])
        .all()
        and (work["analysed_archive_n"] - work["fitted_n"])
        .eq(work["excluded_after_archive_n"])
        .all()
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
        "discontinued_but_followed": int(
            work["discontinued_but_followed_n"].sum()
        ),
        "fitted": int(work["fitted_n"].sum()),
        "fitted_intervention": int(indexed.loc[1, "fitted_n"]),
        "fitted_control": int(indexed.loc[0, "fitted_n"]),
    }


def _kf_itt_causal_sentence(
    population: Mapping[str, int], *, floor_rule: bool = False
) -> str:
    """Selected-population causal wording shared by every single-outcome ITT.

    ``floor_rule`` names the extra qualification the P/N off-floor primaries carry
    (#392): they are a *post-hoc*, data-adaptive contrast within the subgroup observed
    at the floor at baseline, so the population is narrower than the trial's and the
    subgroup was chosen after seeing the data. The review found these emitting the
    ordinary ITT sentence, which understates both.
    """

    label = (
        "This is a post-hoc subgroup available-case modified ITT estimate, not a "
        "full-randomised-cohort ITT estimate. "
        if floor_rule
        else (
            "This is an available-case modified ITT estimate, not a "
            "full-randomised-cohort ITT estimate. "
        )
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
            "Random assignment supports a cause-and-effect reading only under the "
            "available-case assumption: for the "
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
        label
        + scope
        + f"{population['fitted']} fitted children "
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
    output_dir,
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
        raise _KeyFindingsUnavailable(
            "current B model is not one of the validated paired-link fits"
        )
    current = summary.loc[current_model_id]

    def _effect(row: Mapping) -> str:
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


def _kf_rope_sentence(rope: Mapping, *, is_rd: bool) -> str:
    """The magnitude (ROPE) verdict from a ``rope_summary.csv`` row."""
    delta = _kf_float(rope["delta_items"]) * (100.0 if is_rd else 1.0)
    if is_rd:
        unit = "percentage point" if delta == 1 else "percentage points"
    else:
        unit = "item" if delta == 1 else "items"
    p_benefit = _kf_pct(rope["prob_benefit_ge_delta"])
    p_rope = _kf_pct(rope["prob_in_rope"])
    return (
        f"The project agreed after its initial results review that a change of at "
        f"least {delta:g} {unit} would be the smallest difference that matters in "
        f"practice. The probability the benefit reaches that size is {p_benefit}%, "
        f"and the probability the effect is too small to matter either way is "
        f"{p_rope}%; because the threshold is post-hoc, read this beside the "
        f"threshold-sensitivity analysis."
    )


def _kf_itt_missingness_sentence(output_dir, config: Mapping) -> str | None:
    """The mandatory full-57 word-reading sensitivity, when registered."""

    if str(config.get("model_id")) != "lrp-rli-itt-010":
        return None
    frame = _kf_csv(output_dir, "itt_missingness_sensitivity.csv")
    if frame is None or "scenario" not in frame.columns:
        raise _KeyFindingsUnavailable(
            "itt_missingness_sensitivity.csv is absent or malformed"
        )
    indexed = frame.set_index(frame["scenario"].astype(str), drop=False)
    required = (
        "screening_model_observed_profiles",
        "mar_all_57",
        "jump_to_reference_intervention_nonstarter",
    )
    if any(scenario not in indexed.index for scenario in required):
        raise _KeyFindingsUnavailable(
            "the bridge, MAR or jump-to-reference missingness row is absent"
        )
    bridge = indexed.loc[required[0]]
    mar = indexed.loc[required[1]]
    j2r = indexed.loc[required[2]]
    grid = frame.loc[frame["scenario_class"].astype(str) == "arm_specific_delta_grid"]
    if len(grid) != 25:
        raise _KeyFindingsUnavailable("the 25-row missingness delta grid is incomplete")
    grid_medians = pd.to_numeric(grid["effect_items_median"], errors="coerce")
    if not np.isfinite(grid_medians.to_numpy(dtype=float)).all():
        raise _KeyFindingsUnavailable("the missingness delta grid is non-numeric")
    delta_i = pd.to_numeric(grid["delta_intervention_items"], errors="coerce")
    delta_c = pd.to_numeric(grid["delta_control_items"], errors="coerce")
    factual_mar = grid.loc[delta_i.eq(0.0) & delta_c.eq(0.0)]
    if len(factual_mar) != 1:
        raise _KeyFindingsUnavailable(
            "the factual-arm zero-delta MAR completion is absent"
        )
    factual_mar = factual_mar.iloc[0]
    clipping = grid[
        ["clipped_intervention_fraction", "clipped_control_fraction"]
    ].apply(pd.to_numeric, errors="coerce")
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

    def _effect(row: Mapping) -> str:
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


def _kf_itt_attrition_bounds_clause(output_dir, config: Mapping) -> str | None:
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
    except (TypeError, ValueError):
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


def _kf_has_factor_term(output_dir, term: str) -> bool:
    """Whether ``factor_summary.csv`` carries a row for ``term``.

    Used to gate prose on a coefficient the fit actually contains, so a caveat
    about (say) group×ability is not published for a model fitted without it.
    Absent or malformed file → ``False``: a missing caveat is better than one
    that describes a term the reader cannot find in the table."""
    path = os.path.join(str(output_dir), "factor_summary.csv")
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    return "term" in df.columns and bool((df["term"].astype(str) == term).any())


def _kf_strongest_factor(output_dir, *, exclude_roles: tuple[str, ...] = ("causal",)) -> str | None:
    """Family-highlight sentence: the most clearly resolved adjusted association
    in ``factor_summary.csv``, or None when nothing usable is present.

    Ranked by ``|prob_positive - 0.5|`` (how clearly the direction is resolved),
    NOT by ``|median|`` — the factor coefficients sit on different scales (the
    own baseline enters on the raw logit scale, other covariates per SD), so
    magnitudes are not comparable across terms. Interaction terms and
    unlabelled coefficients are skipped — the box must stay readable without a
    code key."""
    path = os.path.join(str(output_dir), "factor_summary.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    needed = {"term", "role", "prob_positive"}
    if df.empty or not needed.issubset(df.columns):
        return None
    rows = df[~df["role"].isin(exclude_roles) & df["term"].isin(_KF_FACTOR_LABELS)]
    probs = pd.to_numeric(rows["prob_positive"], errors="coerce")
    rows = rows[np.isfinite(probs)]
    if rows.empty:
        return None
    top = rows.loc[(pd.to_numeric(rows["prob_positive"]) - 0.5).abs().idxmax()]
    label = _KF_FACTOR_LABELS[str(top["term"])]
    p = float(top["prob_positive"])
    fav = favoured_direction(p)
    ends = (
        "also tended to score higher afterwards"
        if fav["favoured_direction"] == "positive"
        else "tended to score lower afterwards"
    )
    return (
        f"Of the other factors in the model, {label} had the most clearly "
        f"resolved link with the outcome: children higher on it {ends} "
        f"(a {_kf_pct(fav['favoured_direction_prob'])}% probability for that "
        f"direction; an adjusted association, not a cause)."
    )


def _kf_build_itt(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Available-case modified ITT suite: rope card first, then tau fallback."""
    outcome_label = _kf_outcome_label(config)
    population = _kf_itt_analysis_population(output_dir)
    blending_evidence = _kf_blending_link_evidence(output_dir, config)
    score_mean_link = str(
        (config.get("resolved_run_plan") or {}).get("score_mean_link", "logit")
    )
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
        direction_sentence = _kf_direction_words(
            blending_row["prob_effect_positive"], is_rd=False
        )
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
                raise _KeyFindingsUnavailable(
                    "neither rope_summary.csv nor tau_summary.csv is present"
                )
            from language_reading_predictors.statistical_models.measures import MEASURES

            measure = MEASURES.get(config.get("outcome_symbol"))
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
        floor_rule=bool(
            (config.get("resolved_run_plan") or {}).get("floor_rule", False)
        ),
    )
    attrition_clause = _kf_itt_attrition_bounds_clause(output_dir, config)
    if attrition_clause is not None:
        causal_sentence += attrition_clause
    sentences.append(_kf_sentence(causal_sentence, "causal"))
    return sentences


#: Human labels for the moderation-variant treatment-interaction coefficients.
_KF_MODERATION_LABELS: dict[str, str] = {
    "gamma_int_trt_ability": "general cognitive ability (block design)",
    "gamma_int_trt_own": "the child's starting point on this measure",
}


def _kf_moderation_sentences(output_dir) -> list[str]:
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


def _kf_build_gain_factors(output_dir, config: Mapping) -> list[dict[str, str]]:
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


def _kf_level_blending_link_sentence(output_dir, config: Mapping) -> str | None:
    """The level family's paired-link sentence, or ``None`` when not a B fit.

    Mirrors :func:`_kf_blending_link_evidence` for ``level_factors`` (#584
    decision 2). Phoneme blending's ten items are three-alternative forced choice,
    so the ordinary inverse-logit mean can predict below-chance scores and the
    guessing-floor companion cannot; the two estimates are one piece of evidence,
    and a key-findings box that showed either alone would overstate what the fit
    establishes. Fails closed: an unready pair raises, which withholds the box
    rather than publishing a single-link headline.
    """
    if str(config.get("kind")) != "level_factors" or str(
        config.get("outcome_symbol")
    ) != "B":
        return None
    from language_reading_predictors.statistical_models.blending_sensitivity import (
        evaluate_level_blending_link_pair,
    )

    status = evaluate_level_blending_link_pair(output_dir, config=config)
    if not status.get("required"):
        return None
    if not status.get("ready"):
        raise _KeyFindingsUnavailable(
            str(status.get("reason") or "the B link pair is not ready")
        )
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


def _kf_build_level_factors(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Level family: only the t2 group contrast is the randomised
    treated-versus-untreated effect; the later timepoints are randomised
    early-start-versus-delayed-start schedule contrasts (#631 finding 13)."""
    outcome_label = _kf_outcome_label(config)
    rope = _kf_csv_row(output_dir, "rope_summary.csv")
    if rope is None:
        raise _KeyFindingsUnavailable(
            "rope_summary.csv (the t2 items-scale contrast) is not present"
        )
    sentences: list[dict[str, str]] = []
    headline, is_rd = _kf_headline_from_rope(
        rope, outcome_label, "at the end of the randomised period (t2)"
    )
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
            _kf_direction_words(
                rope["pd"], is_rd=is_rd, rd_event="being off the floor at t2"
            ),
            "confidence",
        )
    )
    sentences.append(_kf_sentence(_kf_rope_sentence(rope, is_rd=is_rd), "rope"))
    plan = config.get("resolved_run_plan") or {}
    t1_referenced = str(plan.get("arm_gap_reference", "free")) == "t1" and bool(
        plan.get("group_by_time", True)
    )
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


def _kf_build_did(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Waitlist-crossover arm-by-wave family: the t2 arm contrast is the randomised
    treated-versus-untreated quantity; the t3 gap is a randomised *treatment-schedule*
    contrast and the gap change a description of how the two differ, never an
    identified catch-up mechanism (#576 finding 3). Dose companions (no ``tau_t2``)
    get the honest association wording."""
    outcome_label = _kf_outcome_label(config)
    did = _kf_csv_row(output_dir, "did_summary.csv")
    if did is None:
        raise _KeyFindingsUnavailable("did_summary.csv is not present")
    if "tau_t2_items_median" not in did:
        # Dose companion: no randomised t2 contrast to headline. The pooled
        # variant summarises ``beta_dose``; the period-varying variant
        # (LRPDID07) has no ``beta_dose`` at all — its slopes live in
        # ``dose_slope_summary.csv`` — so detect the family's own
        # ``dose_interpretation`` marker too, not just the pooled column
        # (#390: the period-varying fit regenerated as "predates the
        # arm-by-wave schema" and lost its release decision).
        if "dose_interpretation" in did or any(
            str(k).startswith("beta_dose") for k in did
        ):
            return [
                _kf_sentence(
                    "This companion model estimates how outcomes vary with the "
                    "amount of intervention received; that dose relationship is "
                    "an observational association, not a randomised comparison, "
                    "so no causal treatment-effect headline is reported.",
                    "causal",
                ),
                _kf_sentence(
                    # #576 finding 1: name the one quantity this fit publishes, so
                    # a reader is not left to pick among the several the results
                    # section shows.
                    "See the results section below for the dose estimates and "
                    "their uncertainty. The figure this model publishes is the "
                    "change in the outcome across a step in sessions, averaged "
                    "over the children who were on the intervention.",
                    "note",
                ),
            ]
        raise _KeyFindingsUnavailable(
            "did_summary.csv predates the arm-by-wave schema (no t2 items-scale "
            "contrast); refit or regenerate after a refit"
        )
    off_floor = bool(did.get("off_floor", False))
    sentences: list[dict[str, str]] = []
    if off_floor:
        med = _kf_float(did["tau_t2_items_median"]) * 100.0
        lo = _kf_float(did["tau_t2_items_lo"]) * 100.0
        hi = _kf_float(did["tau_t2_items_hi"]) * 100.0
        sentences.append(
            _kf_sentence(
                f"Best estimate: at t2 — the randomised comparison — being in the "
                f"immediate-intervention group was associated with a "
                f"**{med:+.0f} percentage-point** contrast in the chance of scoring "
                f"above zero on {outcome_label} "
                f"compared with the waiting list "
                f"(89% credible range {lo:+.0f} to {hi:+.0f}).",
                "headline",
            )
        )
    else:
        med = _kf_float(did["tau_t2_items_median"])
        lo = _kf_float(did["tau_t2_items_lo"])
        hi = _kf_float(did["tau_t2_items_hi"])
        higher_lower = "higher" if med >= 0 else "lower"
        sentences.append(
            _kf_sentence(
                f"Best estimate: at t2 — the randomised comparison — children in "
                f"the immediate-intervention group scored **{abs(med):.1f} items "
                f"{higher_lower}** on {outcome_label} than the waiting-list "
                f"children (89% credible range {lo:+.1f} to {hi:+.1f}).",
                "headline",
            )
        )
    sentences.append(
        _kf_sentence(
            # The off-floor DiD outcome is off-floor STATUS at each wave
            # (score > 0) — prevalence, not a floor-exit transition — so the
            # tau_t2 sentence names the status estimand (#490 review follow-up).
            _kf_direction_words(
                did["prob_tau_t2_pos"],
                is_rd=off_floor,
                rd_event="being off the floor at t2",
            ),
            "confidence",
        )
    )
    sentences.append(
        _kf_sentence(
            # #576 finding 3: the t3 quantities are randomised too — of a different
            # exposure. Calling them "descriptive associations" understated their
            # identification while overstating what they can explain.
            "The t2 comparison is randomised, but its cause-and-effect reading is "
            "limited to the fitted available-case t2 population and assumes "
            "outcome and required-covariate observation do not depend jointly on "
            "group and potential outcomes. The t1 gap is a starting-point balance "
            "check. The t3 gap is still a comparison of randomly assigned groups, "
            "but of a different thing — starting the intervention earlier rather "
            "than later, since both groups have been taught by then — so it cannot "
            "be read as the effect of being taught at all.",
            "causal",
        )
    )
    # Prefer the common-population gap change: the wave-specific one averages each
    # leg over its own wave's fitted rows, so where those differ it mixes the change
    # over time with a change in who is being averaged (#576 MQ6).
    common_available = bool(did.get("delta_crossover_items_common_available", False))
    key = (
        "delta_crossover_items_common_median"
        if common_available
        else "delta_crossover_items_median"
    )
    if common_available or bool(did.get("delta_crossover_items_available", False)):
        try:
            catch = _kf_float(did[key])
        except _KeyFindingsUnavailable:
            catch = None
        if catch is not None:
            unit = "percentage points" if off_floor else "items"
            moved = "narrowed" if catch > 0 else "widened"
            scale = 100.0 if off_floor else 1.0
            sentences.append(
                _kf_sentence(
                    f"After the waiting-list children started the intervention, the "
                    f"gap between the groups {moved} by about "
                    f"{abs(catch) * scale:.1f} {unit}. That describes how the "
                    "difference between the two randomly assigned groups changed; "
                    "it does not show why, because a shorter time in the "
                    "intervention, ordinary development, the ceiling of the test "
                    "and the different material each group was taught cannot be "
                    "separated here.",
                    "highlight",
                )
            )
    return sentences


def _kf_joint_pp(value) -> str:
    """A proportion-scale effect as signed percentage points (one decimal)."""
    return f"{100.0 * _kf_float(value):+.1f}"


def _kf_joint_optional_text(value) -> str:
    """A contrast-metadata cell as clean text; empty for an absent/NaN cell."""
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _kf_joint_marginal_phrase(
    symbol: str,
    tau_rows: Mapping[str, Mapping] | None,
    marginal_rows: Mapping[str, Mapping],
) -> str:
    """One outcome's marginal effect, preferring the comparable pp scale."""
    row = (tau_rows or {}).get(symbol)
    if row is not None:
        return (
            f"{symbol} {_kf_joint_pp(row['ame_prob_median'])} percentage points "
            f"(89% {_kf_joint_pp(row['ame_prob_lo'])} to "
            f"{_kf_joint_pp(row['ame_prob_hi'])}; "
            f"P(> 0) = {_kf_pct(row['prob_ame_pos'])}%)"
        )
    row = marginal_rows.get(symbol)
    if row is None:
        raise _KeyFindingsUnavailable(
            f"no joint marginal row for contrast outcome {symbol!r}"
        )
    return (
        f"{symbol} {_kf_float(row['items_median']):+.1f} items "
        f"(89% {_kf_float(row['items_lo']):+.1f} to "
        f"{_kf_float(row['items_hi']):+.1f}; "
        f"P(> 0) = {_kf_pct(row['prob_pos'])}%)"
    )


def _kf_build_joint(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Joint available-case modified ITT: contrast-first when one is declared.

    2026-08-21 joint review, findings 2 and 4. A contrast fit's declared estimand
    (``tau_difference.csv``) leads the box — previously the box reported only the
    two marginal effects, so a model registered to answer "did the intervention
    help taught words *more than* untaught words" headlined the much
    better-resolved marginal instead. The cross-outcome range is stated in
    percentage points (the AME scale built for cross-outcome comparability)
    rather than pooling item units across tests with different denominators,
    falling back to the items scale only for a stored fit without the pp
    columns; and a range including P or B carries the floor-rule /
    response-link qualification the report body already applies.
    """
    df = _kf_csv(output_dir, "joint_treatment_marginal.csv")
    if df is None:
        raise _KeyFindingsUnavailable(
            "joint_treatment_marginal.csv is not present; this fit predates the "
            "joint items-scale pushforward"
        )
    required = {"outcome", "items_median", "items_lo", "items_hi", "prob_pos"}
    if not required.issubset(df.columns):
        raise _KeyFindingsUnavailable(
            "joint_treatment_marginal.csv does not have the expected columns"
        )
    marginal_rows = {
        str(row["outcome"]): row for _, row in df.iterrows()
    }

    tau = _kf_csv(output_dir, "tau_summary.csv")
    pp_cols = {"outcome", "ame_prob_median", "ame_prob_lo", "ame_prob_hi", "prob_ame_pos"}
    tau_rows = (
        {str(row["outcome"]): row for _, row in tau.iterrows()}
        if tau is not None and pp_cols.issubset(tau.columns)
        else None
    )

    contrast = _kf_csv_row(output_dir, "tau_difference.csv")
    contrast_cols = {
        "contrast",
        "diff_prob_median",
        "diff_prob_lo",
        "diff_prob_hi",
        "diff_prob_lo50",
        "diff_prob_hi50",
        "prob_diff_pos",
    }

    sentences: list[dict[str, str]] = []
    if contrast is not None and contrast_cols.issubset(contrast):
        # The declared two-outcome contrast IS this model's estimand, so it
        # leads; the marginal effects underneath it are droppable context.
        pair_a, _, pair_b = str(contrast["contrast"]).partition("_minus_")
        label = _kf_joint_optional_text(contrast.get("contrast_label")) or str(
            contrast["contrast"]
        )
        kind_word = (
            _kf_joint_optional_text(contrast.get("contrast_kind")) or "outcome"
        )
        sentences.append(
            _kf_sentence(
                f"The declared {kind_word} contrast — {label} "
                f"(average marginal effect on {pair_a or contrast['contrast']} minus "
                f"{pair_b or contrast['contrast']}) — is "
                f"**{_kf_joint_pp(contrast['diff_prob_median'])} percentage "
                "points** on the proportion-correct scale (50% interval "
                f"{_kf_joint_pp(contrast['diff_prob_lo50'])} to "
                f"{_kf_joint_pp(contrast['diff_prob_hi50'])}; 89% "
                f"{_kf_joint_pp(contrast['diff_prob_lo'])} to "
                f"{_kf_joint_pp(contrast['diff_prob_hi'])}).",
                "headline",
            )
        )
        p_diff = _kf_float(contrast["prob_diff_pos"])
        fav = favoured_direction(p_diff)
        positive = fav["favoured_direction"] == "positive"
        interpretation = _kf_joint_optional_text(
            contrast.get(
                "positive_interpretation" if positive else "negative_interpretation"
            )
        )
        sentences.append(
            _kf_sentence(
                "The posterior probability of a "
                f"{'positive' if positive else 'negative'} difference is "
                f"{_kf_pct(fav['favoured_direction_prob'])}% — "
                f"{fav['favoured_direction_label']} evidence for that direction."
                + (f" {interpretation}" if interpretation else ""),
                "confidence",
            )
        )
        if pair_a and pair_b:
            sentences.append(
                _kf_sentence(
                    "The two marginal intervention effects underneath it: "
                    f"{_kf_joint_marginal_phrase(pair_a, tau_rows, marginal_rows)} "
                    "and "
                    f"{_kf_joint_marginal_phrase(pair_b, tau_rows, marginal_rows)}.",
                    "note",
                )
            )
        transfer_symbol = _kf_joint_optional_text(contrast.get("transfer_outcome"))
        transfer_read = _kf_joint_optional_text(
            contrast.get("transfer_interpretation")
        )
        if transfer_symbol:
            sentences.append(
                _kf_sentence(
                    (f"{transfer_read} " if transfer_read else "")
                    + "Here that marginal effect is "
                    + _kf_joint_marginal_phrase(
                        transfer_symbol, tau_rows, marginal_rows
                    )
                    + ".",
                    "transfer",
                )
            )
    else:
        if tau_rows is not None:
            medians = [_kf_float(r["ame_prob_median"]) * 100.0 for r in tau_rows.values()]
            lows = [_kf_float(r["ame_prob_lo"]) * 100.0 for r in tau_rows.values()]
            highs = [_kf_float(r["ame_prob_hi"]) * 100.0 for r in tau_rows.values()]
            sentences.append(
                _kf_sentence(
                    f"Across the {len(tau_rows)} outcomes, the joint available-case "
                    "modified ITT estimates ranged from "
                    f"**{min(medians):+.1f} to {max(medians):+.1f} percentage "
                    "points** on each test's proportion-correct scale; the "
                    f"individual 89% credible ranges extended from "
                    f"{min(lows):+.1f} to {max(highs):+.1f} percentage points "
                    "overall.",
                    "headline",
                )
            )
        else:
            # Stored fit without the pp columns: retain the items-scale range.
            medians = [_kf_float(v) for v in df["items_median"]]
            lows = [_kf_float(v) for v in df["items_lo"]]
            highs = [_kf_float(v) for v in df["items_hi"]]
            sentences.append(
                _kf_sentence(
                    f"Across the {len(df)} outcomes, the joint available-case "
                    "modified ITT estimates ranged from "
                    f"**{min(medians):+.1f} to {max(medians):+.1f} items**; the "
                    f"individual 89% credible ranges extended from "
                    f"{min(lows):+.1f} to {max(highs):+.1f} items overall.",
                    "headline",
                )
            )

        clearest = _kf_most_resolved_row(df, prob_col="prob_pos")
        symbol = str(clearest["outcome"])
        label = _kf_measure_label(symbol)
        direction = _kf_direction_words(clearest["prob_pos"], is_rd=False)
        sentences.append(
            _kf_sentence(
                f"For {label}, the clearest directional result: "
                f"{direction[0].lower() + direction[1:]}",
                "confidence",
            )
        )

        outcomes_present = {str(v) for v in df["outcome"]}
        qualified = []
        if "P" in outcomes_present:
            qualified.append(
                "P's graded score is a flagged secondary under the suite floor "
                "rule (its headline is the binary off-floor estimand)"
            )
        if "B" in outcomes_present:
            qualified.append(
                "B's ordinary-logit effect is conditional on the mandatory "
                "lrp-rli-itt-008/108 response-link sensitivity"
            )
        if qualified:
            sentences.append(
                _kf_sentence(
                    " and ".join(qualified)
                    + "; the range includes "
                    + ("them" if len(qualified) > 1 else "it")
                    + " for completeness only.",
                    "note",
                )
            )

        if {"delta_items", "prob_benefit_ge_delta"}.issubset(df.columns):
            deltas = df[
                np.isfinite(pd.to_numeric(df["delta_items"], errors="coerce"))
                & np.isfinite(
                    pd.to_numeric(df["prob_benefit_ge_delta"], errors="coerce")
                )
            ]
            if not deltas.empty:
                probabilities = [
                    _kf_float(v) for v in deltas["prob_benefit_ge_delta"]
                ]
                more_likely_than_not = sum(p >= 0.5 for p in probabilities)
                sentences.append(
                    _kf_sentence(
                        f"Among the {len(deltas)} "
                        f"outcome{'' if len(deltas) == 1 else 's'} with a post-hoc, "
                        f"project-agreed smallest-important difference, "
                        f"{more_likely_than_not} "
                        f"{'was' if more_likely_than_not == 1 else 'were'} "
                        f"more likely than not to reach it; the outcome-specific "
                        f"probabilities ranged from {_kf_pct(min(probabilities))}% to "
                        f"{_kf_pct(max(probabilities))}%.",
                        "rope",
                    )
                )
    sentences.append(
        _kf_sentence(
            "These are available-case modified ITT estimates of randomised-arm "
            "contrasts, not full-randomised-cohort ITT estimates. Their cause-and-effect "
            "reading assumes archive inclusion, "
            "outcome observation and any complete-case restriction do not depend "
            "jointly on arm and potential outcomes; without further missing-data "
            "assumptions they are not effects for all 57 randomised children.",
            "causal",
        )
    )
    return sentences


def _kf_mechanism_shape_caveat(output_dir, config: Mapping) -> dict[str, str] | None:
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


def mechanism_headline_estimand(output_dir) -> dict | None:
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


def _kf_mechanism_slope_sentences(output_dir) -> list[dict[str, str]]:
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


def _kf_build_mechanism(output_dir, config: Mapping) -> list[dict[str, str]]:
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
            f"between the {int(round(100 * float(q_hi)))}th and "
            f"{int(round(100 * float(q_lo)))}th percentile of the fitted exposure "
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
        low, high = ordered.iloc[0], ordered.iloc[-1]
        sentences.append(
            _kf_sentence(
                f"Across the fitted predictor range, its model contribution changed "
                f"from {_kf_float(low['f_mean']):+.2f} logit units "
                f"(89% range {_kf_float(low['f_lo']):+.2f} to "
                f"{_kf_float(low['f_hi']):+.2f}) to "
                f"{_kf_float(high['f_mean']):+.2f} "
                f"({_kf_float(high['f_lo']):+.2f} to "
                f"{_kf_float(high['f_hi']):+.2f}).",
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


#: Display labels for the covariate moderators a mechanism fit may declare
#: (measures take their registered label).
_KF_COVARIATE_MODERATOR_LABELS = {
    "A": "age",
    "erbto": "phonological memory (word/nonword repetition)",
}


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
    output_dir, config: Mapping, *, prob_gamma_int_pos: float
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


def _kf_build_mediation(output_dir, config: Mapping) -> list[dict[str, str]]:
    """One- or two-mediator g-formula decomposition."""
    df = _kf_csv(output_dir, "mediation_summary.csv")
    if df is None or "quantity" not in df.columns:
        raise _KeyFindingsUnavailable("mediation_summary.csv is not present")
    indexed = df.set_index("quantity")
    if "total" not in indexed.index:
        raise _KeyFindingsUnavailable(
            "mediation_summary.csv has no total-effect row"
        )
    total = indexed.loc["total"].to_dict()
    off_floor = str(total.get("off_floor", "false")).lower() in {"true", "1"}
    scale = 100.0 if off_floor else 1.0
    unit = "percentage points" if off_floor else "items"
    med = _kf_float(total["words_median"]) * scale
    lo = _kf_float(total["words_lo"]) * scale
    hi = _kf_float(total["words_hi"]) * scale
    fav = favoured_direction(_kf_float(total["prob_pos"]))
    positive = fav["favoured_direction"] == "positive"
    direction = "positive" if positive else "negative"
    claim = (
        "the intervention improves the outcome under the fitted model"
        if positive
        else "the intervention worsens the outcome under the fitted model"
    )
    sentences = [
        _kf_sentence(
            f"The model-based total intervention contrast was **{med:+.1f} "
            f"{unit}** (89% credible range {lo:+.1f} to {hi:+.1f}).",
            "headline",
        ),
        _kf_sentence(
            f"The posterior probability that this model-based total contrast is "
            f"{direction} is {_kf_pct(fav['favoured_direction_prob'])}% — "
            f"{fav['favoured_direction_label']} evidence that {claim}.",
            "confidence",
        ),
    ]
    indirect_name = next(
        (name for name in ("NIE_joint", "NIE", "IIE") if name in indexed.index),
        None,
    )
    if indirect_name is not None:
        indirect = indexed.loc[indirect_name].to_dict()
        i_med = _kf_float(indirect["words_median"]) * scale
        i_lo = _kf_float(indirect["words_lo"]) * scale
        i_hi = _kf_float(indirect["words_hi"]) * scale
        sentences.append(
            _kf_sentence(
                f"The estimated indirect component ({indirect_name}) was "
                f"{i_med:+.1f} {unit} (89% credible range {i_lo:+.1f} to "
                f"{i_hi:+.1f}).",
                "highlight",
            )
        )
    # Period-stacked fits standardise over ONE window: the only one holding both
    # arms. Say so, so the headline is not read as an all-period average (#585).
    unsupported = (config.get("extra") or {}).get("unsupported_periods") or []
    if unsupported:
        listed = ", ".join(str(period) for period in unsupported)
        sentences.append(
            _kf_sentence(
                "This contrast is averaged over the randomised first period only. "
                f"Period(s) {listed} contain no untreated children after the "
                "wait-list crossover, so an all-period average would extrapolate "
                "an untreated counterfactual the data cannot support.",
                "scale",
            )
        )
    sentences.append(
        _kf_sentence(
            "The direct/indirect split is a model-based g-formula decomposition, "
            "not an identified causal mediation effect: unmeasured "
            "mediator-outcome confounding remains a binding assumption.",
            "causal",
        )
    )
    return sentences


def _kf_build_aligned(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Onset-aligned per-protocol cohort contrast; every term is associative."""
    outcome_label = _kf_outcome_label(config)
    marginal = _kf_csv_row(output_dir, "cohort_marginal.csv")
    if marginal is None:
        raise _KeyFindingsUnavailable("cohort_marginal.csv is not present")
    plan = config.get("resolved_run_plan") or {}
    extra = config.get("extra") or {}
    off_floor = bool(
        plan.get(
            "off_floor",
            plan.get("likelihood", extra.get("likelihood"))
            == "bernoulli_offfloor",
        )
    )
    scale = 100.0 if off_floor else 1.0
    unit = "percentage points" if off_floor else "items"
    med = _kf_float(marginal["trt_items_median"]) * scale
    lo = _kf_float(marginal["trt_items_lo"]) * scale
    hi = _kf_float(marginal["trt_items_hi"]) * scale
    sentences = [
        _kf_sentence(
            f"After aligning children by intervention onset, the immediate cohort "
            f"differed from the waiting-list cohort on {outcome_label} by "
            f"**{med:+.1f} {unit}** (89% credible range {lo:+.1f} to "
            f"{hi:+.1f}).",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                marginal["prob_trt_pos"],
                positive_claim="the immediate cohort tends to score higher",
                negative_claim="the immediate cohort tends to score lower",
            ),
            "confidence",
        ),
        _kf_sentence(
            "This is a per-protocol cohort association, not a randomised treatment "
            "effect; age at onset and cohort timing can confound it.",
            "causal",
        ),
    ]
    highlight = _kf_strongest_factor(output_dir, exclude_roles=())
    if highlight:
        sentences.append(_kf_sentence(highlight, "highlight"))
    return sentences


def _kf_build_adjusted(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Between-child adjusted predictor associations on the items scale."""
    df = _kf_csv(output_dir, "predicted_gain_words.csv")
    if df is None:
        raise _KeyFindingsUnavailable("predicted_gain_words.csv is not present")
    # Missing-data indicators (``{cov}_missing``) are subgroup mean-offsets under
    # the missing-indicator method — nuisance terms the associations table and the
    # priors table already exclude. The pipeline now filters them out of this table
    # too; this guard keeps a stored pre-fix file from headlining "Speech missing
    # (indicator)" as the clearest predictor (2026-08-22 review, finding 3).
    if "predictor" in df.columns:
        df = df[~df["predictor"].astype(str).str.endswith("_missing")]
        if df.empty:
            raise _KeyFindingsUnavailable(
                "predicted_gain_words.csv carries only missing-indicator rows"
            )
    row = _kf_most_resolved_row(df, prob_col="prob_pos")
    label = _kf_plain_label(row.get("label", row.get("predictor", "predictor")))
    # House standard is the posterior median (METHODS.md); the mean was reported
    # here until the August 2026 review, which is why an adjusted headline could
    # disagree with the same fit's tables by a rounding step.
    med = _kf_float(row.get("delta_words_median", row["delta_words_mean"]))
    lo = _kf_float(row["delta_words_lo"])
    hi = _kf_float(row["delta_words_hi"])
    outcome_label = _kf_outcome_label(config)
    # The design is read from the persisted plan: the stacked Byrne transition
    # model pools annual transitions with a child random intercept, so its slopes
    # are repeated-transition associations, not the one-row-per-child
    # between-child contrast of the span designs (2026-08-22 review, finding 6).
    plan = config.get("resolved_run_plan") or {}
    if plan.get("transition_waves"):
        causal = (
            "This is a pooled repeated-transition adjusted association (annual "
            "transitions stacked, with a child random intercept); neither the "
            "temporal ordering nor the random intercept identifies what would "
            "happen if the predictor were changed."
        )
    else:
        causal = (
            "This is a between-child adjusted association; it does not identify "
            "what would happen if the predictor were changed."
        )
    return [
        _kf_sentence(
            f"The clearest adjusted predictor was {label}: a 1-SD increase was "
            f"associated with **{med:+.1f} items** of difference in "
            f"{outcome_label} "
            f"(89% credible range {lo:+.1f} to {hi:+.1f}).",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["prob_pos"],
                positive_claim="higher values accompany greater gain",
                negative_claim="higher values accompany less gain",
            ),
            "confidence",
        ),
        _kf_sentence(causal, "causal"),
    ]


def _kf_build_corr_factor(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Cross-sectional correlated-domain measurement model."""
    correlations = _kf_csv(output_dir, "factor_correlation_summary.csv")
    structural = _kf_csv(output_dir, "structural_summary.csv")
    if correlations is None and structural is None:
        raise _KeyFindingsUnavailable(
            "neither factor_correlation_summary.csv nor structural_summary.csv is present"
        )
    sentences: list[dict[str, str]] = []
    if correlations is not None:
        row = _kf_most_resolved_row(correlations, prob_col="prob_pos")
        pair = (
            f"{_kf_plain_label(row['domain_i'])} and "
            f"{_kf_plain_label(row['domain_j'])}"
        )
        sentences.extend(
            [
                _kf_sentence(
                    f"The clearest latent-domain correlation was between {pair}: "
                    f"**{_kf_float(row['median']):+.2f}** (89% credible range "
                    f"{_kf_float(row['lo']):+.2f} to "
                    f"{_kf_float(row['hi']):+.2f}).",
                    "headline",
                ),
                _kf_sentence(
                    _kf_association_direction(
                        row["prob_pos"],
                        positive_claim="the two latent skill areas tend to move together",
                        negative_claim="the two latent skill areas tend to move oppositely",
                    ),
                    "confidence",
                ),
            ]
        )
    if structural is not None:
        # Only the beta_<domain> factor slopes are structural slopes. Ranking over
        # every row let the beta_age adjustment covariate win the highlight in all
        # four released RLI boxes — displacing beta_code, the errors-in-variables
        # focal slope, in mm-002/102 (2026-08-21 review, finding 2b). A config
        # without factor names in its plan (a legacy stub) keeps the unfiltered
        # ranking rather than failing.
        plan = config.get("resolved_run_plan") or {}
        factors = list(plan.get("structural_factors") or []) or [
            domain[0] for domain in (plan.get("domains") or [])
        ]
        slopes = structural
        if factors:
            wanted = {f"beta_{name}" for name in factors}
            slopes = structural[structural["coefficient"].astype(str).isin(wanted)]
            if slopes.empty:
                raise _KeyFindingsUnavailable(
                    "structural_summary.csv has no factor-slope rows matching the "
                    "resolved plan's structural factors"
                )
        row = _kf_most_resolved_row(slopes, prob_col="prob_pos")
        sentences.append(
            _kf_sentence(
                f"The clearest structural slope was "
                f"{_kf_plain_label(row['coefficient'])}: "
                f"{_kf_float(row['median']):+.2f} logit units (89% credible range "
                f"{_kf_float(row['lo']):+.2f} to "
                f"{_kf_float(row['hi']):+.2f}).",
                "highlight",
            )
        )
        if correlations is None:
            sentences.append(
                _kf_sentence(
                    _kf_association_direction(
                        row["prob_pos"],
                        positive_claim="the linked latent quantities tend to move together",
                        negative_claim="the linked latent quantities tend to move oppositely",
                    ),
                    "confidence",
                )
            )
    sentences.append(
        _kf_sentence(
            "This is a measurement and triangulation model; its factor "
            "correlations and structural slopes are associations, not causal effects.",
            "causal",
        )
    )
    return sentences


def _kf_build_dose_response(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Observational session-dose association."""
    marginal = _kf_csv_row(output_dir, "dose_marginal_summary.csv")
    outcome_label = _kf_outcome_label(config)
    sentences: list[dict[str, str]] = []
    if marginal is not None:
        # Say how big the step actually was and who it was averaged over (#587
        # finding 3). "A 1-SD increase in sessions" hid a 30.7-session step applied to
        # every row, including children with no intervention at all; the repaired
        # contrast is a within-period interquartile move over on-intervention rows,
        # and the sentence names the sessions rather than a standardised unit.
        # A fit written before the support-respecting contrast existed has neither
        # column; fall back to the old wording rather than refusing to render.
        raw_step = marginal.get("contrast_sessions_median")
        step = (
            float(raw_step)
            if raw_step is not None and np.isfinite(pd.to_numeric(raw_step, errors="coerce"))
            else float("nan")
        )
        rows = marginal.get("n_rows")
        if np.isfinite(step):
            opening = (
                f"Among children on the intervention, attending about "
                f"{step:.0f} more sessions in a period — an interquartile step "
                f"within that period's observed attendance — was associated with"
            )
        else:
            opening = "A 1-SD increase in sessions was associated with"
        sentences.append(
            _kf_sentence(
                f"{opening} "
                f"**{_kf_float(marginal['items_median']):+.1f} items** on "
                f"{outcome_label} "
                f"(89% credible range {_kf_float(marginal['items_lo']):+.1f} "
                f"to {_kf_float(marginal['items_hi']):+.1f}"
                + (f"; averaged over {int(rows)} rows)." if rows is not None else ")."),
                "headline",
            )
        )
        sentences.append(
            _kf_sentence(
                _kf_association_direction(
                    marginal["prob_pos"],
                    positive_claim=(
                        "attending more sessions accompanies a higher outcome among "
                        "children already on the intervention"
                    ),
                    negative_claim=(
                        "attending more sessions accompanies a lower outcome among "
                        "children already on the intervention"
                    ),
                ),
                "confidence",
            )
        )
    else:
        slopes = _kf_csv(output_dir, "dose_slope_summary.csv")
        if slopes is None:
            raise _KeyFindingsUnavailable(
                "neither dose_marginal_summary.csv nor dose_slope_summary.csv is present"
            )
        row = slopes.iloc[0].to_dict()
        sentences.append(
            _kf_sentence(
                f"The headline dose slope was "
                f"**{_kf_float(row['median']):+.2f} logit units per 1 SD of "
                f"sessions** (89% credible range {_kf_float(row['lo']):+.2f} "
                f"to {_kf_float(row['hi']):+.2f}).",
                "headline",
            )
        )
        sentences.append(
            _kf_sentence(
                _kf_association_direction(
                    row["p_pos"],
                    positive_claim="higher session dose accompanies a higher outcome",
                    negative_claim="higher session dose accompanies a lower outcome",
                ),
                "confidence",
            )
        )
    slope_table = _kf_csv(output_dir, "dose_slope_summary.csv")
    if slope_table is not None and "on_intervention" in set(slope_table["term"]):
        presence = slope_table[slope_table["term"] == "on_intervention"].iloc[0]
        sentences.append(
            _kf_sentence(
                "Being on the intervention at all is reported separately from how "
                "much was attended: "
                f"**{_kf_float(presence['median']):+.2f} logit units** "
                f"(89% credible range {_kf_float(presence['lo']):+.2f} to "
                f"{_kf_float(presence['hi']):+.2f}). In period 1 that contrast is "
                "randomised — every immediate-arm child attended and every waitlist "
                "child attended none — so it, not the dose slope, is where this "
                "model's randomised evidence sits.",
                "robustness",
            )
        )
    sentences.append(
        _kf_sentence(
            "How many sessions a child attended was not randomised and may reflect "
            "ability, attendance or availability — the study DAG has age, latent "
            "general ability and assigned group all pointing into attendance — so "
            "every dose slope here is an observational association, not evidence "
            "that more sessions cause more progress.",
            "causal",
        )
    )
    if config.get("outcome_symbol") == "B":
        # Phoneme blending has ten three-alternative items, so the ordinary logit mean
        # admits fitted means below the one-third guessing level. `METHODS.md` requires
        # any headline B interpretation to be paired with the guessing-floor link
        # sensitivity, and that pairing is currently built only for the ITT fits
        # (#587 finding 6). Say so on the fit rather than publishing an unqualified
        # blending number; the repo-wide gap is tracked separately.
        sentences.append(
            _kf_sentence(
                "**Qualified result.** Phoneme blending is measured with ten "
                "three-choice items, so a child guessing at random still scores about "
                "a third. This model uses the ordinary link, which allows fitted "
                "means below that guessing level, and the project's required "
                "guessing-floor companion has not been built for this family — only "
                "for the randomised-arm blending fits. Read the blending association "
                "as provisional until that pair exists.",
                "caveat",
            )
        )
    return sentences


def _kf_build_lcsm(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Latent change-score couplings, with an optional randomised-window check."""
    df = _kf_csv(output_dir, "coupling_summary.csv")
    if df is None:
        raise _KeyFindingsUnavailable("coupling_summary.csv is not present")
    # Only the g_/h_ rows are couplings. A contains("->") filter also matched the
    # age slope and the shared adjuster slopes, so a precision covariate could win
    # the "clearest longitudinal coupling" headline with a level-worded confidence
    # sentence (2026-08-21 review, finding 2a) — live in the released 067 box.
    directed = df[df["coefficient"].astype(str).str.match(r"[gh]_")]
    if directed.empty:
        raise _KeyFindingsUnavailable("coupling_summary.csv has no coupling rows")
    row = _kf_most_resolved_row(directed, prob_col="prob_pos")
    name = str(row["coefficient"])
    lagged = name.startswith("h_")
    label = _kf_plain_label(name)
    if "(" in label and label.endswith(")"):
        label = label.split("(", 1)[1][:-1]
    sentences = [
        _kf_sentence(
            f"The clearest longitudinal coupling was {label}: "
            f"**{_kf_float(row['median']):+.2f} logit units** (89% credible range "
            f"{_kf_float(row['lo']):+.2f} to {_kf_float(row['hi']):+.2f}).",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["prob_pos"],
                positive_claim=(
                    "greater earlier change accompanies greater later change"
                    if lagged
                    else "a higher earlier level accompanies greater later change"
                ),
                negative_claim=(
                    "greater earlier change accompanies less later change"
                    if lagged
                    else "a higher earlier level accompanies less later change"
                ),
            ),
            "confidence",
        ),
        _kf_sentence(
            "The couplings are conditional predictive associations among latent "
            "trajectories, not causal skill-to-skill effects.",
            "causal",
        ),
    ]
    itt = _kf_csv(output_dir, "itt_window1_contrast.csv")
    if itt is not None:
        # Quote the model's focal outcome when its row exists, and always name the
        # measure — the unnamed most-resolved row silently attributed another
        # outcome's contrast to the focal measure (finding 2c: 081 quoted W under
        # a taught-vocabulary model, 091 quoted L under a word-reading model).
        focal = str(config.get("outcome_symbol") or "")
        cand = (
            itt[itt["coefficient"].astype(str).str.startswith(f"itt_w1[{focal}]")]
            if focal
            else itt.iloc[0:0]
        )
        check = cand.iloc[0] if len(cand) else _kf_most_resolved_row(itt, prob_col="prob_pos")
        match = re.search(r"itt_w1\[([^\]]+)\]", str(check["coefficient"]))
        measure = _kf_measure_label(match.group(1)) if match else str(check["coefficient"])
        sentences.append(
            _kf_sentence(
                f"The separate randomised window-1 consistency contrast for "
                f"{measure} was {_kf_float(check['median']):+.2f} latent-logit "
                f"units (89% credible range {_kf_float(check['lo']):+.2f} to "
                f"{_kf_float(check['hi']):+.2f}); it is a check, not the coupling "
                f"headline.",
                "highlight",
            )
        )
    return sentences


def _kf_build_horseshoe(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Regularised-horseshoe predictor-ranking sensitivity analysis."""
    df = _kf_csv(output_dir, "predictor_ranking.csv")
    if df is None:
        raise _KeyFindingsUnavailable("predictor_ranking.csv is not present")
    row = df.sort_values("rank").iloc[0].to_dict()
    label = _kf_plain_label(row["predictor"])
    direction = "positive" if _kf_float(row["beta_median"]) >= 0 else "negative"
    return [
        _kf_sentence(
            f"The top-ranked predictor was {label}, with a standardised "
            f"{direction} association of **{_kf_float(row['beta_median']):+.2f} "
            f"logit units** (89% highest-density interval "
            f"{_kf_float(row['beta_hdi_lo']):+.2f} to "
            f"{_kf_float(row['beta_hdi_hi']):+.2f}).",
            "headline",
        ),
        _kf_sentence(
            f"Its probability of exceeding the model's worth-noticing "
            f"coefficient threshold was {_kf_pct(row['p_abs_gt_delta'])}%.",
            "confidence",
        ),
        _kf_sentence(
            "The ranking is an adjusted predictive sensitivity check, not a list "
            "of causal drivers; closely ranked predictors should not be treated as "
            "meaningfully ordered.",
            "causal",
        ),
    ]


def _kf_growth_interaction_sentences(
    gamma_int: pd.DataFrame, gamma: pd.DataFrame
) -> list[dict[str, str]]:
    """Key-findings box for the age x ability interaction growth model (LRP85)."""
    row = _kf_most_resolved_row(gamma_int, prob_col="prob_positive")
    outcome = _kf_measure_label(row["outcome"])
    sentences = [
        _kf_sentence(
            f"For {outcome}, the clearest interaction result, a child +1 SD older "
            f"at entry **and** +1 SD higher in baseline non-verbal ability differed "
            f"in growth rate by **{_kf_float(row['median']):+.2f} logit units** "
            f"beyond the two main effects (89% credible range "
            f"{_kf_float(row['lo89']):+.2f} to {_kf_float(row['hi89']):+.2f}).",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["prob_positive"],
                positive_claim=(
                    "older-and-more-able children progress faster than the main "
                    "effects alone imply"
                ),
                negative_claim=(
                    "the ability-growth association weakens with age at entry"
                ),
            ),
            "confidence",
        ),
    ]
    same = gamma[gamma["outcome"] == row["outcome"]]
    if not same.empty:
        g = same.iloc[0]
        sentences.append(
            _kf_sentence(
                f"The ability main effect (gamma) for the same outcome, at the "
                f"sample-mean entry age, was {_kf_float(g['median']):+.2f} logit "
                f"units (89% credible range {_kf_float(g['lo89']):+.2f} to "
                f"{_kf_float(g['hi89']):+.2f}).",
                "highlight",
            )
        )
    sentences.append(
        _kf_sentence(
            "These trajectory coefficients are adjusted associations, not effects "
            "of changing non-verbal ability.",
            "causal",
        )
    )
    return sentences


def _kf_build_growth(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Multivariate growth: baseline ability association with growth rate."""
    df = _kf_csv(output_dir, "growth_association_summary.csv")
    if df is None:
        raise _KeyFindingsUnavailable("growth_association_summary.csv is not present")
    gamma = df[df["coefficient"] == "gamma"]
    if gamma.empty:
        raise _KeyFindingsUnavailable("growth summary has no gamma rows")
    plan = config.get("resolved_run_plan") or {}
    # The interaction model's registered headline is gamma_int, not gamma
    # (2026-08-21 review, finding 1); a plan that declares the interaction but a
    # summary without its rows is a stale pre-fix artefact, so fail loud.
    if bool(plan.get("age_ability_interaction")):
        gamma_int = df[df["coefficient"] == "gamma_int"]
        if gamma_int.empty:
            raise _KeyFindingsUnavailable(
                "the plan declares the age x ability interaction but the growth "
                "summary has no gamma_int rows; regenerate the summary CSV first"
            )
        return _kf_growth_interaction_sentences(gamma_int, gamma)
    row = _kf_most_resolved_row(gamma, prob_col="prob_positive")
    outcome = _kf_measure_label(row["outcome"])
    study_id = str(config.get("study_id") or "rli")
    baseline_symbol = plan.get("baseline_covariate")
    baseline_label = "non-verbal ability"
    if study_id != "rli" and isinstance(baseline_symbol, str):
        try:
            from language_reading_predictors.statistical_models.datasets import (
                resolve_dataset,
            )

            _dataset, catalogue = resolve_dataset(study_id)
            baseline_label = catalogue[baseline_symbol].label
            if str(row["outcome"]) in catalogue:
                outcome = catalogue[str(row["outcome"])].label
        except (KeyError, TypeError):
            baseline_label = baseline_symbol
    return [
        _kf_sentence(
            f"For {outcome}, the clearest result, a 1-SD higher baseline "
            f"{baseline_label} score was associated with a growth-rate change of "
            f"**{_kf_float(row['median']):+.2f} logit units** (89% credible range "
            f"{_kf_float(row['lo89']):+.2f} to "
            f"{_kf_float(row['hi89']):+.2f}).",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["prob_positive"],
                positive_claim="higher baseline ability accompanies faster growth",
                negative_claim="higher baseline ability accompanies slower growth",
            ),
            "confidence",
        ),
        _kf_sentence(
            "These trajectory coefficients are adjusted associations, not effects of "
            f"changing {baseline_label}.",
            "causal",
        ),
    ]


def _kf_build_historical_growth(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Historical-cohort natural-history reproduction.

    Window-aware (2026-08-21 historical-families review, finding 3). The
    within-group intervals mix complete-case **core** rows with the
    attrition-selected **extension** tail (#338), and the selector's
    ``P(positive)`` metric saturates at 1 for most of them — so ranking alone
    used to headline an extension interval, unflagged, in five of six
    publishable fits. Prefer a core interval when the fit has one, say so when
    the headline is an extension row, and always report the interval's own
    subject count. The between-group contrasts — the estimand the family's prior
    pushforward checks — get their own sentence rather than being filtered out.
    """
    df = _kf_csv(output_dir, "posterior_growth_summary.csv")
    if df is None:
        raise _KeyFindingsUnavailable("posterior_growth_summary.csv is not present")
    labels = df["readgrp_label"].fillna("").astype(str).str.strip()
    within = df[labels.str.len() > 0]
    contrasts = df[labels.str.len() == 0]
    if within.empty:
        within = df
        contrasts = df.iloc[0:0]
    # Prefer the audited core window; fall back to the extension tail only when
    # the fit supports no core interval at all.
    core = (
        within[within["window"].astype(str) == "core"]
        if "window" in within.columns
        else within
    )
    candidates = core if not core.empty else within
    row = _kf_most_resolved_row(candidates, prob_col="p_gt_0")
    group = _kf_plain_label(row.get("readgrp_label", "historical cohort"))
    window = str(row.get("window", "")).strip()
    n_subjects = row.get("n_subjects")
    try:
        n_text = f", {int(float(n_subjects))} children"
    except (TypeError, ValueError):
        n_text = ""
    window_text = (
        " This interval is on the attrition-selected follow-up extension, not "
        "the audited complete-case core, so it describes the children who "
        "remained in the study."
        if window == "extension"
        else ""
    )
    fav = favoured_direction(_kf_float(row["p_gt_0"]))
    positive = fav["favoured_direction"] == "positive"
    direction = "positive" if positive else "negative"
    claim = (
        "scores tend to increase over that interval"
        if positive
        else "scores tend to decrease over that interval"
    )
    sentences = [
        _kf_sentence(
            f"For the {group} group, {_kf_plain_label(row['label'])} was "
            f"**{_kf_float(row['mean']):+.1f} items** (89% credible range "
            f"{_kf_float(row['q_lo']):+.1f} to "
            f"{_kf_float(row['q_hi']):+.1f}{n_text}).{window_text}",
            "headline",
        ),
        _kf_sentence(
            f"The posterior probability that this growth is {direction} is "
            f"{_kf_pct(fav['favoured_direction_prob'])}% — "
            f"{fav['favoured_direction_label']} evidence that {claim}.",
            "confidence",
        ),
    ]
    if not contrasts.empty:
        contrast = _kf_most_resolved_row(contrasts, prob_col="p_gt_0")
        c_fav = favoured_direction(_kf_float(contrast["p_gt_0"]))
        sentences.append(
            _kf_sentence(
                f"Comparing groups over the window every group supports, "
                f"{_kf_plain_label(contrast['label'])} was "
                f"**{_kf_float(contrast['mean']):+.1f} items** (89% credible "
                f"range {_kf_float(contrast['q_lo']):+.1f} to "
                f"{_kf_float(contrast['q_hi']):+.1f}; "
                f"{c_fav['favoured_direction_label']} evidence it is "
                f"{c_fav['favoured_direction']}).",
                "highlight",
            )
        )
    sentences.append(
        _kf_sentence(
            "This is descriptive natural-history growth in a historical cohort, "
            "not an intervention effect or an explanation of group differences.",
            "causal",
        )
    )
    cells = _kf_csv(output_dir, "posterior_cell_summary.csv")
    if cells is not None and "posterior_mean_minus_observed_mean" in cells.columns:
        # The published audit is the complete-case core (Table 2); an extension
        # cell was never in it, so it must not set the reproduction figure.
        audit = (
            cells[cells["window"].astype(str) == "core"]
            if "window" in cells.columns
            else cells
        )
        gaps = [
            abs(_kf_float(v))
            for v in audit["posterior_mean_minus_observed_mean"]
            if np.isfinite(_kf_float(v))
        ]
        if gaps:
            sentences.append(
                _kf_sentence(
                    f"As a reproduction check on the complete-case core window, "
                    f"the largest fitted-minus-observed cell mean gap was "
                    f"{max(gaps):.1f} items.",
                    "highlight",
                )
            )
    return sentences


def _kf_build_historical_joint(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Byrne joint correlated growth: cross-measure coupling headline (#338)."""
    within = _kf_csv(output_dir, "within_measure_correlation_summary.csv")
    if within is not None:
        scales = _kf_csv(output_dir, "within_scale_summary.csv")
        if scales is not None and "pair_resolvable" in within.columns:
            resolvable_pairs = within[
                within["pair_resolvable"]
                .astype(str)
                .str.lower()
                .isin({"true", "1"})
            ]
            if resolvable_pairs.empty:
                strongest = scales.iloc[
                    pd.to_numeric(
                        scales["prob_above_minimum"], errors="coerce"
                    ).argmax()
                ]
                threshold = _kf_float(strongest["minimum_resolvable_sd"])
                return [
                    _kf_sentence(
                        "The model did not resolve a within-child correlation: "
                        "no measure pair had both wave-specific residual standard "
                        f"deviations supported above {threshold:.2f} logits.",
                        "headline",
                    ),
                    _kf_sentence(
                        f"The best-resolved residual scale was "
                        f"{_kf_plain_label(strongest['label'])}, median "
                        f"{_kf_float(strongest['median']):.2f} logits (inner 50% "
                        f"range {_kf_float(strongest['lo50']):.2f} to "
                        f"{_kf_float(strongest['hi50']):.2f}; 89% credible range "
                        f"{_kf_float(strongest['lo']):.2f} to "
                        f"{_kf_float(strongest['hi']):.2f}).",
                        "confidence",
                    ),
                    _kf_sentence(
                        "Non-resolution is itself a conclusion under this fit's "
                        "within-scale prior: that prior decides which measures "
                        "clear the threshold, and the registered wider-prior "
                        "sensitivity must be read beside this result before it is "
                        "treated as settled.",
                        "robustness",
                    ),
                    _kf_sentence(
                        "When a residual scale is not distinguishable from "
                        "measurement noise, its correlation is not substantively "
                        "identified. This is a descriptive information limit, not "
                        "evidence that skills are causally unrelated.",
                        "causal",
                    ),
                ]
            within = resolvable_pairs
        row = _kf_most_resolved_row(within, prob_col="prob_pos")
        pair = (
            f"{_kf_plain_label(row.get('label_i', row['measure_i']))} and "
            f"{_kf_plain_label(row.get('label_j', row['measure_j']))}"
        )
        sentences = [
            _kf_sentence(
                f"The clearest within-child coupling was between {pair}: a "
                f"wave-specific latent-logit correlation of "
                f"**{_kf_float(row['median']):+.2f}** (inner 50% range "
                f"{_kf_float(row['lo50']):+.2f} to "
                f"{_kf_float(row['hi50']):+.2f}; 89% credible range "
                f"{_kf_float(row['lo']):+.2f} to "
                f"{_kf_float(row['hi']):+.2f}).",
                "headline",
            ),
            _kf_sentence(
                _kf_association_direction(
                    row["prob_pos"],
                    positive_claim=(
                        "waves above a child's stable level on one measure tend "
                        "also to be above-level waves on the other"
                    ),
                    negative_claim=(
                        "waves above a child's stable level on one measure tend "
                        "to be below-level waves on the other"
                    ),
                ),
                "confidence",
            ),
        ]
        comparison = _kf_csv(
            output_dir, "between_within_correlation_comparison.csv"
        )
        if comparison is not None:
            matched = comparison[
                (comparison["measure_i"].astype(str) == str(row["measure_i"]))
                & (
                    comparison["measure_j"].astype(str)
                    == str(row["measure_j"])
                )
            ]
            if not matched.empty:
                comp = matched.iloc[0]
                sentences.append(
                    _kf_sentence(
                        "For that pair, the within-minus-between correlation was "
                        f"{_kf_float(comp['within_minus_between_median']):+.2f} "
                        f"(89% credible range "
                        f"{_kf_float(comp['within_minus_between_lo']):+.2f} to "
                        f"{_kf_float(comp['within_minus_between_hi']):+.2f}; "
                        f"P(within > between) = "
                        f"{_kf_float(comp['prob_within_gt_between']):.2f}).",
                        "highlight",
                    )
                )
        sentences.append(
            _kf_sentence(
                "This is descriptive within-child co-movement in a historical "
                "cohort - it does not identify direction, a treatment effect or "
                "a mechanism, and the residual scale must pass prior sensitivity.",
                "causal",
            )
        )
        sentences.append(_kf_sentence(_kf_pair_selection_note(len(within)), "note"))
        return sentences

    df = _kf_csv(output_dir, "measure_correlation_summary.csv")
    if df is None:
        raise _KeyFindingsUnavailable("measure_correlation_summary.csv is not present")
    row = _kf_most_resolved_row(df, prob_col="prob_pos")
    pair_note = _kf_pair_selection_note(len(df))
    pair = (
        f"{_kf_plain_label(row.get('label_i', row['measure_i']))} and "
        f"{_kf_plain_label(row.get('label_j', row['measure_j']))}"
    )
    return [
        _kf_sentence(
            f"The clearest between-child coupling was between {pair}: a stable-"
            f"level correlation of **{_kf_float(row['median']):+.2f}** (89% credible "
            f"range {_kf_float(row['lo']):+.2f} to {_kf_float(row['hi']):+.2f}).",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["prob_pos"],
                positive_claim=(
                    "children who sit higher on one measure tend to sit higher "
                    "on the other"
                ),
                negative_claim=(
                    "children who sit higher on one measure tend to sit lower "
                    "on the other"
                ),
            ),
            "confidence",
        ),
        _kf_sentence(
            "This is a descriptive between-child correlation of stable levels in "
            "a historical cohort - it is not causal and does not say that "
            "changing one skill changes another.",
            "causal",
        ),
        _kf_sentence(pair_note, "note"),
    ]


def _kf_pair_selection_note(n_pairs: int) -> str:
    """Label the leading measure pair as an exploratory, uncertainty-based choice.

    2026-08-23 joint audit, lower-priority reporting correction. The lead pair is
    the one whose ``P(rho > 0)`` sits furthest from 0.5 among those examined -- in
    ``jc-002``, among those first passing the residual-scale resolvability rule. It
    is therefore neither pre-specified nor the largest effect, and calling it "the
    clearest" without saying so invites a reader to treat it as a finding about
    that pair specifically. Naming the selection is the fix; no multiplicity
    adjustment is claimed or implied, and none is applied.
    """
    return (
        f"**Exploratory pair selection.** The leading pair is the one whose "
        f"direction is clearest of the {n_pairs} examined -- chosen after seeing "
        "all of them and on uncertainty, not on effect size, and not "
        "pre-specified. Read the full table rather than this pair alone. No "
        "multiplicity adjustment is applied and none is implied."
    )


def _kf_build_survival(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Discrete-time off-floor hazard model.

    Window-aware (2026-08-21 survival review, finding 1): under the default
    ``treatment_window="randomised"`` the headline tau is the randomised
    first-interval arm contrast and the later intervals are both-arms-treated
    hazards; a stored fit whose plan predates the field was fitted with the
    legacy pooled shift, whose direction beyond interval 1 is prior-mediated —
    the box says so rather than presenting it as data evidence. The ratio word
    follows the link (hazard ratio under cloglog, odds ratio under the logistic
    sensitivity link — finding 4).
    """
    df = _kf_csv(output_dir, "survival_summary.csv")
    if df is None:
        raise _KeyFindingsUnavailable("survival_summary.csv is not present")
    plan = config.get("resolved_run_plan") or {}
    window = str(plan.get("treatment_window", "pooled"))
    link = str(
        plan.get(
            "hazard_link",
            (config.get("extra") or {}).get("hazard_link", "cloglog"),
        )
    )
    ratio_word = "hazard ratio" if link == "cloglog" else "odds ratio"
    effects = df[np.isfinite(pd.to_numeric(df["P(>0)"], errors="coerce"))]
    if effects.empty:
        raise _KeyFindingsUnavailable("survival summary has no directional effects")
    treatment = effects[effects["term"].astype(str).str.startswith("tau")]
    row = (treatment.iloc[0] if not treatment.empty else _kf_most_resolved_row(
        effects, prob_col="P(>0)"
    )).to_dict()
    ratio = np.exp(_kf_float(row["median"]))
    ratio_lo = np.exp(_kf_float(row["ci_low"]))
    ratio_hi = np.exp(_kf_float(row["ci_high"]))
    label = _kf_plain_label(row["term"])
    scope = (
        "in the randomised first interval"
        if window == "randomised"
        else "in an interval"
    )
    if window == "randomised":
        causal_text = (
            "The contrast is a model-based, available-case modified-ITT "
            "assignment contrast in the randomised first interval among children "
            "at the floor at wave 1; later intervals pool both treated arms and "
            "carry no arm contrast. The baseline-subgroup restriction, the "
            "observed-wave-2 requirement, mean-imputed covariates and the "
            "hazard-model form qualify it, and no causal headline is released "
            "(#631 finding 11)."
        )
    else:
        causal_text = (
            "This pooled coefficient is prognostic, not a randomised effect of "
            "record: only the first interval carries an arm contrast, so its "
            "direction beyond that interval is set by the baseline-hazard priors "
            "rather than by observed comparisons."
        )
    sentences = [
        _kf_sentence(
            f"The {label} corresponded to a {ratio_word} of **{ratio:.2f}** "
            f"(89% credible range {ratio_lo:.2f} to {ratio_hi:.2f}) for coming "
            f"off the floor {scope}.",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["P(>0)"],
                positive_claim="the reported term accompanies earlier movement off the floor",
                negative_claim="the reported term accompanies later movement off the floor",
            ),
            "confidence",
        ),
        _kf_sentence(causal_text, "causal"),
    ]
    terms = df["term"].astype(str)
    untreated = df[terms.str.startswith("baseline off-floor prob") & terms.str.contains(r"\(untreated\)", regex=True)]
    treated_cells = df[terms.str.startswith("off-floor prob") & terms.str.contains("both arms treated")]
    legacy_baseline = df[terms.str.startswith("baseline off-floor prob")]
    if window == "randomised" and not untreated.empty:
        first = _kf_float(untreated.iloc[0]["median"])
        text = (
            f"For an untreated child at mean covariates, the fitted first-interval "
            f"off-floor probability was {_kf_pct(first)}%."
        )
        if not treated_cells.empty:
            values = [_kf_float(v) for v in treated_cells["median"]]
            text += (
                f" With both arms treated, the fitted later-interval probabilities "
                f"ranged from {_kf_pct(min(values))}% to {_kf_pct(max(values))}%."
            )
        sentences.append(_kf_sentence(text, "highlight"))
    elif not legacy_baseline.empty:
        values = [_kf_float(v) for v in legacy_baseline["median"]]
        sentences.append(
            _kf_sentence(
                f"The fitted untreated off-floor probability ranged from "
                f"{_kf_pct(min(values))}% to {_kf_pct(max(values))}% across "
                f"intervals; beyond the first interval those untreated values are "
                f"prior-mediated extrapolations (no untreated children were "
                f"observed there).",
                "highlight",
            )
        )
    return sentences


def _kf_build_block_exposure(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Staggered block-2 active-exposure association."""
    row = _kf_csv_row(output_dir, "block_exposure_summary.csv")
    if row is None:
        raise _KeyFindingsUnavailable("block_exposure_summary.csv is not present")
    off_floor = (config.get("extra") or {}).get("likelihood") == "bernoulli_offfloor"
    scale = 100.0 if off_floor else 1.0
    unit = "percentage points" if off_floor else "items"
    outcome_label = _kf_outcome_label(config)
    return [
        _kf_sentence(
            f"When block-2 teaching was active, {outcome_label} differed by "
            f"**{_kf_float(row['delta_items_median']) * scale:+.1f} {unit}** "
            f"(89% credible range "
            f"{_kf_float(row['delta_items_lo']) * scale:+.1f} to "
            f"{_kf_float(row['delta_items_hi']) * scale:+.1f}).",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["prob_delta_pos"],
                positive_claim="active block-2 teaching accompanies a higher outcome",
                negative_claim="active block-2 teaching accompanies a lower outcome",
            ),
            "confidence",
        ),
        _kf_sentence(
            "Block-2 exposure was not randomised; this is a parallel-trends "
            "association comparing block-2-active with block-1-active periods.",
            "causal",
        ),
    ]


def _kf_build_concurrent(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Per-wave mutually-adjusted same-time associations."""
    df = _kf_csv(output_dir, "concurrent_marginals.csv")
    if df is None:
        raise _KeyFindingsUnavailable("concurrent_marginals.csv is not present")
    converged = df["converged"].astype(str).str.lower().isin({"true", "1"})
    rows = df[
        (df["adjustment"] == "adjusted")
        & (df["scale"] == "+1 SD")
        & converged
    ]
    if rows.empty:
        raise _KeyFindingsUnavailable(
            "no converged adjusted +1 SD concurrent marginals are present"
        )
    # Several wave × predictor rows routinely sit at P(>0) ≈ 1 in this family, so
    # the "most resolved row" has to be decided among ties on a stated basis:
    # rows whose P(>0) agree to the nearest 1 % are tied (2 decimals — an order
    # of magnitude above the Monte-Carlo noise in P at 36 000 draws; 3 decimals
    # still flipped on a 1e-4 difference), and ties go to the family's primary
    # wave first (the first declared wave is the primary fit — the largest
    # sample; the later waves are sub-fits), then to the larger items-scale
    # contrast within that wave. Without this the headline wave flipped between
    # two refits of ``lrp-rlm-ca-001`` (t1 → t2; P(>0) 0.99967 / 0.99958 against
    # 0.99944 / 0.99953) on noise below anything the box reports (2026-08-22
    # adjusted-family review, extension).
    rows = rows.assign(
        _kf_timepoint=pd.to_numeric(rows["timepoint"], errors="coerce"),
        _kf_abs_items=pd.to_numeric(rows["items_median"], errors="coerce").abs(),
    )
    row = _kf_most_resolved_row(
        rows,
        prob_col="prob_pos",
        resolution_decimals=2,
        tie_breakers=(("_kf_timepoint", True), ("_kf_abs_items", False)),
    )
    label = _kf_plain_label(row.get("label", row["term"]))
    if config.get("study_id", "rli") == "rli":
        causal_note = (
            "All concurrent coefficients condition on post-treatment skills and "
            "are descriptive associations, not causal pathways. Any fitted "
            "missingness-indicator coefficients are nuisance subgroup offsets, not "
            "skill effects."
        )
    else:
        causal_note = (
            "All concurrent coefficients condition on same-wave skills in an "
            "observational cohort and are descriptive associations, not causal "
            "pathways. Reading-group coefficients are nuisance adjustment, not "
            "group effects."
        )
    return [
        _kf_sentence(
            f"At t{int(_kf_float(row['timepoint']))}, the clearest adjusted "
            f"same-wave predictor was {label}: +1 SD was associated with "
            f"**{_kf_float(row['items_median']):+.1f} outcome items** (89% "
            f"credible range {_kf_float(row['items_lo']):+.1f} to "
            f"{_kf_float(row['items_hi']):+.1f}).",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["prob_pos"],
                positive_claim="the two same-wave skills tend to be higher together",
                negative_claim="the two same-wave skills tend to move oppositely",
            ),
            "confidence",
        ),
        _kf_sentence(
            causal_note,
            "causal",
        ),
    ]


def _kf_build_long_corr_factor(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Longitudinal latent-domain measurement model, using its items translation."""
    df = _kf_csv(output_dir, "latent_items_slopes.csv")
    if df is None:
        raise _KeyFindingsUnavailable("latent_items_slopes.csv is not present")
    row = _kf_most_resolved_row(df, prob_col="prob_pos")
    predictor = _kf_measure_label(row["predictor_indicator"])
    target = _kf_measure_label(row["target_indicator"])
    # Lead with the median (house standard, 2026-08-21 review, finding 10); a
    # stored pre-fix CSV carries only the mean, so fall back rather than fail.
    point = (
        row["items_per_item_median"]
        if "items_per_item_median" in row
        else row["items_per_item_mean"]
    )
    return [
        _kf_sentence(
            f"At wave {int(_kf_float(row['wave']))}, the clearest translated latent "
            f"coupling linked +1 {predictor} item with "
            f"**{_kf_float(point):+.2f} {target} items** "
            f"(89% credible range {_kf_float(row['items_per_item_lo']):+.2f} "
            f"to {_kf_float(row['items_per_item_hi']):+.2f}).",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["prob_pos"],
                positive_claim="the two latent domains tend to move together",
                negative_claim="the two latent domains tend to move oppositely",
            ),
            "confidence",
        ),
        _kf_sentence(
            "This items-scale slope is a linearised measurement-model "
            "association at the average operating point, not a caused gain.",
            "causal",
        ),
    ]


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


def _kf_jm_interval(row) -> str:
    """One coefficient as median + inner 50% + outer reporting interval.

    The house standard is median with an inner 50% and outer 89% equal-tailed
    interval (METHODS.md), which is also #421's acceptance criterion — the first cut
    of this family reported only the outer interval (#427 review).
    """
    return (
        f"**{_kf_float(row['median']):+.2f}** (50% "
        f"{_kf_float(row['lo50']):+.2f} to {_kf_float(row['hi50']):+.2f}; 89% "
        f"{_kf_float(row['lo']):+.2f} to {_kf_float(row['hi']):+.2f})"
    )


def _kf_jm_wave_series(rows: pd.DataFrame, *, decimals: int = 2) -> str:
    """Every wave's median in wave order, as ``t1 -0.47, t2 -0.17, ...``.

    The whole set, never a selection. The previous builder led with the wave whose
    ``P(> 0)`` sat furthest from 0.5 — a headline chosen after seeing which posterior
    was most extreme, which is exactly the selection a reader would have to discount
    (2026-08-23 follow-up review, finding 1).
    """
    return ", ".join(
        f"{str(r['wave'])} {_kf_float(r['median']):+.{decimals}f}"
        for _, r in rows.iterrows()
    )


def _kf_jm_psense_flags(output_dir, rows: pd.DataFrame) -> list[str]:
    """Flagged power-scaling parameters per published wave, in wave order.

    Each wave is read against **its own** table where one exists, so a diagnosis is
    surfaced beside the result it belongs to rather than borrowed from the wave that
    happens to host the fit-level artefacts.
    """
    terms = ("beta_mech[W]", "beta_mech[N]", "delta_ls_decoding", "rho_outcome")
    flags: list[str] = []
    for _, row in rows.iterrows():
        wave = str(row.get("wave", "")).strip()
        filename = str(row.get("psense_file") or "").strip() or "psense_summary.csv"
        flagged = [
            term
            for term in terms
            if _kf_psense_diagnosis(output_dir, term, filename=filename) is not None
        ]
        if flagged:
            flags.append(f"{wave} ({', '.join(flagged)})" if wave else ", ".join(flagged))
    return flags


def _kf_build_joint_mechanism(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Decoding-specificity contrast — and, in the per-wave levels design, the
    conditional-to-marginal slope ratio — from the bivariate joint model (#421 Tier 3).

    Reads ``joint_mechanism_slopes.csv``, which carries one block of rows per fitted
    wave (``t1``…``t4`` for ``design="levels"``, a single ``stacked`` block for
    ``design="transition"``).

    Three things this deliberately does not do (2026-08-23 follow-up review). It does
    not pick a wave to headline: with several waves the whole set is reported, in wave
    order. It does not call the contrast a decoding-use *signature*: the difference is
    measurement-scale dependent, and unequal loadings on a common general-ability
    factor produce a non-zero contrast with no causal letter-sound route at all. And
    it does not read the conditional/marginal ratio as a mediated share: the ratio is
    unbounded and governed by ``conditional_slope_ratio.csv``.
    """
    df = _kf_csv(output_dir, "joint_mechanism_slopes.csv")
    if df is None:
        raise _KeyFindingsUnavailable("joint_mechanism_slopes.csv is missing")
    for column in (
        "wave", "term", "median", "lo50", "hi50", "lo", "hi", "prob_pos", "converged",
    ):
        if column not in df.columns:
            raise _KeyFindingsUnavailable(
                f"joint_mechanism_slopes.csv has no {column!r} column"
            )
    # A wave whose fit did not converge is published flagged in the CSV but must not
    # enter any number in the box. Since the 2026-08-23 review every published wave is
    # also release-gating, so this filter is a second line rather than the only one.
    converged_rows = df["converged"].astype(str).str.lower().isin({"true", "1"})
    excluded_waves = [str(w) for w in df.loc[~converged_rows, "wave"].drop_duplicates()]
    df = df[converged_rows]
    delta = df[df["term"] == "delta_ls_decoding"]
    if delta.empty:
        raise _KeyFindingsUnavailable(
            "joint_mechanism_slopes.csv has no converged delta_ls_decoding row"
        )
    waves = [str(w) for w in df["wave"].drop_duplicates()]
    per_wave = len(waves) > 1
    # The family's own keys live under ``extra``; the top-level ``design`` is the
    # human-readable study-design string, not the "levels"/"transition" switch.
    extra = config.get("extra") or {}
    design = str(extra.get("design", "levels"))
    contrast = extra.get("contrast") or ("N", "W")
    hi_sym, lo_sym = (str(contrast[0]), str(contrast[1]))
    diagnostics = _kf_csv(output_dir, "joint_mechanism_fit_diagnostics.csv")

    # No lead wave. The retired builder headlined the wave whose ``P(Δ > 0)`` sat
    # furthest from 0.5 — a selection made after seeing every posterior — and
    # labelled that choice exploratory (2026-08-23 joint audit, finding 3). Since
    # #591 every fitted wave receives the same full lifecycle and the whole set is
    # reported in wave order, so there is no selection left to label.
    sentences: list[dict[str, str]] = []
    if per_wave:
        headline = (
            f"Letter-sound knowledge tracks the two reading outcomes differently, "
            f"and by how much depends on the wave: Δ = β(LS→{hi_sym}) − "
            f"β(LS→{lo_sym}) is {_kf_jm_wave_series(delta)} logit per SD at the "
            f"{len(waves)} fitted timepoints. All fitted waves are reported; none is "
            "selected as a headline."
        )
    else:
        headline = (
            f"On this model's scale the identified contrast Δ = β(LS→{hi_sym}) − "
            f"β(LS→{lo_sym}) is {_kf_jm_interval(delta.iloc[0])} logit per SD."
        )
    headline += (
        " Both slopes come from one posterior with an explicit cross-outcome "
        "dependence block, so this is a within-model contrast — not the "
        "product-of-marginals sensitivity that separate fits can only bound."
    )
    # 2026-08-23 joint audit, finding 4: the numbers are right; the construct-level
    # reading is not licensed. The two tests differ in item count, score
    # distribution, discrimination, reliability and floor/ceiling behaviour, and the
    # model puts them on no common latent outcome scale — so one shared ability
    # loading differently on the two tests produces a non-zero slope contrast by
    # itself. This is an operational property of the two scores, not a measure of
    # decoding specificity.
    headline += (
        " Read it as an **operational contrast between two adjusted test-score "
        "associations**, not as construct-level decoding specificity: the two tests "
        "differ in item count, score distribution, discrimination, reliability and "
        "floor/ceiling behaviour, and this model calibrates them to no common latent "
        "outcome scale, so a single shared ability that loads differently on them "
        "would produce a non-zero contrast on its own."
    )
    if design == "levels":
        headline += (
            " This is a **levels** contrast (score at the wave); it is a different "
            "estimand from the transition/ANCOVA contrast the Tier-1 note reports, "
            "and the two need not agree in sign."
        )
    else:
        headline += (
            " It is an ANCOVA association — each outcome's post-level given its own "
            "baseline — not a within-child change effect."
        )
    sentences.append(_kf_sentence(headline, "headline"))

    # The direction, read off the fit, with the interpretation limit attached. A
    # positive contrast is consistent with a decoding route; it does not reject an
    # unobserved common factor, because unequal loadings on one general ability
    # already produce a non-zero difference between two differently scaled outcomes.
    probs = [_kf_float(v) for v in delta["prob_pos"]]
    where = " at every fitted wave" if per_wave else ""
    if all(p > 0.5 for p in probs):
        direction = (
            f"letter sounds track {_kf_measure_label(hi_sym)} more closely than "
            f"{_kf_measure_label(lo_sym)}{where}"
        )
    elif all(p < 0.5 for p in probs):
        direction = (
            f"letter sounds track {_kf_measure_label(lo_sym)} more closely than "
            f"{_kf_measure_label(hi_sym)}{where}"
        )
    else:
        direction = "the contrast does not keep one sign across the fitted waves"
    # A levels-scale reversal has a ready non-causal reading; the ANCOVA design's
    # must not borrow it (2026-08-21 review).
    if any(p < 0.5 for p in probs):
        direction += (
            " — which on the levels scale is what a shared reading-development / "
            "general-ability component would produce and what the 6-item nonword "
            "floor would exaggerate"
            if design == "levels"
            else " — a reversal of the Tier-1 contrast, to be read against the "
            "matched mech-096 / mech-101 pair"
        )
    sentences.append(
        _kf_sentence(
            f"Direction: {direction} (P(Δ > 0) = "
            f"{', '.join(f'{p:.2f}' for p in probs)}). Read it as an adjusted, "
            "measurement-scale-dependent association contrast, not a decoding-use "
            "signature: the two outcomes have different item counts, floors and link "
            "discrimination, and unequal loadings on one unobserved general-ability "
            "factor would produce a non-zero contrast with no causal letter-sound "
            "route at all.",
            "confidence",
        )
    )

    # The ratio: one stability rule, applied in the pipeline and reproduced in
    # ``conditional_slope_ratio.csv``. Reported only where it holds, never as a
    # median classified against 0.5, and never as a mediated share (2026-08-23 joint
    # audit, findings 4 and 10; #591 follow-up review, finding 5).
    share = df[df["term"] == "share_retained"]
    if not share.empty:
        stable_flags = (
            share["share_retained_stable"].astype(str).str.lower().isin({"true", "1"})
            if "share_retained_stable" in share.columns
            else pd.Series(True, index=share.index)
        )
        stable = share[stable_flags]
        unstable_waves = [str(w) for w in share.loc[~stable_flags, "wave"]]
        governance = _kf_csv(output_dir, "conditional_slope_ratio.csv")
        regions = ""
        if governance is not None and "prob_in_unit" in governance.columns:
            usable = governance[
                governance["wave"].astype(str).isin(set(stable["wave"].astype(str)))
            ]
            if not usable.empty:
                regions = (
                    " P(0 ≤ ratio ≤ 1) = "
                    + ", ".join(
                        f"{_kf_float(v):.2f}" for v in usable["prob_in_unit"]
                    )
                    + "."
                )
        if not stable.empty:
            sentences.append(
                _kf_sentence(
                    f"Holding latent {_kf_measure_label(hi_sym)} fixed, the ratio of "
                    f"the adjusted letter-sound → {_kf_measure_label(lo_sym)} "
                    "association to its unconditional value is "
                    f"{_kf_jm_wave_series(stable)}.{regions} It is a **ratio of two "
                    "adjusted associations** — unbounded, not a mediation "
                    "proportion, not a causal path fraction, and not evidence that "
                    "the association runs through a decoding channel. It partials "
                    "the *latent* held-fixed skill rather than an observed score.",
                    "detail",
                )
            )
        reduction = df[df["term"] == "abs_slope_reduction"]
        if not reduction.empty:
            sentences.append(
                _kf_sentence(
                    "On the denominator-free scale, holding latent "
                    f"{_kf_measure_label(hi_sym)} fixed reduces the absolute "
                    f"letter-sound → {_kf_measure_label(lo_sym)} slope by "
                    f"{_kf_jm_wave_series(reduction)} logit per SD. This companion "
                    "is reported whether or not the ratio is stable, because a "
                    "difference has no denominator to blow up.",
                    "detail",
                )
            )
        if unstable_waves:
            sentences.append(
                _kf_sentence(
                    f"The ratio is withheld as unstable at "
                    f"{', '.join(unstable_waves)}: the posterior does not put at "
                    "least 95% of its mass on the unconditional slope, or on the "
                    "held-fixed outcome's residual scale, being away from zero, so "
                    "the ratio is heavy-tailed and its summary would describe the "
                    "draws rather than the quantity. Read the two slopes and their "
                    "absolute difference instead.",
                    "note",
                )
            )

    # The transition design publishes one posterior and writes no per-wave diagnostic
    # table, so it is read against the fit-level power-scaling table.
    psense_rows = (
        diagnostics
        if diagnostics is not None
        else pd.DataFrame([{"wave": "", "psense_file": "psense_summary.csv"}])
    )
    flags = _kf_jm_psense_flags(output_dir, psense_rows)
    if flags:
        sentences.append(
            _kf_sentence(
                "Power-scaling sensitivity is flagged for "
                f"{'; '.join(flags)}. Those posteriors move materially when the "
                "prior or the likelihood is reweighted, so read the affected numbers "
                "as prior-dependent until direct alternative-prior fits resolve them.",
                "detail",
            )
        )

    if excluded_waves:
        sentences.append(
            _kf_sentence(
                f"Wave(s) {', '.join(excluded_waves)} did not meet the convergence "
                "gate; their rows are published flagged in "
                "joint_mechanism_slopes.csv but are excluded from every number "
                "above.",
                "detail",
            )
        )

    rho = df[df["term"] == "rho_outcome"]
    if not rho.empty:
        level = "within-wave residual" if design == "levels" else "between-child"
        sentences.append(
            _kf_sentence(
                f"The {level} correlation between the two outcomes is "
                + (
                    _kf_jm_wave_series(rho)
                    if per_wave
                    else _kf_jm_interval(rho.iloc[0])
                )
                + ". This is the dependence block doing the work: an interval "
                "sitting on zero would mean the joint fit buys little over two "
                "separate ones.",
                "detail",
            )
        )

    slopes = df[df["term"].isin([f"beta_mech[{lo_sym}]", f"beta_mech[{hi_sym}]"])]
    if not per_wave and len(slopes) == 2:
        at = {str(r["term"]): r for _, r in slopes.iterrows()}
        sentences.append(
            _kf_sentence(
                f"The two letter-sound slopes: {_kf_measure_label(hi_sym)} "
                f"{_kf_jm_interval(at[f'beta_mech[{hi_sym}]'])} versus "
                f"{_kf_measure_label(lo_sym)} "
                f"{_kf_jm_interval(at[f'beta_mech[{lo_sym}]'])}, on one commensurate "
                "logit-per-SD scale.",
                "detail",
            )
        )
    return sentences


def _kf_build_pooled_levels(output_dir, config: Mapping) -> list[dict[str, str]]:
    """Key findings for the wave-pooled level family.

    The headline is the *decomposition*, not a single slope: a between-child
    coefficient beside a within-child one, because the whole reason the family
    exists is that a random-intercept model with one exposure coefficient returns
    an uninterpretable blend of the two.
    """
    table = _kf_csv(output_dir, "pooled_levels_summary.csv")
    if table is None:
        raise _KeyFindingsUnavailable("pooled_levels_summary.csv is not present")
    plan = config.get("resolved_run_plan") or {}
    rows = {str(r["term"]): r for _, r in table.iterrows()}
    outcome = _kf_measure_label(plan.get("outcome_symbol"))
    exposure = _kf_measure_label(plan.get("mechanism_symbol"))
    # A raw-score covariate exposure (#553) is read in its own units: the fit
    # records how many raw points one SD of the fitted exposure is.
    extra = config.get("extra") or {}
    sd_raw = extra.get("mechanism_exposure_sd_raw")
    if bool(plan.get("mechanism_is_covariate", False)) and sd_raw is not None:
        try:
            unit = f"1 SD ≈ {float(sd_raw):.1f} raw points"
        except (TypeError, ValueError):
            unit = None
        if unit is not None:
            exposure = (
                f"{exposure[:-1]}; {unit})" if exposure.endswith(")") else f"{exposure} ({unit})"
            )

    sentences: list[dict[str, str]] = []
    between = rows.get("beta_between")
    within = rows.get("beta_within")
    if between is None:
        blended = rows.get("beta_mech")
        if blended is None:
            raise _KeyFindingsUnavailable("no exposure coefficient in the summary")
        sentences.append(
            {
                "text": (
                    f"Pooled across waves, a 1 SD higher {exposure} level goes with a "
                    f"**{_kf_float(blended['median']):+.2f}** logit difference in "
                    f"{outcome} (89% {_kf_float(blended['lo']):+.2f} to "
                    f"{_kf_float(blended['hi']):+.2f}). This fit does not separate the "
                    "between-child from the within-child association."
                ),
                "kind": "headline",
            }
        )
        return sentences

    sentences.append(
        {
            "text": (
                f"**Between children**, those sitting 1 SD higher on {exposure} across "
                f"the study sit **{_kf_float(between['median']):+.2f}** logit higher on "
                f"{outcome} (89% {_kf_float(between['lo']):+.2f} to "
                f"{_kf_float(between['hi']):+.2f}; "
                f"P(> 0) = {_kf_float(between['prob_positive']):.3f})."
            ),
            "kind": "headline",
        }
    )
    if within is not None:
        sentences.append(
            {
                "text": (
                    f"**Within a child**, at the waves where they are 1 SD above their "
                    f"own {exposure} average, {outcome} is "
                    f"**{_kf_float(within['median']):+.2f}** logit above their own "
                    f"average (89% {_kf_float(within['lo']):+.2f} to "
                    f"{_kf_float(within['hi']):+.2f}; "
                    f"P(> 0) = {_kf_float(within['prob_positive']):.3f})."
                ),
                "kind": "confidence",
            }
        )
        sentences.append(
            {
                "text": (
                    "The two are different questions. A large between-child coefficient "
                    "beside a small within-child one places the association in stable "
                    "differences between children rather than in a child's own "
                    "movement — the pattern a shared-cause account predicts."
                ),
                "kind": "highlight",
            }
        )
    skills = [str(sk) for sk in (plan.get("skill_symbols") or [])]
    sentences.append(
        {
            "text": (
                "Exposure and outcome are measured at the same wave, so nothing here "
                "orders them in time. Every term is an adjusted association, not a "
                "causal effect."
                + (
                    " The model also holds fixed the same-wave levels of "
                    + ", ".join(_kf_measure_label(sk) for sk in skills)
                    + " — contemporaneous skills that may themselves be affected by "
                    "the intervention, so their coefficients are associations too."
                    if skills
                    else ""
                )
            ),
            "kind": "causal",
        }
    )
    return sentences


_KF_BUILDERS = {
    "itt": _kf_build_itt,
    "joint": _kf_build_joint,
    "joint_mechanism": _kf_build_joint_mechanism,
    "mechanism": _kf_build_mechanism,
    "mediation": _kf_build_mediation,
    "mediation_multi": _kf_build_mediation,
    "did": _kf_build_did,
    "gain_factors": _kf_build_gain_factors,
    "level_factors": _kf_build_level_factors,
    "aligned": _kf_build_aligned,
    "adjusted": _kf_build_adjusted,
    "corr_factor": _kf_build_corr_factor,
    "dose_response": _kf_build_dose_response,
    "lcsm": _kf_build_lcsm,
    "horseshoe": _kf_build_horseshoe,
    "growth": _kf_build_growth,
    "historical_growth": _kf_build_historical_growth,
    "historical_joint": _kf_build_historical_joint,
    "survival": _kf_build_survival,
    "block_exposure": _kf_build_block_exposure,
    "concurrent": _kf_build_concurrent,
    "long_corr_factor": _kf_build_long_corr_factor,
    "pooled_levels": _kf_build_pooled_levels,
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
