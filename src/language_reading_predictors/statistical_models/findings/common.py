# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the common family."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os
from collections.abc import Mapping, Sequence
import numpy as np
import pandas as pd
from dse_research_utils.statistics.evidence import (
    favoured_direction,
)


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


def _kf_float(value: Any) -> float:
    """Return ``value`` as a finite float, else raise (the no-``nan`` guard)."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise _KeyFindingsUnavailable(f"non-numeric value {value!r}") from exc
    if not np.isfinite(v):
        raise _KeyFindingsUnavailable(f"non-finite value {value!r}")
    return v


def _kf_pct(prob: Any) -> str:
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


def _kf_csv_row(output_dir: str | Path, name: str) -> dict | None:
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
    output_dir: str | Path, term: str, *, filename: str = "psense_summary.csv"
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


def _kf_csv(output_dir: str | Path, name: str) -> pd.DataFrame | None:
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


def _kf_plain_label(value: Any) -> str:
    """Make an artefact identifier readable without inventing a construct name."""
    return str(value).replace("_", " ").strip()


def _kf_dag_unit(value: Any) -> str:
    """Readable exposure unit with a leading measure symbol mapped to its DAG
    symbol (#374): e.g. ``'L items'`` -> ``'LS items'``. Leaves units that do not
    begin with a mapped modelling symbol unchanged."""
    from language_reading_predictors.statistical_models.measures import DAG_SYMBOL

    text = _kf_plain_label(value)
    head, sep, tail = text.partition(" ")
    return f"{DAG_SYMBOL.get(head, head)}{sep}{tail}"


def _kf_measure_label(symbol: Any) -> str:
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
    prob_pos: Any,
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
            study_measure = study_measures.get(symbol)
            if study_measure is not None:
                return study_measure.label
        except (KeyError, TypeError):
            measure = None
    if measure is not None:
        return measure.label
    return config.get("title") or symbol or "the outcome"


def _kf_direction_words(
    prob_pos: Any, *, is_rd: bool, rd_event: str = "coming off the floor"
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


def _kf_has_factor_term(output_dir: str | Path, term: str) -> bool:
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


def _kf_strongest_factor(output_dir: str | Path, *, exclude_roles: tuple[str, ...] = ("causal",)) -> str | None:
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


#: Human labels for the moderation-variant treatment-interaction coefficients.
_KF_MODERATION_LABELS: dict[str, str] = {
    "gamma_int_trt_ability": "general cognitive ability (block design)",
    "gamma_int_trt_own": "the child's starting point on this measure",
}


#: Display labels for the covariate moderators a mechanism fit may declare
#: (measures take their registered label).
_KF_COVARIATE_MODERATOR_LABELS = {
    "A": "age",
    "erbto": "phonological memory (word/nonword repetition)",
}
