# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Joint dependence pairing and its measured consequence.

Whether a factorised joint contrast has a release-ready LKJ companion bound
beside it, what the dependence model does to the declared average-marginal-
effect difference, and the historical-joint prior-sensitivity qualification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd
from language_reading_predictors.statistical_models.release.base import (
    RELEASE_DECISION_FILENAME,
    _HISTORICAL_JOINT_PRIOR_BINDING,
    _HISTORICAL_JOINT_PRIOR_SENSITIVE,
    _JOINT_PAIR_BINDING,
    _config_name,
    _finite,
    _load_config,
    _plan,
    _read_csv,
    _read_json,
)
from language_reading_predictors.statistical_models.release.robustness import (
    _AME_CORRELATION_NOISE,
    _CONTRAST_DIRECTION_SHIFT,
)

def _dependence_identification_note(output_dir: Path) -> str:
    """Qualifier when a fitted dependence block never moved off its prior.

    A companion that switches the LKJ residual block on estimates the within-child
    covariance the parent's factorised interval omits. That covariance is the
    *data's* only if the correlation posterior is distinguishable from the
    correlation prior. For the three registered two-outcome companions
    at n = 53 it is not: posterior-to-prior SD ratios of 1.002, 1.008 and 1.001
    (2026-08-22 ITT audit, finding 3). The interval such a fit publishes is the
    LKJ prior's implied correction, and a reader is entitled to be told so beside
    the number rather than having to reconstruct it.

    A note, never a withhold. The fit is valid and its residual SDs *are*
    informed; what is qualified is the interpretation of the correlation. Silent
    when ``dependence_identification.csv`` is absent (every fit without the block,
    and any stored fit written before the table existed), so old decisions
    re-decide identically.
    """
    frame = _read_csv(output_dir, "dependence_identification.csv")
    if frame is None or frame.empty or "verdict" not in frame.columns:
        return ""
    correlations = frame.loc[frame["role"].astype(str) == "residual correlation"]
    if correlations.empty:
        return ""
    dominated = correlations.loc[
        correlations["verdict"].astype(str) == "prior-dominated"
    ]
    if dominated.empty:
        return ""
    names = ", ".join(str(v) for v in dominated["parameter"])
    return (
        "The within-child residual correlation did not move off its prior "
        f"({names}), so the dependence correction this fit applies to the "
        "contrast's interval is the prior's rather than the data's; read the "
        "interval as a prior-informed sensitivity, not as a measured "
        "within-child covariance."
    )


def _required_dependence_companion(config: Mapping[str, Any]) -> str:
    """The companion this fit must be read beside, or ``""``.

    Derived from the **registered** pairing constant first and the stored plan
    second. Deriving it from the stored plan alone left the qualifier dormant on
    every artefact written before ``dependence_companion`` existed — which is all
    three current parent fits (2026-08-23 joint audit, finding 2) — so a stale
    stored plan could bypass a policy the registered module declares. This mirrors
    how the phoneme-blending gate derives its requirement from
    ``BLENDING_LINK_MODELS`` rather than from what a fit happened to record.
    """
    from language_reading_predictors.statistical_models.joint import (
        JOINT_DEPENDENCE_COMPANIONS,
    )

    model_id = str(config.get("model_id") or "")
    registered = JOINT_DEPENDENCE_COMPANIONS.get(model_id, "")
    if registered:
        return registered
    contrast = _plan(config).get("contrast")
    if not isinstance(contrast, Mapping):
        return ""
    return str(contrast.get("dependence_companion") or "")


def _joint_marginal_widths(
    directory: Path, outcomes: tuple[str, str]
) -> dict[str, float] | None:
    """Each contrast outcome's probability-scale AME interval width, or ``None``."""
    frame = _read_csv(directory, "tau_summary.csv")
    if frame is None or frame.empty or "outcome" not in frame.columns:
        return None
    needed = ("ame_prob_lo", "ame_prob_hi")
    if any(column not in frame.columns for column in needed):
        return None
    indexed = frame.set_index(frame["outcome"].astype(str))
    widths: dict[str, float] = {}
    for outcome in outcomes:
        if outcome not in indexed.index:
            return None
        row = indexed.loc[outcome]
        lo, hi = _finite(row["ame_prob_lo"]), _finite(row["ame_prob_hi"])
        if lo is None or hi is None or hi <= lo:
            return None
        widths[outcome] = hi - lo
    return widths


def _joint_width_channels(
    *,
    parent_dir: Path,
    companion_dir: Path,
    outcomes: tuple[str, str],
    parent_width: float,
    companion_width: float,
) -> dict[str, Any]:
    """Split the contrast's width change into marginal and covariance channels.

    2026-08-24 review of the joint audit. Finding 2 asked that the dependence block
    be assessed through its consequence for the declared contrast, which
    :func:`_joint_contrast_consequence` does for the contrast's *location*. But the
    reason three report templates give for running the companion at all is about its
    *width*: that a factorised interval omits within-child cross-outcome covariance,
    so a positive residual correlation leaves it too wide and a negative one too
    narrow. That sign rule describes the covariance term
    ``Var(A - B) = V_A + V_B - 2 Cov(A, B)`` in isolation. It does not describe what
    separates these two fits, because the companion also adds a per-child
    logistic-normal layer whose own parameter uncertainty widens *both* marginals.

    So measure which channel the change came through instead of asserting one. Each
    fit's implied cross-outcome posterior correlation follows from the same identity
    read on equal-tailed interval widths,
    ``r = (W_A^2 + W_B^2 - W_diff^2) / (2 W_A W_B)``; the parent's is structurally
    zero because a factorised fit shares no parameter between outcomes, so its
    measured value is this approximation's own noise floor and is recorded beside
    the companion's for exactly that purpose. ``marginal`` is what the companion's
    wider marginals alone would do at the parent's correlation, and ``covariance``
    is the remainder.

    Returns the record fields, or a ``channel_status`` explaining why the split
    could not be taken. Never raises: this is descriptive provenance attached to a
    release decision, not a gate.
    """
    parent_widths = _joint_marginal_widths(parent_dir, outcomes)
    companion_widths = _joint_marginal_widths(companion_dir, outcomes)
    if parent_widths is None or companion_widths is None:
        return {
            "channel_status": "unavailable",
            "channel_reason": "tau_summary.csv is missing the per-outcome AME interval",
        }
    left, right = outcomes

    def _implied(widths: Mapping[str, float], diff_width: float) -> float | None:
        a, b = widths[left], widths[right]
        value = (a * a + b * b - diff_width * diff_width) / (2 * a * b)
        return value if -1.0 <= value <= 1.0 else None

    parent_r = _implied(parent_widths, parent_width)
    companion_r = _implied(companion_widths, companion_width)
    if parent_r is None or companion_r is None:
        return {
            "channel_status": "unavailable",
            "channel_reason": (
                "the interval widths imply a correlation outside [-1, 1], so the "
                "Gaussian width identity does not describe these posteriors"
            ),
        }
    a, b = companion_widths[left], companion_widths[right]
    marginal_only = float(np.sqrt(max(a * a + b * b - 2 * parent_r * a * b, 0.0)))
    marginal_channel = marginal_only - parent_width
    covariance_channel = companion_width - marginal_only
    moved = abs(marginal_channel) + abs(covariance_channel)
    correlation_change = companion_r - parent_r
    if abs(correlation_change) <= _AME_CORRELATION_NOISE:
        dominant = "marginal_uncertainty"
    elif abs(covariance_channel) > abs(marginal_channel):
        dominant = "cross_outcome_covariance"
    else:
        dominant = "marginal_uncertainty"
    return {
        "channel_status": "measured",
        "parent_marginal_widths": {k: float(v) for k, v in parent_widths.items()},
        "companion_marginal_widths": {
            k: float(v) for k, v in companion_widths.items()
        },
        "parent_implied_ame_correlation": float(parent_r),
        "companion_implied_ame_correlation": float(companion_r),
        "implied_ame_correlation_change": float(correlation_change),
        "marginal_width_channel": float(marginal_channel),
        "covariance_width_channel": float(covariance_channel),
        "covariance_channel_share": (
            float(abs(covariance_channel) / moved) if moved else None
        ),
        "dominant_width_channel": dominant,
    }


def _joint_contrast_consequence(
    parent_dir: Path, companion_dir: Path, *, pair: tuple[str, str] | None = None
) -> tuple[dict[str, Any], str]:
    """Measure what the dependence model does to the **declared contrast**.

    Finding 2's second half. The robustness gate classifies power-scaling rows for
    the conditional-logit ``tau`` vector; clean marginal ``tau`` diagnoses say
    nothing about a nonlinear difference of standardised average marginal effects,
    which is the quantity the findings box actually reports. And requiring every
    nuisance correlation in the LKJ block to be sharply identified is the wrong
    test — at n = 53 it never will be, and it need not be for the contrast to be
    stable. So assess the block *through its consequence for the contrast*: read
    both fits' ``tau_difference.csv`` and compare the declared quantity directly.

    Returns the machine-readable record and a qualifier sentence, empty when the
    dependence model leaves the contrast's conclusion where it was. "Material" is
    a direction flip in the median, or a shift in P(> 0) of at least
    :data:`_CONTRAST_DIRECTION_SHIFT` — deliberately a conclusion-level rule, not
    a threshold on the interval, whose movement *is* the companion's purpose.
    """
    record: dict[str, Any] = {}

    def _unusable(status: str, reason: str) -> tuple[dict[str, Any], str]:
        """The comparison could not be taken, so say so rather than publishing silence.

        Fail-closed, matching the binding checks above: an absent or unreadable
        comparison is not evidence that the dependence model left the contrast
        alone. Without a note the reader sees an unqualified release and has no
        way to tell "checked and unchanged" from "never checked".
        """
        record["status"] = status
        record["reason"] = reason
        return record, (
            "The dependence model's consequence for the declared contrast could "
            f"not be measured ({reason}), so the paired contrast is "
            "dependence-unchecked in substance even though the companion is bound "
            "beside it. Regenerate this decision once both fits carry a readable "
            "contrast summary."
        )

    parent = _read_csv(parent_dir, "tau_difference.csv")
    companion = _read_csv(companion_dir, "tau_difference.csv")
    if parent is None or companion is None or parent.empty or companion.empty:
        return _unusable("unavailable", "one or both fits have no tau_difference.csv")
    p, c = parent.iloc[0], companion.iloc[0]
    if str(p.get("contrast")) != str(c.get("contrast")):
        return _unusable(
            "mismatched",
            f"parent reports {p.get('contrast')!r} and companion "
            f"{c.get('contrast')!r}",
        )
    needed = ("diff_prob_median", "diff_prob_lo", "diff_prob_hi", "prob_diff_pos")
    if any(col not in parent.columns or col not in companion.columns for col in needed):
        return _unusable(
            "unavailable", "tau_difference.csv is missing the contrast columns"
        )
    values = {name: (_finite(p[name]), _finite(c[name])) for name in needed}
    if any(v[0] is None or v[1] is None for v in values.values()):
        return _unusable(
            "unavailable", "tau_difference.csv holds non-finite contrast values"
        )
    p_med, c_med = values["diff_prob_median"]
    p_pos, c_pos = values["prob_diff_pos"]
    p_width = values["diff_prob_hi"][0] - values["diff_prob_lo"][0]
    c_width = values["diff_prob_hi"][1] - values["diff_prob_lo"][1]
    direction_shift = abs(c_pos - p_pos)
    flipped = (p_med > 0) != (c_med > 0)
    record.update(
        {
            "status": "compared",
            "contrast": str(p.get("contrast")),
            "scale": str(p.get("headline_scale") or ""),
            "parent_median": p_med,
            "companion_median": c_med,
            "median_shift": c_med - p_med,
            "parent_prob_positive": p_pos,
            "companion_prob_positive": c_pos,
            "direction_probability_shift": direction_shift,
            "parent_interval_width": p_width,
            "companion_interval_width": c_width,
            "interval_width_ratio": (c_width / p_width) if p_width else None,
            "direction_flipped": bool(flipped),
            "material": bool(flipped or direction_shift >= _CONTRAST_DIRECTION_SHIFT),
        }
    )
    if pair is not None and all(pair):
        record.update(
            _joint_width_channels(
                parent_dir=parent_dir,
                companion_dir=companion_dir,
                outcomes=pair,
                parent_width=p_width,
                companion_width=c_width,
            )
        )
    else:
        record["channel_status"] = "unavailable"
        record["channel_reason"] = "the resolved plan does not name the contrast pair"
    if not record["material"]:
        return record, ""
    cause = (
        "reverses the sign of the contrast median"
        if flipped
        else f"moves P(> 0) by {direction_shift:.2f}"
    )
    return record, (
        "The dependence model materially changes the declared contrast: the LKJ "
        f"companion {cause} (parent P(> 0) = {p_pos:.2f}, companion "
        f"{c_pos:.2f}). Read the paired conclusion from the companion, not from "
        "this fit's factorised interval alone."
    )


def _historical_joint_prior_sensitivity(output_dir: Path) -> str:
    """The measured prior sensitivity of ``sigma_within``, as a phrase or ``""``.

    The qualification below is about a prior whose influence the fit has already
    measured, so quote the measurement rather than asserting that the prior matters.
    """
    frame = _read_csv(output_dir, "psense_summary.csv", index_col=0)
    if frame is None or frame.empty or "prior" not in frame.columns:
        return ""
    rows = frame.loc[frame.index.astype(str).str.startswith("sigma_within")]
    values = pd.to_numeric(rows["prior"], errors="coerce").dropna()
    if values.empty:
        return ""
    top = float(values.max())
    if top < _HISTORICAL_JOINT_PRIOR_SENSITIVE:
        return (
            f" This fit's own power scaling puts the largest sigma_within prior "
            f"sensitivity at {top:.2f}, below ArviZ's "
            f"{_HISTORICAL_JOINT_PRIOR_SENSITIVE:.2f} flag."
        )
    return (
        f" This fit's own power scaling already flags that prior: the largest "
        f"sigma_within prior sensitivity is {top:.2f}, against ArviZ's "
        f"{_HISTORICAL_JOINT_PRIOR_SENSITIVE:.2f} flag threshold."
    )


def _historical_joint_prior_companion_qualifications(
    output_dir: Path, config: Mapping[str, Any]
) -> list[str]:
    """Qualify a within-child historical-joint fit whose prior sensitivity is absent.

    2026-08-23 joint audit, finding 5, completing what #609 registered. The family
    is descriptive, so :func:`gate_applies` excludes it and no robustness verdict is
    produced for it at all — which left the parent publishing a **prior-dependent
    classification** with nothing machine-readable saying so. Which measures clear
    the 0.05-logit resolvability threshold, and therefore which correlations may be
    interpreted, is decided by ``sigma_within``, whose prior the registered
    companion varies; on the stored fit that parameter is also the most
    power-scaling-sensitive quantity in the model.

    A **qualification, never a withhold**: the fit is valid, its convergence gate
    passes and its tables are correct under the declared prior. What is qualified is
    the robustness of the classification those tables carry. Fail-closed on every
    unreadable or unbound path, and silent for a fit the constant does not pair or
    that has no within-child block (so a stored ``jc-001`` decision is untouched).
    """
    from language_reading_predictors.statistical_models.historical_joint import (
        HISTORICAL_JOINT_PRIOR_COMPANIONS,
    )

    if str(config.get("kind") or "") != "historical_joint":
        return []
    model_id = str(config.get("model_id") or "")
    companion = HISTORICAL_JOINT_PRIOR_COMPANIONS.get(model_id, "")
    if not companion or not bool(_plan(config).get("within_correlation")):
        return []
    measured = _historical_joint_prior_sensitivity(output_dir)

    def _note(reason: str) -> list[str]:
        return [
            f"the registered within-scale prior sensitivity ({companion}) is not "
            f"release-ready beside this fit ({reason}), so which measures clear the "
            "resolvability threshold — and therefore which correlations may be read "
            f"at all — is a conclusion under this fit's prior alone.{measured}"
        ]

    try:
        directory = Path(output_dir).resolve()
        config_name = str(config.get("config_name") or "") or _config_name(
            directory, model_id
        )
        if not config_name:
            return _note("this fit's configuration name could not be resolved")
        companion_dir = directory.parent / f"{companion}-{config_name}"
        decision, decision_error = _read_json(
            companion_dir / RELEASE_DECISION_FILENAME
        )
        if decision_error is not None or not isinstance(decision, Mapping):
            return _note("it has not been fitted, or its release decision is unreadable")
        if not bool(decision.get("publishable")):
            return _note("its own release decision withholds publication")
        companion_config = _load_config(companion_dir)
        if not companion_config:
            return _note("its config.json is missing or unreadable")
        if str(companion_config.get("model_id") or "") != companion:
            return _note("the sibling directory does not identify itself as the companion")
        ours = _plan(config).get("sigma_within_prior_sigma")
        theirs = _plan(companion_config).get("sigma_within_prior_sigma")
        if ours is None or theirs is None:
            return _note("the within-scale prior is not recorded on both fits")
        if ours == theirs:
            return _note(
                "it was fitted under the same within-scale prior, so it varies "
                "nothing"
            )
        for description, reader in _HISTORICAL_JOINT_PRIOR_BINDING:
            mine, yours = reader(config), reader(companion_config)
            if mine is None or yours is None:
                return _note(f"{description} is not recorded on both fits")
            if mine != yours:
                return _note(f"{description} differs between the two fits")
        changed = _historical_joint_resolvability_change(directory, companion_dir)
    except Exception as exc:  # noqa: BLE001 - a check that cannot run must fail closed
        return _note(f"it could not be verified: {exc}")
    if changed:
        return [
            f"the within-scale prior sensitivity ({companion}) changes the "
            f"resolvability classification ({changed}), so the interpretable set of "
            "correlations here depends on that prior rather than on the data."
        ]
    return []


def _historical_joint_resolvability_change(
    parent_dir: Path, companion_dir: Path
) -> str:
    """Which measures the wider prior reclassifies, as a phrase or ``""``.

    The classification *is* the conclusion for this family, so comparing it across
    the two independently sampled fits is the comparison that matters — not pairing
    draws, which are unrelated between chains fitted under different priors.
    """
    parent = _read_csv(parent_dir, "within_scale_summary.csv")
    companion = _read_csv(companion_dir, "within_scale_summary.csv")
    if parent is None or companion is None:
        return "one of the two fits has no within_scale_summary.csv"
    needed = {"measure", "resolvable"}
    if not needed <= set(parent.columns) or not needed <= set(companion.columns):
        return "within_scale_summary.csv does not record the classification"

    def _flags(frame: pd.DataFrame) -> dict[str, bool]:
        return {
            str(row["measure"]): str(row["resolvable"]).strip().lower()
            in {"true", "1"}
            for _, row in frame.iterrows()
        }

    mine, theirs = _flags(parent), _flags(companion)
    if set(mine) != set(theirs):
        return "the two fits classify different measure sets"
    moved = sorted(name for name in mine if mine[name] != theirs[name])
    if not moved:
        return ""
    return ", ".join(
        f"{name}: {'resolvable' if mine[name] else 'unresolvable'} here, "
        f"{'resolvable' if theirs[name] else 'unresolvable'} under the wider prior"
        for name in moved
    )


def _joint_dependence_companion_note(
    output_dir: Path, config: Mapping[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    """Qualifying note when a factorised joint contrast's dependence companion is
    not release-ready **and bound** beside it (2026-08-21 joint review, finding 3;
    binding and contrast consequence added by the 2026-08-23 joint audit, finding 2).

    The three contrast parents' ``dependence_note`` prose has always said the
    contrast is dependence-checked only once the registered LKJ companion
    (lrp-rli-itt-215/315/216, #551) has passed the house gate. Verifying that the
    companion is publishable is necessary but not sufficient: a *different*
    companion fit — other outcomes, the reversed contrast, other rows, other
    sampling settings, another commit — would satisfy it just as well. So the pair
    is now bound field by field through :data:`_JOINT_PAIR_BINDING`, and the
    dependence block is assessed through its consequence for the declared contrast
    rather than through whether every nuisance correlation is sharply identified.

    Deliberately a **qualify-note**, not a withhold: the parent's per-outcome
    marginal effects are fully valid without the companion — only the paired
    contrast's interval is dependence-unchecked — so the failure attaches the
    caveat sentence to the findings box rather than withholding valid marginals.
    During a fresh sweep a parent can finalise before its companion has been
    fitted; the note then attaches and is cleared by regenerating the decision
    (``scripts/regenerate_key_findings.py``) once the companion completes.
    Fail-closed: any error verifying the companion, and any binding field that
    cannot be read on both sides, attaches the note with the reason rather than
    silently releasing an unchecked pairing.

    Returns ``(note, contrast_record)``; the record is the machine-readable
    contrast comparison persisted in ``release_decision.json``.
    """
    if str(config.get("kind") or "") != "joint":
        return "", None
    companion = _required_dependence_companion(config)
    if not companion or bool(_plan(config).get("use_residual_correlation")):
        return "", None

    def _note(reason: str) -> str:
        return (
            f"The declared contrast's dependence-model companion ({companion}) is "
            f"not release-ready beside this fit ({reason}), so the paired contrast "
            "is dependence-unchecked: its interval omits within-child cross-outcome "
            "covariance and is not automatically conservative. Regenerate this "
            "decision once the companion has passed the house gate."
        )

    try:
        directory = Path(output_dir).resolve()
        model_id = str(config.get("model_id") or "")
        config_name = str(config.get("config_name") or "") or _config_name(
            directory, model_id
        )
        if not config_name:
            return _note("this fit's configuration name could not be resolved"), None
        companion_dir = directory.parent / f"{companion}-{config_name}"
        decision, decision_error = _read_json(
            companion_dir / RELEASE_DECISION_FILENAME
        )
        if decision_error is not None or not isinstance(decision, Mapping):
            return _note("its release decision is missing or unreadable"), None
        if not bool(decision.get("publishable")):
            return _note("its own release decision withholds publication"), None
        companion_config = _load_config(companion_dir)
        if not companion_config:
            return _note("its config.json is missing or unreadable"), None
        if str(companion_config.get("model_id") or "") != companion:
            return (
                _note("the sibling directory does not identify itself as the companion"),
                None,
            )
        if not bool(_plan(companion_config).get("use_residual_correlation")):
            return (
                _note("it is not a residual-correlated fit, so it is not a dependence model"),
                None,
            )
        for description, reader in _JOINT_PAIR_BINDING:
            ours, theirs = reader(config), reader(companion_config)
            if ours is None or theirs is None:
                return _note(f"{description} is not recorded on both fits"), None
            if ours != theirs:
                return _note(f"{description} differs between the two fits"), None
    except Exception as exc:  # noqa: BLE001 - a gate that cannot run must fail closed
        return _note(f"the companion could not be verified: {exc}"), None

    declared = _plan(config).get("contrast")
    pair: tuple[str, str] | None = None
    if isinstance(declared, Mapping):
        left, right = str(declared.get("left") or ""), str(declared.get("right") or "")
        pair = (left, right) if left and right else None
    contrast_record, contrast_note = _joint_contrast_consequence(
        directory, companion_dir, pair=pair
    )
    contrast_record["companion"] = companion
    return contrast_note, contrast_record
