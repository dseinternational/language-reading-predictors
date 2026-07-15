# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Post-hoc floor rule for heavily-floored ITT outcomes (issues #119, #341).

Two RCT-phase outcomes — phonetic spelling (``P``) and nonword reading (``N``) —
are heavily floored at t2: most children still score zero, so a graded
Beta-Binomial treatment effect is leveraged by a handful of dispersed tail values
rather than by the arm contrast. This reanalysis chose the rule after inspecting
the outcome distributions and earlier model results, so it is **post-hoc and
data-adaptive**, not pre-registered or prospectively specified. Applying the gate
arm-blind avoids using treatment labels in the mechanical classification, but
does not turn the choice into prospective pre-specification. An outcome with at
least :data:`FLOOR_THRESHOLD` of its post-scores at zero (computed arm-blind)

1. drops the (degenerate) own-baseline precision term and uses an age-only
   predictor ``alpha + tau*G + gamma_A*A_std``; and
2. reports the exploratory binary transition estimand
   ``Pr(post > 0 | observed pre == 0)`` via a Bernoulli/logistic ``tau`` among
   eligible children, retaining the graded Beta-Binomial ``tau`` only as a
   flagged, detection-limited secondary.

Because eligibility is defined by an observed pre-randomisation score, the arm
contrast remains causal for that observed subgroup. Missing baseline-floor status
is excluded and must be reported explicitly by arm.

See ``notes/202606251124-lrpitt-floored-outcomes-nonword-spelling.md`` and the
"Floored outcomes" section of issue #119.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.measures import MEASURES
from language_reading_predictors.statistical_models.preprocessing import PreparedData

FLOOR_THRESHOLD: float = 0.40
"""An outcome is "floored" if at least this fraction of its post-scores are zero
at t2 (a post-hoc, data-adaptive, arm-blind rule in this reanalysis)."""


def proportion_at_zero(prepared: PreparedData, symbol: str) -> float:
    """Fraction of (non-missing) post-scores equal to zero for ``symbol``.

    Computed **arm-blind** (pooling both groups) over ``prepared.post_counts``.
    For ``phase_mode="itt"`` these are the t2 post-scores. Returns ``nan`` if
    there are no non-missing post-scores.
    """
    if symbol not in prepared.post_counts:
        raise KeyError(f"{symbol!r} not in prepared.post_counts")
    post = np.asarray(prepared.post_counts[symbol], dtype=float)
    finite = post[np.isfinite(post)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite == 0.0))


def is_floored(
    prepared: PreparedData,
    symbol: str,
    threshold: float = FLOOR_THRESHOLD,
) -> bool:
    """Apply the arm-blind floor gate: ``proportion_at_zero >= threshold``."""
    p0 = proportion_at_zero(prepared, symbol)
    return bool(np.isfinite(p0) and p0 >= threshold)


def baseline_floor_eligibility_by_arm(
    prepared: PreparedData,
    symbol: str,
) -> pd.DataFrame:
    """Count observed-baseline floor eligibility separately by randomised arm.

    The headline-within-reanalysis transition estimand requires both an observed
    post-score and an observed baseline score of zero. The returned table makes
    the otherwise silent exclusion of missing baseline-floor status auditable.
    Counts are taken from the supplied prepared frame; under the registered ITT
    loader this is the outcome-available t2 analysis frame.
    """
    if symbol not in prepared.pre_logit:
        raise ValueError(
            f"cannot classify baseline-floor eligibility for {symbol!r}: its "
            "pre-score is not loaded"
        )
    if symbol not in prepared.post_counts:
        raise KeyError(f"{symbol!r} not in prepared.post_counts")

    pre = np.asarray(prepared.pre_logit[symbol], dtype=float)
    post = np.asarray(prepared.post_counts[symbol], dtype=float)
    group = np.asarray(prepared.G, dtype=int)
    if not (len(pre) == len(post) == len(group)):
        raise ValueError("pre, post and group arrays must have the same length")

    n_trials = prepared.n_trials[symbol]
    floor_logit = float(np.log(0.5 / (n_trials + 0.5)))
    pre_observed = np.isfinite(pre)
    post_observed = np.isfinite(post)
    at_floor = pre_observed & np.isclose(pre, floor_logit)
    above_floor = pre_observed & ~at_floor

    rows: list[dict[str, str | int]] = []
    for g, label in ((1, "intervention"), (0, "control")):
        arm = group == g
        post_available = arm & post_observed
        rows.append(
            {
                "arm": label,
                "n_loaded": int(arm.sum()),
                "n_post_observed": int(post_available.sum()),
                "n_pre_observed": int((post_available & pre_observed).sum()),
                "n_pre_missing": int((post_available & ~pre_observed).sum()),
                "n_pre_floor": int((post_available & at_floor).sum()),
                "n_pre_above_floor": int((post_available & above_floor).sum()),
                "n_exploratory_eligible": int((post_available & at_floor).sum()),
            }
        )
    return pd.DataFrame(rows)


def baseline_floor_status_bounds(
    prepared: PreparedData,
    symbol: str,
) -> pd.DataFrame:
    """Bound the raw off-floor risk difference over unknown floor eligibility.

    Children with an observed t2 score but missing t1 score could either have
    started at the floor (and belong in the transition estimand) or above it. The
    lower bound assigns unknown baseline-zero outcomes to the intervention
    subgroup and unknown baseline-positive outcomes to the control subgroup; the
    upper bound reverses those adverse choices. Unknown children whose assignment
    would move a rate in the opposite direction are left ineligible. This
    enumerates the extrema over all possible baseline-floor classifications while
    keeping their observed t2 off-floor status fixed.

    The result is a transparent sensitivity for missing *eligibility*, not a
    posterior interval and not a correction for the three randomised children
    absent from the repository.
    """

    if symbol not in prepared.pre_logit or symbol not in prepared.post_counts:
        raise KeyError(f"{symbol!r} requires loaded pre and post scores")

    pre = np.asarray(prepared.pre_logit[symbol], dtype=float)
    post = np.asarray(prepared.post_counts[symbol], dtype=float)
    group = np.asarray(prepared.G, dtype=int)
    n_trials = int(prepared.n_trials[symbol])
    floor_logit = float(np.log(0.5 / (n_trials + 0.5)))
    post_observed = np.isfinite(post)
    known_eligible = np.isfinite(pre) & np.isclose(pre, floor_logit) & post_observed
    unknown = ~np.isfinite(pre) & post_observed
    event = post > 0

    arm: dict[int, dict[str, float | int]] = {}
    for g in (1, 0):
        known = known_eligible & (group == g)
        unknown_arm = unknown & (group == g)
        known_n = int(known.sum())
        known_events = int((known & event).sum())
        unknown_events = int((unknown_arm & event).sum())
        unknown_zeros = int((unknown_arm & ~event).sum())
        if known_n == 0:
            raise ValueError(
                f"Cannot bound {symbol!r}: arm G={g} has no known baseline-floor rows"
            )
        observed_rate = known_events / known_n
        min_rate = known_events / (known_n + unknown_zeros)
        max_rate = (known_events + unknown_events) / (known_n + unknown_events)
        arm[g] = {
            "known_eligible_n": known_n,
            "known_events_n": known_events,
            "unknown_eligibility_n": unknown_events + unknown_zeros,
            "unknown_events_n": unknown_events,
            "unknown_zeros_n": unknown_zeros,
            "observed_rate": observed_rate,
            "min_rate": min_rate,
            "max_rate": max_rate,
        }

    intervention = arm[1]
    control = arm[0]
    return pd.DataFrame(
        [
            {
                "outcome": symbol,
                "scale": "off_floor_risk_difference",
                "observed_known_eligibility_difference": (
                    float(intervention["observed_rate"])
                    - float(control["observed_rate"])
                ),
                "eligibility_status_lower": (
                    float(intervention["min_rate"]) - float(control["max_rate"])
                ),
                "eligibility_status_upper": (
                    float(intervention["max_rate"]) - float(control["min_rate"])
                ),
                "intervention_known_eligible_n": int(
                    intervention["known_eligible_n"]
                ),
                "control_known_eligible_n": int(control["known_eligible_n"]),
                "intervention_unknown_eligibility_n": int(
                    intervention["unknown_eligibility_n"]
                ),
                "control_unknown_eligibility_n": int(
                    control["unknown_eligibility_n"]
                ),
                "interpretation": (
                    "Raw risk-difference extrema over all classifications of "
                    "observed-post children with missing baseline-floor status; "
                    "not a posterior interval."
                ),
            }
        ]
    )


def _validated_raw_transition_frame(
    prepared: PreparedData,
    symbol: str,
) -> pd.DataFrame:
    """Load validated t1/t2 scores for every child in the archived CSV.

    The fitted :class:`PreparedData` has already removed rows with a missing t2
    outcome, which is exactly the information a missingness bound must recover.
    This helper therefore returns to ``prepared.data_path`` and uses the central
    variable/measure mappings rather than duplicating column names. Missing scores
    are allowed; malformed group codes and non-integer or out-of-range observed
    scores fail loudly before any extrema are calculated.
    """

    if symbol not in MEASURES:
        raise KeyError(f"Unknown bounded-score outcome {symbol!r}")
    if not prepared.data_path:
        raise ValueError(
            "PreparedData.data_path is required for the floor-transition "
            "missingness bound"
        )

    path = Path(prepared.data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Floor-transition source data not found: {path}")
    data = pd.read_csv(path)
    measure = MEASURES[symbol]
    required = {V.SUBJECT_ID, V.TIME, V.GROUP, measure.column}
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise ValueError(
            "Floor-transition source data are missing required column(s): "
            f"{missing_columns}"
        )
    if data[V.SUBJECT_ID].isna().any():
        raise ValueError("Floor-transition source data contain a missing subject_id")

    raw_group = pd.to_numeric(data[V.GROUP], errors="coerce")
    invalid_group = data[V.GROUP].notna() & raw_group.isna()
    valid_group = raw_group.notna() & raw_group.isin((1.0, 2.0))
    if invalid_group.any() or not valid_group.all():
        invalid = data.loc[~valid_group, V.GROUP].drop_duplicates().tolist()
        raise ValueError(
            "Group codes must be exactly 1 (immediate intervention) or 2 "
            f"(wait-list control); found invalid raw value(s) {invalid}"
        )

    time = pd.to_numeric(data[V.TIME], errors="coerce")
    if (data[V.TIME].notna() & time.isna()).any():
        invalid = data.loc[time.isna(), V.TIME].drop_duplicates().tolist()
        raise ValueError(f"Time contains non-numeric value(s) {invalid}")
    transition = data.loc[time.isin((1.0, 2.0))].copy()
    transition["_time"] = time.loc[transition.index].astype(int)
    if transition.duplicated([V.SUBJECT_ID, "_time"]).any():
        duplicate = transition.loc[
            transition.duplicated([V.SUBJECT_ID, "_time"], keep=False),
            [V.SUBJECT_ID, "_time"],
        ].drop_duplicates()
        raise ValueError(
            "Floor-transition source data contain duplicate child/time rows: "
            f"{duplicate.to_dict(orient='records')}"
        )

    raw_score = transition[measure.column]
    score = pd.to_numeric(raw_score, errors="coerce")
    non_numeric = raw_score.notna() & score.isna()
    if non_numeric.any():
        invalid = raw_score.loc[non_numeric].drop_duplicates().tolist()
        raise ValueError(
            f"Measure {symbol!r} ({measure.column}) contains non-numeric "
            f"value(s) {invalid}"
        )
    finite = score.dropna().to_numpy(dtype=float)
    if finite.size and np.any(finite != np.rint(finite)):
        invalid = np.unique(finite[finite != np.rint(finite)]).tolist()
        raise ValueError(
            f"Measure {symbol!r} ({measure.column}) must contain integer counts; "
            f"found fractional value(s) {invalid}"
        )
    if finite.size and (finite.min() < 0 or finite.max() > measure.n_trials):
        raise ValueError(
            f"Measure {symbol!r} ({measure.column}) must lie in "
            f"[0, {measure.n_trials}]; found range "
            f"[{finite.min():g}, {finite.max():g}]"
        )
    transition["_score"] = score

    group_by_child = pd.DataFrame(
        {
            V.SUBJECT_ID: data[V.SUBJECT_ID],
            "_group": raw_group.astype(int),
        }
    )
    if (group_by_child.groupby(V.SUBJECT_ID)["_group"].nunique() != 1).any():
        raise ValueError("Group assignment changes across waves for at least one child")
    group_by_child = group_by_child.drop_duplicates(V.SUBJECT_ID).set_index(V.SUBJECT_ID)

    wide = transition.pivot(
        index=V.SUBJECT_ID,
        columns="_time",
        values="_score",
    ).reindex(group_by_child.index)
    wide = wide.reindex(columns=[1, 2])
    return pd.DataFrame(
        {
            "subject_id": group_by_child.index.astype(str),
            "G": (2 - group_by_child["_group"].to_numpy(dtype=int)).astype(int),
            "pre": wide[1].to_numpy(dtype=float),
            "post": wide[2].to_numpy(dtype=float),
        }
    )


def _transition_components(frame: pd.DataFrame, g: int) -> dict[str, int]:
    """Classify the observed and unknown pieces of one arm's transition risk."""

    arm = frame.loc[frame["G"] == g]
    pre = arm["pre"].to_numpy(dtype=float)
    post = arm["post"].to_numpy(dtype=float)
    pre_observed = np.isfinite(pre)
    post_observed = np.isfinite(post)
    known_eligible = pre_observed & (pre == 0)
    fixed = known_eligible & post_observed
    unknown_eligibility_observed = ~pre_observed & post_observed
    return {
        "archive_n": int(len(arm)),
        "fixed_eligible_observed_n": int(fixed.sum()),
        "fixed_events_n": int((fixed & (post > 0)).sum()),
        "known_eligible_missing_post_n": int((known_eligible & ~post_observed).sum()),
        "unknown_eligibility_observed_event_n": int(
            (unknown_eligibility_observed & (post > 0)).sum()
        ),
        "unknown_eligibility_observed_zero_n": int(
            (unknown_eligibility_observed & (post == 0)).sum()
        ),
        "unknown_eligibility_missing_post_n": int(
            ((~pre_observed) & (~post_observed)).sum()
        ),
        "known_ineligible_n": int((pre_observed & (pre > 0)).sum()),
    }


def _arm_transition_extrema(
    components: Mapping[str, int],
    *,
    absent_randomised_n: int,
) -> tuple[float, float]:
    """Sharp risk extrema under arbitrary missing eligibility/outcome values."""

    fixed_n = int(components["fixed_eligible_observed_n"])
    fixed_events = int(components["fixed_events_n"])
    if fixed_n == 0:
        raise ValueError("No fixed observed baseline-floor denominator in one trial arm")
    known_missing = int(components["known_eligible_missing_post_n"])
    unknown_events = int(components["unknown_eligibility_observed_event_n"])
    unknown_zeros = int(components["unknown_eligibility_observed_zero_n"])
    unknown_joint = (
        int(components["unknown_eligibility_missing_post_n"])
        + absent_randomised_n
    )

    # Minimum: missing outcomes for known-eligible children are zero; unknown
    # observed zeros and all jointly unknown children are eligible zeros; unknown
    # observed events are ineligible. Maximum reverses each admissible choice.
    minimum = fixed_events / (fixed_n + known_missing + unknown_zeros + unknown_joint)
    maximum = (
        fixed_events + known_missing + unknown_events + unknown_joint
    ) / (fixed_n + known_missing + unknown_events + unknown_joint)
    return float(minimum), float(maximum)


def binary_transition_missingness_bounds(
    prepared: PreparedData,
    symbol: str,
    *,
    randomised_by_g: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    """Bound the same-estimand off-floor risk difference under all missing values.

    Two sharp, assumption-free identification intervals are returned:

    ``archived_dataset``
        Completes missing t1 eligibility and/or t2 event status among every child
        in the archived CSV. A known baseline zero is always eligible; a missing
        baseline may be zero (eligible) or positive (ineligible).

    ``full_randomised_population``
        Adds the 57-to-54 absent participants. Their baseline eligibility and t2
        event are both allowed to take whichever values minimise or maximise the
        arm-specific transition risk.

    In both scopes the estimand remains the intervention-minus-control difference
    in ``Pr(t2 > 0 | t1 = 0)``. The endpoints can correspond to different completed
    eligibility sets, as an identification bound should. They are not posterior
    intervals and make no missing-at-random assumption.
    """

    if randomised_by_g is None:
        from language_reading_predictors.statistical_models.itt_audit import (
            RLI_RANDOMISED_BY_G,
        )

        randomised_by_g = RLI_RANDOMISED_BY_G
    if set(randomised_by_g) != {0, 1}:
        raise ValueError("randomised_by_g must provide exactly the G=0 and G=1 arms")

    frame = _validated_raw_transition_frame(prepared, symbol)
    components = {g: _transition_components(frame, g) for g in (1, 0)}
    absent: dict[int, int] = {}
    for g in (1, 0):
        randomised_n = int(randomised_by_g[g])
        archive_n = components[g]["archive_n"]
        if randomised_n < archive_n:
            raise ValueError(
                f"Randomised G={g} count {randomised_n} is below archived count "
                f"{archive_n}"
            )
        absent[g] = randomised_n - archive_n

    for g in (1, 0):
        if components[g]["fixed_eligible_observed_n"] == 0:
            raise ValueError(
                f"No observed baseline-floor child with observed t2 in arm G={g}"
            )
    observed_rd = (
        components[1]["fixed_events_n"]
        / components[1]["fixed_eligible_observed_n"]
        - components[0]["fixed_events_n"]
        / components[0]["fixed_eligible_observed_n"]
    )
    rows = []
    for scope, include_absent in (
        ("archived_dataset", False),
        ("full_randomised_population", True),
    ):
        arm_bounds = {
            g: _arm_transition_extrema(
                components[g],
                absent_randomised_n=absent[g] if include_absent else 0,
            )
            for g in (1, 0)
        }
        row: dict[str, str | int | float] = {
            "outcome": symbol,
            "scope": scope,
            "scale": "off_floor_risk_difference",
            "observed_complete_case_difference": float(observed_rd),
            "intervention_min_risk": arm_bounds[1][0],
            "intervention_max_risk": arm_bounds[1][1],
            "control_min_risk": arm_bounds[0][0],
            "control_max_risk": arm_bounds[0][1],
            "risk_difference_lower": arm_bounds[1][0] - arm_bounds[0][1],
            "risk_difference_upper": arm_bounds[1][1] - arm_bounds[0][0],
        }
        for g, label in ((1, "intervention"), (0, "control")):
            for key, value in components[g].items():
                row[f"{label}_{key}"] = value
            row[f"{label}_absent_randomised_n"] = absent[g]
            row[f"{label}_scope_n"] = (
                int(randomised_by_g[g])
                if include_absent
                else components[g]["archive_n"]
            )
        row["interpretation"] = (
            "Sharp extrema for the binary transition risk difference over missing "
            "baseline-floor eligibility and t2 event status"
            + (
                ", including participants absent from the archive."
                if include_absent
                else " within the archived dataset."
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)
