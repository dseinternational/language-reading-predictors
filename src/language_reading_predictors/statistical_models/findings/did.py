# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the did family."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from language_reading_predictors.statistical_models.findings.common import (
    _KeyFindingsUnavailable,
    _kf_csv_row,
    _kf_direction_words,
    _kf_float,
    _kf_outcome_label,
    _kf_sentence,
)


def _kf_build_did(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
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
