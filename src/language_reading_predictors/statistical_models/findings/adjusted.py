# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the adjusted family."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from language_reading_predictors.statistical_models.findings.common import (
    _KeyFindingsUnavailable,
    _kf_association_direction,
    _kf_csv,
    _kf_float,
    _kf_most_resolved_row,
    _kf_outcome_label,
    _kf_plain_label,
    _kf_sentence,
)


def _kf_build_adjusted(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
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
            raise _KeyFindingsUnavailable("predicted_gain_words.csv carries only missing-indicator rows")
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
