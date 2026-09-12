# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the aligned family."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from language_reading_predictors.statistical_models.findings.common import (
    _KeyFindingsUnavailable,
    _kf_association_direction,
    _kf_csv_row,
    _kf_float,
    _kf_outcome_label,
    _kf_sentence,
    _kf_strongest_factor,
)


def _kf_build_aligned(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
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
