# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the concurrent family."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import pandas as pd
from language_reading_predictors.statistical_models.findings.common import (
    _KeyFindingsUnavailable,
    _kf_association_direction,
    _kf_csv,
    _kf_float,
    _kf_most_resolved_row,
    _kf_plain_label,
    _kf_sentence,
)


def _kf_build_concurrent(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
    """Per-wave mutually-adjusted same-time associations."""
    df = _kf_csv(output_dir, "concurrent_marginals.csv")
    if df is None:
        raise _KeyFindingsUnavailable("concurrent_marginals.csv is not present")
    converged = df["converged"].astype(str).str.lower().isin({"true", "1"})
    rows = df[(df["adjustment"] == "adjusted") & (df["scale"] == "+1 SD") & converged]
    if rows.empty:
        raise _KeyFindingsUnavailable("no converged adjusted +1 SD concurrent marginals are present")
    # Round direction resolution to 1% to avoid selecting on tiny differences
    # near P(>0) = 1. Break ties by the earliest wave, then the largest absolute
    # items contrast. The earliest wave need not be the pipeline's anchor fit.
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
