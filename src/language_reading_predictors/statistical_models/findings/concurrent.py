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
