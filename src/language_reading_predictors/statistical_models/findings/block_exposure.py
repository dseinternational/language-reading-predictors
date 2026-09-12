# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the block exposure family."""

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
)


def _kf_build_block_exposure(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
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
