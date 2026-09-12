# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the long corr factor family."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from language_reading_predictors.statistical_models.findings.common import (
    _KeyFindingsUnavailable,
    _kf_association_direction,
    _kf_csv,
    _kf_float,
    _kf_measure_label,
    _kf_most_resolved_row,
    _kf_sentence,
)


def _kf_build_long_corr_factor(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
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
