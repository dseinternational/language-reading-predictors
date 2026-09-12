# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the horseshoe family."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from language_reading_predictors.statistical_models.findings.common import (
    _KeyFindingsUnavailable,
    _kf_csv,
    _kf_float,
    _kf_pct,
    _kf_plain_label,
    _kf_sentence,
)


def _kf_build_horseshoe(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
    """Regularised-horseshoe predictor-ranking sensitivity analysis."""
    df = _kf_csv(output_dir, "predictor_ranking.csv")
    if df is None:
        raise _KeyFindingsUnavailable("predictor_ranking.csv is not present")
    row = df.sort_values("rank").iloc[0].to_dict()
    label = _kf_plain_label(row["predictor"])
    direction = "positive" if _kf_float(row["beta_median"]) >= 0 else "negative"
    return [
        _kf_sentence(
            f"The top-ranked predictor was {label}, with a standardised "
            f"{direction} association of **{_kf_float(row['beta_median']):+.2f} "
            f"logit units** (89% highest-density interval "
            f"{_kf_float(row['beta_hdi_lo']):+.2f} to "
            f"{_kf_float(row['beta_hdi_hi']):+.2f}).",
            "headline",
        ),
        _kf_sentence(
            f"Its probability of exceeding the model's worth-noticing "
            f"coefficient threshold was {_kf_pct(row['p_abs_gt_delta'])}%.",
            "confidence",
        ),
        _kf_sentence(
            "The ranking is an adjusted predictive sensitivity check, not a list "
            "of causal drivers; closely ranked predictors should not be treated as "
            "meaningfully ordered.",
            "causal",
        ),
    ]
