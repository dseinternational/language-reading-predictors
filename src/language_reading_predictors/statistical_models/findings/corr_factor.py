# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the corr factor family."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from language_reading_predictors.statistical_models.findings.common import (
    _KeyFindingsUnavailable,
    _kf_association_direction,
    _kf_csv,
    _kf_float,
    _kf_most_resolved_row,
    _kf_plain_label,
    _kf_sentence,
)


def _kf_build_corr_factor(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
    """Cross-sectional correlated-domain measurement model."""
    correlations = _kf_csv(output_dir, "factor_correlation_summary.csv")
    structural = _kf_csv(output_dir, "structural_summary.csv")
    if correlations is None and structural is None:
        raise _KeyFindingsUnavailable("neither factor_correlation_summary.csv nor structural_summary.csv is present")
    sentences: list[dict[str, str]] = []
    if correlations is not None:
        row = _kf_most_resolved_row(correlations, prob_col="prob_pos")
        pair = f"{_kf_plain_label(row['domain_i'])} and {_kf_plain_label(row['domain_j'])}"
        sentences.extend(
            [
                _kf_sentence(
                    f"The clearest latent-domain correlation was between {pair}: "
                    f"**{_kf_float(row['median']):+.2f}** (89% credible range "
                    f"{_kf_float(row['lo']):+.2f} to "
                    f"{_kf_float(row['hi']):+.2f}).",
                    "headline",
                ),
                _kf_sentence(
                    _kf_association_direction(
                        row["prob_pos"],
                        positive_claim="the two latent skill areas tend to move together",
                        negative_claim="the two latent skill areas tend to move oppositely",
                    ),
                    "confidence",
                ),
            ]
        )
    if structural is not None:
        # Only the beta_<domain> factor slopes are structural slopes. Ranking over
        # every row let the beta_age adjustment covariate win the highlight in all
        # four released RLI boxes — displacing beta_code, the errors-in-variables
        # focal slope, in mm-002/102 (2026-08-21 review, finding 2b). A config
        # without factor names in its plan (a legacy stub) keeps the unfiltered
        # ranking rather than failing.
        plan = config.get("resolved_run_plan") or {}
        factors = list(plan.get("structural_factors") or []) or [domain[0] for domain in (plan.get("domains") or [])]
        slopes = structural
        if factors:
            wanted = {f"beta_{name}" for name in factors}
            slopes = structural[structural["coefficient"].astype(str).isin(wanted)]
            if slopes.empty:
                raise _KeyFindingsUnavailable(
                    "structural_summary.csv has no factor-slope rows matching the resolved plan's structural factors"
                )
        row = _kf_most_resolved_row(slopes, prob_col="prob_pos")
        sentences.append(
            _kf_sentence(
                f"The clearest structural slope was "
                f"{_kf_plain_label(row['coefficient'])}: "
                f"{_kf_float(row['median']):+.2f} logit units (89% credible range "
                f"{_kf_float(row['lo']):+.2f} to "
                f"{_kf_float(row['hi']):+.2f}).",
                "highlight",
            )
        )
        if correlations is None:
            sentences.append(
                _kf_sentence(
                    _kf_association_direction(
                        row["prob_pos"],
                        positive_claim="the linked latent quantities tend to move together",
                        negative_claim="the linked latent quantities tend to move oppositely",
                    ),
                    "confidence",
                )
            )
    sentences.append(
        _kf_sentence(
            "This is a measurement and triangulation model; its factor "
            "correlations and structural slopes are associations, not causal effects.",
            "causal",
        )
    )
    return sentences
