# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the historical growth family."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import numpy as np
from dse_research_utils.statistics.evidence import (
    favoured_direction,
)
from language_reading_predictors.statistical_models.findings.common import (
    _KeyFindingsUnavailable,
    _kf_csv,
    _kf_float,
    _kf_most_resolved_row,
    _kf_pct,
    _kf_plain_label,
    _kf_sentence,
)


def _kf_build_historical_growth(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
    """Historical-cohort natural-history reproduction.

    Window-aware (2026-08-21 historical-families review, finding 3). The
    within-group intervals mix complete-case **core** rows with the
    attrition-selected **extension** tail (#338), and the selector's
    ``P(positive)`` metric saturates at 1 for most of them — so ranking alone
    used to headline an extension interval, unflagged, in five of six
    publishable fits. Prefer a core interval when the fit has one, say so when
    the headline is an extension row, and always report the interval's own
    subject count. The between-group contrasts — the estimand the family's prior
    pushforward checks — get their own sentence rather than being filtered out.
    """
    df = _kf_csv(output_dir, "posterior_growth_summary.csv")
    if df is None:
        raise _KeyFindingsUnavailable("posterior_growth_summary.csv is not present")
    labels = df["readgrp_label"].fillna("").astype(str).str.strip()
    within = df[labels.str.len() > 0]
    contrasts = df[labels.str.len() == 0]
    if within.empty:
        within = df
        contrasts = df.iloc[0:0]
    # Prefer the audited core window; fall back to the extension tail only when
    # the fit supports no core interval at all.
    core = within[within["window"].astype(str) == "core"] if "window" in within.columns else within
    candidates = core if not core.empty else within
    row = _kf_most_resolved_row(candidates, prob_col="p_gt_0")
    group = _kf_plain_label(row.get("readgrp_label", "historical cohort"))
    window = str(row.get("window", "")).strip()
    n_subjects = row.get("n_subjects")
    try:
        n_text = f", {int(_kf_float(n_subjects))} children"
    except _KeyFindingsUnavailable, TypeError, ValueError:
        n_text = ""
    window_text = (
        " This interval is on the attrition-selected follow-up extension, not "
        "the audited complete-case core, so it describes the children who "
        "remained in the study."
        if window == "extension"
        else ""
    )
    fav = favoured_direction(_kf_float(row["p_gt_0"]))
    positive = fav["favoured_direction"] == "positive"
    direction = "positive" if positive else "negative"
    claim = "scores tend to increase over that interval" if positive else "scores tend to decrease over that interval"
    sentences = [
        _kf_sentence(
            f"For the {group} group, {_kf_plain_label(row['label'])} was "
            f"**{_kf_float(row['mean']):+.1f} items** (89% credible range "
            f"{_kf_float(row['q_lo']):+.1f} to "
            f"{_kf_float(row['q_hi']):+.1f}{n_text}).{window_text}",
            "headline",
        ),
        _kf_sentence(
            f"The posterior probability that this growth is {direction} is "
            f"{_kf_pct(fav['favoured_direction_prob'])}% — "
            f"{fav['favoured_direction_label']} evidence that {claim}.",
            "confidence",
        ),
    ]
    if not contrasts.empty:
        contrast = _kf_most_resolved_row(contrasts, prob_col="p_gt_0")
        c_fav = favoured_direction(_kf_float(contrast["p_gt_0"]))
        sentences.append(
            _kf_sentence(
                f"Comparing groups over the window every group supports, "
                f"{_kf_plain_label(contrast['label'])} was "
                f"**{_kf_float(contrast['mean']):+.1f} items** (89% credible "
                f"range {_kf_float(contrast['q_lo']):+.1f} to "
                f"{_kf_float(contrast['q_hi']):+.1f}; "
                f"{c_fav['favoured_direction_label']} evidence it is "
                f"{c_fav['favoured_direction']}).",
                "highlight",
            )
        )
    sentences.append(
        _kf_sentence(
            "This is descriptive natural-history growth in a historical cohort, "
            "not an intervention effect or an explanation of group differences.",
            "causal",
        )
    )
    cells = _kf_csv(output_dir, "posterior_cell_summary.csv")
    if cells is not None and "posterior_mean_minus_observed_mean" in cells.columns:
        # The published audit is the complete-case core (Table 2); an extension
        # cell was never in it, so it must not set the reproduction figure.
        audit = cells[cells["window"].astype(str) == "core"] if "window" in cells.columns else cells
        gaps = [abs(_kf_float(v)) for v in audit["posterior_mean_minus_observed_mean"] if np.isfinite(_kf_float(v))]
        if gaps:
            sentences.append(
                _kf_sentence(
                    f"As a reproduction check on the complete-case core window, "
                    f"the largest fitted-minus-observed cell mean gap was "
                    f"{max(gaps):.1f} items.",
                    "highlight",
                )
            )
    return sentences
