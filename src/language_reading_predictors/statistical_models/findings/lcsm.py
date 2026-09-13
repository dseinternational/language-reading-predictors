# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the lcsm family."""

from __future__ import annotations

from pathlib import Path
import re
from collections.abc import Mapping
from language_reading_predictors.statistical_models.findings.common import (
    _KeyFindingsUnavailable,
    _kf_association_direction,
    _kf_csv,
    _kf_float,
    _kf_measure_label,
    _kf_most_resolved_row,
    _kf_plain_label,
    _kf_sentence,
)


def _kf_build_lcsm(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
    """Latent change-score couplings, with an optional randomised-window check."""
    df = _kf_csv(output_dir, "coupling_summary.csv")
    if df is None:
        raise _KeyFindingsUnavailable("coupling_summary.csv is not present")
    # Only the g_/h_ rows are couplings. A contains("->") filter also matched the
    # age slope and the shared adjuster slopes, so a precision covariate could win
    # the "clearest longitudinal coupling" headline with a level-worded confidence
    # sentence (2026-08-21 review, finding 2a) — live in the released 067 box.
    directed = df[df["coefficient"].astype(str).str.match(r"[gh]_")]
    if directed.empty:
        raise _KeyFindingsUnavailable("coupling_summary.csv has no coupling rows")
    row = _kf_most_resolved_row(directed, prob_col="prob_pos")
    name = str(row["coefficient"])
    lagged = name.startswith("h_")
    label = _kf_plain_label(name)
    if "(" in label and label.endswith(")"):
        label = label.split("(", 1)[1][:-1]
    sentences = [
        _kf_sentence(
            f"The clearest longitudinal coupling was {label}: "
            f"**{_kf_float(row['median']):+.2f} logit units** (89% credible range "
            f"{_kf_float(row['lo']):+.2f} to {_kf_float(row['hi']):+.2f}).",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["prob_pos"],
                positive_claim=(
                    "greater earlier change accompanies greater later change"
                    if lagged
                    else "a higher earlier level accompanies greater later change"
                ),
                negative_claim=(
                    "greater earlier change accompanies less later change"
                    if lagged
                    else "a higher earlier level accompanies less later change"
                ),
            ),
            "confidence",
        ),
        _kf_sentence(
            "The couplings are conditional predictive associations among latent "
            "trajectories, not causal skill-to-skill effects.",
            "causal",
        ),
    ]
    itt = _kf_csv(output_dir, "itt_window1_contrast.csv")
    if itt is not None:
        # Quote the model's focal outcome when its row exists, and always name the
        # measure — the unnamed most-resolved row silently attributed another
        # outcome's contrast to the focal measure (finding 2c: 081 quoted W under
        # a taught-vocabulary model, 091 quoted L under a word-reading model).
        focal = str(config.get("outcome_symbol") or "")
        cand = itt[itt["coefficient"].astype(str).str.startswith(f"itt_w1[{focal}]")] if focal else itt.iloc[0:0]
        check = cand.iloc[0] if len(cand) else _kf_most_resolved_row(itt, prob_col="prob_pos")
        match = re.search(r"itt_w1\[([^\]]+)\]", str(check["coefficient"]))
        measure = _kf_measure_label(match.group(1)) if match else str(check["coefficient"])
        sentences.append(
            _kf_sentence(
                f"The separate randomised window-1 consistency contrast for "
                f"{measure} was {_kf_float(check['median']):+.2f} latent-logit "
                f"units (89% credible range {_kf_float(check['lo']):+.2f} to "
                f"{_kf_float(check['hi']):+.2f}); it is a check, not the coupling "
                f"headline.",
                "highlight",
            )
        )
    return sentences
