# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the historical joint family."""

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


def _kf_build_historical_joint(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
    """Byrne joint correlated growth: cross-measure coupling headline (#338)."""
    within = _kf_csv(output_dir, "within_measure_correlation_summary.csv")
    if within is not None:
        scales = _kf_csv(output_dir, "within_scale_summary.csv")
        if scales is not None and "pair_resolvable" in within.columns:
            resolvable_pairs = within[within["pair_resolvable"].astype(str).str.lower().isin({"true", "1"})]
            if resolvable_pairs.empty:
                strongest = scales.iloc[pd.to_numeric(scales["prob_above_minimum"], errors="coerce").argmax()]
                threshold = _kf_float(strongest["minimum_resolvable_sd"])
                return [
                    _kf_sentence(
                        "The model did not resolve a within-child correlation: "
                        "no measure pair had both wave-specific residual standard "
                        f"deviations supported above {threshold:.2f} logits.",
                        "headline",
                    ),
                    _kf_sentence(
                        f"The best-resolved residual scale was "
                        f"{_kf_plain_label(strongest['label'])}, median "
                        f"{_kf_float(strongest['median']):.2f} logits (inner 50% "
                        f"range {_kf_float(strongest['lo50']):.2f} to "
                        f"{_kf_float(strongest['hi50']):.2f}; 89% credible range "
                        f"{_kf_float(strongest['lo']):.2f} to "
                        f"{_kf_float(strongest['hi']):.2f}).",
                        "confidence",
                    ),
                    _kf_sentence(
                        "Non-resolution is itself a conclusion under this fit's "
                        "within-scale prior: that prior decides which measures "
                        "clear the threshold, and the registered wider-prior "
                        "sensitivity must be read beside this result before it is "
                        "treated as settled.",
                        "robustness",
                    ),
                    _kf_sentence(
                        "When a residual scale is not distinguishable from "
                        "measurement noise, its correlation is not substantively "
                        "identified. This is a descriptive information limit, not "
                        "evidence that skills are causally unrelated.",
                        "causal",
                    ),
                ]
            within = resolvable_pairs
        row = _kf_most_resolved_row(within, prob_col="prob_pos")
        pair = (
            f"{_kf_plain_label(row.get('label_i', row['measure_i']))} and "
            f"{_kf_plain_label(row.get('label_j', row['measure_j']))}"
        )
        sentences = [
            _kf_sentence(
                f"The clearest within-child coupling was between {pair}: a "
                f"wave-specific latent-logit correlation of "
                f"**{_kf_float(row['median']):+.2f}** (inner 50% range "
                f"{_kf_float(row['lo50']):+.2f} to "
                f"{_kf_float(row['hi50']):+.2f}; 89% credible range "
                f"{_kf_float(row['lo']):+.2f} to "
                f"{_kf_float(row['hi']):+.2f}).",
                "headline",
            ),
            _kf_sentence(
                _kf_association_direction(
                    row["prob_pos"],
                    positive_claim=(
                        "waves above a child's stable level on one measure tend "
                        "also to be above-level waves on the other"
                    ),
                    negative_claim=(
                        "waves above a child's stable level on one measure tend to be below-level waves on the other"
                    ),
                ),
                "confidence",
            ),
        ]
        comparison = _kf_csv(output_dir, "between_within_correlation_comparison.csv")
        if comparison is not None:
            matched = comparison[
                (comparison["measure_i"].astype(str) == str(row["measure_i"]))
                & (comparison["measure_j"].astype(str) == str(row["measure_j"]))
            ]
            if not matched.empty:
                comp = matched.iloc[0]
                sentences.append(
                    _kf_sentence(
                        "For that pair, the within-minus-between correlation was "
                        f"{_kf_float(comp['within_minus_between_median']):+.2f} "
                        f"(89% credible range "
                        f"{_kf_float(comp['within_minus_between_lo']):+.2f} to "
                        f"{_kf_float(comp['within_minus_between_hi']):+.2f}; "
                        f"P(within > between) = "
                        f"{_kf_float(comp['prob_within_gt_between']):.2f}).",
                        "highlight",
                    )
                )
        sentences.append(
            _kf_sentence(
                "This is descriptive within-child co-movement in a historical "
                "cohort - it does not identify direction, a treatment effect or "
                "a mechanism, and the residual scale must pass prior sensitivity.",
                "causal",
            )
        )
        sentences.append(_kf_sentence(_kf_pair_selection_note(len(within)), "note"))
        return sentences

    df = _kf_csv(output_dir, "measure_correlation_summary.csv")
    if df is None:
        raise _KeyFindingsUnavailable("measure_correlation_summary.csv is not present")
    row = _kf_most_resolved_row(df, prob_col="prob_pos")
    pair_note = _kf_pair_selection_note(len(df))
    pair = (
        f"{_kf_plain_label(row.get('label_i', row['measure_i']))} and "
        f"{_kf_plain_label(row.get('label_j', row['measure_j']))}"
    )
    return [
        _kf_sentence(
            f"The clearest between-child coupling was between {pair}: a stable-"
            f"level correlation of **{_kf_float(row['median']):+.2f}** (89% credible "
            f"range {_kf_float(row['lo']):+.2f} to {_kf_float(row['hi']):+.2f}).",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["prob_pos"],
                positive_claim=("children who sit higher on one measure tend to sit higher on the other"),
                negative_claim=("children who sit higher on one measure tend to sit lower on the other"),
            ),
            "confidence",
        ),
        _kf_sentence(
            "This is a descriptive between-child correlation of stable levels in "
            "a historical cohort - it is not causal and does not say that "
            "changing one skill changes another.",
            "causal",
        ),
        _kf_sentence(pair_note, "note"),
    ]


def _kf_pair_selection_note(n_pairs: int) -> str:
    """Label the leading measure pair as an exploratory, uncertainty-based choice.

    2026-08-23 joint audit, lower-priority reporting correction. The lead pair is
    the one whose ``P(rho > 0)`` sits furthest from 0.5 among those examined -- in
    ``jc-002``, among those first passing the residual-scale resolvability rule. It
    is therefore neither pre-specified nor the largest effect, and calling it "the
    clearest" without saying so invites a reader to treat it as a finding about
    that pair specifically. Naming the selection is the fix; no multiplicity
    adjustment is claimed or implied, and none is applied.
    """
    return (
        f"**Exploratory pair selection.** The leading pair is the one whose "
        f"direction is clearest of the {n_pairs} examined -- chosen after seeing "
        "all of them and on uncertainty, not on effect size, and not "
        "pre-specified. Read the full table rather than this pair alone. No "
        "multiplicity adjustment is applied and none is implied."
    )
