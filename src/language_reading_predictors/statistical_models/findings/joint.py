# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the joint family."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Mapping
import numpy as np
import pandas as pd
from dse_research_utils.statistics.evidence import (
    favoured_direction,
)
from language_reading_predictors.statistical_models.findings.common import (
    _KeyFindingsUnavailable,
    _kf_csv,
    _kf_csv_row,
    _kf_direction_words,
    _kf_float,
    _kf_measure_label,
    _kf_most_resolved_row,
    _kf_pct,
    _kf_sentence,
)


def _kf_joint_pp(value: Any) -> str:
    """A proportion-scale effect as signed percentage points (one decimal)."""
    return f"{100.0 * _kf_float(value):+.1f}"


def _kf_joint_optional_text(value: Any) -> str:
    """A contrast-metadata cell as clean text; empty for an absent/NaN cell."""
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _kf_joint_marginal_phrase(
    symbol: str,
    tau_rows: Mapping[str, Mapping | pd.Series] | None,
    marginal_rows: Mapping[str, Mapping | pd.Series],
) -> str:
    """One outcome's marginal effect, preferring the comparable pp scale."""
    row = (tau_rows or {}).get(symbol)
    if row is not None:
        return (
            f"{symbol} {_kf_joint_pp(row['ame_prob_median'])} percentage points "
            f"(89% {_kf_joint_pp(row['ame_prob_lo'])} to "
            f"{_kf_joint_pp(row['ame_prob_hi'])}; "
            f"P(> 0) = {_kf_pct(row['prob_ame_pos'])}%)"
        )
    row = marginal_rows.get(symbol)
    if row is None:
        raise _KeyFindingsUnavailable(f"no joint marginal row for contrast outcome {symbol!r}")
    return (
        f"{symbol} {_kf_float(row['items_median']):+.1f} items "
        f"(89% {_kf_float(row['items_lo']):+.1f} to "
        f"{_kf_float(row['items_hi']):+.1f}; "
        f"P(> 0) = {_kf_pct(row['prob_pos'])}%)"
    )


def _kf_build_joint(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
    """Joint available-case modified ITT: contrast-first when one is declared.

    2026-08-21 joint review, findings 2 and 4. A contrast fit's declared estimand
    (``tau_difference.csv``) leads the box — previously the box reported only the
    two marginal effects, so a model registered to answer "did the intervention
    help taught words *more than* untaught words" headlined the much
    better-resolved marginal instead. The cross-outcome range is stated in
    percentage points (the AME scale built for cross-outcome comparability)
    rather than pooling item units across tests with different denominators,
    falling back to the items scale only for a stored fit without the pp
    columns; and a range including P or B carries the floor-rule /
    response-link qualification the report body already applies.
    """
    df = _kf_csv(output_dir, "joint_treatment_marginal.csv")
    if df is None:
        raise _KeyFindingsUnavailable(
            "joint_treatment_marginal.csv is not present; this fit predates the joint items-scale pushforward"
        )
    required = {"outcome", "items_median", "items_lo", "items_hi", "prob_pos"}
    if not required.issubset(df.columns):
        raise _KeyFindingsUnavailable("joint_treatment_marginal.csv does not have the expected columns")
    marginal_rows = {str(row["outcome"]): row for _, row in df.iterrows()}

    tau = _kf_csv(output_dir, "tau_summary.csv")
    pp_cols = {"outcome", "ame_prob_median", "ame_prob_lo", "ame_prob_hi", "prob_ame_pos"}
    tau_rows = (
        {str(row["outcome"]): row for _, row in tau.iterrows()}
        if tau is not None and pp_cols.issubset(tau.columns)
        else None
    )

    contrast = _kf_csv_row(output_dir, "tau_difference.csv")
    contrast_cols = {
        "contrast",
        "diff_prob_median",
        "diff_prob_lo",
        "diff_prob_hi",
        "diff_prob_lo50",
        "diff_prob_hi50",
        "prob_diff_pos",
    }

    sentences: list[dict[str, str]] = []
    if contrast is not None and contrast_cols.issubset(contrast):
        # The declared two-outcome contrast IS this model's estimand, so it
        # leads; the marginal effects underneath it are droppable context.
        pair_a, _, pair_b = str(contrast["contrast"]).partition("_minus_")
        label = _kf_joint_optional_text(contrast.get("contrast_label")) or str(contrast["contrast"])
        kind_word = _kf_joint_optional_text(contrast.get("contrast_kind")) or "outcome"
        sentences.append(
            _kf_sentence(
                f"The declared {kind_word} contrast — {label} "
                f"(average marginal effect on {pair_a or contrast['contrast']} minus "
                f"{pair_b or contrast['contrast']}) — is "
                f"**{_kf_joint_pp(contrast['diff_prob_median'])} percentage "
                "points** on the proportion-correct scale (50% interval "
                f"{_kf_joint_pp(contrast['diff_prob_lo50'])} to "
                f"{_kf_joint_pp(contrast['diff_prob_hi50'])}; 89% "
                f"{_kf_joint_pp(contrast['diff_prob_lo'])} to "
                f"{_kf_joint_pp(contrast['diff_prob_hi'])}).",
                "headline",
            )
        )
        p_diff = _kf_float(contrast["prob_diff_pos"])
        fav = favoured_direction(p_diff)
        positive = fav["favoured_direction"] == "positive"
        interpretation = _kf_joint_optional_text(
            contrast.get("positive_interpretation" if positive else "negative_interpretation")
        )
        sentences.append(
            _kf_sentence(
                "The posterior probability of a "
                f"{'positive' if positive else 'negative'} difference is "
                f"{_kf_pct(fav['favoured_direction_prob'])}% — "
                f"{fav['favoured_direction_label']} evidence for that direction."
                + (f" {interpretation}" if interpretation else ""),
                "confidence",
            )
        )
        if pair_a and pair_b:
            sentences.append(
                _kf_sentence(
                    "The two marginal intervention effects underneath it: "
                    f"{_kf_joint_marginal_phrase(pair_a, tau_rows, marginal_rows)} "
                    "and "
                    f"{_kf_joint_marginal_phrase(pair_b, tau_rows, marginal_rows)}.",
                    "note",
                )
            )
        transfer_symbol = _kf_joint_optional_text(contrast.get("transfer_outcome"))
        transfer_read = _kf_joint_optional_text(contrast.get("transfer_interpretation"))
        if transfer_symbol:
            sentences.append(
                _kf_sentence(
                    (f"{transfer_read} " if transfer_read else "")
                    + "Here that marginal effect is "
                    + _kf_joint_marginal_phrase(transfer_symbol, tau_rows, marginal_rows)
                    + ".",
                    "transfer",
                )
            )
    else:
        if tau_rows is not None:
            medians = [_kf_float(r["ame_prob_median"]) * 100.0 for r in tau_rows.values()]
            lows = [_kf_float(r["ame_prob_lo"]) * 100.0 for r in tau_rows.values()]
            highs = [_kf_float(r["ame_prob_hi"]) * 100.0 for r in tau_rows.values()]
            sentences.append(
                _kf_sentence(
                    f"Across the {len(tau_rows)} outcomes, the joint available-case "
                    "modified ITT estimates ranged from "
                    f"**{min(medians):+.1f} to {max(medians):+.1f} percentage "
                    "points** on each test's proportion-correct scale; the "
                    f"individual 89% credible ranges extended from "
                    f"{min(lows):+.1f} to {max(highs):+.1f} percentage points "
                    "overall.",
                    "headline",
                )
            )
        else:
            # Stored fit without the pp columns: retain the items-scale range.
            medians = [_kf_float(v) for v in df["items_median"]]
            lows = [_kf_float(v) for v in df["items_lo"]]
            highs = [_kf_float(v) for v in df["items_hi"]]
            sentences.append(
                _kf_sentence(
                    f"Across the {len(df)} outcomes, the joint available-case "
                    "modified ITT estimates ranged from "
                    f"**{min(medians):+.1f} to {max(medians):+.1f} items**; the "
                    f"individual 89% credible ranges extended from "
                    f"{min(lows):+.1f} to {max(highs):+.1f} items overall.",
                    "headline",
                )
            )

        clearest = _kf_most_resolved_row(df, prob_col="prob_pos")
        symbol = str(clearest["outcome"])
        label = _kf_measure_label(symbol)
        direction = _kf_direction_words(clearest["prob_pos"], is_rd=False)
        sentences.append(
            _kf_sentence(
                f"For {label}, the clearest directional result: {direction[0].lower() + direction[1:]}",
                "confidence",
            )
        )

        outcomes_present = {str(v) for v in df["outcome"]}
        qualified = []
        if "P" in outcomes_present:
            qualified.append(
                "P's graded score is a flagged secondary under the suite floor "
                "rule (its headline is the binary off-floor estimand)"
            )
        if "B" in outcomes_present:
            qualified.append(
                "B's ordinary-logit effect is conditional on the mandatory "
                "lrp-rli-itt-008/108 response-link sensitivity"
            )
        if qualified:
            sentences.append(
                _kf_sentence(
                    " and ".join(qualified)
                    + "; the range includes "
                    + ("them" if len(qualified) > 1 else "it")
                    + " for completeness only.",
                    "note",
                )
            )

        if {"delta_items", "prob_benefit_ge_delta"}.issubset(df.columns):
            deltas = df[
                np.isfinite(pd.to_numeric(df["delta_items"], errors="coerce"))
                & np.isfinite(pd.to_numeric(df["prob_benefit_ge_delta"], errors="coerce"))
            ]
            if not deltas.empty:
                probabilities = [_kf_float(v) for v in deltas["prob_benefit_ge_delta"]]
                more_likely_than_not = sum(p >= 0.5 for p in probabilities)
                sentences.append(
                    _kf_sentence(
                        f"Among the {len(deltas)} "
                        f"outcome{'' if len(deltas) == 1 else 's'} with a post-hoc, "
                        f"project-agreed smallest-important difference, "
                        f"{more_likely_than_not} "
                        f"{'was' if more_likely_than_not == 1 else 'were'} "
                        f"more likely than not to reach it; the outcome-specific "
                        f"probabilities ranged from {_kf_pct(min(probabilities))}% to "
                        f"{_kf_pct(max(probabilities))}%.",
                        "rope",
                    )
                )
    sentences.append(
        _kf_sentence(
            "These are available-case modified ITT estimates of randomised-arm "
            "contrasts, not full-randomised-cohort ITT estimates. Their cause-and-effect "
            "reading assumes archive inclusion, "
            "outcome observation and any complete-case restriction do not depend "
            "jointly on arm and potential outcomes; without further missing-data "
            "assumptions they are not effects for all 57 randomised children.",
            "causal",
        )
    )
    return sentences
