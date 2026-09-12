# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain-language findings for the growth family."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
import pandas as pd
from language_reading_predictors.statistical_models.findings.common import (
    _KeyFindingsUnavailable,
    _kf_association_direction,
    _kf_csv,
    _kf_float,
    _kf_measure_label,
    _kf_most_resolved_row,
    _kf_sentence,
)


def _kf_growth_interaction_sentences(gamma_int: pd.DataFrame, gamma: pd.DataFrame) -> list[dict[str, str]]:
    """Key-findings box for the age x ability interaction growth model (LRP85)."""
    row = _kf_most_resolved_row(gamma_int, prob_col="prob_positive")
    outcome = _kf_measure_label(row["outcome"])
    sentences = [
        _kf_sentence(
            f"For {outcome}, the clearest interaction result, a child +1 SD older "
            f"at entry **and** +1 SD higher in baseline non-verbal ability differed "
            f"in growth rate by **{_kf_float(row['median']):+.2f} logit units** "
            f"beyond the two main effects (89% credible range "
            f"{_kf_float(row['lo89']):+.2f} to {_kf_float(row['hi89']):+.2f}).",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["prob_positive"],
                positive_claim=("older-and-more-able children progress faster than the main effects alone imply"),
                negative_claim=("the ability-growth association weakens with age at entry"),
            ),
            "confidence",
        ),
    ]
    same = gamma[gamma["outcome"] == row["outcome"]]
    if not same.empty:
        g = same.iloc[0]
        sentences.append(
            _kf_sentence(
                f"The ability main effect (gamma) for the same outcome, at the "
                f"sample-mean entry age, was {_kf_float(g['median']):+.2f} logit "
                f"units (89% credible range {_kf_float(g['lo89']):+.2f} to "
                f"{_kf_float(g['hi89']):+.2f}).",
                "highlight",
            )
        )
    sentences.append(
        _kf_sentence(
            "These trajectory coefficients are adjusted associations, not effects of changing non-verbal ability.",
            "causal",
        )
    )
    return sentences


def _kf_build_growth(output_dir: str | Path, config: Mapping) -> list[dict[str, str]]:
    """Multivariate growth: baseline ability association with growth rate."""
    df = _kf_csv(output_dir, "growth_association_summary.csv")
    if df is None:
        raise _KeyFindingsUnavailable("growth_association_summary.csv is not present")
    gamma = df[df["coefficient"] == "gamma"]
    if gamma.empty:
        raise _KeyFindingsUnavailable("growth summary has no gamma rows")
    plan = config.get("resolved_run_plan") or {}
    # The interaction model's registered headline is gamma_int, not gamma
    # (2026-08-21 review, finding 1); a plan that declares the interaction but a
    # summary without its rows is a stale pre-fix artefact, so fail loud.
    if bool(plan.get("age_ability_interaction")):
        gamma_int = df[df["coefficient"] == "gamma_int"]
        if gamma_int.empty:
            raise _KeyFindingsUnavailable(
                "the plan declares the age x ability interaction but the growth "
                "summary has no gamma_int rows; regenerate the summary CSV first"
            )
        return _kf_growth_interaction_sentences(gamma_int, gamma)
    row = _kf_most_resolved_row(gamma, prob_col="prob_positive")
    outcome = _kf_measure_label(row["outcome"])
    study_id = str(config.get("study_id") or "rli")
    baseline_symbol = plan.get("baseline_covariate")
    baseline_label = "non-verbal ability"
    if study_id != "rli" and isinstance(baseline_symbol, str):
        try:
            from language_reading_predictors.statistical_models.datasets import (
                resolve_dataset,
            )

            _dataset, catalogue = resolve_dataset(study_id)
            baseline_label = catalogue[baseline_symbol].label
            if str(row["outcome"]) in catalogue:
                outcome = catalogue[str(row["outcome"])].label
        except KeyError, TypeError:
            baseline_label = baseline_symbol
    return [
        _kf_sentence(
            f"For {outcome}, the clearest result, a 1-SD higher baseline "
            f"{baseline_label} score was associated with a growth-rate change of "
            f"**{_kf_float(row['median']):+.2f} logit units** (89% credible range "
            f"{_kf_float(row['lo89']):+.2f} to "
            f"{_kf_float(row['hi89']):+.2f}).",
            "headline",
        ),
        _kf_sentence(
            _kf_association_direction(
                row["prob_positive"],
                positive_claim="higher baseline ability accompanies faster growth",
                negative_claim="higher baseline ability accompanies slower growth",
            ),
            "confidence",
        ),
        _kf_sentence(
            f"These trajectory coefficients are adjusted associations, not effects of changing {baseline_label}.",
            "causal",
        ),
    ]
