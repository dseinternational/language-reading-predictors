# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT25 - available-case modified ITT estimate for basic concept knowledge (F, CELF).

Standalone available-case modified ITT estimate for basic concepts (#228 suite-gap
Tier-1): F previously had only gain-factor / level-factor / aligned models.
The uniform DAG-faithful model (issue #119) requires no adjustment set for tau;
the own baseline and linear age are precision terms only and no cross-baselines
enter. Basic concepts is one of the eight standardised available-case modified ITT
outcomes but has no education-lead-agreed ROPE delta, so the report gives the
estimate tau (size / direction / probability) but omits the P(benefit >= delta)
meaningful-benefit table the ROPE-anchored outcomes carry. Sign convention:
positive tau means the intervention raises the outcome.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.pipeline import fit_itt

SPEC = ModelSpec(
    model_id="lrp-rli-itt-025",
    kind="itt",
    title=(
        "Available-case modified ITT estimate of the assigned-arm contrast in "
        "basic concept knowledge (F)"
    ),
    outcome_symbol="F",
    model_settings=IttModelSettings(),
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
