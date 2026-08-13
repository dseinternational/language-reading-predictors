# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT26 - available-case modified ITT estimate for receptive grammar (T, TROG-2).

Standalone available-case modified ITT estimate for receptive grammar (#228
suite-gap Tier-1): T previously had only gain-factor / level-factor / aligned models.
The uniform DAG-faithful model (issue #119) requires no adjustment set for tau;
the own baseline and linear age are precision terms only and no cross-baselines
enter. Receptive grammar is one of the eight standardised available-case modified
ITT outcomes but has no education-lead-agreed ROPE delta, so the report gives the
estimate tau but omits the P(benefit >= delta) meaningful-benefit table. Sign
convention: positive tau means the intervention raises the outcome.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.pipelines.itt import fit_itt

SPEC = ModelSpec(
    model_id="lrp-rli-itt-026",
    kind="itt",
    title=(
        "Available-case modified ITT estimate of the assigned-arm contrast in "
        "receptive grammar (T)"
    ),
    outcome_symbol="T",
    model_settings=IttModelSettings(),
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
