# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT06 - available-case modified ITT estimate for standardised expressive vocabulary (E, EOWPVT).

Uniform DAG-faithful available-case modified ITT model (issue #119). Under the
locked DAG the assigned-arm coefficient requires no adjustment set, so the own
baseline and linear age are PRECISION terms only and no cross-baselines enter.
Sign convention: positive ``tau`` means the intervention raises the outcome.
Supersedes LRP54.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.pipeline import fit_itt

SPEC = ModelSpec(
    model_id="lrp-rli-itt-006",
    kind="itt",
    title=(
        "Available-case modified ITT estimate of the assigned-arm contrast in "
        "standardised expressive vocabulary (E)"
    ),
    outcome_symbol="E",
    model_settings=IttModelSettings(),
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
