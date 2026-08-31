# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT10 - available-case modified ITT estimate for word reading (W, EWRSWR).

Uniform DAG-faithful available-case modified ITT model (issue #119). Under the
locked DAG the assigned-arm coefficient requires no adjustment set, so the own
baseline and linear age are PRECISION terms only and no cross-baselines enter.
Sign convention: positive ``tau`` means the intervention raises the outcome.

Word reading is ~40% floored at *baseline* (t1) but not at the post-score (t2),
so it keeps the graded own-baseline specification (the floor rule, which gates on
the t2 post-score, does not trip). The report carries a baseline floor diagnostic.
Supersedes LRP52.
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.pipelines.itt import fit_itt

SPEC = ModelSpec(
    model_id="lrp-rli-itt-010",
    kind="itt",
    title=(
        "Available-case modified ITT estimate of the assigned-arm contrast in "
        "word reading (W)"
    ),
    outcome_symbol="W",
    model_settings=IttModelSettings(missingness_sensitivity=True),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_itt(SPEC, config=config)
