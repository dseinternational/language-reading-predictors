# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT03 - available-case modified ITT estimate for not-taught receptive vocabulary, block 1 (UR).

Uniform DAG-faithful available-case modified ITT model (issue #119). Under the
locked DAG the assigned-arm coefficient requires no adjustment set, so the own
baseline and linear age are PRECISION terms only and no cross-baselines enter.
Sign convention: positive ``tau`` means the intervention raises the outcome.

The not-taught set's item count (n_trials = 12) is the observed maximum, flagged
unconfirmed in ``measures.py`` (Burgoyne et al. 2012, Table 3, documents only the
24-item taught tests); probability-scale summaries are therefore approximate.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.pipeline import fit_itt

SPEC = ModelSpec(
    model_id="lrp-rli-itt-003",
    kind="itt",
    title=(
        "Available-case modified ITT estimate of the assigned-arm contrast in "
        "not-taught receptive vocabulary, block 1 (UR)"
    ),
    outcome_symbol="UR",
    model_settings=IttModelSettings(),
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
