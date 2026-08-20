# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT04 - available-case modified ITT estimate for not-taught expressive vocabulary, block 1 (UE).

Uniform DAG-faithful available-case modified ITT model (issue #119). Under the
locked DAG the assigned-arm coefficient requires no adjustment set, so the own
baseline and linear age are PRECISION terms only and no cross-baselines enter.
Sign convention: positive ``tau`` means the intervention raises the outcome.

The not-taught set's item count (n_trials = 12) has been confirmed since #214
(the taught-word list resolved the 24 taught / 12 not-taught split per modality;
``measures.py`` records it with ``n_trials_confirmed=True``) — an earlier version
of this docstring predated that confirmation (corrected in the 2026-08-20 ITT
code review).
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.itt import IttModelSettings
from language_reading_predictors.statistical_models.pipelines.itt import fit_itt

SPEC = ModelSpec(
    model_id="lrp-rli-itt-004",
    kind="itt",
    title=(
        "Available-case modified ITT estimate of the assigned-arm contrast in "
        "not-taught expressive vocabulary, block 1 (UE)"
    ),
    outcome_symbol="UE",
    model_settings=IttModelSettings(),
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
