# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT04 - ITT effect on not-taught expressive vocabulary, block 1 (UE).

Uniform DAG-faithful ITT model (issue #119). Under the locked DAG the effect of
randomised assignment is identified by the *empty* adjustment set, so the own
baseline and linear age are PRECISION terms only and no cross-baselines enter.
Sign convention: positive ``tau`` means the intervention raises the outcome.

The not-taught set's item count (n_trials = 12) is the observed maximum, flagged
unconfirmed in ``measures.py``; probability-scale summaries are approximate.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_itt

SPEC = ModelSpec(
    model_id="lrpitt04",
    kind="itt",
    title="ITT effect of group assignment on not-taught expressive vocabulary, block 1 (UE)",
    outcome_symbol="UE",
    extra={
        "outcomes": ("UE",),
        "cross_symbols": (),
        "use_age_gp": False,
        "use_own_baseline_gp": False,
        "use_age_linear": True,
        "use_own_baseline": True,
    },
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
