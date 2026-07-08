# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPITT07 - ITT effect on letter-sound knowledge (L, YARC-LSK).

Uniform DAG-faithful ITT model (issue #119). Under the locked DAG the effect of
randomised assignment is identified by the *empty* adjustment set, so the own
baseline and linear age are PRECISION terms only and no cross-baselines enter.
Sign convention: positive ``tau`` means the intervention raises the outcome.
Letter sounds carried the strongest treatment effect in the earlier joint model.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_itt

SPEC = ModelSpec(
    model_id="lrp-rli-itt-007",
    kind="itt",
    title="ITT effect of group assignment on letter-sound knowledge (L)",
    outcome_symbol="L",
    extra={
        "outcomes": ("L",),
        "cross_symbols": (),
        "use_age_gp": False,
        "use_own_baseline_gp": False,
        "use_age_linear": True,
        "use_own_baseline": True,
    },
)


def fit(config: str = "dev"):
    return fit_itt(SPEC, config=config)
