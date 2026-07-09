# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF10 - gain factors for taught expressive vocabulary (TE).

DAG-focused gain-factors model (#224): associations with how much children gain
in taught expressive vocabulary (block 1, ``b1extau``) across the three period
transitions (ANCOVA, Beta-Binomial logit, child random intercept). Only the
randomised on-intervention term is causal; the own baseline, age, ability
(blocks) and the upstream DAG parent taught receptive vocabulary (TR) are
adjusted associations under the DAG (confounded by latent GA; the child
intercept repairs the time-invariant part). TR is TE's one measured DAG parent
(``TR -> TE``); the standardised transfer measures (RV/EV) sit downstream and so
are not conditioned on. SES excluded (non-DAG / redundant).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrp-rli-gf-010",
    kind="gain_factors",
    title="Gain factors for taught expressive vocabulary (TE)",
    outcome_symbol="TE",
    extra={
        "skill_symbols": ("TR",),
        "ability_covariate": V.BLOCKS,
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": False,
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
