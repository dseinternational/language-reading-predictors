# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF01 - gain factors for word reading (W).

DAG-focused factor model (#127): what is associated with how much children gain
in word reading across the three period transitions (ANCOVA gain, Beta-Binomial
on the logit scale, child random intercept). Under the locked DAG only the
randomised on-intervention term is causal (its coefficient is identified by the
period-1 contrast, so it reproduces the ITT tau); the own baseline, age, cognitive
ability (blocks), and the upstream DAG skills letter sounds (L) and receptive
vocabulary (R) are *adjusted associations* confounded by latent general ability,
which the child random intercept repairs up to shrinkage. SES is excluded - not a
DAG node and statistically redundant. Focal interactions: group x ability,
group x own-baseline, age x ability.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrpgf01",
    kind="gain_factors",
    title="Gain factors for word reading (W)",
    outcome_symbol="W",
    extra={
        "skill_symbols": ("L", "R"),
        "ability_covariate": V.BLOCKS,
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": False,
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
