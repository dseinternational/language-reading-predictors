# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF07 - gain factors for CELF basic concepts (F).

DAG-focused gain-factors model (#127): associations with how much children
gain in CELF basic concepts across the three period transitions (ANCOVA, Beta-Binomial
logit, child random intercept). Only the randomised on-intervention term is
causal; the own baseline, age, ability (blocks), and the upstream DAG skill receptive vocabulary (R) are adjusted
associations under the DAG (confounded by latent GA; the child intercept
repairs the time-invariant part). SES excluded (non-DAG / redundant).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrp-rli-gf-007",
    kind="gain_factors",
    title="Gain factors for CELF basic concepts (F)",
    outcome_symbol="F",
    extra={
        "skill_symbols": ("R",),
        "ability_covariate": V.BLOCKS,
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": False,
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
