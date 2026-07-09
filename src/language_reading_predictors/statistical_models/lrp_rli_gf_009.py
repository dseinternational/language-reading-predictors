# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF09 - gain factors for taught receptive vocabulary (TR).

DAG-focused gain-factors model (#224): associations with how much children gain
in taught receptive vocabulary (block 1, ``b1retau``) across the three period
transitions (ANCOVA, Beta-Binomial logit, child random intercept). Only the
randomised on-intervention term is causal; the own baseline, age and ability
(blocks) are adjusted associations under the DAG (confounded by latent GA; the
child intercept repairs the time-invariant part). No upstream skill is
conditioned on: under the locked DAG the standardised transfer measures (RV/EV)
sit *downstream* of taught vocabulary (``TR -> RV``), so TR's only measured
handles are age and ability. SES excluded (non-DAG / redundant).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrp-rli-gf-009",
    kind="gain_factors",
    title="Gain factors for taught receptive vocabulary (TR)",
    outcome_symbol="TR",
    extra={
        "skill_symbols": (),
        "ability_covariate": V.BLOCKS,
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": False,
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
