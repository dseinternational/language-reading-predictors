# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF05b - gain factors for phonetic spelling (P), off-floor, treated-only.

The off-floor LRPGF05 restricted to on-intervention rows ("off-floor while on
intervention"): the constant treatment indicator and its interactions are dropped,
leaving the adjusted associations (own baseline, age, ability, skills L/B,
age x ability) of coming off the phonetic-spelling floor among the treated. SES
excluded (non-DAG / redundant).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrpgf05b",
    kind="gain_factors",
    title="Gain factors for phonetic spelling (P), off-floor, treated-only",
    outcome_symbol="P",
    extra={
        "skill_symbols": ("L", "B"),
        "ability_covariate": V.BLOCKS,
        "interactions": (("trt", "ability"), ("trt", "own"), ("age", "ability")),
        "treated_only": True,
        "likelihood": "bernoulli_offfloor",
    },
)


def fit(config: str = "dev"):
    return fit_gain_factors(SPEC, config=config)
