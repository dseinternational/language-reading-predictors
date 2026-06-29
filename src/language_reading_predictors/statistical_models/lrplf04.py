# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPLF04 - level factors for letter sounds (L).

DAG-focused level-factors model (#127): associations with the letter sounds score
level at each of the four timepoints (Beta-Binomial logit, child random
intercept; no own baseline). group x time is a per-timepoint group effect
(trajectory divergence) - the clean randomised contrast lives only at t2
(b_grp_time[1]); ability x time and group x ability complete the focal set.
Every non-t2 coefficient is an adjusted association under the DAG. SES
excluded (non-DAG / redundant).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_level_factors

SPEC = ModelSpec(
    model_id="lrplf04",
    kind="level_factors",
    title="Level factors for letter sounds (L)",
    outcome_symbol="L",
    extra={
        "ability_covariate": V.BLOCKS,
        "group_by_time": True,
        "ability_by_time": True,
        "group_ability": True,
    },
)


def fit(config: str = "dev"):
    return fit_level_factors(SPEC, config=config)
