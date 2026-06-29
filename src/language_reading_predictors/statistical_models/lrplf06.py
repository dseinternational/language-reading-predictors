# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPLF06 - level factors for phoneme blending (blending) (B).

DAG-focused level-factors model (#127): associations with the phoneme blending (blending) score
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
    model_id="lrplf06",
    kind="level_factors",
    title="Level factors for phoneme blending (blending) (B)",
    outcome_symbol="B",
    extra={
        "ability_covariate": V.BLOCKS,
        "group_by_time": True,
        "ability_by_time": True,
        "group_ability": True,
    },
)


def fit(config: str = "dev"):
    return fit_level_factors(SPEC, config=config)
