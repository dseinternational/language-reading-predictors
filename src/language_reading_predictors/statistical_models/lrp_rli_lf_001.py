# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPLF01 - level factors for word reading (W).

DAG-focused factor model (#127): what is associated with the word-reading score
*level* at each of the four timepoints (Beta-Binomial on the logit scale, child
random intercept; no own baseline - not autoregressive). The focal interactions
are modelled over categorical time: ``group x time`` as a per-timepoint group
effect (trajectory divergence) and ``ability x time`` as a per-timepoint ability
effect, plus ``group x ability``. **Level-model caveat:** after t2 the waitlist
crosses over, so the group effect across the four timepoints is not a clean ITT
contrast - the clean randomised contrast lives only at t2 (``b_grp_time[1]``);
every other coefficient is an adjusted association under the DAG. SES is excluded
(not a DAG node; redundant).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_level_factors

SPEC = ModelSpec(
    model_id="lrp-rli-lf-001",
    kind="level_factors",
    title="Level factors for word reading (W)",
    outcome_symbol="W",
    extra={
        "ability_covariate": V.BLOCKS,
        "group_by_time": True,
        "ability_by_time": True,
        "group_ability": True,
    },
)


def fit(config: str = "dev"):
    return fit_level_factors(SPEC, config=config)
