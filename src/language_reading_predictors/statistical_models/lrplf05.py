# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPLF05 - level factors for phonetic spelling (P), off-floor.

Phonetic spelling is heavily floored, so the level model uses a **Bernoulli on the
off-floor indicator** (score > 0 at each timepoint) rather than a graded
Beta-Binomial. group x time is a per-timepoint off-floor log-odds for the group
(the clean randomised contrast lives only at t2, b_grp_time[1]); ability x time
and group x ability complete the focal set. Every non-t2 coefficient is an
adjusted association under the DAG. SES excluded (non-DAG / redundant).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_level_factors

SPEC = ModelSpec(
    model_id="lrplf05",
    kind="level_factors",
    title="Level factors for phonetic spelling (P), off-floor",
    outcome_symbol="P",
    extra={
        "ability_covariate": V.BLOCKS,
        "group_by_time": True,
        "ability_by_time": True,
        "group_ability": True,
        "likelihood": "bernoulli_offfloor",
    },
)


def fit(config: str = "dev"):
    return fit_level_factors(SPEC, config=config)
