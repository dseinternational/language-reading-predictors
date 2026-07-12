# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPLF11 - level factors for nonword reading (N), off-floor.

DAG-focused level-factors model (#224): associations with the nonword-reading off-floor
status at each of the four timepoints. Nonword reading is heavily floored, so the level
outcome is the off-floor indicator (score > 0) at each timepoint — being off the floor
**at post** (post>0) pools zero→positive movement, persistence above zero and
return-to-floor (not merely "coming off the floor") — modelled by a Bernoulli rather
than a graded Beta-Binomial. group x time is a per-timepoint off-floor log-odds for the
group (the clean randomised contrast lives only at t2, b_grp_time[1]); ability x time
and group x ability complete the focal set. Every non-t2 coefficient is an adjusted
association under the DAG. SES excluded (non-DAG / redundant).

Revised-DAG update (#247; adjustment set re-derived against
``dag/dag-language-reading.dagitty``, 2026-07-10): NW's exogenous non-measure confounder
parents — speech production (SP) and phonological memory (RW) — enter via ``adjust_for``
(``deapp_c``, ``erbto``); hearing (``hs``) is NOT a NW parent in the DAG, so it is not
conditioned on. Measured skill parents are deliberately NOT conditioned on: in a levels
model a contemporaneous skill level is a post-treatment mediator of the group×time
effect, so adjusting for it would bias the very trajectory the model estimates. The
clean randomised contrast remains the t2 group effect (``b_grp_time[1]``); every other
coefficient is an adjusted association, and the child random intercept is a partial
shrunken stand-in for between-child heterogeneity that does not control latent general
ability.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_level_factors

SPEC = ModelSpec(
    model_id="lrp-rli-lf-011",
    kind="level_factors",
    title="Level factors for nonword reading (N), off-floor",
    outcome_symbol="N",
    extra={
        "ability_covariate": V.BLOCKS,
        "adjust_for": ("deapp_c", "deapp_c_missing", "erbto", "erbto_missing"),
        "group_by_time": True,
        "ability_by_time": True,
        "group_ability": True,
        "likelihood": "bernoulli_offfloor",
    },
)


def fit(config: str = "dev"):
    return fit_level_factors(SPEC, config=config)
