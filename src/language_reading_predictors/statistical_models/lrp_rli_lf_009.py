# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPLF09 - level factors for taught receptive vocabulary (TR).

DAG-focused level-factors model (#224): associations with the taught receptive
vocabulary score level (block 1, ``b1retau``) at each of the four timepoints
(Beta-Binomial logit, child random intercept; no own baseline). group x time is a
per-timepoint group effect (trajectory divergence) - the clean randomised contrast
lives only at t2 (``d_grp_time[t2]``, see below); ability x time and group x ability complete the focal
set. Every non-t2 coefficient is an adjusted association under the DAG. SES excluded
(non-DAG / redundant).

Revised-DAG update (#247; adjustment set re-derived against
``dag/dag-language-reading.dagitty``, 2026-07-10): TR's exogenous non-measure confounder
parents — hearing (HS) and phonological memory (RW) — enter via ``adjust_for``
(``hs``, ``erbto``). Measured skill parents are deliberately NOT conditioned on: in a
levels model a contemporaneous skill level is a post-treatment mediator of the
group×time effect, so adjusting for it would bias the very trajectory the model
estimates. The clean randomised contrast remains the t2 group effect
(``d_grp_time[t2]``, see below); every other coefficient is an adjusted association, and the child
random intercept is a partial shrunken stand-in for between-child heterogeneity that
does not control latent general ability.

Arm-gap parameterisation (#552): the per-timepoint group coefficient is centred
on the timepoint-1 arm gap — ``arm_gap_t1`` (the covariate-adjusted
pre-randomisation gap, a balance quantity, never an effect) plus the change in
that gap at each later wave, ``d_grp_time[t]`` — so the clean randomised contrast
is the **t2 change ``d_grp_time[t2]``**, a difference-in-differences of adjusted
levels; ``d_grp_time[t3]`` / ``[t4]`` are post-crossover associations and the
per-wave gaps ``b_grp_time[t]`` are kept as a derived levels view. The former free
per-timepoint vector (whose t2 element ``b_grp_time[1]`` carried the adjusted
chance t1 imbalance) is retained only as the ``arm_gap_reference="free"``
comparator.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.level_factors import (
    LevelFactorsModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.level_factors import fit_level_factors

SPEC = ModelSpec(
    model_id="lrp-rli-lf-009",
    kind="level_factors",
    title="Factors associated with the level of taught receptive vocabulary (TR)",
    outcome_symbol="TR",
    model_settings=LevelFactorsModelSettings(
        ability_covariate=V.BLOCKS,
        adjust_for=("hs", "hs_missing", "erbto", "erbto_missing"),
        group_by_time=True,
        ability_by_time=True,
        group_ability=True,
        arm_gap_reference="t1",
    ),
)


def fit(config: str = "dev"):
    return fit_level_factors(SPEC, config=config)
