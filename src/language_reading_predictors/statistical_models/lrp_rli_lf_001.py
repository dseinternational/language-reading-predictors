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
contrast - the clean randomised contrast lives only at t2 (``d_grp_time[t2]``, see below);
every other coefficient is an adjusted association under the DAG. SES is excluded
(not a DAG node; redundant).

Revised-DAG update (#247; adjustment set re-derived against
``dag/dag-language-reading.dagitty``, 2026-07-10): this outcome's exogenous non-measure
confounder parents — hearing (HS), speech production (SP) and/or phonological memory
(RW), where the DAG has such an edge — enter via ``adjust_for``. Measured skill parents
are deliberately NOT conditioned on: in a levels model their contemporaneous level is a
post-treatment mediator of the group×time effect, so adjusting for them would bias the
very trajectory the model estimates. The clean randomised contrast remains the t2 group
effect (``d_grp_time[t2]``, see below); every other coefficient is an adjusted association, and the
child random intercept is a partial shrunken stand-in for between-child heterogeneity
that does not control latent general ability.

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
from language_reading_predictors.statistical_models.pipelines.level_factors import fit_level_factors

SPEC = ModelSpec(
    model_id="lrp-rli-lf-001",
    kind="level_factors",
    title="Factors associated with the level of word reading (W)",
    outcome_symbol="W",
    extra={
        "ability_covariate": V.BLOCKS,
        "adjust_for": (),
        "group_by_time": True,
        "ability_by_time": True,
        "group_ability": True,
        "arm_gap_reference": "t1",
    },
)


def fit(config: str = "dev"):
    return fit_level_factors(SPEC, config=config)
