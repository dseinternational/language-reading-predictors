# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPLF01 - level factors for word reading (W).

DAG-focused factor model (#127): what is associated with the word-reading score
*level* at each of the four timepoints (Beta-Binomial on the logit scale, child
random intercept; no own baseline - not autoregressive). The focal interactions
are modelled over categorical time: ``group x time`` as a per-timepoint group
effect (trajectory divergence) and ``ability x time`` as a per-timepoint ability
effect, plus ``group x ability``. **Level-model caveat:** the t2 arm contrast compares
intervention with no
intervention yet. Later contrasts compare randomised treatment schedules after
waitlist crossover. All use the longitudinal working model; other coefficients
are adjusted associations under the DAG. SES is excluded
(not a DAG node; redundant).

Revised-DAG update (#247; adjustment set re-derived against
``dag/dag-language-reading.dagitty``, 2026-07-10): this outcome's exogenous non-measure
confounder parents — hearing (HS), speech production (SP) and/or phonological memory
(RW), where the DAG has such an edge — enter via ``adjust_for``. Measured skill parents
are deliberately NOT conditioned on: in a levels model their contemporaneous level is a
post-treatment mediator of the group×time effect. Adjusting for these skills
could block a treatment-mediated path and change the estimand. The t2 arm contrast
compares treatment with no treatment yet. Later arm
contrasts compare randomised treatment schedules. Other coefficients are adjusted
associations. The child random intercept is a partial shrunken stand-in for
between-child heterogeneity
that does not control latent general ability.

The arm-gap parameterisation (#552) uses ``arm_gap_t1`` for the covariate-adjusted
pre-randomisation arm gap. This measures baseline balance. At later waves,
``d_grp_time[t]`` gives the change from that gap. The randomised-window contrast
is the **t2 change ``d_grp_time[t2]``**, a difference-in-differences of adjusted
levels. The later changes compare randomised early-start and delayed-start
treatment schedules, as explained in :mod:`level_factors`. Shared parameters
allow all waves to inform these estimates; compare the t1/t2-only companion
when assessing dependence on the longitudinal working model. The per-wave gaps
``b_grp_time[t]`` remain a derived levels view. The former free
per-timepoint vector (whose t2 element ``b_grp_time[1]`` carried the adjusted
chance t1 imbalance) is retained only as the ``arm_gap_reference="free"``
comparator.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.level_factors import (
    LevelFactorsModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.level_factors import fit_level_factors

SPEC = ModelSpec(
    model_id="lrp-rli-lf-001",
    kind="level_factors",
    title="Factors associated with the level of word reading (W)",
    outcome_symbol="W",
    model_settings=LevelFactorsModelSettings(
        ability_covariate=V.BLOCKS,
        adjust_for=(),
        group_by_time=True,
        ability_by_time=True,
        group_ability=True,
        arm_gap_reference="t1",
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_level_factors(SPEC, config=config)
