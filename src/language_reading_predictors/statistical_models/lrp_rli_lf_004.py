# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPLF04 - level factors for letter sounds (L).

DAG-focused level-factors model (#127): associations with the letter sounds score
level at each of the four timepoints (Beta-Binomial logit, child random
intercept; no own baseline). group x time is a per-timepoint group effect
(trajectory divergence) - the t2 change is ``d_grp_time[t2]``; later changes
compare randomised treatment schedules; ability x time and group x ability complete the focal set.
Other coefficients are adjusted associations under the DAG. SES
excluded (non-DAG / redundant).

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
    model_id="lrp-rli-lf-004",
    kind="level_factors",
    title="Factors associated with the level of letter sounds (L)",
    outcome_symbol="L",
    model_settings=LevelFactorsModelSettings(
        ability_covariate=V.BLOCKS,
        adjust_for=("hs", "hs_missing", "deapp_c", "deapp_c_missing"),
        group_by_time=True,
        ability_by_time=True,
        group_ability=True,
        arm_gap_reference="t1",
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_level_factors(SPEC, config=config)
