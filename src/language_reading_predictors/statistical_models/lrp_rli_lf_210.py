# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPLF210 - taught expressive vocabulary (TE) levels, randomised-window (t1/t2) comparator.

The registered comparator to ``lrp-rli-lf-010`` (#584 decision 3). Identical
model, identical adjustment set, identical estimand — restricted to the two waves
at which randomisation alone separates the arms: the pre-randomisation baseline
t1 and the end of the randomised period t2.

Why it exists: in the four-wave model of record the post-crossover t3/t4
likelihood reaches the reported t2 change through parameters the waves share —
the balance term ``arm_gap_t1`` the changes are measured from, the child random
intercept, the dispersion, and the single time-invariant ``group x ability``
term. Across the stored suite the posterior correlation between ``arm_gap_t1``
and ``d_grp_time[t2]`` runs from -0.07 to -0.44. The contrast is therefore
randomisation-anchored but longitudinal-model-dependent, and calling it simply
"the clean randomised contrast" overstates the separation (#584 finding 5).

Here there is nothing to borrow: no post-crossover row enters the likelihood, so
``d_grp_time[t2]`` is identified from the randomised window alone. The four-wave
fit remains the model of record — it is the levels *view*, and showing all four
waves on one scale is its purpose — and this comparator is reported beside it so
a reader can see how much the longitudinal working model moved the answer.
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
    model_id="lrp-rli-lf-210",
    kind="level_factors",
    title="Taught expressive vocabulary (TE): randomised-window (t1/t2) comparator",
    outcome_symbol="TE",
    model_settings=LevelFactorsModelSettings(
        ability_covariate=V.BLOCKS,
        adjust_for=("hs", "hs_missing", "deapp_c", "deapp_c_missing", "erbto", "erbto_missing"),
        group_by_time=True,
        ability_by_time=True,
        group_ability=True,
        arm_gap_reference="t1",
        waves=("t1", "t2"),
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_level_factors(SPEC, config=config)
