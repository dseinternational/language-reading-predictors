# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPLF106 - phoneme blending (B) levels under the one-in-three guessing floor.

The registered response-link companion to ``lrp-rli-lf-006`` (#584 decision 2),
mirroring the ITT family's ``lrp-rli-itt-008`` / ``lrp-rli-itt-108`` pair. Same
data, same adjustment set, same arm-gap parameterisation and the same randomised
contrast ``d_grp_time[t2]``; the single difference is the score mean.

Phoneme blending has ten **three-alternative forced-choice** items, so a child
answering at random scores about 3.3 out of 10. The ordinary Beta-Binomial
inverse-logit mean does not know that: it permits expected scores anywhere in
(0, 1), and the fitted lf-006 posterior uses that room — 24 of 215 rows have
posterior-mean expected proportions below one third, and at timepoint 2 it is 8 of
54 rows and 16% of the posterior mass. This companion constrains the mean to
``1/3 + 2/3 * expit(eta)``, so the model cannot predict below-chance performance.

Because the two links can disagree about the size of the effect, **neither fit is
sufficient release evidence on its own**: the plan marks both as
link-sensitivity-required, and the release gate refuses to publish either without
the other. The empirical-Bayes intercept anchor is mapped back through the link
(``invert_score_mean_link``), so the anchor sits where the floor link needs it
rather than 1.1 logits away.
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
    model_id="lrp-rli-lf-106",
    kind="level_factors",
    title="Level of phoneme blending (B) under the one-in-three guessing floor",
    outcome_symbol="B",
    model_settings=LevelFactorsModelSettings(
        ability_covariate=V.BLOCKS,
        adjust_for=("hs", "hs_missing", "deapp_c", "deapp_c_missing", "erbto", "erbto_missing"),
        group_by_time=True,
        ability_by_time=True,
        group_ability=True,
        arm_gap_reference="t1",
        score_mean_link="three_choice_guessing_floor",
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_level_factors(SPEC, config=config)
