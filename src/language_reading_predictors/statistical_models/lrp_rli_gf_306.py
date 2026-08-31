# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPGF06f - phoneme blending (B) gains under the one-in-three guessing floor.

The registered response-link companion to ``lrp-rli-gf-006`` (#596), mirroring the
ITT family's ``lrp-rli-itt-008`` / ``lrp-rli-itt-108`` pair and the level family's
``lrp-rli-lf-006`` / ``lrp-rli-lf-106``. Same data, same adjustment set, same
period stacking, same priors and the same randomised term ``beta_trt``; the single
difference is the score mean.

Phoneme blending has ten **three-alternative forced-choice** items, so a child
answering at random scores about 3.3 out of 10. The ordinary Beta-Binomial
inverse-logit mean does not know that: it permits expected scores anywhere in
(0, 1), and the fitted gf-006 posterior uses that room — 15 of its 161 fitted rows
have posterior-mean expected proportions below one third, 10.7 % of the row-by-draw
posterior mass sits below chance, and the worst single row puts 99.8 % of its mass
there (`notes/202608251100-gain-blending-guessing-floor-596.md`). This companion
constrains the mean to ``1/3 + 2/3 * expit(eta)``, so the model cannot predict
below-chance performance.

Because the two links can disagree about the size of the effect — in the one pair
fitted under both, the ITT items estimate halved and its 89 % interval crossed zero
— **neither fit is sufficient release evidence on its own**: the plan marks both
link-sensitivity-required, and the release gate refuses to publish either without
the other.

Two structural notes, both deliberate:

*No intercept anchor to remap.* The level companion had to map its empirical-Bayes
intercept anchor back through the link (it moved 1.1 logits). The gain family has
no such anchor: ``_alpha_sigma_for`` tiers the intercept prior's **SD** and leaves
its mean at zero, because a gain ANCOVA carries the outcome level in
``gamma_own * logit(y_pre)`` rather than in the intercept. There is nothing to
remap, and adding an anchor here would be a second change to a sensitivity fit
whose whole purpose is to isolate the first.

*``gamma_own`` is unchanged.* Its ``Normal(1, 0.25)`` prior is centred on
"post-logit tracks pre-logit 1:1", and under the floor link ``eta`` is the logit of
the rescaled mean ``(mu - 1/3) / (2/3)`` rather than of ``mu`` itself, so that 1:1
reading is no longer exact. It is kept as-is because ``lrp-rli-itt-108`` — the
archive-grade pair, which has the same own-baseline structure — keeps it, and a
link sensitivity that also moved a baseline prior would no longer isolate the link.
Recorded here so the assumption is visible rather than implied.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.gain_factors import (
    GainFactorsModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.gain_factors import fit_gain_factors

SPEC = ModelSpec(
    model_id="lrp-rli-gf-306",
    kind="gain_factors",
    title=(
        "Factors associated with gains in phoneme blending (blending) (B): "
        "three-choice guessing-floor link sensitivity"
    ),
    outcome_symbol="B",
    # Deliberately a one-key diff from lrp-rli-gf-006's declaration, in the same
    # style, so a reader can verify by eye that the pair differs only in the link.
    model_settings=GainFactorsModelSettings(
        skill_symbols=("L", "E", "TE"),
        ability_covariate=V.BLOCKS,
        adjust_for=(
            "hs",
            "hs_missing",
            "deapp_c",
            "deapp_c_missing",
            "erbto",
            "erbto_missing",
        ),
        interactions=(("age", "ability"),),
        treated_only=False,
        score_mean_link="three_choice_guessing_floor",
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_gain_factors(SPEC, config=config)
