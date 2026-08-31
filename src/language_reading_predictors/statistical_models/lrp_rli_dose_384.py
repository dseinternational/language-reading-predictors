# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP84f - dose -> phoneme blending (B) under the one-in-three guessing floor.

The registered response-link companion to ``lrp-rli-dose-084`` (#619, under the
#608 policy), mirroring ``lrp-rli-itt-008`` / ``lrp-rli-itt-108`` and the level,
gain, aligned and concurrent pairs. Same children, same rows, same period-resolved
dose design, same adjustment set and the same focal slope; the single difference is
the score mean.

Phoneme blending has ten **three-alternative forced-choice** items, so a child
answering at random scores about 3.3 out of 10. The ordinary Beta-Binomial
inverse-logit mean does not know that: the fitted dose-084 posterior puts 8 of its
160 rows below one third in posterior mean and 7.0 % of its row-by-draw mass below
chance, with a worst row at 100 %. This companion constrains the mean to
``1/3 + 2/3 * expit(eta)``, so the model cannot predict below-chance performance.

**Why this family in particular.** The #608 decision named the dose family as the
case that defeats the "observational families are exempt" argument. `METHODS.md`
defines every dose fit's focal estimand as the **natural-scale treated-row dose
marginal** — a quantity published in items — so it inherits the link exactly as a
randomised treatment contrast does. What identifies a quantity and what scale it is
reported on are different questions, and only the second is what the link decides.

Nothing about the causal labelling changes. There is still **no clean back-door
set** for ``sessions -> outcome``: latent general ability is unmeasured, and every
dose slope stays an *adjusted association*, never "dose drives gains". The
extensive margin (``theta_treated``), the between/within split and the arm term all
keep the meanings ``coefficient_meanings()`` records for them.

Two structural notes. The family has **no empirical-Bayes intercept anchor** to map
back through the link, so unlike the level companion there is nothing to remap. And
``dose_marginal_draws`` — the one transform behind both the posterior marginal and
its prior pushforward — takes the link, so the "check" cannot end up comparing two
quantities on different scales.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.dose_response import (
    DoseResponseModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.dose_response import fit_dose_response

SPEC = ModelSpec(
    model_id="lrp-rli-dose-384",
    kind="dose_response",
    title=(
        "Period-resolved dose-response: intervention dose -> phoneme blending (B): "
        "three-choice guessing-floor link sensitivity"
    ),
    outcome_symbol="B",
    adjustment=["G", "A", "B_pre"],
    # Deliberately a one-key diff from lrp-rli-dose-084's declaration, in the same
    # style, so a reader can verify by eye that the pair differs only in the link.
    model_settings=DoseResponseModelSettings(
        adjust_baseline_symbol="B",
        dose_covariate="attend",
        period_varying_dose=True,
        use_subject_random_intercept=True,
        outcomes=("B",),
        score_mean_link="three_choice_guessing_floor",
    ),
        # Matched to lrp-rli-dose-084 so the pair differs only in the link: the same
        # period-varying dose geometry needs the same sampler setting, and a
        # companion that quietly sampled at a different target_accept would confound
        # the link comparison with a sampling-quality one.
    target_accept=0.97,
)


def fit(config: str = "dev"):
    return fit_dose_response(SPEC, config=config)
