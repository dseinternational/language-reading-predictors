# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP-CA-307 - concurrent skills -> phoneme blending under the one-in-three guessing floor.

The registered response-link companion to ``lrp-rli-ca-007`` (#619, under the #608
policy), mirroring ``lrp-rli-itt-008`` / ``lrp-rli-itt-108``, ``lrp-rli-lf-006`` /
``lrp-rli-lf-106``, ``lrp-rli-gf-006`` / ``lrp-rli-gf-306`` and ``lrp-rli-al-006`` /
``lrp-rli-al-306``. Same children, same waves, same predictors, same trait
covariates, same slope prior; the single difference is the outcome's score mean.

Phoneme blending has ten **three-alternative forced-choice** items, so a child
answering at random scores about 3.3 out of 10. The ordinary Beta-Binomial
inverse-logit mean does not know that, and the ca-007 posterior uses the room it
leaves: 4 of its 54 primary-wave rows have posterior-mean expected proportions below
one third and 9.7 % of the row-by-draw posterior mass sits below chance. This
companion constrains the mean to ``1/3 + 2/3 * expit(eta)``, so the model cannot
predict below-chance performance.

**Nothing about the causal labelling changes**, and that is exactly why the pairing
still binds. Every coefficient here is an *adjusted association*, never a causal
effect — the family exists to describe the conditional joint distribution of skill
levels at each wave. The #608 decision is explicit that the response link binds
association and contrast alike: the link determines the mapping from the latent
scale to the reported one, so any quantity published on the natural scale inherits
it regardless of what identifies it. ``concurrent_marginals.csv`` reports items, so
it inherits it.

The link applies to blending as the **outcome**. Where B is a *predictor* (ca-001 to
ca-006 and the sibling models) it enters as a standardised same-wave logit covariate
rather than as a modelled score mean, so those fits are untouched — the resolver
rejects the floor link for any non-B outcome.

Because the family fits one model per wave — and, per predictor, an additional
single-skill sub-fit — the link is threaded into every build, so an adjusted and a
bivariate sub-fit of the same wave cannot disagree about the response scale.
"""

from language_reading_predictors.statistical_models.concurrent import (
    ConcurrentModelSettings,
)
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipelines.concurrent import fit_concurrent

SPEC = ModelSpec(
    model_id="lrp-rli-ca-307",
    kind="concurrent",
    title=(
        "Concurrent conditional associations: concurrent skills -> phoneme blending "
        "(per wave): three-choice guessing-floor link sensitivity"
    ),
    outcome_symbol="B",
    family="concurrent",
    design="per-wave cross-sectional conditional associations",
    estimand_type="association",
    causal_status="none",
    # Deliberately a one-key diff from lrp-rli-ca-007's declaration, in the same
    # style, so a reader can verify by eye that the pair differs only in the link.
    model_settings=ConcurrentModelSettings(
        predictor_symbols=("W", "L", "TR", "TE", "R", "E"),
        covariates=(
            "blocks",
            "hs",
            "hs_missing",
            "deapp_c",
            "deapp_c_missing",
            "erbto",
            "erbto_missing",
        ),
        include_age=True,
        include_group=True,
        predictor_slope_sigma=0.3,
        score_mean_link="three_choice_guessing_floor",
    ),
)


def fit(config: str = "dev"):
    return fit_concurrent(SPEC, config=config)
