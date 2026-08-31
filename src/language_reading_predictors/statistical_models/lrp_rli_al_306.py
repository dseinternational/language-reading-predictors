# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPAL06f - aligned phoneme blending (B) under the one-in-three guessing floor.

The registered response-link companion to ``lrp-rli-al-006`` (#619, under the #608
policy), mirroring ``lrp-rli-itt-008`` / ``lrp-rli-itt-108``, ``lrp-rli-lf-006`` /
``lrp-rli-lf-106`` and ``lrp-rli-gf-006`` / ``lrp-rli-gf-306``. Same children, same
onset alignment, same adjustment set and the same cohort contrast ``beta_cohort``;
the single difference is the score mean.

Phoneme blending has ten **three-alternative forced-choice** items, so a child
answering at random scores about 3.3 out of 10. The ordinary Beta-Binomial
inverse-logit mean does not know that: it permits expected scores anywhere in
(0, 1), and the fitted al-006 posterior uses that room — 2 of its 54 rows have
posterior-mean expected proportions below one third and 4.9 % of the row-by-draw
posterior mass sits below chance, with a worst row at 98.0 %. This companion
constrains the mean to ``1/3 + 2/3 * expit(eta)``, so the model cannot predict
below-chance performance.

Because the two links can disagree about the size of the estimate — in the two
pairs fitted under both, the items number fell by about 40 % and the interval
crossed zero — **neither fit is sufficient release evidence on its own**: the plan
marks both link-sensitivity-required, and the release gate refuses to publish
either without the other.

Nothing about the causal labelling changes. This is a **per-protocol** design:
``beta_cohort`` contrasts the two arms at their own onset-aligned endpoints and is
confounded by age-at-onset and cohort/timing, so no coefficient here is a clean
treatment effect — all are associations, under either link. The #608 decision is
explicit that the link binds association and contrast alike, because any quantity
reported on the natural scale inherits the link dependence regardless of what
identifies it.

Like the gain family, this family has **no empirical-Bayes intercept anchor** to map
back through the link (``build_aligned_model`` gives ``alpha`` a zero-centred prior;
the outcome level is carried by ``gamma_own``), so there is nothing to remap.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.aligned import (
    AlignedModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.aligned import fit_aligned

SPEC = ModelSpec(
    model_id="lrp-rli-al-306",
    kind="aligned",
    title=(
        "Aligned-40-week per-protocol gain for phoneme blending (B): "
        "three-choice guessing-floor link sensitivity"
    ),
    outcome_symbol="B",
    # Deliberately a one-key diff from lrp-rli-al-006's declaration, in the same
    # style, so a reader can verify by eye that the pair differs only in the link.
    model_settings=AlignedModelSettings(
        ability_covariate=V.BLOCKS,
        use_cohort=True,
        use_dose=False,
        score_mean_link="three_choice_guessing_floor",
    ),
)


def fit(config: str = "dev"):
    return fit_aligned(SPEC, config=config)
