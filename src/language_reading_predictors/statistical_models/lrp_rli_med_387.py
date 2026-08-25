# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP87f - blending (B) via letter sounds (L) under the one-in-three guessing floor.

The registered response-link companion to ``lrp-rli-med-087`` (#619, under the #608
policy), completing the policy across every family. Same children, same rows, same
mediator, same adjustment set and the same g-formula decomposition; the single
difference is the **outcome** leg's score mean.

Phoneme blending has ten **three-alternative forced-choice** items, so a child
answering at random scores about 3.3 out of 10. The ordinary Beta-Binomial
inverse-logit mean does not know that: the fitted med-087 posterior puts 5 of its 53
rows below one third in posterior mean and 12.1 % of its row-by-draw mass below
chance, with a worst row at 99.1 %. That is the **highest** share of any registered
``B`` fit — higher than LRPITT08 (8.9 %), the fit the policy was written for. This
companion constrains the outcome mean to ``1/3 + 2/3 * expit(eta)``.

**Why the link has to enter the simulation, not a summary.** Every NDE, NIE and
total this family publishes is a difference of *simulated outcome means*: the
g-formula accumulates ``E[Y(g, M(g'))]`` over units and mediator replicates, then
differences the cells. There is no downstream summary to correct — running the
decomposition on the ordinary inverse logit against a floor-link posterior would
produce items the fitted model never implied, cell by cell. So ``score_mean_link``
is threaded into ``decompose``'s ``outcome_p``, which is the one place the latent
scale becomes a score.

**The link governs the outcome only.** The mediator here is letter sounds, a
different measure with its own leg and its own denominator; no registered mediation
model has phoneme blending as its mediator, and the resolver rejects the floor link
for any non-B outcome.

Nothing about the identification changes, and none of it was ever strong. The
binding unverifiable assumption is still **no unmeasured L -> B confounding**, which
latent general ability violates; intervention sessions (``IS``) remain a
treatment-affected recanting witness that no adjustment set rescues. This is a
model-based g-formula decomposition under stated cross-world assumptions, wide at
n ~ 53 — not an identified natural effect. The link fixes the response scale; it
does not touch any of that.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.mediation_settings import (
    MediationModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.mediation import fit_mediation

SPEC = ModelSpec(
    model_id="lrp-rli-med-387",
    kind="mediation",
    title=(
        "Mediation: does the intervention raise phoneme blending (B) via "
        "letter-sound knowledge (L)? Three-choice guessing-floor link sensitivity"
    ),
    outcome_symbol="B",
    mechanism_symbol="L",  # the mediator
    # Deliberately identical to lrp-rli-med-087's declaration apart from the one
    # settings key, so a reader can verify by eye that the pair differs only in the
    # outcome's response link.
    adjustment=[
        "G", "A", "W_pre", "L_t1", "W",
        "hs", "hs_missing", "deapp_c", "deapp_c_missing",
    ],
    model_settings=MediationModelSettings(
        outcomes=("B", "L", "W"),
        score_mean_link="three_choice_guessing_floor",
    ),
)


def fit(config: str = "dev"):
    return fit_mediation(SPEC, config=config)
