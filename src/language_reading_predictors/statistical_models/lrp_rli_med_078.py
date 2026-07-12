# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP78 - interventional-effects decomposition of word reading via letter sounds.

Every other mediation model reports **natural** direct/indirect effects (NDE/NIE).
The natural effects rely on a *cross-world* quantity, and under this DAG they are
**not identified** — latent general ability confounds the mediator->outcome path,
and the intervention dose ``IS`` is a treatment-induced mediator-outcome confounder
(ID-2, §11). The randomised **interventional** analogue (IDE / IIE; VanderWeele,
Vansteelandt & Robins 2014) invokes no cross-world quantity and so escapes the
dose obstacle, but it still assumes **no unmeasured mediator-outcome confounding**
(Hejazi et al. 2022, assumption A5) — a weaker-assumption target, not an identified
one. See METHODS.md and #260.

LRP78 is the interventional-*interpretation* companion to LRP59: the **same** joint
mediator + outcome model (letter sounds L -> word reading W) with the **same**
adjustment set, but the decomposition is labelled interventional (IDE / IIE). The
mediator is drawn from its fitted covariate-conditional law ``P(M | C, g)`` within
strata — exactly what the g-formula already simulates — so in this fully parametric
model with no unit-level latent terms the interventional draw **coincides
numerically** with LRP59's natural-branch computation. The difference is
interpretive (a weaker-assumption estimand class), not a different number.

Do **not** read an IIE-vs-NIE gap as a dose-distortion diagnostic: ``IS`` never
enters the fitted model, so the interventional functional can neither repair nor
detect dose confounding. (An earlier version drew the mediator from the *marginal*
population distribution by permuting it across units — a cruder estimand whose
divergence from LRP59 reflected covariate decoupling, not dose confounding — which
is corrected in ``mediation.decompose``; #268.) The t3 temporal-ordering
sensitivity is skipped (a natural-effect construction). ``n ~ 53`` -> wide
intervals; IDE/IIE remain adjusted associations under stated assumptions.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mediation

SPEC = ModelSpec(
    model_id="lrp-rli-med-078",
    kind="mediation",
    title=(
        "Interventional-effects decomposition: word reading (W) via letter sounds "
        "(L) - the interventional-interpretation companion to LRP59"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",  # the mediator
    # Mirror LRP59 exactly (post-#259) so the IDE/IIE-vs-NDE/NIE comparison is
    # like-for-like: same adjustment set, only the estimand class differs (#268).
    adjustment=[
        "G", "A", "E", "R", "L_t1", "W_pre",
        "hs", "hs_missing", "deapp_c", "deapp_c_missing",
    ],
    extra={"estimand": "interventional"},
)


def fit(config: str = "dev"):
    return fit_mediation(SPEC, config=config)
