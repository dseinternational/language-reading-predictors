# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP78 - interventional-effects decomposition of word reading via letter sounds.

Every other mediation model reports **natural** direct/indirect effects (NDE/NIE).
But the DAG's own identification analysis (ID-2, §11) states that dose ``IS`` is a
**treatment-induced mediator-outcome confounder** of the letter-sound -> reading
path — and under a treatment-induced confounder the natural effects are formally
**not identified** (no adjustment set recovers them). The randomised
**interventional** analogue effects (Vansteelandt & VanderWeele; VanderWeele,
Vansteelandt & Tchetgen Tchetgen 2014) remain well-defined in exactly that setting,
because the mediator is set to a random draw from its *population* arm-g
distribution rather than the unit's own natural counterfactual — severing the
unit-level link to the treatment-induced confounder.

LRP78 is therefore the estimand-class repair of LRP59: the same joint mediator +
outcome model (letter sounds L -> word reading W, adjustment {G, A, E, R, W_pre,
L_t1}), but the g-formula returns the **interventional direct/indirect effects**
(IDE / IIE) instead of NDE / NIE. Compare its IIE against LRP59's NIE: agreement
means the natural decomposition was not badly distorted by the dose confounder;
divergence flags that it was.

Implementation: `mediation.decompose(interventional=True)` permutes the simulated
mediator across units per replicate (a fresh population draw for each
counterfactual cell). The t3 temporal-ordering sensitivity is skipped (it is a
natural-effect construction; the interventional estimand is the point here).

Honesty: the interventional effects fix the *treatment-induced*-confounder problem,
but the mediator-outcome relationship is **still `GA`-confounded** (ID-2's other
leg), so IDE/IIE remain adjusted associations under stated assumptions — a *better*
estimand class, not a causal route. ``n ~ 53`` -> wide intervals.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_mediation

SPEC = ModelSpec(
    model_id="lrp-rli-med-078",
    kind="mediation",
    title=(
        "Interventional-effects decomposition: word reading (W) via letter sounds "
        "(L), robust to the treatment-induced dose confounder"
    ),
    outcome_symbol="W",
    mechanism_symbol="L",  # the mediator
    adjustment=["G", "A", "E", "R", "W_pre", "L_t1"],
    extra={"estimand": "interventional"},
)


def fit(config: str = "dev"):
    return fit_mediation(SPEC, config=config)
