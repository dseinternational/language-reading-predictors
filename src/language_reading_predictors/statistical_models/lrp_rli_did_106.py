# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPDID106 - dispersion-prior sensitivity for LRPDID05 (receptive vocabulary, R).

The **high-denominator** half of the family's dispersion check (#576 material
qualification 2); LRPDID105 is the low-denominator half. Receptive vocabulary is the
170-item test the qualification is computed on: under the family default
``kappa ~ HalfNormal(50)`` the prior median already implies roughly 5.9 times
Binomial variance at that length, and coming within 10 % of Binomial variance would
need ``kappa`` beyond anything the prior gives real mass to. Whatever the posterior
says about dispersion there, the prior is doing much of the saying.

This companion refits LRPDID05 with ``1 / sqrt(kappa) ~ HalfNormal(0.25)`` — the
dispersion-scale parameterisation the ITT and level families adopted for the same
reason — which can reach the near-Binomial limit. Nothing else changes.

Read it beside LRPDID105 (18 items), where the same default prior is far more
permissive. A shift here with none there localises the effect to the denominator, and
is a reason to revisit the family default rather than to reinterpret one outcome.
Note also that receptive vocabulary is a distal outcome whose arm gaps are wide in
both directions, so this sensitivity is about the *dispersion* the model is willing
to estimate, not about resolving that outcome's direction.

Reading rules are LRPDID05's; the dispersion prior changes no term's causal status.
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.did import DiDModelSettings
from language_reading_predictors.statistical_models.pipelines.did import fit_did

SPEC = ModelSpec(
    model_id="lrp-rli-did-106",
    kind="did",
    title=(
        "Dispersion-prior sensitivity for the receptive-vocabulary arm-by-wave "
        "contrasts (ROWPVT) (R)"
    ),
    outcome_symbol="R",
    family="did",
    design="waitlist-crossover arm-by-wave levels, inverse-sqrt dispersion prior",
    estimand_type="mixed",
    causal_status="t2 randomised; t3 a randomised treatment-schedule contrast",
    model_settings=DiDModelSettings(
        # Identical to LRPDID05 except kappa_prior_family.
        outcomes=("R",),
        waves=(0, 1, 2),
        use_child_re=True,
        use_age=True,
        dose=False,
        kappa_prior_family="halfnormal_inverse_sqrt",
    ),
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_did(SPEC, config=config)
