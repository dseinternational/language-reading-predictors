# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP84 - period-resolved dose-response: intervention dose -> phoneme blending (B).

The phoneme-blending companion to ``lrp-rli-dose-077`` (word reading), completing the
dose-response family's coverage of the two largest available-case modified ITT estimates (L and B) (#228 item
2). Same observational estimand and causal structure as dose-077 -- see that module's
docstring for the full treatment; only the outcome changes.

Estimand: among rows **on the intervention**, how blending **conditional change** relates
to how many sessions were attended, with partial pooling across the three periods and a
test of whether that slope varies by period. The outcome is the Beta-Binomial
post-count of B conditional on its own baseline logit (``adjust_baseline_symbol =
"B"``, ``n_trials = 10``) -- conditional change, never raw change.

Causal structure (revised DAG): the focal edge is ``sessions -> outcome``. There is **no
clean back-door set** -- ``IS`` has parents ``A -> IS``, ``GA -> IS`` and ``IG -> IS``, all
of which also point into the outcomes. Age and assigned group are measured and adjusted;
**latent general ability GA is not**, and Frank's 2012 caveat (poorest attenders were the
least able to learn) is exactly that edge. B_pre is the autoregression / RTM control
(parameterisation, not back-door blocking). The cumulative prior dose (``attend_cumul``) is
a descendant of the ``IS`` collider and is **not** conditioned on (#269).

As in LRP77, presence and intensity are separated (#587): sessions are centred and
standardised over the on-intervention rows only, ``theta_treated`` carries the extensive
margin (randomised when read in period 1, where arm and session count correlate at 0.970),
arm enters only from period 2 as intervention order, and the exposure is split into
between-child and within-child components. Every dose slope is an **adjusted association,
not "dose drives gains"**. G is coded ``G = 2 - group`` (G=1 = immediate-intervention,
G=0 = waitlist; positive = benefit).
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.dose_response import (
    DoseResponseModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.dose_response import fit_dose_response

SPEC = ModelSpec(
    model_id="lrp-rli-dose-084",
    kind="dose_response",
    title="Period-resolved dose-response: intervention dose -> phoneme blending (B)",
    outcome_symbol="B",
    adjustment=["G", "A", "B_pre"],
    model_settings=DoseResponseModelSettings(
        adjust_baseline_symbol="B",
        dose_covariate="attend",
        period_varying_dose=True,
        use_subject_random_intercept=True,
        outcomes=("B",),
    ),
    extra={
        # Same period-varying dose geometry as LRP77, but far milder here: 1 divergence
        # at the reporting preset's 0.95, 0 at 0.97 (R-hat 1.001, min ESS 6,720). Kept
        # at the value actually validated rather than raised to the family's 0.99 — the
        # default seed is fixed, so this reproduces the stored fit exactly. See
        # notes/202608050649-reporting-refit-predictive-checks.md.
        "target_accept": 0.97,
    },
)


def fit(config: str = "dev"):
    return fit_dose_response(SPEC, config=config)
