# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPMM101 - prior-geometry sensitivity companion to the LRPMM01 measurement model.

LRPMM01 is the reference fit and (since #383) uses the **communality-scale**
loading / residual parameterisation: each indicator's communality is the free
parameter, ``c ~ Beta(2, 2)`` on (0, 1), with ``lambda = sqrt(c)`` and
``sigma = sqrt(1 - c)`` derived so ``lambda**2 + sigma**2 = 1`` exactly — the
unit-variance budget the standardised indicators imply. LRPMM101 is **identical in
every respect except that geometry**: it keeps the legacy free pair, ``lambda ~
HalfNormal(1)`` and ``sigma ~ HalfNormal(1)``, which leaves the budget
unconstrained and implies ``communality ~ Beta(1/2, 1/2)`` (an arcsine prior with
mass piled on both singular corners, and ~32% of loading mass above 1).

Everything else -- the marginal MVN measurement likelihood, the conjugate conditional
for the factor scores, the LKJ factor correlation, the Beta-Binomial structural leg,
the data, and ``target_accept = 0.999`` -- matches LRPMM01. Holding the sampler equal
is the point: the loading-prior geometry is the single free variable, so any
posterior difference is attributable to it and nothing else. Both geometries share
the prior median communality of 0.5, so the contrast isolates the *shape* (bounded
mid-mass Beta(2, 2) vs boundary-loving unbounded pair), not the central tendency.

**History.** This slot originally held the #261 "recalibrated" free pair
(``lambda ~ TruncatedNormal(0.6, 0.5, lower=0)``, ``sigma ~ HalfNormal(0.5)``),
registered to un-confound #261's prior change from its measure-preserving
marginalisation. That ablation (reporting tier, 6 chains x 6000 draws; recorded in
``notes/202607101638-mm-001-convergence-reparameterisation.md``) found the
recalibration **neither necessary nor sufficient** for convergence — at
``target_accept`` 0.95 both prior sets fail (571 vs 528 divergences), at 0.999 both
pass — while moving prior-implied median communality 0.50 -> 0.79 and
``P(communality > 0.8)`` 0.29 -> 0.49: a substantive commitment about exactly the
quantity the model estimates, bought for nothing. It was rejected as a default, and
its question is settled. #383 then replaced the default geometry itself (bounded
communality scale, same 0.5 median, no corner mass), so the live sensitivity
question this companion now answers is: **do the reported quantities — factor
correlations, communalities, structural slopes — depend on that geometry change?**
The prior-predictive indicator-scale check (#381) quantifies the two geometries'
calibration directly: the legacy pair generates indicator data ~1.4x too dispersed;
the communality scale is calibrated at ~1.0 by construction.

Same caveats as LRPMM01: a **measurement / triangulation** model, not causal. Per
ID-2 every factor->gain slope is a latent-ability-confounded **adjusted association**.
"""

from language_reading_predictors.statistical_models.context import (
    ModelSpec,
    StatisticalFitContext,
)
from language_reading_predictors.statistical_models.corr_factor import (
    CorrFactorModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.corr_factor import fit_correlated_factor

SPEC = ModelSpec(
    model_id="lrp-rli-mm-101",
    kind="corr_factor",
    title=(
        "Prior sensitivity for the correlated-domain-factor measurement model "
        "(legacy free loading / residual pair)"
    ),
    outcome_symbol="W",
    model_settings=CorrFactorModelSettings(
        domains=(
            ("vocabulary", ("R", "E")),
            ("code", ("L", "B")),
            ("grammar", ("F", "T")),
        ),
        structural_covariates=("blocks",),
        use_age=True,
        loading_prior="free",
    ),
        # The legacy geometry at its knob DEFAULTS (HalfNormal(1) pair): LRPMM01
        # uses the factory-default communality parameterisation, so this companion
        # varies only the geometry, not the knob values.
        # Matched to LRPMM01 so the loading-prior geometry is the ONLY difference
        # between the fits.
    target_accept=0.999,
)


def fit(config: str = "dev") -> StatisticalFitContext:
    return fit_correlated_factor(SPEC, config=config)
