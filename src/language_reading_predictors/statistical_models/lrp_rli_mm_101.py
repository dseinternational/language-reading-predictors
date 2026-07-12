# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRPMM101 - prior-sensitivity companion to the LRPMM01 measurement model.

LRPMM01 is the reference fit and uses the model's original priors
(``lambda ~ HalfNormal(1)``, ``sigma ~ HalfNormal(1)``). LRPMM101 is **identical in
every respect except the loading and residual priors**, which take the recalibrated
values that an earlier revision of #261 had proposed as part of the convergence fix:

- ``lambda_load ~ TruncatedNormal(mu=0.6, sigma=0.5, lower=0)`` -- a positive-mode
  loading prior, moving mass off the ``lambda -> 0`` neck.
- ``sigma_indicator ~ HalfNormal(0.5)`` -- a tighter residual prior.

Everything else -- the marginal MVN measurement likelihood, the conjugate conditional
for the factor scores, the LKJ factor correlation, the Beta-Binomial structural leg,
the data, and ``target_accept = 0.999`` -- matches LRPMM01. Holding the sampler equal
is the point: the prior is the single free variable, so any posterior difference is
attributable to it and nothing else.

**Why this model exists.** #261 originally marginalised the factor scores *and*
recalibrated these priors in one step, then claimed the posterior was unchanged and
"only the sampler geometry" differed. The first half of that is true (the conjugate
rewrite is measure-preserving); the second half is not (a prior change is not). The
two were confounded, so neither the convergence fix nor the prior change could be
assessed. This companion separates them.

**What the ablation found** (reporting tier, 6 chains x 6000 draws; recorded in
``notes/202607101638-mm-001-convergence-reparameterisation.md``):

| priors | target_accept 0.95 | target_accept 0.999 |
| --- | --- | --- |
| original HalfNormal(1) | FAIL - 571 divergences, min ESS 370 | PASS - 0 divergences, min ESS 2200 |
| recalibrated | FAIL - 528 divergences, min ESS 850 | PASS - 0 divergences, min ESS 1700 |

So the prior recalibration is **neither necessary nor sufficient** for convergence:
at 0.95 both prior sets fail by a comparable margin, and at 0.999 both pass. What
clears the gate is the marginalisation (which repairs BFMI, 0.21 -> ~0.87) plus the
raised ``target_accept`` (which clears the boundary divergences). And the posteriors
agree to within Monte-Carlo error -- factor correlations within 0.02, every
structural slope to the third decimal.

Meanwhile the recalibration is **not** free: it moves the prior-implied median
communality from 0.50 to 0.79, and ``P(communality > 0.8)`` from 0.29 to 0.49. With
only two indicators per factor at ``n ~ 51`` that is a substantive prior commitment
about how well the indicators measure their domain -- exactly the quantity the model
is meant to estimate. Buying nothing and costing that, it was reverted in LRPMM01 and
retained here as the sensitivity fit.

Same caveats as LRPMM01: a **measurement / triangulation** model, not causal. Per
ID-2 every factor->gain slope is a latent-ability-confounded **adjusted association**.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.pipeline import fit_correlated_factor

SPEC = ModelSpec(
    model_id="lrp-rli-mm-101",
    kind="corr_factor",
    title=(
        "Prior sensitivity for the correlated-domain-factor measurement model "
        "(recalibrated loading / residual priors)"
    ),
    outcome_symbol="W",
    extra={
        "domains": {
            "vocabulary": ("R", "E"),
            "code": ("L", "B"),
            "grammar": ("F", "T"),
        },
        "structural_covariates": ("blocks",),
        "use_age": True,
        # The recalibrated priors. LRPMM01 uses the factory defaults, which are the
        # original HalfNormal(1) pair.
        "loading_mu": 0.6,
        "loading_sigma": 0.5,
        "residual_sigma": 0.5,
        # Matched to LRPMM01 so the prior is the ONLY difference between the fits.
        "target_accept": 0.999,
    },
)


def fit(config: str = "dev"):
    return fit_correlated_factor(SPEC, config=config)
