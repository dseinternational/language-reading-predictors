# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""LRP304 - dispersion prior-family sensitivity for L -> W (n = 79) (#605).

Companion to **LRP101**, identical in rows, outcome, baseline, adjustment set and
coefficient priors, differing in exactly one thing: the Beta-Binomial concentration
is put on the **dispersion scale**, ``1 / sqrt(kappa) ~ HalfNormal(0.25)``, instead
of the family's shared ``kappa ~ HalfNormal(50)``.

**Why.** On a high-denominator outcome ``HalfNormal(50)`` is not a weak prior. With
``alpha + beta = kappa`` the Beta-Binomial variance inflation over Binomial is

    VIF = (kappa + n) / (kappa + 1) = 1 + (n - 1) / (kappa + 1)

so being within 10% of Binomial variance needs ``kappa >= 10 (n - 1) - 1``. At the
prior's own median (``kappa`` about 33.7) that is a **3.25x** inflation for word
reading (n = 79) and would need ``kappa > 779`` to avoid - a region
``HalfNormal(50)`` gives vanishing mass. The prior therefore *enforces* a floor on
overdispersion, and the near-Binomial limit - for a bounded count the perfectly
ordinary hypothesis "this measure shows no extra-Binomial variation beyond the child
random intercept" - is excluded a priori. That is a substantive modelling assumption
presented as a nuisance prior.

The dispersion-scale parameterisation fixes the prior's *shape*, not its scale: on
``u = 1 / sqrt(kappa)`` the no-extra-dispersion limit is simply ``u = 0``, so the
tail reaches it. It is the same constructor the ITT family offers as a sensitivity
(``lrp-rli-itt-006``, ``lrp-rli-itt-022``) and ``level_factors`` adopted as its
default under #584 decision 4.

**Read.** Compare this fit's ``beta_mech`` and the family's declared items-scale
headline contrast against LRP101's, on the same rows and the same estimand, and read
``kappa``'s posterior against its prior together with the implied variance-inflation
factor published in ``dispersion_summary.csv``. If the estimand does not move, the
registered prior is not binding here and the family default stands; if it does, the
default is carrying the answer and the family should adopt the dispersion scale, as
``level_factors`` did. Nothing here is a release gate - it is recorded evidence about
a prior choice.
"""

from language_reading_predictors.statistical_models.context import ModelSpec
from language_reading_predictors.statistical_models.mechanism import (
    MechanismModelSettings,
)
from language_reading_predictors.statistical_models.pipelines.mechanism import fit_mechanism

SPEC = ModelSpec(
    model_id="lrp-rli-mech-304",
    kind="mechanism",
    title="Dispersion prior sensitivity: letter sounds (L) -> word reading (W)",
    outcome_symbol="W",
    mechanism_symbol="L",
    adjustment=["G", "A", "W_pre"],
    model_settings=MechanismModelSettings(
        adjust_baseline_symbol="W",
        outcomes=("W", "L"),
        adjust_for=("hs", "hs_missing", "attend", "deapp_c", "deapp_c_missing"),
        linear_mechanism=True,
        kappa_prior_family="halfnormal_inverse_sqrt",
        use_age_gp=False,
        use_subject_random_intercept=True,
    ),
)


def fit(config: str = "dev"):
    return fit_mechanism(SPEC, config=config)
