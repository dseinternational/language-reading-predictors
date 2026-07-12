# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Guard against pipeline prior-scale fallbacks drifting from the factories.

The pipeline sources each prior-scale ``spec.extra.get(key, ...)`` fallback from
the factory signature via ``pipeline._default_of`` (issue #209 review), so the
factory is the single source of truth. This test locks the reconciled factory
defaults themselves (prior-critical-review 2026-07-07, recommendations 2-3): with
``_default_of`` feeding the pipeline, locking the factory defaults makes the
"fallback lags the factory" drift Copilot caught structurally impossible to
reintroduce silently. Change a value here only alongside a deliberate,
documented prior recalibration.
"""

import pytest

from language_reading_predictors.statistical_models import factories, priors
from language_reading_predictors.statistical_models.pipeline import _default_of

# (factory, keyword, reconciled default). Every entry the pipeline reads through
# ``_default_of``; the growth entries are locked at source too (its fit path uses
# the factory defaults directly).
RECONCILED_FACTORY_DEFAULTS = [
    (factories.build_adjusted_model, "predictor_slope_sigma", 0.3),
    (factories.build_lcsm_model, "coupling_prior_sigma", 0.3),
    (factories.build_lcsm_model, "covariate_prior_sigma", 0.3),
    (factories.build_historical_growth_model, "eta_prior_sigma", 1.5),
    (factories.build_historical_growth_model, "sigma_subject_prior_sigma", 0.5),
    (factories.build_historical_growth_model, "kappa_prior_sigma", 50.0),
    # The ORIGINAL HalfNormal(1)/HalfNormal(1) pair: TruncatedNormal(mu=0, sigma=1,
    # lower=0) is exactly HalfNormal(1). A revision of #261 briefly recalibrated
    # these to (0.6, 0.5)/0.5 alongside the marginalisation, but the LRPMM101 2x2
    # ablation showed the recalibration neither clears the gate nor moves the
    # posterior, while it does shift prior-implied median communality 0.50 -> 0.79.
    # It was reverted; LRPMM101 carries it as a prior-sensitivity fit. These
    # defaults must therefore stay at the original values.
    (factories.build_correlated_factor_model, "loading_mu", 0.0),
    (factories.build_correlated_factor_model, "loading_sigma", 1.0),
    (factories.build_correlated_factor_model, "residual_sigma", 1.0),
    (factories.build_correlated_factor_model, "predictor_slope_sigma", 0.5),
    (factories.build_growth_model, "assoc_prior_sigma", 0.3),
    (factories.build_growth_model, "re_intercept_prior_sigma", 0.5),
]


@pytest.mark.parametrize(
    "fn,param,expected",
    RECONCILED_FACTORY_DEFAULTS,
    ids=[f"{fn.__name__}.{p}" for fn, p, _ in RECONCILED_FACTORY_DEFAULTS],
)
def test_factory_prior_default_reconciled(fn, param, expected):
    assert _default_of(fn, param) == expected


def test_default_of_raises_on_unknown_param():
    # A renamed/removed factory param must fail loudly (not fall back to a stale
    # literal), which is the whole point of sourcing the default from the factory.
    with pytest.raises(KeyError):
        _default_of(factories.build_lcsm_model, "not_a_real_param")


def test_shared_constructor_scales_reconciled():
    # The other half of the reconciliation lives in the shared constructors.
    assert priors.prior_info_for_rv("gamma_own")["distribution"] == "Normal(1, 0.25)"
    assert _default_of(priors.predictor_slope_prior, "sigma") == 0.3
    assert priors.ALPHA_SIGMA_PROXIMAL == 1.5
    assert priors.ALPHA_SIGMA_DISTAL == 1.0
