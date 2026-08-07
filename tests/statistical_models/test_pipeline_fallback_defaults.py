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
    # Widened 0.5 -> 1.0 (#383, prior-critical-review 2026-07-21): the DS
    # verbal/reading sigma_subject posteriors (1.25-1.39) sat at/beyond the
    # HalfNormal(0.5) 99th percentile — a genuine prior-data conflict.
    (factories.build_historical_growth_model, "sigma_subject_prior_sigma", 1.0),
    (factories.build_rlm_joint_growth_model, "sigma_subject_prior_sigma", 1.0),
    (factories.build_historical_growth_model, "kappa_prior_sigma", 50.0),
    # Default loading geometry is the communality scale (#383): communality ~
    # Beta(2, 2) with lambda = sqrt(c), sigma = sqrt(1 - c), enforcing the
    # lambda**2 + sigma**2 = 1 budget standardised indicators imply while keeping
    # the prior median communality of 0.5 that the LRPMM101 ablation defended
    # (the rejected 0.6/0.5 recalibration shifted it to 0.79).
    (factories.build_correlated_factor_model, "loading_prior", "communality"),
    (factories.build_correlated_factor_model, "comm_alpha", 2.0),
    (factories.build_correlated_factor_model, "comm_beta", 2.0),
    # The legacy free-pair knobs stay at the ORIGINAL HalfNormal(1)/HalfNormal(1)
    # values: TruncatedNormal(mu=0, sigma=1, lower=0) is exactly HalfNormal(1),
    # and LRPMM101 (loading_prior="free" at these defaults) is the
    # geometry-sensitivity companion, so they must not drift.
    (factories.build_correlated_factor_model, "loading_mu", 0.0),
    (factories.build_correlated_factor_model, "loading_sigma", 1.0),
    (factories.build_correlated_factor_model, "residual_sigma", 1.0),
    # Structural-slope prior reconciled 0.5 -> 0.3 to match the shared
    # predictor_slope_prior default (review finding B4, 2026-07-13).
    (factories.build_correlated_factor_model, "predictor_slope_sigma", 0.3),
    # #382 item 1: unset by default — only the LRPMM102 sensitivity companion
    # widens the focal beta_factor / beta_G pair to the N(0, 1) mechanism scale.
    (factories.build_correlated_factor_model, "focal_slope_sigma", None),
    # #383 follow-up: the longitudinal CFA takes the pooled-budget communality
    # parameterisation — communality ~ Beta(2, 2) with lambda / sigma derived so
    # the model-implied POOLED indicator variance is exactly 1 (the exact budget
    # is lambda**2 + sigma**2 = 1 / (1 + c V), V being the wave-mean spread, since
    # pooled standardisation puts 5-18% of the unit variance between waves). The
    # legacy free-pair knobs stay at the original HalfNormal(1) values so a
    # geometry-only sensitivity contrast remains constructible.
    (factories.build_longitudinal_corr_factor_model, "loading_prior", "communality"),
    (factories.build_longitudinal_corr_factor_model, "comm_alpha", 2.0),
    (factories.build_longitudinal_corr_factor_model, "comm_beta", 2.0),
    (factories.build_longitudinal_corr_factor_model, "loading_sigma", 1.0),
    (factories.build_longitudinal_corr_factor_model, "residual_sigma", 1.0),
    (factories.build_growth_model, "assoc_prior_sigma", 0.3),
    (factories.build_growth_model, "re_intercept_prior_sigma", 0.5),
    # #389 finding 2: the level family's zero-sum wave-deviation scale (sized so
    # the largest observed wave deviation, ~0.85 logits, sits within ~1.3
    # marginal prior SD) and the sweep-only override for the focal t2 contrast
    # (None = the outcome-tier default; only the treatment-prior sweep sets it).
    (factories.build_level_factors_model, "alpha_time_prior_sigma", 0.75),
    (factories.build_level_factors_model, "tau_prior_sigma", None),
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


@pytest.mark.parametrize(
    "extra,match",
    [
        # Free-pair knobs under the (default) communality parameterisation would
        # be silently ignored by the fitted model — reject before any IO (#383).
        ({"loading_mu": 0.6}, "loading_prior='free'"),
        ({"loading_prior": "communality", "residual_sigma": 0.5}, "loading_prior='free'"),
        # And the converse: Beta shapes under the legacy free pair.
        ({"loading_prior": "free", "comm_alpha": 3.0}, "communality"),
        # Unknown parameterisations are rejected at the spec boundary too.
        ({"loading_prior": "bounded"}, "loading_prior"),
    ],
    ids=["mu-under-communality", "residual-under-communality", "comm-under-free", "unknown-mode"],
)
def test_corr_factor_loading_settings_coherence(extra, match):
    """fit_correlated_factor rejects loading knobs its parameterisation ignores,
    before make_context resets the output directory (#383, the #455 principle)."""
    from language_reading_predictors.statistical_models.context import ModelSpec
    from language_reading_predictors.statistical_models.pipeline import (
        fit_correlated_factor,
    )

    spec = ModelSpec(
        model_id="lrp-test-corr-guard",
        kind="corr_factor",
        title="guard",
        outcome_symbol="W",
        extra=dict(extra),
    )
    with pytest.raises(ValueError, match=match):
        fit_correlated_factor(spec, config="dev")


@pytest.mark.parametrize(
    "extra,match",
    [
        # Free-pair knobs under the (default) pooled-budget communality
        # parameterisation would be silently ignored — reject before any IO
        # (#383 follow-up; the lcf builder has no loading_mu knob).
        ({"loading_sigma": 0.5}, "loading_prior='free'"),
        ({"loading_prior": "communality", "residual_sigma": 0.5}, "loading_prior='free'"),
        ({"loading_prior": "free", "comm_beta": 3.0}, "communality"),
        ({"loading_prior": "bounded"}, "loading_prior"),
    ],
    ids=["sigma-under-communality", "residual-under-communality", "comm-under-free", "unknown-mode"],
)
def test_lcf_loading_settings_coherence(extra, match):
    """fit_longitudinal_corr_factor rejects loading knobs its parameterisation
    ignores, before make_context resets the output directory (#383 follow-up)."""
    from language_reading_predictors.statistical_models.context import ModelSpec
    from language_reading_predictors.statistical_models.pipeline import (
        fit_longitudinal_corr_factor,
    )

    spec = ModelSpec(
        model_id="lrp-test-lcf-guard",
        kind="long_corr_factor",
        title="guard",
        outcome_symbol=None,
        extra=dict(extra),
    )
    with pytest.raises(ValueError, match=match):
        fit_longitudinal_corr_factor(spec, config="dev")
