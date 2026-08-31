# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Adjusted between-child span model construction, RLI and Byrne cohorts.

Carved out of the 8,506-line ``factories.py`` by #637 stage 3, which is why
every name here is still re-exported from ``factories``. Every family module
depends only on :mod:`factories.base`; nothing crosses between families.
"""

from __future__ import annotations


from typing import TYPE_CHECKING, Iterable

import numpy as np
import pymc as pm
import pytensor.tensor as pt

if TYPE_CHECKING:
    pass


from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.fitted_payloads import (
    EmptyPayload,
)
from language_reading_predictors.statistical_models.likelihood import (
    beta_binomial_from_logit,
)
from language_reading_predictors.statistical_models.preprocessing import (
    _subset,
    PreparedData,
)
from language_reading_predictors.statistical_models.factories.base import (
    BuiltModel,
    _add_child_random_intercept,
    _alpha_sigma_for,
    _resolve_adjusted_predictor,
    _rlm_dispersion_kappa,
    _rlm_group_nuisance,
)



def build_adjusted_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str = "W",
    predictors: Iterable[str] = ("L", "lang", "B", "age", "blocks", "behav"),
    language_composite_symbols: Iterable[str] = ("R", "E", "F"),
    predictor_slope_sigma: float = 0.3,
    gamma_own_sigma: float = 0.25,
) -> BuiltModel[EmptyPayload]:
    """Between-child adjusted model: standardised T1 baselines -> word-reading gain.

    One row per child (``prepared.phase_mode`` in ``{"span", "itt"}``). The outcome post-score
    (``outcome_symbol`` at the span's later wave) is conditioned on its own T1
    baseline via ``gamma_own`` - the gain framing shared with the mechanism
    models. Each predictor enters as a single **standardised** linear term with a
    fixed weakly-informative ``Normal(0, predictor_slope_sigma)`` slope. There is
    **no** phase intercept and **no** child random intercept: with one row per
    child the coefficients are genuinely between-child associations (a random
    intercept would tilt them toward the within-child question - see the LRP65
    docstring). Passing a single-element ``predictors`` gives the bivariate
    (baseline-only-adjusted) association used for the shared-variance comparison.

        eta_i = alpha + gamma_own * logit(W_pre_i) + sum_k beta_k * z_{k,i}

    with a Beta-Binomial likelihood on the outcome post-count.

    ``gamma_own_sigma`` is the SD of the own-baseline coupling prior
    (``Normal(1, gamma_own_sigma)``, default 0.25); the family's prior sweep
    refits at the wider 0.5, the "required 0.25-vs-0.5 sensitivity" of
    :func:`priors.gamma_own_prior` (2026-08-22 review, finding 5).
    """
    if prepared.phase_mode not in {"span", "itt"}:
        raise ValueError(
            "Adjusted (between-child) model requires phase_mode in {'span', 'itt'} "
            f"(one row per child); got {prepared.phase_mode!r}"
        )
    if outcome_symbol not in prepared.pre_logit:
        raise KeyError(f"Outcome {outcome_symbol!r} missing from prepared data")

    # One row per child: drop children missing the outcome post-score.
    post = prepared.post_counts[outcome_symbol]
    keep = ~np.isnan(post)
    if not keep.all():
        prepared = _subset(prepared, keep)

    post = prepared.post_counts[outcome_symbol].astype(np.int64)
    N = prepared.n_trials[outcome_symbol]
    own_pre_logit = prepared.pre_logit[outcome_symbol]
    language_symbols = tuple(language_composite_symbols)
    resolved = [
        _resolve_adjusted_predictor(prepared, k, language_symbols) for k in predictors
    ]

    coords = {"obs_id": np.arange(prepared.n_obs)}
    with pm.Model(coords=coords) as model:
        own_pre_d = pm.Data("own_pre_logit", own_pre_logit, dims="obs_id")
        alpha = _priors.alpha_prior(
            sigma=_alpha_sigma_for(outcome_symbol)
        ).to_pymc("alpha")
        gamma_own = _priors.gamma_own_prior(sigma=gamma_own_sigma).to_pymc(
            "gamma_own"
        )
        eta = alpha + gamma_own * own_pre_d

        for coef_name, vec, _label in resolved:
            x_d = pm.Data(f"x_{coef_name}", vec, dims="obs_id")
            beta = _priors.predictor_slope_prior(predictor_slope_sigma).to_pymc(
                coef_name
            )
            eta = eta + beta * x_d

        eta = pm.Deterministic("eta", eta, dims="obs_id")
        kappa = _priors.kappa_prior().to_pymc("kappa")
        beta_binomial_from_logit(
            "y_post", eta, n_trials=N, kappa=kappa, observed=post, dims="obs_id"
        )

    return BuiltModel(model=model, prepared=prepared, payload=EmptyPayload())


def build_rlm_adjusted_model(
    frame,
    *,
    predictors: Iterable[str] | None = None,
    predictor_slope_sigma: float = 0.3,
    gamma_own_sigma: float = 0.25,
    dispersion_prior_sigma: float = 0.25,
) -> BuiltModel[EmptyPayload]:
    """Byrne between-child adjusted model (#338 Phase D, ``lrp-rlm-adj-001``).

    The RLI ``build_adjusted_model`` ported to the Byrne
    :class:`preprocessing.RlmSpanFrame`: one row per child, the outcome's later
    -wave count conditioned on its own pre-wave Haldane logit (``gamma_own`` -
    the gain framing), each standardised wave-1 predictor as a single
    ``Normal(0, predictor_slope_sigma)`` slope, plus non-interpretable group-
    nuisance dummies (``readgrp`` is an observational cohort factor - nothing
    here is causal, and there is no treatment term to protect). No child random
    intercept: one row per child keeps the coefficients genuinely between-child.
    Passing a single-element ``predictors`` gives the bivariate comparison fit.

    ``gamma_own_sigma`` is the own-baseline coupling prior SD (``Normal(1, ·)``,
    default 0.25; the family's sweep refits at 0.5). The Beta-Binomial
    concentration takes the **dispersion-scale** prior of the RLM historical
    families — ``1/sqrt(kappa) ~ HalfNormal(dispersion_prior_sigma)`` with
    ``kappa`` kept as a Deterministic — rather than ``kappa_prior``'s
    ``HalfNormal(50)``: at the Byrne denominators of the confirmed-input outcomes
    (BPVS 32, TROG 20, digit recall 34) a HalfNormal on the concentration gives
    the near-Binomial limit ``kappa >> n`` essentially no prior mass, and the
    stored ``adj-003``/``004``/``005`` posteriors sat against that prior (posterior
    SD 0.90-0.96 of the prior's, P(kappa > 100) 0.31-0.45 against the prior's
    0.046). See :func:`priors.inv_sqrt_kappa_prior` for the calibration
    (2026-08-22 adjusted-family review, finding 4).
    """
    keys = list(predictors) if predictors is not None else list(frame.predictors)
    missing = [k for k in keys if k not in frame.predictors]
    if missing:
        raise KeyError(f"Predictors {missing} not in frame (have {list(frame.predictors)}).")

    outcome = frame.outcome
    post = frame.post_counts[outcome].astype(np.int64)
    N = frame.n_trials[outcome]

    coords = {"obs_id": np.arange(frame.n_obs)}
    with pm.Model(coords=coords) as model:
        own_pre_d = pm.Data("own_pre_logit", frame.pre_logit[outcome], dims="obs_id")
        alpha = _priors.alpha_prior(sigma=1.5).to_pymc("alpha")
        gamma_own = _priors.gamma_own_prior(sigma=gamma_own_sigma).to_pymc(
            "gamma_own"
        )
        eta = alpha + gamma_own * own_pre_d

        for k in keys:
            x_d = pm.Data(f"x_{k}", frame.predictors[k], dims="obs_id")
            beta = _priors.predictor_slope_prior(predictor_slope_sigma).to_pymc(
                f"beta_{k}"
            )
            eta = eta + beta * x_d

        eta = _rlm_group_nuisance(frame, eta)
        eta = pm.Deterministic("eta", eta, dims="obs_id")
        kappa = _rlm_dispersion_kappa(dispersion_prior_sigma)
        beta_binomial_from_logit(
            "y_post", eta, n_trials=N, kappa=kappa, observed=post, dims="obs_id"
        )

    return BuiltModel(model=model, prepared=frame, payload=EmptyPayload())


def build_rlm_transition_adjusted_model(
    frame,
    *,
    predictors: Iterable[str] | None = None,
    predictor_slope_sigma: float = 0.3,
    varying_slopes: bool = False,
    gamma_own_sigma: float = 0.25,
    dispersion_prior_sigma: float = 0.25,
) -> BuiltModel[EmptyPayload]:
    """Pooled annual-transition Byrne ANCOVA with repeated-child dependence.

    The primary form shares each predictor slope over transitions while allowing
    a separate transition intercept, conditioning on the transition-start outcome
    and adding a non-centred child random intercept. ``varying_slopes=True`` is the
    pre-specified stability sensitivity: it replaces the pooled scalar slopes with
    independent transition-specific slopes under the same prior. Predictors have
    already been standardised within transition by the loader.

    ``loo_child_idx`` is deliberately separate from the model's ``child_idx``. It
    marks this likelihood for child-level aggregation in the shared PSIS-LOO code
    without changing the established row-level LOO of other repeated-row families.

    ``gamma_own_sigma`` and ``dispersion_prior_sigma`` are as in
    :func:`build_rlm_adjusted_model`: the own-baseline prior SD (0.25, swept at
    0.5) and the dispersion-scale Beta-Binomial prior shared with the RLM
    historical families (2026-08-22 adjusted-family review, findings 4 and 5).
    """
    keys = list(predictors) if predictors is not None else list(frame.predictors)
    missing = [key for key in keys if key not in frame.predictors]
    if missing:
        raise KeyError(
            f"Predictors {missing} not in frame (have {list(frame.predictors)})."
        )

    outcome = frame.outcome
    post = frame.post_counts[outcome].astype(np.int64)
    n_trials = frame.n_trials[outcome]
    coords = {
        "obs_id": np.arange(frame.n_obs),
        "child": np.arange(frame.n_children),
        "transition": frame.transition_labels,
        "predictor": keys,
    }
    with pm.Model(coords=coords) as model:
        phase_d = pm.Data("phase_idx", frame.phase, dims="obs_id")
        child_d = pm.Data("child_idx", frame.child_idx, dims="obs_id")
        pm.Data("loo_child_idx", frame.child_idx, dims="obs_id")
        own_pre_d = pm.Data(
            "own_pre_logit", frame.pre_logit[outcome], dims="obs_id"
        )
        alpha_transition = pm.Normal(
            "alpha_transition", mu=0.0, sigma=1.5, dims="transition"
        )
        gamma_own = _priors.gamma_own_prior(sigma=gamma_own_sigma).to_pymc(
            "gamma_own"
        )
        eta_fixed = alpha_transition[phase_d] + gamma_own * own_pre_d

        if varying_slopes:
            X = pm.Data(
                "X_predictor",
                np.column_stack([frame.predictors[key] for key in keys]),
                dims=("obs_id", "predictor"),
            )
            beta_transition = pm.Normal(
                "beta_transition",
                mu=0.0,
                sigma=predictor_slope_sigma,
                dims=("transition", "predictor"),
            )
            eta_fixed = eta_fixed + pt.sum(
                X * beta_transition[phase_d], axis=1
            )
        else:
            for key in keys:
                x_d = pm.Data(f"x_{key}", frame.predictors[key], dims="obs_id")
                beta = _priors.predictor_slope_prior(
                    predictor_slope_sigma
                ).to_pymc(f"beta_{key}")
                eta_fixed = eta_fixed + beta * x_d

        eta_fixed = _rlm_group_nuisance(frame, eta_fixed)
        eta_fixed = pm.Deterministic("eta_fixed", eta_fixed, dims="obs_id")
        eta = _add_child_random_intercept(eta_fixed, child_d)
        eta = pm.Deterministic("eta", eta, dims="obs_id")
        kappa = _rlm_dispersion_kappa(dispersion_prior_sigma)
        beta_binomial_from_logit(
            "y_post",
            eta,
            n_trials=n_trials,
            kappa=kappa,
            observed=post,
            dims="obs_id",
        )

    return BuiltModel(model=model, prepared=frame, payload=EmptyPayload())
