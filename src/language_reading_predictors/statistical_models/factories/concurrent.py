# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Concurrent (same-wave) association model construction.

Carved out of the 8,506-line ``factories.py`` by #637 stage 3, which is why
every name here is still re-exported from ``factories``. Every family module
depends only on :mod:`factories.base`; nothing crosses between families.
"""

from __future__ import annotations


from typing import TYPE_CHECKING, Iterable

import numpy as np
import pymc as pm

if TYPE_CHECKING:
    pass


from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.fitted_payloads import (
    ConcurrentPayload,
    EmptyPayload,
)
from language_reading_predictors.statistical_models.likelihood import (
    ScoreMeanLink,
    beta_binomial_from_logit,
    beta_binomial_from_score_mean_link,
)
from language_reading_predictors.statistical_models.preprocessing import (
    _subset,
    PreparedData,
    filter_informative_covariates,
    logit_safe,
    standardise,
)
from language_reading_predictors.statistical_models.factories.base import (
    BuiltModel,
    _alpha_sigma_for,
    _rlm_dispersion_kappa,
    _rlm_group_nuisance,
)

def build_concurrent_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str = "W",
    predictor_symbols: Iterable[str] = ("L", "B", "TR", "TE", "R", "E"),
    covariates: Iterable[str] = (),
    include_age: bool = True,
    include_group: bool = True,
    predictor_slope_sigma: float = 0.3,
    # #619: the phoneme-blending response link, applied to the OUTCOME's score mean.
    # ``"logit"`` is the ordinary Beta-Binomial inverse-logit mean;
    # ``"three_choice_guessing_floor"`` maps it onto [1/3, 1] for the ten
    # three-alternative forced-choice blending items, whose expected score cannot
    # fall below chance. B outcomes only, and released only beside the paired
    # ordinary-link fit. A B *predictor* is unaffected: it enters as a standardised
    # logit covariate, not as a modelled score.
    score_mean_link: ScoreMeanLink = "logit",
) -> BuiltModel[ConcurrentPayload]:
    """Concurrent conditional-associations model for ONE wave (#312).

    Expects a single-wave subset of the ``phase_mode="levels"`` frame (the pipeline
    slices ``prepared.phase == wave_idx`` before calling), so there is exactly one
    row per child. The focal outcome's post-count level is conditioned on the
    standardised same-wave logits of ``predictor_symbols`` (each a mutually-adjusted
    ``beta_{sym}`` on the raw-logit's standardised scale), optionally standardised age
    (``beta_age``) and a group nuisance term (``beta_group_nuisance``, flagged
    non-interpretable — it only absorbs arm composition):

        eta_i = alpha + Σ_k beta_k · z_k(logit predictor_k)_i
                     [+ beta_age · z(age)_i] [+ beta_group_nuisance · G_i]

    with a Beta-Binomial likelihood on the outcome post-count. Missing predictor
    values are mean-imputed (0 on the standardised scale) — PyMC cannot take NaN
    inputs and the associations are a descriptive read. Mean imputation changes the
    predictor distribution and can bias a conditional coefficient; the direction is
    not guaranteed when missingness relates to the predictor, outcome, or other
    skills. Rows missing the focal OUTCOME are dropped by the caller (an outcome
    cannot be imputed).

    Regularising ``Normal(0, predictor_slope_sigma)`` slopes are essential: with
    n ≈ 53 and a strongly inter-correlated predictor cluster, the mutually-adjusted
    coefficients are collinearity-shrunk, and each answers a *different* conditional
    question (the Table-2 fallacy — see the report). ``beta_{sym}`` is per-SD of the
    raw same-wave logit; the pipeline records each logit's SD so a ``+k items``
    marginal can be pushed through :func:`reporting.concurrent_marginals`.
    """
    if prepared.phase_mode != "levels":
        raise ValueError(
            "Concurrent model requires a phase_mode='levels' subset (one wave); "
            f"got {prepared.phase_mode!r}"
        )
    if outcome_symbol not in prepared.post_counts:
        raise KeyError(f"Outcome {outcome_symbol!r} missing from prepared data")

    # One row per child at this wave: drop children missing the focal outcome.
    keep = ~np.isnan(prepared.post_counts[outcome_symbol])
    if not keep.all():
        prepared = _subset(prepared, keep)
    if prepared.n_obs != prepared.n_children:
        raise ValueError(
            "Concurrent model expects one row per child (a single wave); got "
            f"{prepared.n_obs} rows over {prepared.n_children} children — pass a "
            "single-wave subset."
        )

    from language_reading_predictors.statistical_models.preprocessing import (
        logit_safe,
        standardise,
    )

    post = prepared.post_counts[outcome_symbol].astype(np.int64)
    N = prepared.n_trials[outcome_symbol]
    predictor_symbols = tuple(predictor_symbols)
    covariates = tuple(covariates)
    _, effective_covariates, dropped_covariates = filter_informative_covariates(
        prepared, covariates
    )
    if dropped_covariates:
        raise ValueError(
            "Concurrent covariates must be present and vary on the final fitted rows; "
            "filter them after the wave and focal-outcome masks before building. "
            f"Invalid: {', '.join(dropped_covariates)}"
        )
    if effective_covariates != covariates:
        raise ValueError(
            "Concurrent covariates must be unique and preserve their declared order"
        )

    coords = {"obs_id": np.arange(prepared.n_obs)}
    with pm.Model(coords=coords) as model:
        alpha = _priors.alpha_prior(
            sigma=_alpha_sigma_for(outcome_symbol)
        ).to_pymc("alpha")
        eta = alpha

        for sym in predictor_symbols:
            if sym not in prepared.post_counts:
                raise KeyError(
                    f"Concurrent predictor {sym!r} missing from prepared data"
                )
            z, _ = standardise(logit_safe(prepared.post_counts[sym], prepared.n_trials[sym]))
            z = np.nan_to_num(z)  # mean-impute missing (0 on the standardised scale)
            z_d = pm.Data(f"z_{sym}", z, dims="obs_id")
            beta = _priors.predictor_slope_prior(predictor_slope_sigma).to_pymc(
                f"beta_{sym}"
            )
            eta = eta + beta * z_d

        if include_age:
            age_d = pm.Data(
                "z_age", np.nan_to_num(np.asarray(prepared.A_std, dtype=float)),
                dims="obs_id",
            )
            beta_age = _priors.predictor_slope_prior(predictor_slope_sigma).to_pymc(
                "beta_age"
            )
            eta = eta + beta_age * age_d

        if include_group:
            # Group as a NON-INTERPRETABLE nuisance: absorbs arm composition at this
            # wave so it does not leak into the skill coefficients. A wide Normal(0, 1)
            # (not the regularising association prior) — it is not an association we
            # report, just a composition control. The report flags it as such.
            g_d = pm.Data("G", prepared.G.astype(float), dims="obs_id")
            beta_group = pm.Normal("beta_group_nuisance", mu=0.0, sigma=1.0)
            eta = eta + beta_group * g_d

        # Trait covariates (e.g. non-verbal ability, hearing, speech, phonological
        # memory), passed by the pipeline as t1 baselines broadcast across the waves
        # (via ``baseline_covariates=``) so the levels panel conditions on the same
        # variable set as the gains panel. Each enters as a standardised linear
        # ``gamma_{c}`` with the regularising cross-coupling prior. The caller has
        # already removed absent or fitted-row-constant terms; the fail-closed check
        # above prevents a missingness indicator from aliasing the intercept.
        for c in covariates:
            cov_vec = np.nan_to_num(np.asarray(prepared.covariates[c], dtype=float))
            cov_d = pm.Data(f"z_{c}", cov_vec, dims="obs_id")
            gamma = _priors.predictor_slope_prior(predictor_slope_sigma).to_pymc(
                f"gamma_{c}"
            )
            eta = eta + gamma * cov_d

        eta = pm.Deterministic("eta", eta, dims="obs_id")
        kappa = _priors.kappa_prior().to_pymc("kappa")
        beta_binomial_from_score_mean_link(
            "y_post", eta, n_trials=N, kappa=kappa, observed=post, dims="obs_id",
            score_mean_link=score_mean_link,
        )

    return BuiltModel(
        model=model,
        prepared=prepared,
        payload=ConcurrentPayload(score_mean_link=score_mean_link),
    )


def build_rlm_concurrent_model(
    frame,
    *,
    predictor_symbols: Iterable[str],
    include_age: bool = True,
    include_group: bool = True,
    predictor_slope_sigma: float = 0.3,
    dispersion_prior_sigma: float = 0.25,
) -> BuiltModel[EmptyPayload]:
    """One-wave Byrne concurrent conditional-associations model (#409 C1).

    This is the RLM adapter for :func:`build_concurrent_model`: a Beta-Binomial
    regression of the focal count on mutually adjusted same-wave predictor logits,
    optional age, and observational reading-group nuisance dummies. Predictor
    missingness is mean-imputed at zero after within-wave standardisation, matching
    the established family policy. Every reported slope is descriptive.

    The Beta-Binomial concentration takes the Byrne families' **dispersion-scale**
    prior (``1/sqrt(kappa) ~ HalfNormal(dispersion_prior_sigma)``, ``kappa`` a
    Deterministic, via :func:`_rlm_dispersion_kappa`) rather than
    ``kappa_prior``'s ``HalfNormal(50)``, which at these denominators excludes
    the near-Binomial limit a priori (2026-08-21 historical review, finding 8;
    extended to the adjusted, horseshoe and concurrent RLM factories on
    2026-08-22).
    """
    outcome = frame.outcome
    if outcome not in frame.post_counts:
        raise KeyError(f"Outcome {outcome!r} missing from RLM concurrent frame")
    keys = tuple(predictor_symbols)
    missing = [key for key in keys if key not in frame.post_counts]
    if missing:
        raise KeyError(f"Predictors {missing} missing from RLM concurrent frame")

    post = np.asarray(frame.post_counts[outcome], dtype=np.int64)
    n_trials = frame.n_trials[outcome]
    coords = {"obs_id": np.arange(frame.n_obs)}
    with pm.Model(coords=coords) as model:
        alpha = _priors.alpha_prior(sigma=1.5).to_pymc("alpha")
        eta = alpha
        for key in keys:
            z, _ = standardise(
                logit_safe(frame.post_counts[key], frame.n_trials[key])
            )
            predictor = pm.Data(
                f"z_{key}", np.nan_to_num(z), dims="obs_id"
            )
            beta = _priors.predictor_slope_prior(predictor_slope_sigma).to_pymc(
                f"beta_{key}"
            )
            eta = eta + beta * predictor

        if include_age:
            age = pm.Data(
                "z_age",
                np.nan_to_num(np.asarray(frame.A_std, dtype=float)),
                dims="obs_id",
            )
            beta_age = _priors.predictor_slope_prior(
                predictor_slope_sigma
            ).to_pymc("beta_age")
            eta = eta + beta_age * age

        if include_group:
            eta = _rlm_group_nuisance(frame, eta)

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
