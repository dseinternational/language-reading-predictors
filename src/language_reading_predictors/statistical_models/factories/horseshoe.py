# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regularised-horseshoe predictor-ranking model construction.

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
    _resolve_adjusted_predictor,
    _rlm_dispersion_kappa,
    _rlm_group_nuisance,
    _scalar_prior,
)

def _build_horseshoe_betas(
    *, tau0: float, slab_scale: float, slab_df: float
) -> pt.TensorVariable:
    """Regularized ("Finnish") horseshoe coefficient vector over the ``predictor`` coord.

    Call inside a ``pm.Model`` that declares a ``predictor`` coord. Returns the
    Deterministic ``beta`` (dims ``predictor``); see ``priors.horseshoe_*`` for the
    spec (Piironen & Vehtari 2017). Non-centred (``hs_z``) for a healthy geometry.
    """
    tau = _priors.horseshoe_tau_prior(tau0).to_pymc("hs_tau")
    c2 = _priors.horseshoe_slab_prior(slab_scale, slab_df).to_pymc("hs_c2")
    lam = _priors.horseshoe_local_prior().to_pymc("hs_lambda", dims="predictor")
    lam_tilde = pt.sqrt(c2 * lam**2 / (c2 + tau**2 * lam**2))
    z = _priors.declare(
            pm.Normal("hs_z", mu=0.0, sigma=1.0, dims="predictor"),
            role="nuisance",
            rationale=(
                "Standard-normal non-centred horseshoe coefficient offset; scaled "
                "by tau * lambda_tilde to give beta (LRPHS)."
            ),
        )
    return pm.Deterministic("beta", tau * lam_tilde * z, dims="predictor")


def _resolve_level_predictor(prepared: PreparedData, key: str) -> tuple[str, np.ndarray, str]:
    """Concurrent (same-wave) standardised predictor for the levels-mode horseshoe.

    ``age`` -> standardised age; a covariate column -> that standardised covariate;
    a measure symbol -> standardised concurrent logit of its post-count. Missing
    values are mean-imputed (0 on the standardised scale) since PyMC — unlike
    LightGBM — cannot take NaN inputs; the ranking is a sensitivity read, not a
    calibrated fit, so mean-imputation is acceptable and is noted in the report.
    Caveat (Group-C): zero-imputation shrinks a predictor's realised variance in
    proportion to its missingness, biasing that coefficient toward zero — the ranking
    therefore systematically disadvantages patchier predictors, which the report
    should flag alongside the ordering.
    """
    from language_reading_predictors.statistical_models.measures import MEASURES
    from language_reading_predictors.statistical_models.preprocessing import (
        logit_safe,
        standardise,
    )

    coef = f"beta_{key}"
    if key == "age":
        return coef, np.nan_to_num(np.asarray(prepared.A_std, dtype=float)), "Age"
    if key in prepared.covariates:
        return coef, np.nan_to_num(np.asarray(prepared.covariates[key], dtype=float)), key
    if key in prepared.post_counts:
        m = MEASURES[key]
        z, _ = standardise(logit_safe(prepared.post_counts[key], m.n_trials))
        return coef, np.nan_to_num(z), m.label
    raise KeyError(f"Unknown horseshoe level-predictor key {key!r}")


def build_horseshoe_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str = "W",
    predictors: Iterable[str],
    gain: bool = True,
    tau0: float = 0.1,
    slab_scale: float = 2.0,
    slab_df: float = 4.0,
    language_composite_symbols: Iterable[str] = ("R", "E", "F"),
    include_age: bool = True,
    use_subject_random_intercept: bool = True,
    sigma_child_prior_sigma: float = 0.5,
) -> BuiltModel[EmptyPayload]:
    """Regularized-horseshoe sparse regression for a predictor-ranking cross-check.

    An independent Bayesian read on which predictors carry signal for a single
    outcome, to sanity-check the gradient-boosting permutation/SHAP ranking
    (#116 Phase E). All predictors enter as **standardised** linear terms whose
    coefficients share a regularized-horseshoe prior (:func:`_build_horseshoe_betas`),
    so noise predictors collapse toward zero while genuinely predictive ones
    escape via the heavy local tail. Ranking = posterior ``P(|beta_k| > delta)``.

    ``gain=True`` mirrors the LRP65 between-child gain framing (``phase_mode`` in
    ``{"span", "itt"}``): outcome post-count conditioned on its own baseline
    (``gamma_own``), predictors are standardised **T1 baselines**. ``gain=False``
    is the concurrent level framing (``phase_mode="levels"``): outcome level with a
    subject random intercept for the repeated waves, predictors are standardised
    **same-wave** levels. Beta-Binomial likelihood on the outcome count either way.

    Age handling: when ``"age"`` is in ``predictors`` it is ranked as a horseshoe
    coefficient like any other construct (so it appears in the ranking and is
    comparable to the GB ranking, which includes age). In that case the separate
    unshrunk ``gamma_A`` age slope is **suppressed** even if ``include_age=True``,
    to avoid double-counting age; ``gamma_A`` is only added when age is adjusted for
    but not itself ranked.
    """
    if gain and prepared.phase_mode not in {"span", "itt"}:
        raise ValueError(
            f"horseshoe gain model needs phase_mode in {{'span','itt'}}; got {prepared.phase_mode!r}"
        )
    if not gain and prepared.phase_mode != "levels":
        raise ValueError(
            f"horseshoe level model needs phase_mode='levels'; got {prepared.phase_mode!r}"
        )
    if outcome_symbol not in prepared.post_counts:
        raise KeyError(f"Outcome {outcome_symbol!r} missing from prepared data")

    keep = ~np.isnan(prepared.post_counts[outcome_symbol])
    if not keep.all():
        prepared = _subset(prepared, keep)

    post = prepared.post_counts[outcome_symbol].astype(np.int64)
    N = prepared.n_trials[outcome_symbol]
    predictors = list(predictors)
    lang = tuple(language_composite_symbols)
    if gain:
        resolved = [_resolve_adjusted_predictor(prepared, k, lang) for k in predictors]
        resolved = [(c, np.nan_to_num(v), lbl) for c, v, lbl in resolved]
    else:
        resolved = [_resolve_level_predictor(prepared, k) for k in predictors]
    names = [c.removeprefix("beta_") for c, _v, _lbl in resolved]
    X = np.column_stack([v for _c, v, _lbl in resolved])

    coords = {"obs_id": np.arange(prepared.n_obs), "predictor": names}
    if not gain:
        coords["child"] = np.arange(prepared.n_children)
    with pm.Model(coords=coords) as model:
        X_d = pm.Data("X", X, dims=("obs_id", "predictor"))
        alpha = _scalar_prior("alpha", _priors.alpha_prior)
        eta = alpha
        if gain:
            own_pre_d = pm.Data(
                "own_pre_logit", prepared.pre_logit[outcome_symbol], dims="obs_id"
            )
            gamma_own = _priors.gamma_own_prior().to_pymc("gamma_own")
            eta = eta + gamma_own * own_pre_d
        else:
            child_idx_d = pm.Data(
                "child_idx", prepared.child_idx.astype(np.int64), dims="obs_id"
            )
            # Age is ranked as a horseshoe predictor when it is in ``predictors``
            # (so it competes under the same shrinkage as every other construct and
            # appears in the ranking). Adding a separate unshrunk ``gamma_A`` too
            # would double-count age and make its ranking uninterpretable (#160
            # review); only add the fixed age slope when age is *not* a predictor.
            if include_age and "age" not in predictors:
                A_std_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
                gamma_A = _priors.gamma_age_prior().to_pymc("gamma_A")
                eta = eta + gamma_A * A_std_d
            if use_subject_random_intercept:
                eta = _add_child_random_intercept(
                    eta, child_idx_d, sigma_prior_sigma=sigma_child_prior_sigma
                )

        beta = _build_horseshoe_betas(tau0=tau0, slab_scale=slab_scale, slab_df=slab_df)
        eta = eta + pt.dot(X_d, beta)
        eta = pm.Deterministic("eta", eta, dims="obs_id")
        kappa = _priors.kappa_prior().to_pymc("kappa")
        beta_binomial_from_logit(
            "y_post", eta, n_trials=N, kappa=kappa, observed=post, dims="obs_id"
        )

    return BuiltModel(model=model, prepared=prepared, payload=EmptyPayload())


def build_rlm_horseshoe_model(
    frame,
    *,
    predictors: Iterable[str] | None = None,
    tau0: float = 0.1,
    slab_scale: float = 2.0,
    slab_df: float = 4.0,
    dispersion_prior_sigma: float = 0.25,
) -> BuiltModel[EmptyPayload]:
    """Byrne regularised-horseshoe ranking model (#338 Phase D, ``lrp-rlm-hs-001``).

    The Beta-Binomial concentration takes the Byrne families' dispersion-scale
    prior (``1/sqrt(kappa) ~ HalfNormal(dispersion_prior_sigma)``, ``kappa`` a
    Deterministic; :func:`_rlm_dispersion_kappa`), as the RLM adjusted partner of
    each horseshoe frame does since 2026-08-22 — the two fits share a frame and
    must share the dispersion prior.

    The RLI gain-framing ``build_horseshoe_model`` on the Byrne span frame: the
    outcome's later-wave count conditioned on its own pre-wave logit
    (``gamma_own``), with every standardised wave-1 predictor (age included)
    sharing a regularised-horseshoe prior so noise predictors collapse toward
    zero. Group-nuisance dummies stay **outside** the horseshoe (unshrunk
    ``Normal(0, 1)``): the cohort factor is a nuisance, not a ranked construct,
    and shrinking it would leak group composition into the skill ranking.
    Ranking = posterior ``P(|beta_k| > delta)``, as in the RLI family.
    """
    keys = list(predictors) if predictors is not None else list(frame.predictors)
    missing = [k for k in keys if k not in frame.predictors]
    if missing:
        raise KeyError(f"Predictors {missing} not in frame (have {list(frame.predictors)}).")

    outcome = frame.outcome
    post = frame.post_counts[outcome].astype(np.int64)
    N = frame.n_trials[outcome]
    X = np.column_stack([frame.predictors[k] for k in keys])

    coords = {"obs_id": np.arange(frame.n_obs), "predictor": keys}
    with pm.Model(coords=coords) as model:
        X_d = pm.Data("X", X, dims=("obs_id", "predictor"))
        own_pre_d = pm.Data("own_pre_logit", frame.pre_logit[outcome], dims="obs_id")
        alpha = _scalar_prior("alpha", _priors.alpha_prior)
        gamma_own = _priors.gamma_own_prior().to_pymc("gamma_own")
        eta = alpha + gamma_own * own_pre_d
        eta = _rlm_group_nuisance(frame, eta)

        beta = _build_horseshoe_betas(tau0=tau0, slab_scale=slab_scale, slab_df=slab_df)
        eta = eta + pt.dot(X_d, beta)
        eta = pm.Deterministic("eta", eta, dims="obs_id")
        kappa = _rlm_dispersion_kappa(dispersion_prior_sigma)
        beta_binomial_from_logit(
            "y_post", eta, n_trials=N, kappa=kappa, observed=post, dims="obs_id"
        )

    return BuiltModel(model=model, prepared=frame, payload=EmptyPayload())
