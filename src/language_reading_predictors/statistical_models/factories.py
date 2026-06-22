# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model factories for LRP52-LRP60.

Three factories are provided:

- :func:`build_itt_model` — LRP52, LRP53, LRP54, LRP60 (one outcome, RCT phase).
- :func:`build_joint_model` — LRP55 (eight outcomes, RCT phase, LKJ Σ).
- :func:`build_mechanism_model` — LRP56, LRP57, LRP58 (adjustment-set
  mechanism regressions on ``W_post`` using all phases).

All three return a tuple ``(model, variables)`` where ``variables`` is a dict
mapping names to PyMC variables, used by the pipeline to extract posterior
draws and assemble report tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.hsgp import (
    build_hsgp_1d,
    build_tau_modifier,
)
from language_reading_predictors.statistical_models.likelihood import (
    beta_binomial_from_logit,
)
from language_reading_predictors.statistical_models.measures import ITT_OUTCOMES
from language_reading_predictors.statistical_models.preprocessing import PreparedData


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _scalar_prior(name: str, prior_ctor) -> pt.TensorVariable:
    return prior_ctor().to_pymc(name)


@dataclass
class BuiltModel:
    model: pm.Model
    variables: dict[str, pt.TensorVariable]
    prepared: PreparedData
    """The (possibly row-subset) prepared data that the model was built on.

    Factories may drop rows with missing post-scores or missing confounder
    values; this attribute exposes the actually-used data so the pipeline can
    align posterior indices to input rows.
    """


# ---------------------------------------------------------------------------
# LRP52-LRP54: ITT factory
# ---------------------------------------------------------------------------


def build_itt_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str,
    use_age_gp: bool = True,
    use_own_baseline_gp: bool = True,
    use_varying_tau: bool = False,
    adjust_for: Iterable[str] = (),
) -> BuiltModel:
    """
    Build the single-outcome ITT model used by LRP52, LRP53, LRP54.

    The linear predictor is

        eta_i = alpha
              + tau * G_i
              + gamma_own * logit(y_pre_i, N_own)
              + sum_{k != own} gamma_k * logit(k_pre_i, N_k)
              + f_A(A_std_i)                          # optional HSGP
              + f_ypre(logit(y_pre_i, N_own))         # optional HSGP

    with Beta-Binomial likelihood on the post-score count (``N_own`` trials).

    Parameters
    ----------
    prepared
        Output of :func:`preprocessing.load_and_prepare` with ``phase_mode="itt"``.
    outcome_symbol
        Target measure (``"W"``, ``"R"``, ``"E"``, ...).
    use_age_gp, use_own_baseline_gp
        Toggles for the two HSGP main effects.
    use_varying_tau
        If True, the treatment effect is modelled as ``tau0 + g_tauA(A_std)``
        via a :func:`build_tau_modifier` GP with the tight ``HalfNormal(0.3)``
        amplitude prior.
    adjust_for
        Standardised non-outcome covariates from ``prepared.covariates`` to add
        as linear adjustment terms. Coefficients use the same weak
        ``Normal(0, 0.3)`` prior as cross-baseline couplings.
    """
    if prepared.phase_mode != "itt":
        raise ValueError(
            f"build_itt_model expects phase_mode='itt', got {prepared.phase_mode!r}"
        )
    if outcome_symbol not in prepared.pre_logit:
        raise KeyError(f"Outcome {outcome_symbol!r} missing from prepared data")

    own = outcome_symbol
    adjust_for = tuple(adjust_for)
    missing_adjusters = [c for c in adjust_for if c not in prepared.covariates]
    if missing_adjusters:
        raise KeyError(
            "Requested adjustment covariates missing from prepared data: "
            f"{missing_adjusters}"
        )
    cross = [s for s in ITT_OUTCOMES if s != own]

    post = prepared.post_counts[own]
    if np.any(np.isnan(post)):
        keep = ~np.isnan(post)
        if not keep.all():
            prepared = _subset(prepared, keep)
            post = prepared.post_counts[own]

    post = post.astype(np.int64)
    y_pre_logit = prepared.pre_logit[own]

    coords = {"obs_id": np.arange(prepared.n_obs)}
    G_f = prepared.G.astype(float)

    with pm.Model(coords=coords) as model:
        pm.Data("A_std", prepared.A_std, dims="obs_id")
        G_d = pm.Data("G", G_f, dims="obs_id")
        own_pre_d = pm.Data("own_pre_logit", y_pre_logit, dims="obs_id")
        cross_pre_data: dict[str, pt.TensorVariable] = {}
        for s in cross:
            cross_pre_data[s] = pm.Data(
                f"{s}_pre_logit", prepared.pre_logit[s], dims="obs_id"
            )
        adjust_data: dict[str, pt.TensorVariable] = {}
        for c in adjust_for:
            adjust_data[c] = pm.Data(f"{c}_std", prepared.covariates[c], dims="obs_id")

        alpha = _scalar_prior("alpha", _priors.alpha_prior)
        tau0 = _scalar_prior("tau", _priors.tau_prior)
        gamma_own = _scalar_prior("gamma_own", _priors.gamma_own_prior)

        gamma_cross: dict[str, pt.TensorVariable] = {}
        for s in cross:
            gamma_cross[s] = _priors.gamma_cross_prior().to_pymc(f"gamma_{s}")

        cross_contrib: pt.TensorVariable | float = 0.0
        for s in cross:
            cross_contrib = cross_contrib + gamma_cross[s] * cross_pre_data[s]

        adjust_contrib: pt.TensorVariable | float = 0.0
        for c in adjust_for:
            gamma_c = _priors.gamma_cross_prior().to_pymc(f"gamma_{c}")
            adjust_contrib = adjust_contrib + gamma_c * adjust_data[c]

        eta = alpha + gamma_own * own_pre_d + cross_contrib + adjust_contrib

        if use_age_gp:
            f_A = build_hsgp_1d("f_A", prepared.A_std)
            eta = eta + f_A
        if use_own_baseline_gp:
            f_ypre = build_hsgp_1d("f_ypre", y_pre_logit)
            eta = eta + f_ypre

        if use_varying_tau:
            g_tauA = build_tau_modifier("g_tauA", prepared.A_std)
            tau_i = pm.Deterministic("tau_i", tau0 + g_tauA, dims="obs_id")
            eta = eta + tau_i * G_d
        else:
            eta = eta + tau0 * G_d

        eta = pm.Deterministic("eta", eta, dims="obs_id")

        kappa = _scalar_prior("kappa", _priors.kappa_prior)

        beta_binomial_from_logit(
            "y_post",
            eta,
            n_trials=prepared.n_trials[own],
            kappa=kappa,
            observed=post,
            dims="obs_id",
        )

    variables = _variables_dict(model)
    return BuiltModel(model=model, variables=variables, prepared=prepared)


# ---------------------------------------------------------------------------
# LRP55: joint model
# ---------------------------------------------------------------------------


def build_joint_model(
    prepared: PreparedData,
    *,
    outcomes: Iterable[str] = ITT_OUTCOMES,
    use_age_gp: bool = False,
    partial_pool_age_gp: bool = True,
    use_residual_correlation: bool = False,
) -> BuiltModel:
    """
    Build the LRP55 joint Beta-Binomial model.

    For each child i and outcome k, the model is

        eta_{k,i} = alpha_k + tau_k * G_i
                    + gamma_own_k * logit(k_pre_i, N_k)
                    + sum_{j != k} gamma_{k,j} * logit(j_pre_i, N_j)
                    [ + f_A_k(A_std_i)         if use_age_gp ]
                    [ + u_{k,i}                if use_residual_correlation ]

    with per-outcome Beta-Binomial likelihood on the post-score count.

    ``use_age_gp`` (default False): when True, adds an HSGP on standardised
    age. With ``partial_pool_age_gp=True`` this is a partial-pooled age GP
    ``f_A_k = mu_A + delta_A_k`` (shared mean GP + outcome-specific
    deviations with a tight HalfNormal(0.3) amplitude); with False it is
    ``K`` independent ``f_A_k`` GPs. Turned off by default after the
    2026-04-18 LRP55 follow-up fit showed the age-GP amplitudes were the
    residual source of ~8 % divergent transitions; LOO does not prefer a
    model with the GP included. See
    notes/202604181700-lrp55-age-gp-drop.md for the rationale.

    ``use_residual_correlation`` (default False): when True, adds an
    ``u_i ~ MvNormal(0, Sigma)`` residual with ``Sigma = diag(sigma) Corr
    diag(sigma)`` and ``Corr ~ LKJCorr(eta=4)``, non-centred via
    ``pm.LKJCholeskyCov`` + ``z_raw``. Turned off by default after the
    2026-04-18 LRP55 fit showed the LKJ block was prior-dominated (all
    off-diagonal correlation CIs spanning zero, sigma_outcome CIs reaching
    zero). See notes/202604181600-lrp52-58-findings.md for the rationale.
    Keep both flags available for explicit sensitivity fits.
    """
    outcomes = tuple(outcomes)
    for s in outcomes:
        if s not in prepared.pre_logit:
            raise KeyError(f"Outcome {s!r} missing from prepared data")
    if prepared.phase_mode != "itt":
        raise ValueError("LRP55 joint model requires phase_mode='itt'")

    K = len(outcomes)
    N_obs = prepared.n_obs

    # Observation masks (per outcome) for rows with observed post values.
    mask = np.stack(
        [~np.isnan(prepared.post_counts[s]) for s in outcomes], axis=1
    )  # (N_obs, K)
    post_counts_int = np.stack(
        [np.nan_to_num(prepared.post_counts[s], nan=0.0).astype(np.int64) for s in outcomes],
        axis=1,
    )  # (N_obs, K)
    n_trials_vec = np.array([prepared.n_trials[s] for s in outcomes], dtype=int)

    coords = {
        "obs_id": np.arange(N_obs),
        "outcome": list(outcomes),
        "baseline": list(outcomes),
        # Second outcome axis for outcome×outcome quantities (residual
        # correlation). Cannot reuse "outcome" because PyMC requires
        # distinct dim names per axis.
        "outcome2": list(outcomes),
    }

    G_f = prepared.G.astype(float)

    with pm.Model(coords=coords) as model:
        pm.Data("A_std", prepared.A_std, dims="obs_id")
        G_d = pm.Data("G", G_f, dims="obs_id")

        # Pre-score matrix (N_obs, K) - same order as ``outcomes``.
        pre_logit = np.stack([prepared.pre_logit[s] for s in outcomes], axis=1)
        pre_logit_data = pm.Data(
            "pre_logit", pre_logit, dims=("obs_id", "baseline")
        )

        # Per-outcome scalar parameters
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.5, dims="outcome")
        tau = pm.Normal("tau", mu=0.0, sigma=0.5, dims="outcome")
        gamma_own = pm.Normal("gamma_own", mu=1.0, sigma=0.5, dims="outcome")

        # Cross-baseline couplings: (K outcomes) x K baselines; mask the
        # diagonal to enforce "own baseline handled separately".
        gamma_cross_mat = pm.Normal(
            "gamma_cross", mu=0.0, sigma=0.3, dims=("outcome", "baseline")
        )
        mask_offdiag = 1.0 - np.eye(K)
        gamma_cross_eff = pm.Deterministic(
            "gamma_cross_eff",
            gamma_cross_mat * mask_offdiag,
            dims=("outcome", "baseline"),
        )

        # Own-baseline contribution: (N_obs, K) - elementwise by outcome index.
        own_contrib = gamma_own[None, :] * pre_logit_data
        # Cross-baseline contribution: sum over baselines for each outcome.
        cross_contrib = pt.dot(pre_logit_data, gamma_cross_eff.T)

        eta_core = (
            alpha[None, :]
            + tau[None, :] * pt.shape_padright(G_d)
            + own_contrib
            + cross_contrib
        )

        if use_age_gp:
            if partial_pool_age_gp:
                mu_A = build_hsgp_1d("mu_A", prepared.A_std)
                deltas = []
                for s in outcomes:
                    deltas.append(
                        build_hsgp_1d(
                            f"delta_A_{s}",
                            prepared.A_std,
                            amplitude_prior=_priors.eta_partial_pool_prior(),
                        )
                    )
                f_A = pt.stack([mu_A + deltas[k] for k in range(K)], axis=1)
            else:
                f_A = pt.stack(
                    [build_hsgp_1d(f"f_A_{s}", prepared.A_std) for s in outcomes],
                    axis=1,
                )
            eta_core = eta_core + f_A

        if use_residual_correlation:
            # Non-centred MvNormal on u_i = L z_i where L is the Cholesky
            # factor of the residual covariance Σ. ``pm.LKJCholeskyCov``
            # with ``sd_dist=HalfNormal(0.5)`` already bakes per-outcome
            # standard deviations into ``chol`` (Σ = chol @ chol.T), so
            # there is no separate outer sigma_outcome term — a previous
            # version multiplied chol by an independent HalfNormal which
            # double-scaled Σ and made the block unidentified.
            chol, corr, sigmas = pm.LKJCholeskyCov(
                "u_chol",
                n=K,
                eta=4.0,
                sd_dist=pm.HalfNormal.dist(0.5),
                compute_corr=True,
            )
            # u_corr is outcome × outcome (not outcome × baseline) — use
            # the dedicated ``outcome2`` coord to label the second axis.
            pm.Deterministic("u_corr", corr, dims=("outcome", "outcome2"))
            pm.Deterministic("sigma_outcome", sigmas, dims="outcome")
            z_raw = pm.Normal(
                "u_z", mu=0.0, sigma=1.0, dims=("obs_id", "outcome")
            )
            # u_i = chol @ z_i ⇒ rowwise U = Z @ chol.T.
            u = pm.Deterministic(
                "u", pt.dot(z_raw, chol.T), dims=("obs_id", "outcome")
            )
            eta = eta_core + u
        else:
            eta = eta_core

        eta = pm.Deterministic("eta", eta, dims=("obs_id", "outcome"))

        kappa = pm.HalfNormal("kappa", sigma=50.0, dims="outcome")

        mu = pm.math.sigmoid(eta)
        from dse_research_utils.math.constants import EPSILON  # local import

        mu_clip = pm.math.clip(mu, EPSILON, 1 - EPSILON)
        alpha_bb = mu_clip * kappa[None, :]
        beta_bb = (1 - mu_clip) * kappa[None, :]

        # Flatten using explicit nonzero indices - robust across pytensor versions.
        idx_row, idx_col = np.nonzero(mask)
        flat_alpha = alpha_bb[idx_row, idx_col]
        flat_beta = beta_bb[idx_row, idx_col]
        flat_n = n_trials_vec[idx_col]
        flat_obs = post_counts_int[idx_row, idx_col]

        pm.BetaBinomial(
            "y_post",
            n=flat_n,
            alpha=flat_alpha,
            beta=flat_beta,
            observed=flat_obs,
        )

    return BuiltModel(model=model, variables=_variables_dict(model), prepared=prepared)


# ---------------------------------------------------------------------------
# LRP56-LRP58: mechanism factory
# ---------------------------------------------------------------------------


def build_mechanism_model(
    prepared: PreparedData,
    *,
    mechanism_symbol: str,
    outcome_symbol: str = "W",
    adjust_baseline_symbol: str = "W",
    confounder_symbols: Iterable[str] = (),
    use_age_gp: bool = False,
    phase_specific_mechanism: bool = False,
    use_subject_random_intercept: bool = True,
    sigma_child_prior_sigma: float = 0.5,
    moderator_symbol: str | None = None,
    moderator_is_covariate: bool = False,
    include_interaction: bool = True,
    linear_mechanism: bool = False,
) -> BuiltModel:
    """
    Mechanism model on the outcome post-score.

    Uses all three phase transitions (``prepared.phase_mode == "all"``) with
    phase-specific intercepts to absorb between-period level shifts. The
    mechanism variable is the *post* score of ``mechanism_symbol`` entered as
    a HSGP on its logit-safe transform. Confounders are additional covariates
    (on their logit scale for measures, or raw for group/age) that appear as
    linear terms.

    The outcome baseline ``adjust_baseline_symbol`` (default ``W_pre``) enters
    linearly on the logit scale.

    ``use_subject_random_intercept`` (default True): add a non-centred
    child-level random intercept ``u_child = sigma_child * u_child_raw`` with
    ``u_child_raw ~ Normal(0, 1, dims="child")`` and
    ``sigma_child ~ HalfNormal(sigma_child_prior_sigma)``. Required for
    honest standard errors on β_G, γ's, and f_mech because the 157 rows per
    mechanism fit are three phase-transitions per child and therefore not
    independent. See notes/202604181800-mechanism-random-intercepts.md.

    ``moderator_symbol`` (default None): when set, adds a LINEAR moderation of
    the mechanism effect by the moderator post-score M. Two standardised terms
    enter the linear predictor alongside the nonparametric ``f_mech``:

        ... + gamma_mod * z(M) + gamma_int * z(logit L_post) * z(M)

    where ``z(.)`` is the standardised (mean-0, sd-1) transform computed on the
    kept rows. ``gamma_int > 0`` means letter-sound converts to the outcome
    *more* strongly for higher-M children. Both coefficients use the regularising
    ``gamma_cross_prior()`` (Normal(0, 0.3)). The HSGP ``f_mech`` is unchanged
    (it stays a function of the *raw* ``logit L_post``); the GP-varying-slope
    refinement is deliberately deferred. The caller is responsible for not also
    passing the moderator as a plain confounder (else its main effect would be
    represented twice and be collinear) — the pipeline strips it from
    ``confounder_symbols`` before calling.

    ``linear_mechanism`` (default False): when True the mechanism enters the
    linear predictor as ``beta_mech * z(logit L_post)`` (a single slope with the
    ``beta_mech_prior``) instead of the HSGP ``f_mech``. Intended for low /
    floored count outcomes (e.g. nonword decoding, LRP72) where a nonparametric
    dose-response is not identifiable. No ``f_mech`` variable is created, so the
    pipeline's mechanism-curve step is skipped. Orthogonal to
    ``moderator_symbol``: with both set, the model is
    ``beta_mech*z(L) + gamma_mod*z(M) + gamma_int*z(L)*z(M)``.

    ``moderator_is_covariate`` (default False): treat the moderator as a
    continuous covariate (currently age) rather than a bounded-count measure.
    ``z(M)`` is then the standardised ``prepared.A_std`` instead of
    ``z(logit(M_post/N))``; the measure guard and the moderator NaN keep-mask are
    skipped. Used by LRP73 (age moderation). Default-off so count moderators
    (LRP71 E, LRP72 B) are unaffected.

    ``include_interaction`` (default True): when False, only the moderator main
    effect ``gamma_mod * z(M)`` is added (no ``gamma_int``). Used to build a
    clean no-interaction baseline (e.g. LRP73base) that differs from the full
    model by exactly the interaction term, for a nested PSIS-LOO comparison.
    """
    if prepared.phase_mode != "all":
        raise ValueError("Mechanism factory requires phase_mode='all'")
    if mechanism_symbol not in prepared.pre_logit:
        raise KeyError(f"Mechanism {mechanism_symbol!r} missing from prepared data")
    if outcome_symbol not in prepared.pre_logit:
        raise KeyError(f"Outcome {outcome_symbol!r} missing from prepared data")
    if (
        moderator_symbol is not None
        and not moderator_is_covariate
        and moderator_symbol not in prepared.pre_logit
    ):
        raise KeyError(f"Moderator {moderator_symbol!r} missing from prepared data")

    # Outcome post (target) and mechanism post (predictor) are both needed.
    outcome_post = prepared.post_counts[outcome_symbol]
    mechanism_post = prepared.post_counts[mechanism_symbol]

    keep = ~(np.isnan(outcome_post) | np.isnan(mechanism_post))
    if moderator_symbol is not None and not moderator_is_covariate:
        keep = keep & ~np.isnan(prepared.post_counts[moderator_symbol])
    for s in confounder_symbols:
        if s not in prepared.pre_logit and s not in {"G", "A"}:
            raise KeyError(f"Confounder {s!r} not recognised")
        if s in prepared.post_counts:
            keep = keep & ~np.isnan(prepared.post_counts[s])
    prepared = _subset(prepared, keep)

    from language_reading_predictors.statistical_models.preprocessing import (
        logit_safe,
        standardise,
    )

    outcome_post = prepared.post_counts[outcome_symbol].astype(np.int64)
    N_outcome = prepared.n_trials[outcome_symbol]
    N_mechanism = prepared.n_trials[mechanism_symbol]

    mech_post_logit = logit_safe(prepared.post_counts[mechanism_symbol], N_mechanism)

    own_pre_logit = prepared.pre_logit[adjust_baseline_symbol]

    # Standardised inputs for the LINEAR moderation term, computed on the kept
    # rows so the mean/sd match the data the model is fit to. ``z_L`` re-uses the
    # mechanism logit (a *centred* version, so gamma_mod reads as the moderator
    # effect at the mean of L); ``f_mech`` keeps the raw logit. The keep-mask
    # above guarantees the moderator post-score has no NaNs at this point.
    z_L: np.ndarray | None = None
    z_M: np.ndarray | None = None
    if moderator_symbol is not None or linear_mechanism:
        z_L, _ = standardise(mech_post_logit)
    if moderator_symbol is not None:
        if moderator_is_covariate:
            # Continuous covariate moderator (currently age): use the
            # already-standardised A_std, re-standardised on the kept rows so
            # gamma_mod reads as the moderator effect at mean L and gamma_int is
            # unit-free — consistent with the count-moderator path.
            z_M, _ = standardise(prepared.A_std)
        else:
            moderator_post_logit = logit_safe(
                prepared.post_counts[moderator_symbol],
                prepared.n_trials[moderator_symbol],
            )
            z_M, _ = standardise(moderator_post_logit)

    coords = {
        "obs_id": np.arange(prepared.n_obs),
        "phase": np.arange(prepared.n_phases),
        "child": np.arange(prepared.n_children),
    }

    with pm.Model(coords=coords) as model:
        A_std_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
        G_d = pm.Data("G", prepared.G.astype(float), dims="obs_id")
        own_pre_d = pm.Data("own_pre_logit", own_pre_logit, dims="obs_id")
        pm.Data("mech_post_logit", mech_post_logit, dims="obs_id")
        phase_d = pm.Data(
            "phase_idx", prepared.phase.astype(np.int64), dims="obs_id"
        )
        child_idx_d = pm.Data(
            "child_idx", prepared.child_idx.astype(np.int64), dims="obs_id"
        )
        z_L_d = z_M_d = None
        if z_L is not None:
            z_L_d = pm.Data("z_mech_logit", z_L, dims="obs_id")
        if moderator_symbol is not None:
            z_M_d = pm.Data("z_moderator", z_M, dims="obs_id")
        confounder_data: dict[str, pt.TensorVariable] = {}
        for s in confounder_symbols:
            if s in {"G", "A"}:
                continue
            if s not in prepared.post_counts:
                raise KeyError(
                    f"Confounder {s!r} has no post-score in prepared data"
                )
            c_val_np = logit_safe(prepared.post_counts[s], prepared.n_trials[s])
            confounder_data[s] = pm.Data(
                f"{s}_post_logit", c_val_np, dims="obs_id"
            )

        alpha = _scalar_prior("alpha", _priors.alpha_prior)
        alpha_phase = pm.Normal(
            "alpha_phase", mu=0.0, sigma=0.5, dims="phase"
        )
        beta_G = _priors.tau_prior().to_pymc("beta_G")
        gamma_own = _priors.gamma_own_prior().to_pymc("gamma_own")

        eta = (
            alpha
            + alpha_phase[phase_d]
            + beta_G * G_d
            + gamma_own * own_pre_d
        )

        if use_subject_random_intercept:
            sigma_child = pm.HalfNormal("sigma_child", sigma=sigma_child_prior_sigma)
            u_child_raw = pm.Normal("u_child_raw", mu=0.0, sigma=1.0, dims="child")
            u_child = pm.Deterministic(
                "u_child", sigma_child * u_child_raw, dims="child"
            )
            eta = eta + u_child[child_idx_d]

        # Confounder linear terms (on logit scale for measures)
        for s in confounder_symbols:
            if s in {"G", "A"}:
                continue  # G already in beta_G; A handled via age GP
            gamma_c = _priors.gamma_cross_prior().to_pymc(f"gamma_{s}")
            eta = eta + gamma_c * confounder_data[s]

        # Linear moderation of the mechanism effect by the moderator M.
        # ``gamma_mod`` is the moderator main effect (also serves as the
        # adjustment for M when M is a DAG confounder); ``gamma_int`` is the
        # interaction. Fixed names (never gamma_{M}) so they cannot collide with
        # the confounder loop above. Both are free RVs with the Normal(0, 0.3)
        # cross-coupling prior.
        if moderator_symbol is not None:
            gamma_mod = _priors.gamma_cross_prior().to_pymc("gamma_mod")
            eta = eta + gamma_mod * z_M_d
            if include_interaction:
                gamma_int = _priors.gamma_cross_prior().to_pymc("gamma_int")
                eta = eta + gamma_int * (z_L_d * z_M_d)

        if use_age_gp:
            f_A = build_hsgp_1d("f_A", prepared.A_std)
            eta = eta + f_A

        # Age confounder. ``A`` is a declared confounder for every mechanism
        # model (the DAG lists it), but the two confounder loops above skip
        # {"G", "A"} and the age GP is off by default — so without the linear
        # term here, age would silently never enter ``eta``. That was the bug
        # that left LRP56-58 / LRP71 / LRP72 unadjusted for the age confounder;
        # see notes/202606172100-mechanism-age-adjustment.md. When age is the
        # moderator (LRP73, ``moderator_is_covariate``) its main effect
        # ``gamma_mod * z(age)`` already represents it, so a second linear term
        # would be collinear — skip it in that case.
        age_is_moderator = moderator_symbol == "A" and moderator_is_covariate
        age_linear_added = (
            "A" in confounder_symbols and not use_age_gp and not age_is_moderator
        )
        if age_linear_added:
            gamma_A = _priors.gamma_cross_prior().to_pymc("gamma_A")
            eta = eta + gamma_A * A_std_d

        # Invariant: every declared confounder must reach ``eta``. ``G`` is in
        # ``beta_G``; measure confounders are added in the loop above; ``A`` is
        # the age GP, the age-moderator main effect, or the linear ``gamma_A``
        # term. Raise otherwise, so a future spec cannot silently drop a
        # declared confounder the way the original age handling did.
        represented = {"G"} | set(confounder_data)
        if use_age_gp or age_is_moderator or age_linear_added:
            represented.add("A")
        missing = [s for s in confounder_symbols if s not in represented]
        if missing:
            raise ValueError(
                f"Declared confounder(s) {missing!r} have no representation in "
                "the mechanism-model linear predictor."
            )

        # Mechanism GP (the estimand). The HSGP basis size depends on the
        # numeric range of the input, so it is constructed against the
        # numpy array; the registered ``mech_post_logit`` Data node is
        # kept for documentation / introspection.
        if linear_mechanism:
            # Linear mechanism: beta_mech * z(logit L) instead of the HSGP.
            # Used for low / floored count outcomes (e.g. nonword decoding) where
            # a full GP dose-response is not identifiable. No f_mech variable is
            # created, so the mechanism-curve step is skipped downstream.
            beta_mech = _priors.beta_mech_prior().to_pymc("beta_mech")
            eta = eta + beta_mech * z_L_d
        elif phase_specific_mechanism:
            phase_specific = []
            for p in range(prepared.n_phases):
                phase_specific.append(
                    build_hsgp_1d(f"f_mech_phase{p}", mech_post_logit)
                )
            f_mech = pt.stack(phase_specific, axis=1)[
                np.arange(prepared.n_obs), phase_d
            ]
            eta = eta + f_mech
        else:
            f_mech = build_hsgp_1d("f_mech", mech_post_logit)
            eta = eta + f_mech

        eta = pm.Deterministic("eta", eta, dims="obs_id")
        kappa = _priors.kappa_prior().to_pymc("kappa")

        beta_binomial_from_logit(
            "y_post",
            eta,
            n_trials=N_outcome,
            kappa=kappa,
            observed=outcome_post,
            dims="obs_id",
        )

    return BuiltModel(model=model, variables=_variables_dict(model), prepared=prepared)


# ---------------------------------------------------------------------------
# LRP59: mediation factory (joint mediator + outcome, ITT phase)
# ---------------------------------------------------------------------------


@dataclass
class MediationData:
    """Row-aligned phase-0 arrays + mediator metadata for the g-formula.

    Carried alongside the BuiltModel so :func:`mediation.decompose` can
    re-simulate counterfactuals from the posterior using the exact inputs the
    model saw. ``mediator_kind`` selects which mediator sub-model to simulate:

    - ``"beta_binomial"`` (LRP59): a single count mediator (``L_t2`` out of
      ``n_trials_L``) conditioned on ``logit(L_t1)``; ``med_mean`` / ``med_sd``
      standardise its logit for the outcome model.
    - ``"gaussian_composite"`` (LRP62): a continuous standardised phonics-route
      composite; the baseline composite is ``M_pre_std`` and the mediator is
      drawn from a Normal, so the count-specific fields are unused.
    """

    # Shared across mediator kinds.
    G: np.ndarray
    W1_logit: np.ndarray
    A_std: np.ndarray
    E1_logit: np.ndarray
    R1_logit: np.ndarray
    n_trials_W: int
    mediator_kind: str = "beta_binomial"
    # Beta-Binomial single mediator (LRP59).
    L1_logit: np.ndarray | None = None
    L2_count: np.ndarray | None = None
    W2_count: np.ndarray | None = None
    n_trials_L: int = 0
    med_mean: float = 0.0
    med_sd: float = 1.0
    # Gaussian composite mediator (LRP62).
    M_pre_std: np.ndarray | None = None
    route_symbols: tuple[str, ...] = ()


def build_mediation_model(
    prepared: PreparedData,
    *,
    mediator_symbol: str = "L",
    outcome_symbol: str = "W",
    confounder_symbols: Iterable[str] = ("E", "R"),
    mediator_kind: str = "beta_binomial",
    route_symbols: Iterable[str] = (),
) -> tuple[BuiltModel, MediationData]:
    """Joint mediator + outcome model for the ITT-phase (phase 0) decomposition.

    ``mediator_kind`` selects the mediator sub-model:

    - ``"beta_binomial"`` (LRP59, default): a single count mediator
      (``mediator_symbol``, e.g. letter-sound L) — documented below.
    - ``"gaussian_composite"`` (LRP62): the mediator is an equal-weight
      standardised-logit composite of ``route_symbols`` (the phonics route,
      e.g. ``("L", "B")``), modelled as ``Normal(mu_M, sigma_M)``. The outcome
      model is identical to the LRP59 case; only the mediator leg changes. See
      :func:`_build_route_composite_model`.

    Two Beta-Binomial likelihoods on the logit scale share the randomised
    treatment ``G`` and a baseline-covariate adjustment set:

    - Mediator: ``logit(L_t2) ~ a0 + a_G·G + a_L·logit(L_t1) + sum a_c·C_t1``
    - Outcome:  ``logit(W_t2) ~ b0 + b_G·G + b_M·z(logit L_t2)
                  + b_GM·G·z(logit L_t2) + b_W·logit(W_t1) + sum b_c·C_t1``

    The ``G×M`` interaction (``b_GM``) is included so the natural direct/indirect
    decomposition admits exposure-mediator interaction (the general g-formula
    case). NDE/NIE are NOT read off coefficients — they are computed by
    counterfactual simulation from the posterior (see ``mediation.decompose``);
    this factory only builds the joint likelihood and returns the row-aligned
    inputs needed for that simulation.

    Confounders ``C`` are taken at **baseline (t1)** on their logit scale, not at
    post (t2): a mediator-outcome confounder must not itself be affected by
    treatment (the cross-world assumption), so post-treatment values are
    inadmissible here. Documented in the report.

    Requires ``prepared.phase_mode == 'itt'`` (the single randomised t1->t2
    transition; one row per child, so no subject random intercept).
    """
    if prepared.phase_mode != "itt":
        raise ValueError("Mediation factory requires phase_mode='itt'")
    if mediator_kind == "gaussian_composite":
        return _build_route_composite_model(
            prepared,
            outcome_symbol=outcome_symbol,
            confounder_symbols=tuple(confounder_symbols),
            route_symbols=tuple(route_symbols),
        )
    if mediator_kind != "beta_binomial":
        raise ValueError(f"Unknown mediator_kind {mediator_kind!r}")
    needed = {mediator_symbol, outcome_symbol, *confounder_symbols}
    for s in needed:
        if s not in prepared.pre_logit:
            raise KeyError(f"Symbol {s!r} missing from prepared data")

    from language_reading_predictors.statistical_models.preprocessing import (
        logit_safe,
        standardise,
    )

    med_post = prepared.post_counts[mediator_symbol]
    out_post = prepared.post_counts[outcome_symbol]
    keep = ~(np.isnan(med_post) | np.isnan(out_post))
    if not keep.all():
        prepared = _subset(prepared, keep)

    N_med = prepared.n_trials[mediator_symbol]
    N_out = prepared.n_trials[outcome_symbol]
    L2_count = prepared.post_counts[mediator_symbol].astype(np.int64)
    W2_count = prepared.post_counts[outcome_symbol].astype(np.int64)

    # Standardised mediator (z of logit L_t2) — the regressor in the outcome
    # model; the standardiser is reused for counterfactual mediator draws.
    med_logit = logit_safe(L2_count, N_med)
    z_med, med_scaler = standardise(med_logit)

    L1 = prepared.pre_logit[mediator_symbol]
    W1 = prepared.pre_logit[outcome_symbol]
    conf_logit = {s: prepared.pre_logit[s] for s in confounder_symbols}

    coords = {"obs_id": np.arange(prepared.n_obs)}
    G_f = prepared.G.astype(float)

    with pm.Model(coords=coords) as model:
        G_d = pm.Data("G", G_f, dims="obs_id")
        L1_d = pm.Data("L_pre_logit", L1, dims="obs_id")
        W1_d = pm.Data("W_pre_logit", W1, dims="obs_id")
        A_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
        conf_d = {
            s: pm.Data(f"{s}_pre_logit", conf_logit[s], dims="obs_id")
            for s in confounder_symbols
        }
        z_med_d = pm.Data("z_med", z_med, dims="obs_id")

        # --- Mediator model: logit(L_t2) ---
        a0 = _priors.alpha_prior().to_pymc("a0")
        a_G = _priors.tau_prior().to_pymc("a_G")
        a_L = _priors.gamma_own_prior().to_pymc("a_L")
        a_A = _priors.gamma_cross_prior().to_pymc("a_A")
        mu_M = a0 + a_G * G_d + a_L * L1_d + a_A * A_d
        for s in confounder_symbols:
            a_c = _priors.gamma_cross_prior().to_pymc(f"a_{s}")
            mu_M = mu_M + a_c * conf_d[s]
        mu_M = pm.Deterministic("mu_M", mu_M, dims="obs_id")
        kappa_M = _priors.kappa_prior().to_pymc("kappa_M")
        beta_binomial_from_logit(
            "L_post", mu_M, n_trials=N_med, kappa=kappa_M,
            observed=L2_count, dims="obs_id",
        )

        # --- Outcome model: logit(W_t2) ---
        b0 = _priors.alpha_prior().to_pymc("b0")
        b_G = _priors.tau_prior().to_pymc("b_G")
        b_M = _priors.b_path_prior().to_pymc("b_M")
        b_GM = _priors.gamma_cross_prior().to_pymc("b_GM")
        b_W = _priors.gamma_own_prior().to_pymc("b_W")
        b_A = _priors.gamma_cross_prior().to_pymc("b_A")
        eta_Y = (
            b0
            + b_G * G_d
            + b_M * z_med_d
            + b_GM * (G_d * z_med_d)
            + b_W * W1_d
            + b_A * A_d
        )
        for s in confounder_symbols:
            b_c = _priors.gamma_cross_prior().to_pymc(f"b_{s}")
            eta_Y = eta_Y + b_c * conf_d[s]
        eta_Y = pm.Deterministic("eta", eta_Y, dims="obs_id")
        kappa_Y = _priors.kappa_prior().to_pymc("kappa_Y")
        beta_binomial_from_logit(
            "y_post", eta_Y, n_trials=N_out, kappa=kappa_Y,
            observed=W2_count, dims="obs_id",
        )

    med_data = MediationData(
        G=prepared.G.astype(float),
        L1_logit=L1,
        W1_logit=W1,
        A_std=prepared.A_std,
        E1_logit=conf_logit.get("E", np.zeros(prepared.n_obs)),
        R1_logit=conf_logit.get("R", np.zeros(prepared.n_obs)),
        L2_count=L2_count,
        W2_count=W2_count,
        n_trials_L=int(N_med),
        n_trials_W=int(N_out),
        med_mean=float(med_scaler.mean),
        med_sd=float(med_scaler.sd),
    )
    built = BuiltModel(model=model, variables=_variables_dict(model), prepared=prepared)
    return built, med_data


# ---------------------------------------------------------------------------
# LRP62: reading-route composite mediation (Gaussian mediator, ITT phase)
# ---------------------------------------------------------------------------


def _build_route_composite(
    prepared: PreparedData, route_symbols: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-weight standardised-logit phonics-route composite.

    For each route symbol, the Haldane-logit of the count is standardised on its
    *post* (t2) distribution and that same scaler is applied to the pre (t1)
    value, so the baseline and post composites share one scale. Each child's
    composite is the equal-weight mean across symbols; the post composite is then
    standardised to mean 0 / SD 1 (the scaler reused for the baseline composite,
    so ``a_comp`` is a like-for-like autoregressive coupling). Returns
    ``(C_pre_std, C_post_std)``.
    """
    from language_reading_predictors.statistical_models.preprocessing import (
        logit_safe,
        standardise,
    )

    pre_cols, post_cols = [], []
    for s in route_symbols:
        post_logit = logit_safe(prepared.post_counts[s], prepared.n_trials[s])
        z_post, scaler = standardise(post_logit)
        pre_cols.append(scaler(prepared.pre_logit[s]))  # same scaler maps baseline
        post_cols.append(z_post)
    c_pre_raw = np.mean(np.stack(pre_cols, axis=1), axis=1)
    c_post_raw = np.mean(np.stack(post_cols, axis=1), axis=1)
    c_post_std, comp_scaler = standardise(c_post_raw)
    return comp_scaler(c_pre_raw), c_post_std


def _build_route_composite_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str,
    confounder_symbols: tuple[str, ...],
    route_symbols: tuple[str, ...],
) -> tuple[BuiltModel, MediationData]:
    """LRP62 reading-route mediation: a continuous phonics-route composite mediator.

    Same ITT-phase joint design and the *same* Beta-Binomial outcome model as
    :func:`build_mediation_model`, but the single count mediator is replaced by
    an equal-weight standardised composite of ``route_symbols`` modelled as
    ``Normal(mu_M, sigma_M)``. NDE/NIE are still computed by counterfactual
    simulation in :func:`mediation.decompose` (the ``gaussian_composite`` branch),
    not from coefficients. Confounders are taken at baseline (cross-world
    assumption), exactly as in LRP59.
    """
    if not route_symbols:
        raise ValueError("gaussian_composite requires non-empty route_symbols")
    needed = {outcome_symbol, *route_symbols, *confounder_symbols}
    for s in needed:
        if s not in prepared.pre_logit:
            raise KeyError(f"Symbol {s!r} missing from prepared data")

    # Keep rows with the outcome post and every route post observed.
    keep = ~np.isnan(prepared.post_counts[outcome_symbol])
    for s in route_symbols:
        keep = keep & ~np.isnan(prepared.post_counts[s])
    if not keep.all():
        prepared = _subset(prepared, keep)

    N_out = prepared.n_trials[outcome_symbol]
    W2_count = prepared.post_counts[outcome_symbol].astype(np.int64)
    c_pre_std, c_post_std = _build_route_composite(prepared, route_symbols)

    W1 = prepared.pre_logit[outcome_symbol]
    conf_logit = {s: prepared.pre_logit[s] for s in confounder_symbols}

    coords = {"obs_id": np.arange(prepared.n_obs)}
    G_f = prepared.G.astype(float)

    with pm.Model(coords=coords) as model:
        G_d = pm.Data("G", G_f, dims="obs_id")
        Mpre_d = pm.Data("M_pre_std", c_pre_std, dims="obs_id")
        Mpost_d = pm.Data("M_post_std", c_post_std, dims="obs_id")
        W1_d = pm.Data("W_pre_logit", W1, dims="obs_id")
        A_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
        conf_d = {
            s: pm.Data(f"{s}_pre_logit", conf_logit[s], dims="obs_id")
            for s in confounder_symbols
        }

        # --- Mediator model: standardised route composite ~ Normal ---
        a0 = _priors.alpha_prior().to_pymc("a0")
        a_G = _priors.tau_prior().to_pymc("a_G")
        a_comp = _priors.gamma_own_prior().to_pymc("a_comp")
        a_A = _priors.gamma_cross_prior().to_pymc("a_A")
        mu_M = a0 + a_G * G_d + a_comp * Mpre_d + a_A * A_d
        for s in confounder_symbols:
            a_c = _priors.gamma_cross_prior().to_pymc(f"a_{s}")
            mu_M = mu_M + a_c * conf_d[s]
        mu_M = pm.Deterministic("mu_M", mu_M, dims="obs_id")
        sigma_M = _priors.sigma_mediator_prior().to_pymc("sigma_M")
        pm.Normal("M_post", mu=mu_M, sigma=sigma_M, observed=Mpost_d, dims="obs_id")

        # --- Outcome model: logit(W_t2) (identical to the LRP59 outcome leg) ---
        b0 = _priors.alpha_prior().to_pymc("b0")
        b_G = _priors.tau_prior().to_pymc("b_G")
        b_M = _priors.b_path_prior().to_pymc("b_M")
        b_GM = _priors.gamma_cross_prior().to_pymc("b_GM")
        b_W = _priors.gamma_own_prior().to_pymc("b_W")
        b_A = _priors.gamma_cross_prior().to_pymc("b_A")
        eta_Y = (
            b0
            + b_G * G_d
            + b_M * Mpost_d
            + b_GM * (G_d * Mpost_d)
            + b_W * W1_d
            + b_A * A_d
        )
        for s in confounder_symbols:
            b_c = _priors.gamma_cross_prior().to_pymc(f"b_{s}")
            eta_Y = eta_Y + b_c * conf_d[s]
        eta_Y = pm.Deterministic("eta", eta_Y, dims="obs_id")
        kappa_Y = _priors.kappa_prior().to_pymc("kappa_Y")
        beta_binomial_from_logit(
            "y_post", eta_Y, n_trials=N_out, kappa=kappa_Y,
            observed=W2_count, dims="obs_id",
        )

    med_data = MediationData(
        G=prepared.G.astype(float),
        W1_logit=W1,
        A_std=prepared.A_std,
        E1_logit=conf_logit.get("E", np.zeros(prepared.n_obs)),
        R1_logit=conf_logit.get("R", np.zeros(prepared.n_obs)),
        n_trials_W=int(N_out),
        mediator_kind="gaussian_composite",
        W2_count=W2_count,
        M_pre_std=c_pre_std,
        route_symbols=route_symbols,
    )
    built = BuiltModel(model=model, variables=_variables_dict(model), prepared=prepared)
    return built, med_data


# ---------------------------------------------------------------------------
# LRP65: between-child adjusted model (T1 baselines -> word-reading gain)
# ---------------------------------------------------------------------------


def _t1_language_composite(
    prepared: PreparedData, symbols: Iterable[str]
) -> np.ndarray:
    """Equal-weight standardised-logit language composite at T1.

    Each symbol's Haldane-logit baseline is standardised; the equal-weight mean
    is then standardised again so the composite is a unit-SD predictor. The
    pooled-framing analogue is LRP62's ``_build_route_composite``; that helper
    standardises on the *post* distribution and carries a paired baseline, which
    the T1-only between-child design does not need.
    """
    from language_reading_predictors.statistical_models.preprocessing import (
        standardise,
    )

    cols = []
    for s in symbols:
        if s not in prepared.pre_logit:
            raise KeyError(f"Language-composite symbol {s!r} not in prepared data")
        z, _ = standardise(prepared.pre_logit[s])
        cols.append(z)
    comp = np.mean(np.stack(cols, axis=1), axis=1)
    z_comp, _ = standardise(comp)
    return z_comp


def _resolve_adjusted_predictor(
    prepared: PreparedData, key: str, language_symbols: tuple[str, ...]
) -> tuple[str, np.ndarray, str]:
    """Map an LRP65 predictor key to ``(coef_name, standardised_vector, label)``.

    Keys: a measure symbol (``"L"``, ``"B"``) -> standardised T1 logit;
    ``"lang"`` -> the language composite; ``"age"`` -> the standardised T1 age;
    a covariate column already on ``prepared.covariates`` (``"blocks"``,
    ``"behav"``, ``"mumedupost16"``) -> that standardised covariate. Every key
    maps to coefficient ``beta_<key>``.
    """
    from language_reading_predictors.statistical_models.measures import MEASURES
    from language_reading_predictors.statistical_models.preprocessing import (
        standardise,
    )

    coef = f"beta_{key}"
    if key == "lang":
        return coef, _t1_language_composite(prepared, language_symbols), (
            "Language composite (" + "+".join(language_symbols) + ", T1)"
        )
    if key == "age":
        return coef, np.asarray(prepared.A_std, dtype=float), "Age (T1)"
    if key in prepared.covariates:
        return coef, np.asarray(prepared.covariates[key], dtype=float), f"{key} (T1)"
    if key in prepared.pre_logit:
        z, _ = standardise(prepared.pre_logit[key])
        label = MEASURES[key].label if key in MEASURES else key
        return coef, z, f"{label} (T1)"
    raise KeyError(f"Unknown LRP65 predictor key {key!r}")


def build_adjusted_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str = "W",
    predictors: Iterable[str] = ("L", "lang", "B", "age", "blocks", "behav"),
    language_composite_symbols: Iterable[str] = ("R", "E", "F"),
    predictor_slope_sigma: float = 0.5,
) -> BuiltModel:
    """Between-child adjusted model: standardised T1 baselines -> word-reading gain.

    One row per child (``prepared.phase_mode == "span"``). The outcome post-score
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
    """
    if prepared.phase_mode not in {"span", "itt"}:
        raise ValueError(
            "Adjusted (between-child) model requires phase_mode='span' "
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
        alpha = _scalar_prior("alpha", _priors.alpha_prior)
        gamma_own = _priors.gamma_own_prior().to_pymc("gamma_own")
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

    return BuiltModel(model=model, variables=_variables_dict(model), prepared=prepared)


# ---------------------------------------------------------------------------
# LRP66: latent general-ability factor model (g + specific paths -> gain)
# ---------------------------------------------------------------------------


def build_factor_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str = "W",
    indicator_symbols: Iterable[str] = ("L", "R", "E", "F", "B"),
    indicator_covariates: Iterable[str] = ("blocks",),
    language_composite_symbols: Iterable[str] = ("R", "E", "F"),
    observed_direct: Iterable[str] = ("L", "lang", "age"),
    use_language_specific: bool = False,
    language_specific_symbols: Iterable[str] = ("R", "E", "F"),
    loading_sigma: float = 1.0,
    predictor_slope_sigma: float = 0.5,
) -> BuiltModel:
    """Between-child latent general-ability model (LRP66, Tier 2).

    A one-factor measurement model estimates a latent general ability ``g`` from
    the standardised T1 skill indicators (Gaussian CFA on the standardised
    Haldane-logits / standardised covariates), and a structural Beta-Binomial
    leg regresses word-reading gain (``outcome`` post conditioned on its T1
    baseline via ``gamma_own``) on ``g`` plus the ``observed_direct`` skills —
    whose coefficients given ``g`` are the "signal beyond general ability".

    Orientation/scale are fixed by ``g ~ Normal(0, 1)`` and **positive** loadings
    (``HalfNormal``) — all indicators are positive-manifold cognitive measures, so
    a common factor raises them all. Indicators are standardised (SD 1), so a
    loading is the indicator-``g`` correlation and ``lambda^2 + sigma^2 ~ 1``.

    ``use_language_specific`` (robustness arm): the language indicators
    (``language_specific_symbols``, identified by having >= 2 measures) additionally
    load on an orthogonal language-specific factor ``s_lang``; the structural leg
    then carries ``beta_lang_specific * s_lang`` (the latent "beyond-g" language
    path) and ``"lang"`` is dropped from ``observed_direct`` to avoid double count.
    Single-indicator skills (letter sounds, blending, block design) cannot support
    a specific factor, so their "beyond-g" test stays observed-direct.

    This is between-child (one row per child) and small-n (n ~ 51): a latent model
    here is fragile and priors do real work — read it as triangulation against
    LRP65, not a definitive decomposition.
    """
    from language_reading_predictors.statistical_models.preprocessing import standardise

    if prepared.phase_mode not in {"span", "itt"}:
        raise ValueError(
            "Factor (between-child) model requires phase_mode in {'span', 'itt'} "
            f"(one row per child); got {prepared.phase_mode!r}"
        )

    # One row per child: drop children missing the outcome post-score.
    post = prepared.post_counts[outcome_symbol]
    keep = ~np.isnan(post)
    if not keep.all():
        prepared = _subset(prepared, keep)

    post = prepared.post_counts[outcome_symbol].astype(np.int64)
    N = prepared.n_trials[outcome_symbol]
    own_pre_logit = prepared.pre_logit[outcome_symbol]
    lang_symbols = tuple(language_composite_symbols)

    # Standardised indicator matrix Z (n_obs, J): count skills (z of T1 logit)
    # followed by continuous covariates (already standardised on load).
    ind_names: list[str] = []
    cols: list[np.ndarray] = []
    for s in indicator_symbols:
        if s not in prepared.pre_logit:
            raise KeyError(f"Indicator {s!r} missing from prepared data")
        z, _ = standardise(prepared.pre_logit[s])
        cols.append(z)
        ind_names.append(s)
    for c in indicator_covariates:
        if c not in prepared.covariates:
            raise KeyError(f"Indicator covariate {c!r} missing from prepared data")
        cols.append(np.asarray(prepared.covariates[c], dtype=float))
        ind_names.append(c)
    Z = np.stack(cols, axis=1)
    J = len(ind_names)

    observed_direct = tuple(observed_direct)
    lang_idx = np.array([], dtype=int)
    if use_language_specific:
        observed_direct = tuple(k for k in observed_direct if k != "lang")
        lang_idx = np.array(
            [ind_names.index(s) for s in language_specific_symbols], dtype=int
        )

    coords = {"obs_id": np.arange(prepared.n_obs), "indicator": ind_names}
    with pm.Model(coords=coords) as model:
        Z_d = pm.Data("Z", Z, dims=("obs_id", "indicator"))
        own_pre_d = pm.Data("own_pre_logit", own_pre_logit, dims="obs_id")

        # --- Measurement: one-factor (+ optional language-specific) ---
        g = pm.Normal("g", mu=0.0, sigma=1.0, dims="obs_id")
        lam = pm.HalfNormal("lambda_load", sigma=loading_sigma, dims="indicator")
        sigma_ind = pm.HalfNormal("sigma_indicator", sigma=1.0, dims="indicator")
        mu_Z = lam[None, :] * g[:, None]
        if use_language_specific:
            s_lang = pm.Normal("s_lang", mu=0.0, sigma=1.0, dims="obs_id")
            gam = pm.HalfNormal(
                "lambda_lang_spec", sigma=loading_sigma, shape=len(lang_idx)
            )
            lang_full = pt.set_subtensor(pt.zeros(J)[lang_idx], gam)
            mu_Z = mu_Z + s_lang[:, None] * lang_full[None, :]
        pm.Normal(
            "Z_obs", mu=mu_Z, sigma=sigma_ind[None, :], observed=Z_d,
            dims=("obs_id", "indicator"),
        )

        # --- Structural: gain ~ g + observed-direct skills (+ latent lang) ---
        alpha = _scalar_prior("alpha", _priors.alpha_prior)
        gamma_own = _priors.gamma_own_prior().to_pymc("gamma_own")
        beta_g = _priors.predictor_slope_prior(predictor_slope_sigma).to_pymc("beta_g")
        eta = alpha + gamma_own * own_pre_d + beta_g * g

        for k in observed_direct:
            coef_name, vec, _label = _resolve_adjusted_predictor(
                prepared, k, lang_symbols
            )
            x_d = pm.Data(f"x_{coef_name}", vec, dims="obs_id")
            beta = _priors.predictor_slope_prior(predictor_slope_sigma).to_pymc(
                coef_name
            )
            eta = eta + beta * x_d

        if use_language_specific:
            beta_lang_specific = _priors.predictor_slope_prior(
                predictor_slope_sigma
            ).to_pymc("beta_lang_specific")
            eta = eta + beta_lang_specific * s_lang

        eta = pm.Deterministic("eta", eta, dims="obs_id")
        kappa = _priors.kappa_prior().to_pymc("kappa")
        beta_binomial_from_logit(
            "y_post", eta, n_trials=N, kappa=kappa, observed=post, dims="obs_id"
        )

    return BuiltModel(model=model, variables=_variables_dict(model), prepared=prepared)


# ---------------------------------------------------------------------------
# Private
# ---------------------------------------------------------------------------


def _subset(prepared: PreparedData, keep: np.ndarray) -> PreparedData:
    """Return a copy of ``prepared`` restricted to rows where ``keep`` is True.

    Built with :func:`dataclasses.replace` so every row-indexed field is
    bound to a freshly-sliced array and the per-row dicts are rebuilt.
    Scalars and per-symbol metadata (``column_map``, ``n_trials``,
    ``age_scaler``, ``covariate_scalers``) are intentionally aliased
    from the parent — they do not depend on the row set.
    """
    if bool(keep.all()):
        return prepared
    from dataclasses import replace

    subject_ids = prepared.subject_ids[keep]
    # Re-index children so child_idx is dense 0..n_children-1.
    _, child_idx = np.unique(subject_ids, return_inverse=True)
    child_idx = child_idx.astype(np.int64)

    return replace(
        prepared,
        subject_ids=subject_ids,
        child_idx=child_idx,
        phase=prepared.phase[keep],
        G=prepared.G[keep],
        A_months=prepared.A_months[keep],
        A_std=prepared.A_std[keep],
        pre_logit={s: v[keep] for s, v in prepared.pre_logit.items()},
        post_counts={s: v[keep] for s, v in prepared.post_counts.items()},
        covariates={s: v[keep] for s, v in prepared.covariates.items()},
        n_obs=int(keep.sum()),
        n_children=int(len(np.unique(child_idx))),
    )


def _variables_dict(model: pm.Model) -> dict[str, pt.TensorVariable]:
    out: dict[str, pt.TensorVariable] = {}
    for rv in model.free_RVs:
        out[rv.name] = rv
    for det in model.deterministics:
        out[det.name] = det
    for rv in model.observed_RVs:
        out[rv.name] = rv
    return out
