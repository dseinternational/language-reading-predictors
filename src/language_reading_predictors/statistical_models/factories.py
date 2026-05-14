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
    """
    if prepared.phase_mode != "all":
        raise ValueError("Mechanism factory requires phase_mode='all'")
    if mechanism_symbol not in prepared.pre_logit:
        raise KeyError(f"Mechanism {mechanism_symbol!r} missing from prepared data")
    if outcome_symbol not in prepared.pre_logit:
        raise KeyError(f"Outcome {outcome_symbol!r} missing from prepared data")

    # Outcome post (target) and mechanism post (predictor) are both needed.
    outcome_post = prepared.post_counts[outcome_symbol]
    mechanism_post = prepared.post_counts[mechanism_symbol]

    keep = ~(np.isnan(outcome_post) | np.isnan(mechanism_post))
    for s in confounder_symbols:
        if s not in prepared.pre_logit and s not in {"G", "A"}:
            raise KeyError(f"Confounder {s!r} not recognised")
        if s in prepared.post_counts:
            keep = keep & ~np.isnan(prepared.post_counts[s])
    prepared = _subset(prepared, keep)

    from language_reading_predictors.statistical_models.preprocessing import logit_safe

    outcome_post = prepared.post_counts[outcome_symbol].astype(np.int64)
    N_outcome = prepared.n_trials[outcome_symbol]
    N_mechanism = prepared.n_trials[mechanism_symbol]

    mech_post_logit = logit_safe(prepared.post_counts[mechanism_symbol], N_mechanism)

    own_pre_logit = prepared.pre_logit[adjust_baseline_symbol]

    coords = {
        "obs_id": np.arange(prepared.n_obs),
        "phase": np.arange(prepared.n_phases),
        "child": np.arange(prepared.n_children),
    }

    with pm.Model(coords=coords) as model:
        pm.Data("A_std", prepared.A_std, dims="obs_id")
        G_d = pm.Data("G", prepared.G.astype(float), dims="obs_id")
        own_pre_d = pm.Data("own_pre_logit", own_pre_logit, dims="obs_id")
        pm.Data("mech_post_logit", mech_post_logit, dims="obs_id")
        phase_d = pm.Data(
            "phase_idx", prepared.phase.astype(np.int64), dims="obs_id"
        )
        child_idx_d = pm.Data(
            "child_idx", prepared.child_idx.astype(np.int64), dims="obs_id"
        )
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

        if use_age_gp:
            f_A = build_hsgp_1d("f_A", prepared.A_std)
            eta = eta + f_A

        # Mechanism GP (the estimand). The HSGP basis size depends on the
        # numeric range of the input, so it is constructed against the
        # numpy array; the registered ``mech_post_logit`` Data node is
        # kept for documentation / introspection.
        if phase_specific_mechanism:
            phase_specific = []
            for p in range(prepared.n_phases):
                phase_specific.append(
                    build_hsgp_1d(f"f_mech_phase{p}", mech_post_logit)
                )
            f_mech = pt.stack(phase_specific, axis=1)[
                np.arange(prepared.n_obs), phase_d
            ]
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
# Private
# ---------------------------------------------------------------------------


def _subset(prepared: PreparedData, keep: np.ndarray) -> PreparedData:
    """Return a copy of ``prepared`` restricted to rows where ``keep`` is True."""
    if bool(keep.all()):
        return prepared
    from copy import copy

    new = copy(prepared)
    new.subject_ids = prepared.subject_ids[keep]
    new.child_idx = prepared.child_idx[keep]
    # Re-index children so child_idx is dense 0..n_children-1.
    _, new.child_idx = np.unique(new.subject_ids, return_inverse=True)
    new.child_idx = new.child_idx.astype(np.int64)
    new.phase = prepared.phase[keep]
    new.G = prepared.G[keep]
    new.A_months = prepared.A_months[keep]
    new.A_std = prepared.A_std[keep]
    new.pre_logit = {s: v[keep] for s, v in prepared.pre_logit.items()}
    new.post_counts = {s: v[keep] for s, v in prepared.post_counts.items()}
    new.covariates = {s: v[keep] for s, v in prepared.covariates.items()}
    new.n_obs = int(keep.sum())
    new.n_children = int(len(np.unique(new.child_idx)))
    return new


def _variables_dict(model: pm.Model) -> dict[str, pt.TensorVariable]:
    out: dict[str, pt.TensorVariable] = {}
    for rv in model.free_RVs:
        out[rv.name] = rv
    for det in model.deterministics:
        out[det.name] = det
    for rv in model.observed_RVs:
        out[rv.name] = rv
    return out
