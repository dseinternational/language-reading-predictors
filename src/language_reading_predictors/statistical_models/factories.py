# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model factories for the statistical models.

Three factories are provided:

- :func:`build_itt_model` — the LRPITT ITT suite (one outcome, RCT phase) and its
  SES-adjusted companions; the floored outcomes use its ``bernoulli_offfloor``
  likelihood mode for the off-floor primary estimand.
- :func:`build_joint_model` — the joint model (LRPITT12) and the two-outcome
  generalisation contrasts (LRPITT15/15b), RCT phase, optional LKJ Σ.
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
from language_reading_predictors.statistical_models.preprocessing import (
    PreparedData,
    WavePanel,
    standardise,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _scalar_prior(name: str, prior_ctor) -> pt.TensorVariable:
    return prior_ctor().to_pymc(name)


@dataclass
class BuiltModel:
    model: pm.Model
    variables: dict[str, pt.TensorVariable]
    prepared: PreparedData | WavePanel
    """The (possibly row-subset) prepared data that the model was built on.

    Factories may drop rows with missing post-scores or missing confounder
    values; this attribute exposes the actually-used data so the pipeline can
    align posterior indices to input rows.
    """


# ---------------------------------------------------------------------------
# ITT factory (LRPITT suite)
# ---------------------------------------------------------------------------


def build_itt_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str,
    use_age_gp: bool = False,
    use_own_baseline_gp: bool = False,
    use_varying_tau: bool = False,
    adjust_for: Iterable[str] = (),
    cross_symbols: Iterable[str] | None = None,
    use_age_linear: bool = False,
    use_own_baseline: bool = True,
    likelihood: str = "beta_binomial",
    tau_moderator_symbol: str | None = None,
    tau_moderator_is_covariate: bool = False,
    tau_moderator_interaction: bool = True,
) -> BuiltModel:
    """
    Build the single-outcome ITT model used by the LRPITT suite (and its SES
    companions).

    The linear predictor is

        eta_i = alpha
              + tau * G_i
              + gamma_own * logit(y_pre_i, N_own)     # if use_own_baseline
              + gamma_A * A_std_i                      # if use_age_linear
              + sum_{k in cross} gamma_k * logit(k_pre_i, N_k)
              + sum_c gamma_c * z(c_i)                 # adjust_for covariates
              + f_A(A_std_i)                           # optional HSGP
              + f_ypre(logit(y_pre_i, N_own))          # optional HSGP
              + gamma_tau_mod * z(M_i)                 # optional tau-moderator main
              + gamma_tau_int * G_i * z(M_i)           # optional tau-moderator interaction

    The observation node is a Beta-Binomial on the post count (``likelihood=
    "beta_binomial"``) or, for the floored-outcome floor rule, a Bernoulli on the
    binary "off-floor at t2" indicator ``post > 0`` (``likelihood=
    "bernoulli_offfloor"``).

    Parameters
    ----------
    prepared
        Output of :func:`preprocessing.load_and_prepare` with ``phase_mode="itt"``.
    outcome_symbol
        Target measure (``"W"``, ``"R"``, ``"E"``, ...).
    use_age_gp, use_own_baseline_gp
        Toggles for the two HSGP main effects. **Default False** — the
        2026-04-18 LRP52 sensitivity fit found LOO did not prefer them and the
        GP amplitudes produced an ``eta -> basis-weight`` funnel (~1-8 %
        divergences); they are kept as opt-in flags for per-outcome sensitivity
        fits. This matches the ``build_joint_model`` / ``build_mechanism_model``
        default-off convention, so a spec that omits the flags no longer
        silently fits two unidentifiable GPs (notes/202604181445-lrp52-gp-sensitivity.md).
    use_varying_tau
        If True, the treatment effect is modelled as ``tau0 + g_tauA(A_std)``
        via a :func:`build_tau_modifier` GP with the tight ``HalfNormal(0.3)``
        amplitude prior.
    adjust_for
        Standardised non-outcome covariates from ``prepared.covariates`` to add
        as linear adjustment terms. Coefficients use the same weak
        ``Normal(0, 0.3)`` prior as cross-baseline couplings.
    cross_symbols
        Symbols whose baselines enter as cross-baseline couplings (the
        ``sum_{k != own} gamma_k`` term). ``None`` (default) reproduces the
        legacy behaviour of conditioning on every *other* ITT outcome
        (``ITT_OUTCOMES``). Pass an explicit (possibly empty) iterable to
        condition on a chosen subset instead. The LRPITT suite passes ``()`` —
        under the locked DAG the ITT effect is identified by the empty adjustment
        set, so cross-baselines are dropped. Every requested symbol must be in
        ``prepared.pre_logit``; ``own`` is removed if present.
    use_age_linear
        If True, add a plain linear age main effect ``gamma_A * A_std``
        (``gamma_age_prior``). A precision term only (the DAG identifies ``tau``
        without it). Mutually exclusive with ``use_age_gp`` (the GP already
        absorbs the smooth age effect) — setting both raises ``ValueError``. The
        LRPITT suite uses this in place of the (off-by-default) age GP.
    use_own_baseline
        If True (default), add the own-baseline precision term
        ``gamma_own * logit(y_pre)``. Set False for the age-only specification
        used by the floor-rule outcomes (``P``/``N``) and post-only outcomes
        (``N``): the factory then never indexes ``prepared.pre_logit[own]``, so
        a degenerate or missing baseline cannot enter or drop rows.
    likelihood
        ``"beta_binomial"`` (default) models the graded post count. The floor
        rule (#119) uses ``"bernoulli_offfloor"`` for its PRIMARY estimand: a
        Bernoulli/logistic ``tau`` on the binary off-floor indicator
        ``post > 0`` (no ``kappa``), which targets where the randomised signal
        verifiably lives for heavily-floored outcomes.
    tau_moderator_symbol, tau_moderator_is_covariate, tau_moderator_interaction
        Part B (HTE) plumbing: moderate ``tau`` by a **pre-randomisation**
        quantity ``M`` (so the interaction stays randomisation-respecting). With
        ``tau_moderator_is_covariate=True`` the moderator is ``"A"`` (age,
        ``A_std``) or a key of ``prepared.covariates`` (e.g. SES); otherwise it
        is an outcome symbol whose **baseline logit** ``prepared.pre_logit[M]``
        is used. ``M`` is standardised on the fitted (kept) rows, enters as a
        main effect ``gamma_tau_mod * z(M)``, and — when
        ``tau_moderator_interaction`` (default) — an interaction
        ``gamma_tau_int * G * z(M)``; both use the regularising
        ``Normal(0, 0.3)`` prior. Set ``tau_moderator_interaction=False`` for the
        nested no-interaction baseline used in the PSIS-LOO comparison.
    """
    if prepared.phase_mode != "itt":
        raise ValueError(
            f"build_itt_model expects phase_mode='itt', got {prepared.phase_mode!r}"
        )
    if likelihood not in ("beta_binomial", "bernoulli_offfloor"):
        raise ValueError(
            "likelihood must be 'beta_binomial' or 'bernoulli_offfloor', "
            f"got {likelihood!r}"
        )
    if use_age_gp and use_age_linear:
        raise ValueError(
            "use_age_gp and use_age_linear are mutually exclusive: the age GP "
            "already absorbs the smooth age effect; choose one."
        )

    own = outcome_symbol
    need_own_pre = use_own_baseline or use_own_baseline_gp
    if own not in prepared.post_counts:
        raise KeyError(f"Outcome {own!r} missing from prepared data (post_counts)")
    if need_own_pre and own not in prepared.pre_logit:
        raise KeyError(
            f"Outcome {own!r} has no baseline in prepared data, but "
            "use_own_baseline / use_own_baseline_gp is set. Load it with a "
            "pre-score, or pass use_own_baseline=False for an age-only model."
        )

    adjust_for = tuple(adjust_for)
    missing_adjusters = [c for c in adjust_for if c not in prepared.covariates]
    if missing_adjusters:
        raise KeyError(
            "Requested adjustment covariates missing from prepared data: "
            f"{missing_adjusters}"
        )
    if cross_symbols is None:
        cross = [s for s in ITT_OUTCOMES if s != own]
    else:
        cross = [s for s in cross_symbols if s != own]
        missing_cross = [s for s in cross if s not in prepared.pre_logit]
        if missing_cross:
            raise KeyError(
                f"Cross-baseline symbols missing from prepared data: {missing_cross}"
            )

    # Validate the tau-moderator (Part B). It must be a pre-randomisation
    # quantity — a baseline logit or a covariate — never a post-outcome.
    if tau_moderator_symbol is not None:
        if tau_moderator_is_covariate:
            if (
                tau_moderator_symbol != "A"
                and tau_moderator_symbol not in prepared.covariates
            ):
                raise KeyError(
                    f"tau moderator covariate {tau_moderator_symbol!r} not in "
                    "prepared.covariates (and is not 'A' for age)"
                )
        elif tau_moderator_symbol not in prepared.pre_logit:
            raise KeyError(
                f"tau moderator baseline {tau_moderator_symbol!r} not in "
                "prepared.pre_logit"
            )

    post = prepared.post_counts[own]
    if np.any(np.isnan(post)):
        keep = ~np.isnan(post)
        if not keep.all():
            prepared = _subset(prepared, keep)
            post = prepared.post_counts[own]

    post = post.astype(np.int64)
    y_pre_logit = prepared.pre_logit[own] if need_own_pre else None

    # Resolve the moderator vector on the kept rows (after the post NaN drop) so
    # gamma_tau_mod reads as the effect at the mean of the fitted sample.
    z_M: np.ndarray | None = None
    if tau_moderator_symbol is not None:
        if tau_moderator_is_covariate:
            raw_M = (
                prepared.A_std
                if tau_moderator_symbol == "A"
                else prepared.covariates[tau_moderator_symbol]
            )
        else:
            raw_M = prepared.pre_logit[tau_moderator_symbol]
        z_M, _ = standardise(raw_M)

    coords = {"obs_id": np.arange(prepared.n_obs)}
    G_f = prepared.G.astype(float)

    with pm.Model(coords=coords) as model:
        A_std_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
        G_d = pm.Data("G", G_f, dims="obs_id")
        cross_pre_data: dict[str, pt.TensorVariable] = {}
        for s in cross:
            cross_pre_data[s] = pm.Data(
                f"{s}_pre_logit", prepared.pre_logit[s], dims="obs_id"
            )
        adjust_data: dict[str, pt.TensorVariable] = {}
        for c in adjust_for:
            adjust_data[c] = pm.Data(f"{c}_std", prepared.covariates[c], dims="obs_id")
        z_M_d = (
            pm.Data("z_tau_moderator", z_M, dims="obs_id") if z_M is not None else None
        )

        alpha = _scalar_prior("alpha", _priors.alpha_prior)
        tau0 = _scalar_prior("tau", _priors.tau_prior)

        eta: pt.TensorVariable | float = alpha

        if use_own_baseline:
            own_pre_d = pm.Data("own_pre_logit", y_pre_logit, dims="obs_id")
            gamma_own = _scalar_prior("gamma_own", _priors.gamma_own_prior)
            eta = eta + gamma_own * own_pre_d

        for s in cross:
            gamma_s = _priors.gamma_cross_prior().to_pymc(f"gamma_{s}")
            eta = eta + gamma_s * cross_pre_data[s]

        for c in adjust_for:
            gamma_c = _priors.gamma_cross_prior().to_pymc(f"gamma_{c}")
            eta = eta + gamma_c * adjust_data[c]

        if use_age_linear:
            gamma_A = _priors.gamma_age_prior().to_pymc("gamma_A")
            eta = eta + gamma_A * A_std_d

        if use_age_gp:
            f_A = build_hsgp_1d("f_A", prepared.A_std)
            eta = eta + f_A
        if use_own_baseline_gp:
            f_ypre = build_hsgp_1d("f_ypre", y_pre_logit)
            eta = eta + f_ypre

        # Treatment effect, with the optional linear tau-moderator (Part B). The
        # moderator main effect enters once; the interaction is the G * z(M) term.
        if z_M_d is not None:
            gamma_tau_mod = _priors.gamma_cross_prior().to_pymc("gamma_tau_mod")
            eta = eta + gamma_tau_mod * z_M_d

        if use_varying_tau:
            g_tauA = build_tau_modifier("g_tauA", prepared.A_std)
            tau_i = pm.Deterministic("tau_i", tau0 + g_tauA, dims="obs_id")
            eta = eta + tau_i * G_d
        else:
            eta = eta + tau0 * G_d

        if z_M_d is not None and tau_moderator_interaction:
            gamma_tau_int = _priors.gamma_cross_prior().to_pymc("gamma_tau_int")
            eta = eta + gamma_tau_int * (G_d * z_M_d)

        eta = pm.Deterministic("eta", eta, dims="obs_id")

        if likelihood == "beta_binomial":
            kappa = _scalar_prior("kappa", _priors.kappa_prior)
            beta_binomial_from_logit(
                "y_post",
                eta,
                n_trials=prepared.n_trials[own],
                kappa=kappa,
                observed=post,
                dims="obs_id",
            )
        else:  # bernoulli_offfloor: PRIMARY estimand for the floor rule
            off_floor = (post > 0).astype(np.int64)
            pm.Bernoulli("y_offfloor", logit_p=eta, observed=off_floor, dims="obs_id")

    variables = _variables_dict(model)
    return BuiltModel(model=model, variables=variables, prepared=prepared)


# ---------------------------------------------------------------------------
# Joint model (LRPITT12 joint; LRPITT15/15b generalisation contrasts)
# ---------------------------------------------------------------------------


def build_joint_model(
    prepared: PreparedData,
    *,
    outcomes: Iterable[str] = ITT_OUTCOMES,
    use_age_gp: bool = False,
    partial_pool_age_gp: bool = True,
    use_residual_correlation: bool = False,
    use_cross_baselines: bool = True,
    use_age_linear: bool = False,
) -> BuiltModel:
    """
    Build the joint Beta-Binomial model (LRPITT12; the LRPITT15/15b contrasts).

    For each child i and outcome k, the model is

        eta_{k,i} = alpha_k + tau_k * G_i
                    + gamma_own_k * logit(k_pre_i, N_k)
                    [ + sum_{j != k} gamma_{k,j} * logit(j_pre_i, N_j)  if use_cross_baselines ]
                    [ + gamma_A_k * A_std_i                            if use_age_linear ]
                    [ + f_A_k(A_std_i)                                 if use_age_gp ]
                    [ + u_{k,i}                                        if use_residual_correlation ]

    with per-outcome Beta-Binomial likelihood on the post-score count.

    ``use_cross_baselines`` (default True): include the off-diagonal cross-baseline
    couplings (the historical LRP55 behaviour). The DAG-faithful LRPITT joint
    (LRPITT12) and the generalisation contrasts (LRPITT15/15b) set this **False**,
    so the joint mirrors the single-outcome suite — the ITT effect is identified by
    the empty adjustment set, with own baseline + linear age as precision terms.

    ``use_age_linear`` (default False): add a per-outcome linear age term
    ``gamma_A_k * A_std_i`` (the suite's age precision term); mutually exclusive
    with ``use_age_gp``.

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
        raise ValueError("joint model requires phase_mode='itt'")
    if use_age_gp and use_age_linear:
        raise ValueError(
            "use_age_gp and use_age_linear are mutually exclusive in the joint model."
        )

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

        # Per-outcome scalar parameters — shared constructors (priors.py) so
        # the joint model cannot drift from the ITT / mechanism factories (issue #79).
        alpha = _priors.alpha_prior().to_pymc("alpha", dims="outcome")
        tau = _priors.tau_prior().to_pymc("tau", dims="outcome")
        gamma_own = _priors.gamma_own_prior().to_pymc("gamma_own", dims="outcome")

        # Own-baseline contribution: (N_obs, K) - elementwise by outcome index.
        own_contrib = gamma_own[None, :] * pre_logit_data

        eta_core = (
            alpha[None, :]
            + tau[None, :] * pt.shape_padright(G_d)
            + own_contrib
        )

        # Cross-baseline couplings: (K outcomes) x K baselines; mask the diagonal
        # to enforce "own baseline handled separately". The DAG-faithful LRPITT
        # joint (LRPITT12) and the generalisation contrasts drop these so the joint
        # mirrors the single-outcome suite; kept available for a richer
        # sensitivity fit (the historical LRP55 behaviour).
        if use_cross_baselines:
            gamma_cross_mat = _priors.gamma_cross_prior().to_pymc(
                "gamma_cross", dims=("outcome", "baseline")
            )
            mask_offdiag = 1.0 - np.eye(K)
            gamma_cross_eff = pm.Deterministic(
                "gamma_cross_eff",
                gamma_cross_mat * mask_offdiag,
                dims=("outcome", "baseline"),
            )
            # Cross-baseline contribution: sum over baselines for each outcome.
            eta_core = eta_core + pt.dot(pre_logit_data, gamma_cross_eff.T)

        # Linear age main effect (per outcome), mirroring the single-outcome suite.
        if use_age_linear:
            gamma_A = _priors.gamma_age_prior().to_pymc("gamma_A", dims="outcome")
            eta_core = eta_core + gamma_A[None, :] * prepared.A_std[:, None]

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

        kappa = _priors.kappa_prior().to_pymc("kappa", dims="outcome")

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
# Waitlist-crossover / difference-in-differences factory (kind="did")
# ---------------------------------------------------------------------------


def build_did_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str,
    periods: Iterable[int] = (0, 1),
    use_child_re: bool = True,
    use_age: bool = True,
    dose: bool = False,
) -> BuiltModel:
    """Waitlist-crossover / difference-in-differences model for one outcome.

    Stacks the early phase transitions (P1 = t1->t2, P2 = t2->t3) for **both**
    arms and asks whether being *currently treated* lifts the period gain, with
    each child as their own control (a child random intercept). Under the trial's
    waitlist design the only **untreated** cell is the waitlist arm in P1; the
    immediate arm is treated in both periods and so anchors the period
    (time / maturation) trend. The treatment coefficient ``delta`` is therefore a
    difference-in-differences estimate of the ITT effect, identified jointly by
    the period-1 between-arm contrast and the waitlist's own P1->P2 crossover.

    Requires ``prepared.phase_mode == "all"`` (the phase-stacked frame). Built on
    the same Beta-Binomial-on-logit / ANCOVA convention as the ITT suite, so the
    logit link absorbs ceiling effects: an immediate-arm child near the top of a
    bounded test makes a small P2 gain *expected*, not a spurious negative trend
    (the failure mode of a raw-gain difference-in-differences).

    Linear predictor (binary, ``dose=False``):

        eta_{i,p} = alpha + beta_period * [p == P2]
                  + delta * Treated_{i,p}
                  + gamma_own * logit(pre_{i,p})
                  [ + gamma_A * A_std_{i,p}    if use_age ]
                  [ + u_child_i                if use_child_re ]

    where ``Treated_{i,p} = 1`` when child *i* is receiving the intervention in
    period *p* (immediate: both periods; waitlist: P2 only). With ``dose=True`` the
    binary ``delta * Treated`` is replaced by ``beta_dose * z(attend)`` (the
    standardised intervention-session count for that period) — a dose-response
    sensitivity; the caller must have loaded ``covariates=("attend",)``.

    Parameters
    ----------
    periods
        Phase indices to keep (default ``(0, 1)`` = P1, P2). ``beta_period`` is
        the P2-vs-P1 contrast.
    use_child_re
        Add the non-centred child random intercept (the own-control). Default True.
    use_age
        Add a linear standardised-age precision term. Default True.
    dose
        Use the standardised session count as a continuous treatment intensity
        instead of the binary treated indicator.
    """
    if prepared.phase_mode != "all":
        raise ValueError("build_did_model requires phase_mode='all'")
    own = outcome_symbol
    if own not in prepared.post_counts or own not in prepared.pre_logit:
        raise KeyError(f"Outcome {own!r} missing pre/post in prepared data")
    periods = tuple(int(p) for p in periods)
    if dose and "attend" not in prepared.covariates:
        raise KeyError("dose=True requires the 'attend' covariate to be loaded")

    post = prepared.post_counts[own]
    keep = np.isin(prepared.phase, periods) & ~np.isnan(post)
    if dose:
        keep = keep & np.isfinite(prepared.covariates["attend"])
    prepared = _subset(prepared, keep)

    post = prepared.post_counts[own].astype(np.int64)
    pre_logit = prepared.pre_logit[own]
    # P2 indicator (time); Treated indicator (immediate both periods, waitlist P2).
    is_p2 = (prepared.phase >= 1).astype(float)
    treated = ((prepared.G == 1) | (prepared.phase >= 1)).astype(float)
    n_trials = prepared.n_trials[own]

    coords = {
        "obs_id": np.arange(prepared.n_obs),
        "child": np.arange(prepared.n_children),
    }
    with pm.Model(coords=coords) as model:
        period_d = pm.Data("period", is_p2, dims="obs_id")
        own_pre_d = pm.Data("own_pre_logit", pre_logit, dims="obs_id")

        alpha = _scalar_prior("alpha", _priors.alpha_prior)
        beta_period = _priors.tau_prior().to_pymc("beta_period")
        gamma_own = _priors.gamma_own_prior().to_pymc("gamma_own")

        eta = alpha + beta_period * period_d + gamma_own * own_pre_d

        if use_age:
            A_std_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
            gamma_A = _priors.gamma_age_prior().to_pymc("gamma_A")
            eta = eta + gamma_A * A_std_d

        if use_child_re:
            sigma_child = pm.HalfNormal("sigma_child", sigma=0.5)
            u_child = pm.Deterministic(
                "u_child",
                sigma_child * pm.Normal("u_child_raw", 0.0, 1.0, dims="child"),
                dims="child",
            )
            child_idx_d = pm.Data(
                "child_idx", prepared.child_idx.astype(np.int64), dims="obs_id"
            )
            eta = eta + u_child[child_idx_d]

        # Linear predictor without the treatment term, so the pipeline can read
        # the off-treatment baseline for the average-marginal-effect translation.
        eta_base = pm.Deterministic("eta_base", eta, dims="obs_id")

        if dose:
            z_attend = pm.Data("z_attend", prepared.covariates["attend"], dims="obs_id")
            beta_dose = _priors.tau_prior().to_pymc("beta_dose")
            eta_full = eta_base + beta_dose * z_attend
        else:
            treated_d = pm.Data("treated", treated, dims="obs_id")
            delta = _priors.tau_prior().to_pymc("delta")
            eta_full = eta_base + delta * treated_d

        eta_full = pm.Deterministic("eta", eta_full, dims="obs_id")
        kappa = _scalar_prior("kappa", _priors.kappa_prior)
        beta_binomial_from_logit(
            "y_post",
            eta_full,
            n_trials=n_trials,
            kappa=kappa,
            observed=post,
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
    - ``"gaussian_composite"`` (LRP62): a continuous standardised code-based-route
      composite; the baseline composite is ``M_pre_std`` and the mediator is
      drawn from a Normal, so the count-specific fields are unused.

    Confounders are carried generically in ``conf_logit`` (baseline t1 logits
    keyed by symbol), with ``confounder_symbols`` recording the fitted set, so
    :func:`mediation.decompose` adjusts for exactly the confounders the model
    was fitted with — no symbol can drift between the fit and the g-formula.
    """

    # Shared across mediator kinds.
    G: np.ndarray
    W1_logit: np.ndarray
    A_std: np.ndarray
    conf_logit: dict[str, np.ndarray]
    n_trials_W: int
    mediator_kind: str = "beta_binomial"
    confounder_symbols: tuple[str, ...] = ()
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


def _build_outcome_leg(
    *,
    mediator_node,
    G_d,
    W1_d,
    A_d,
    conf_d: dict,
    confounder_symbols: Iterable[str],
    N_out,
    W2_count,
):
    """Shared Beta-Binomial outcome leg for the single-mediator-design factories.

    Both LRP59 (:func:`build_mediation_model`) and LRP62
    (:func:`_build_route_composite_model`) regress ``logit(W_t2)`` on treatment,
    the standardised post mediator and its ``G`` interaction, baseline word
    reading, age, and the baseline confounders — identical save for
    ``mediator_node`` (``z_med`` for LRP59, the route composite for LRP62). Must
    be called inside an open ``pm.Model`` context so the nodes register.
    """
    b0 = _priors.alpha_prior().to_pymc("b0")
    b_G = _priors.tau_prior().to_pymc("b_G")
    b_M = _priors.b_path_prior().to_pymc("b_M")
    b_GM = _priors.gamma_cross_prior().to_pymc("b_GM")
    b_W = _priors.gamma_own_prior().to_pymc("b_W")
    b_A = _priors.gamma_cross_prior().to_pymc("b_A")
    eta_Y = (
        b0
        + b_G * G_d
        + b_M * mediator_node
        + b_GM * (G_d * mediator_node)
        + b_W * W1_d
        + b_A * A_d
    )
    for s in confounder_symbols:
        b_c = _priors.gamma_cross_prior().to_pymc(f"b_{s}")
        eta_Y = eta_Y + b_c * conf_d[s]
    eta_Y = pm.Deterministic("eta", eta_Y, dims="obs_id")
    kappa_Y = _priors.kappa_prior().to_pymc("kappa_Y")
    return beta_binomial_from_logit(
        "y_post", eta_Y, n_trials=N_out, kappa=kappa_Y,
        observed=W2_count, dims="obs_id",
    )


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
      standardised-logit composite of ``route_symbols`` (the code-based route,
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
    confounder_symbols = tuple(confounder_symbols)
    if mediator_kind == "gaussian_composite":
        return _build_route_composite_model(
            prepared,
            outcome_symbol=outcome_symbol,
            confounder_symbols=confounder_symbols,
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
        _build_outcome_leg(
            mediator_node=z_med_d,
            G_d=G_d,
            W1_d=W1_d,
            A_d=A_d,
            conf_d=conf_d,
            confounder_symbols=confounder_symbols,
            N_out=N_out,
            W2_count=W2_count,
        )

    med_data = MediationData(
        G=prepared.G.astype(float),
        L1_logit=L1,
        W1_logit=W1,
        A_std=prepared.A_std,
        conf_logit={s: conf_logit[s] for s in confounder_symbols},
        confounder_symbols=confounder_symbols,
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
    """Equal-weight standardised-logit code-based-route composite.

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
    """LRP62 reading-route mediation: a continuous code-based-route composite mediator.

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

        # --- Outcome model: logit(W_t2) (shared with the LRP59 outcome leg) ---
        _build_outcome_leg(
            mediator_node=Mpost_d,
            G_d=G_d,
            W1_d=W1_d,
            A_d=A_d,
            conf_d=conf_d,
            confounder_symbols=confounder_symbols,
            N_out=N_out,
            W2_count=W2_count,
        )

    med_data = MediationData(
        G=prepared.G.astype(float),
        W1_logit=W1,
        A_std=prepared.A_std,
        conf_logit={s: conf_logit[s] for s in confounder_symbols},
        confounder_symbols=tuple(confounder_symbols),
        n_trials_W=int(N_out),
        mediator_kind="gaussian_composite",
        W2_count=W2_count,
        M_pre_std=c_pre_std,
        route_symbols=route_symbols,
    )
    built = BuiltModel(model=model, variables=_variables_dict(model), prepared=prepared)
    return built, med_data


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


# ---------------------------------------------------------------------------
# LRPGF / LRPLF: gain-factors and level-factors factories (#127)
#
# DAG-focused exploratory factor models (Beta-Binomial-on-logit, child random
# intercept). Under the locked DAG (notes/202606231600-dag-revision-consolidated.md)
# only the randomised group / on-intervention term is causal; every other
# coefficient is an adjusted association confounded by latent general ability
# (GA), which the child random intercept repairs up to shrinkage. Cognitive
# ability (``blocks``) is the observed handle on GA. SES is intentionally NOT a
# factor: it is not a DAG node and was found statistically redundant, so it is
# excluded from the core sets (it can be added later as a robustness companion,
# as for the ITT suite's lrpitt13). Attendance / dose (``IS``) is a DAG collider
# and is never conditioned on.
# ---------------------------------------------------------------------------


def _interaction_product(term_vecs: dict[str, np.ndarray], a: str, b: str) -> np.ndarray:
    """Elementwise product of two named (already-standardised) factor terms."""
    return np.asarray(term_vecs[a], dtype=float) * np.asarray(term_vecs[b], dtype=float)


def build_gain_factors_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str,
    skill_symbols: Iterable[str] = (),
    ability_covariate: str | None = None,
    interactions: Iterable[tuple[str, str]] = (),
    treated_only: bool = False,
    likelihood: str = "beta_binomial",
    use_subject_random_intercept: bool = True,
    sigma_child_prior_sigma: float = 0.5,
) -> BuiltModel:
    """Gain-factors model (LRPGF): what is associated with how much children gain.

    Repeated measures over the three period transitions (``phase_mode="all"``):
    the outcome is the period post-count given its pre-count (an ANCOVA "gain").
    Linear predictor (logit scale):

        eta = alpha + alpha_phase[p]
            + beta_trt * OnIntervention           # causal (ITT) — period-1 contrast ~ tau
            + gamma_own * logit(own_pre)          # own baseline (precision)
            + gamma_A * A_std                      # age (precision)
            + gamma_ability * z(ability)           # observed GA handle (blocks)
            + sum_s gamma_s * logit(skill_pre_s)   # upstream DAG skills (adjusted assoc.)
            + sum interactions                     # focal, pre-specified
            + u_child[i]                           # GA repair (RI-CLPM)

    ``OnIntervention`` is derived from the data: the immediate arm (``G == 1``) is
    on from period 1; the waitlist (``G == 0``) is off in period 1 only and on once
    it crosses over (``phase >= 1``). Its coefficient is identified almost entirely
    by the period-1 (randomised) contrast, so ``beta_trt`` reproduces the ITT
    ``tau`` (a verification anchor).

    ``treated_only=True`` excludes the waitlist arm's untreated period 1 ("gains
    while on intervention"). Every remaining row is then on-intervention, so the
    treatment term and any treatment interaction are constant — they are dropped
    automatically (the model becomes the factor-association model among the
    treated).

    ``skill_symbols`` are the outcome's measured, repeated-available DAG-upstream
    skills (e.g. L, R for word reading), entered as their period baseline logit.
    ``ability_covariate`` is a ``prepared.covariates`` key (``blocks``).
    ``interactions`` is a set of ``(term_a, term_b)`` pairs over the controlled
    vocabulary ``{"trt", "age", "ability", "own", <skill symbols>}``; each adds a
    ``gamma_int_<a>_<b>`` coefficient on the product of the two standardised terms.
    All non-causal coefficients are adjusted associations under the DAG.
    """
    if prepared.phase_mode != "all":
        raise ValueError("build_gain_factors_model requires phase_mode='all'")
    if likelihood not in ("beta_binomial", "bernoulli_offfloor"):
        raise ValueError(
            "likelihood must be 'beta_binomial' or 'bernoulli_offfloor', "
            f"got {likelihood!r}"
        )
    own = outcome_symbol
    if own not in prepared.post_counts or own not in prepared.pre_logit:
        raise KeyError(f"Outcome {own!r} needs pre+post scores in prepared data")
    skill_symbols = tuple(skill_symbols)
    for s in skill_symbols:
        if s not in prepared.pre_logit:
            raise KeyError(f"Skill {s!r} has no baseline in prepared data")
    if ability_covariate is not None and ability_covariate not in prepared.covariates:
        raise KeyError(f"ability_covariate {ability_covariate!r} not in prepared.covariates")

    valid_terms = {"trt", "age", "own", *skill_symbols}
    if ability_covariate is not None:
        valid_terms.add("ability")
    interactions = tuple(tuple(p) for p in interactions)
    for pair in interactions:
        for k in pair:
            if k not in valid_terms:
                raise KeyError(f"interaction term {k!r} not available; have {sorted(valid_terms)}")

    # Drop rows missing the outcome post, the own baseline, or any skill baseline.
    keep = ~np.isnan(prepared.post_counts[own]) & ~np.isnan(prepared.pre_logit[own])
    for s in skill_symbols:
        keep = keep & ~np.isnan(prepared.pre_logit[s])
    on_intervention = (prepared.G == 1) | (prepared.phase >= 1)
    if treated_only:
        keep = keep & on_intervention
    prepared = _subset(prepared, keep)

    post = prepared.post_counts[own].astype(np.int64)
    trt = ((prepared.G == 1) | (prepared.phase >= 1)).astype(float)
    # In treated_only the treatment indicator is constant -> not identified; drop
    # it and any interaction involving it.
    include_trt = not treated_only
    active_interactions = [
        pair for pair in interactions if include_trt or "trt" not in pair
    ]

    term_vecs: dict[str, np.ndarray] = {"trt": trt, "age": prepared.A_std}
    if ability_covariate is not None:
        term_vecs["ability"] = prepared.covariates[ability_covariate]  # already z-scored
    term_vecs["own"], _ = standardise(prepared.pre_logit[own])
    for s in skill_symbols:
        term_vecs[s], _ = standardise(prepared.pre_logit[s])

    coords = {
        "obs_id": np.arange(prepared.n_obs),
        "phase": np.arange(prepared.n_phases),
        "child": np.arange(prepared.n_children),
    }
    with pm.Model(coords=coords) as model:
        phase_d = pm.Data("phase_idx", prepared.phase.astype(np.int64), dims="obs_id")
        child_idx_d = pm.Data("child_idx", prepared.child_idx.astype(np.int64), dims="obs_id")
        A_std_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
        own_pre_d = pm.Data("own_pre_logit", prepared.pre_logit[own], dims="obs_id")
        trt_d = pm.Data("on_intervention", trt, dims="obs_id") if include_trt else None
        ability_d = (
            pm.Data(f"{ability_covariate}_std", term_vecs["ability"], dims="obs_id")
            if ability_covariate is not None
            else None
        )
        skill_d = {
            s: pm.Data(f"{s}_pre_logit", prepared.pre_logit[s], dims="obs_id")
            for s in skill_symbols
        }
        int_d = {
            pair: pm.Data(f"int_{pair[0]}_{pair[1]}", _interaction_product(term_vecs, *pair), dims="obs_id")
            for pair in active_interactions
        }

        alpha = _scalar_prior("alpha", _priors.alpha_prior)
        alpha_phase = pm.Normal("alpha_phase", mu=0.0, sigma=0.5, dims="phase")
        gamma_own = _priors.gamma_own_prior().to_pymc("gamma_own")
        gamma_A = _priors.gamma_age_prior().to_pymc("gamma_A")

        eta = alpha + alpha_phase[phase_d] + gamma_own * own_pre_d + gamma_A * A_std_d

        if include_trt:
            beta_trt = _priors.tau_prior().to_pymc("beta_trt")
            eta = eta + beta_trt * trt_d
        if ability_d is not None:
            gamma_ability = _priors.gamma_cross_prior().to_pymc("gamma_ability")
            eta = eta + gamma_ability * ability_d
        for s in skill_symbols:
            gamma_s = _priors.gamma_cross_prior().to_pymc(f"gamma_{s}")
            eta = eta + gamma_s * skill_d[s]
        for pair in active_interactions:
            gi = _priors.gamma_cross_prior().to_pymc(f"gamma_int_{pair[0]}_{pair[1]}")
            eta = eta + gi * int_d[pair]

        if use_subject_random_intercept:
            sigma_child = pm.HalfNormal("sigma_child", sigma=sigma_child_prior_sigma)
            u_child_raw = pm.Normal("u_child_raw", mu=0.0, sigma=1.0, dims="child")
            u_child = pm.Deterministic("u_child", sigma_child * u_child_raw, dims="child")
            eta = eta + u_child[child_idx_d]

        eta = pm.Deterministic("eta", eta, dims="obs_id")
        if likelihood == "beta_binomial":
            kappa = _scalar_prior("kappa", _priors.kappa_prior)
            beta_binomial_from_logit(
                "y_post", eta, n_trials=prepared.n_trials[own], kappa=kappa,
                observed=post, dims="obs_id",
            )
        else:  # bernoulli_offfloor: PRIMARY estimand for floored outcomes (e.g. P)
            pm.Bernoulli(
                "y_offfloor", logit_p=eta,
                observed=(post > 0).astype(np.int64), dims="obs_id",
            )

    return BuiltModel(model=model, variables=_variables_dict(model), prepared=prepared)


def build_level_factors_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str,
    ability_covariate: str | None = None,
    group_by_time: bool = True,
    ability_by_time: bool = True,
    group_ability: bool = True,
    likelihood: str = "beta_binomial",
    use_subject_random_intercept: bool = True,
    sigma_child_prior_sigma: float = 0.5,
) -> BuiltModel:
    """Level-factors model (LRPLF): what is associated with achievement levels.

    Repeated measures over the four timepoints (``phase_mode="levels"``): the
    outcome is the score *level* at each timepoint (no own baseline / not
    autoregressive). Linear predictor (logit scale):

        eta = alpha + alpha_time[t]
            + b_grp[t] * group            # group x time (trajectory divergence)
            + gamma_A * A_std_t            # age at t (precision)
            + g_ability[t] * z(ability)    # ability x time (observed GA handle)
            + gamma_grp_ability * group * z(ability)   # group x ability
            + u_child[i]                   # GA repair (RI-CLPM)

    **Level-model caveat (baked into the parameterisation + report):** after t2
    the waitlist crosses over, so the group effect across the four timepoints is
    *not* a clean ITT contrast. The focal ``group x time`` interaction is therefore
    modelled as a per-timepoint group effect ``b_grp[t]`` (dims ``phase`` = the
    timepoint index) — read as trajectory divergence — and the **clean randomised
    contrast lives only at t2** (``b_grp[1]``). ``ability x time`` is likewise a
    per-timepoint ability effect ``g_ability[t]``. Set ``group_by_time`` /
    ``ability_by_time`` False to collapse either to a single time-invariant
    coefficient. Only the randomised contrast is causal; all other terms are
    adjusted associations under the DAG.
    """
    if prepared.phase_mode != "levels":
        raise ValueError("build_level_factors_model requires phase_mode='levels'")
    if likelihood not in ("beta_binomial", "bernoulli_offfloor"):
        raise ValueError(
            "likelihood must be 'beta_binomial' or 'bernoulli_offfloor', "
            f"got {likelihood!r}"
        )
    own = outcome_symbol
    if own not in prepared.post_counts:
        raise KeyError(f"Outcome {own!r} missing from prepared data (post_counts)")
    if ability_covariate is not None and ability_covariate not in prepared.covariates:
        raise KeyError(f"ability_covariate {ability_covariate!r} not in prepared.covariates")
    if group_ability and ability_covariate is None:
        raise ValueError("group_ability interaction requires an ability_covariate")

    keep = ~np.isnan(prepared.post_counts[own])
    prepared = _subset(prepared, keep)

    post = prepared.post_counts[own].astype(np.int64)
    G_f = prepared.G.astype(float)
    ability = prepared.covariates[ability_covariate] if ability_covariate is not None else None

    coords = {
        "obs_id": np.arange(prepared.n_obs),
        "phase": np.arange(prepared.n_phases),
        "child": np.arange(prepared.n_children),
    }
    with pm.Model(coords=coords) as model:
        phase_d = pm.Data("phase_idx", prepared.phase.astype(np.int64), dims="obs_id")
        child_idx_d = pm.Data("child_idx", prepared.child_idx.astype(np.int64), dims="obs_id")
        A_std_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
        G_d = pm.Data("G", G_f, dims="obs_id")
        ability_d = (
            pm.Data(f"{ability_covariate}_std", ability, dims="obs_id")
            if ability is not None
            else None
        )

        alpha = _scalar_prior("alpha", _priors.alpha_prior)
        alpha_time = pm.Normal("alpha_time", mu=0.0, sigma=0.5, dims="phase")
        gamma_A = _priors.gamma_age_prior().to_pymc("gamma_A")
        eta = alpha + alpha_time[phase_d] + gamma_A * A_std_d

        # group x time (or single group main effect).
        if group_by_time:
            b_grp = _priors.tau_prior().to_pymc("b_grp_time", dims="phase")
            eta = eta + b_grp[phase_d] * G_d
        else:
            beta_grp = _priors.tau_prior().to_pymc("beta_grp")
            eta = eta + beta_grp * G_d

        # ability main / ability x time.
        if ability_d is not None:
            if ability_by_time:
                g_ab = _priors.gamma_cross_prior().to_pymc("gamma_ability_time", dims="phase")
                eta = eta + g_ab[phase_d] * ability_d
            else:
                gamma_ability = _priors.gamma_cross_prior().to_pymc("gamma_ability")
                eta = eta + gamma_ability * ability_d

        # group x ability cross term.
        if group_ability:
            ga_prod = pm.Data("int_group_ability", G_f * np.asarray(ability, dtype=float), dims="obs_id")
            gamma_grp_ability = _priors.gamma_cross_prior().to_pymc("gamma_grp_ability")
            eta = eta + gamma_grp_ability * ga_prod

        if use_subject_random_intercept:
            sigma_child = pm.HalfNormal("sigma_child", sigma=sigma_child_prior_sigma)
            u_child_raw = pm.Normal("u_child_raw", mu=0.0, sigma=1.0, dims="child")
            u_child = pm.Deterministic("u_child", sigma_child * u_child_raw, dims="child")
            eta = eta + u_child[child_idx_d]

        eta = pm.Deterministic("eta", eta, dims="obs_id")
        if likelihood == "beta_binomial":
            kappa = _scalar_prior("kappa", _priors.kappa_prior)
            beta_binomial_from_logit(
                "y_post", eta, n_trials=prepared.n_trials[own], kappa=kappa,
                observed=post, dims="obs_id",
            )
        else:  # bernoulli_offfloor: PRIMARY estimand for floored outcomes (e.g. P)
            pm.Bernoulli(
                "y_offfloor", logit_p=eta,
                observed=(post > 0).astype(np.int64), dims="obs_id",
            )

    return BuiltModel(model=model, variables=_variables_dict(model), prepared=prepared)


def build_aligned_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str,
    ability_covariate: str | None = None,
    use_cohort: bool = True,
    use_dose: bool = False,
    likelihood: str = "beta_binomial",
) -> BuiltModel:
    """Per-protocol onset-aligned single-gain ANCOVA (LRPAL).

    A cross-sectional Beta-Binomial ANCOVA of the aligned post-score on its own
    onset baseline, age-at-onset and cognitive ability, plus -- optionally -- the
    cohort indicator (immediate vs wait-list) and the cumulative session dose.
    One row per child (``phase_mode="aligned"``), so there is **no child random
    intercept**.

    The cohort term (``beta_cohort``) is **not** a randomised effect: it contrasts
    the two arms at their own onset-aligned endpoints, confounded by age-at-onset
    and cohort/timing -- report it as a per-protocol association, never as the ITT
    treatment effect. ``use_dose`` adds a within-arm cumulative-session covariate,
    a collider descendant of group and ability -- a sensitivity variant, not the
    primary adjustment set.
    """
    if prepared.phase_mode != "aligned":
        raise ValueError("build_aligned_model requires phase_mode='aligned'")
    if likelihood not in ("beta_binomial", "bernoulli_offfloor"):
        raise ValueError(
            "likelihood must be 'beta_binomial' or 'bernoulli_offfloor', "
            f"got {likelihood!r}"
        )
    own = outcome_symbol
    if own not in prepared.post_counts or own not in prepared.pre_logit:
        raise KeyError(f"Outcome {own!r} needs pre+post scores in prepared data")
    if ability_covariate is not None and ability_covariate not in prepared.covariates:
        raise KeyError(f"ability_covariate {ability_covariate!r} not in prepared.covariates")
    if use_dose and "dose" not in prepared.covariates:
        raise KeyError(
            "use_dose=True requires a 'dose' covariate "
            "(load with load_and_prepare_aligned(include_dose=True))"
        )

    keep = ~np.isnan(prepared.post_counts[own]) & ~np.isnan(prepared.pre_logit[own])
    prepared = _subset(prepared, keep)

    post = prepared.post_counts[own].astype(np.int64)
    cohort = prepared.G.astype(float)
    own_pre_std, _ = standardise(prepared.pre_logit[own])

    coords = {"obs_id": np.arange(prepared.n_obs)}
    with pm.Model(coords=coords) as model:
        own_pre_d = pm.Data("own_pre_logit", own_pre_std, dims="obs_id")
        A_std_d = pm.Data("A_std", prepared.A_std, dims="obs_id")

        alpha = _scalar_prior("alpha", _priors.alpha_prior)
        gamma_own = _priors.gamma_own_prior().to_pymc("gamma_own")
        gamma_A = _priors.gamma_age_prior().to_pymc("gamma_A")
        eta = alpha + gamma_own * own_pre_d + gamma_A * A_std_d

        if use_cohort:
            cohort_d = pm.Data("cohort", cohort, dims="obs_id")
            beta_cohort = _priors.tau_prior().to_pymc("beta_cohort")
            eta = eta + beta_cohort * cohort_d
        if ability_covariate is not None:
            ability_d = pm.Data(
                f"{ability_covariate}_std",
                prepared.covariates[ability_covariate], dims="obs_id",
            )
            gamma_ability = _priors.gamma_cross_prior().to_pymc("gamma_ability")
            eta = eta + gamma_ability * ability_d
        if use_dose:
            dose_d = pm.Data("dose_std", prepared.covariates["dose"], dims="obs_id")
            gamma_dose = _priors.gamma_cross_prior().to_pymc("gamma_dose")
            eta = eta + gamma_dose * dose_d

        eta = pm.Deterministic("eta", eta, dims="obs_id")
        if likelihood == "beta_binomial":
            kappa = _scalar_prior("kappa", _priors.kappa_prior)
            beta_binomial_from_logit(
                "y_post", eta, n_trials=prepared.n_trials[own], kappa=kappa,
                observed=post, dims="obs_id",
            )
        else:  # bernoulli_offfloor: PRIMARY estimand for floored outcomes (e.g. P)
            pm.Bernoulli(
                "y_offfloor", logit_p=eta,
                observed=(post > 0).astype(np.int64), dims="obs_id",
            )

    return BuiltModel(model=model, variables=_variables_dict(model), prepared=prepared)


# ---------------------------------------------------------------------------
# Longitudinal dynamic factories (LRP67 LCSM, LRP68 RI-CLPM)
# ---------------------------------------------------------------------------


def build_lcsm_model(
    panel: WavePanel,
    *,
    reading_symbol: str = "W",
    coupling_prior_sigma: float = 0.5,
    self_prior_sigma: float = 0.5,
    intercept_prior_sigma: float = 1.5,
    covariate_prior_sigma: float = 0.5,
    use_process_noise: bool = True,
    shared_process_noise: bool = False,
    sigma_proc_prior_sigma: float = 0.5,
    sigma_init_prior_sigma: float = 1.0,
    kappa_prior_sigma: float = 50.0,
) -> BuiltModel:
    """Full coupled latent change-score model (LRP67) on the logit scale.

    A latent logit true-score ``x_m[i, t]`` is modelled for each measure ``m``
    (default ``W`` reading, ``L`` letter-sounds, ``E`` expressive vocabulary),
    child ``i`` and wave ``t``. The within-child trajectory follows a McArdle
    latent change-score recursion with **process noise**::

        x_m[i, 1] = mu1_m + sigma1_m * z1_m[i]                      (non-centred)
        x_m[i, t] = x_m[i, t-1] + Delta_m[i, t]
        Delta_m[i, t] = mean_Delta_m[i, t] + sigma_proc_m * zproc_m[i, t]

    The headline coupling is on the **reading** change: prior-wave letter-sounds
    and vocabulary predict subsequent reading change (the longitudinal,
    within-trajectory analogue of LRP65's between-child story)::

        mean_Delta_W = a_W + b_W * x_W[t-1]
                     + sum_{c != W} g_c * x_c[t-1]
                     + d_age_W * age[t-1] + d_dose_W * dose[t]

    Non-reading measures get a self-proportional change plus age / dose only.
    All change coefficients are **time-invariant** (pooled across the 3
    transitions) — a deliberate constraint at n~54. Everything is non-centred
    for sampling.

    The observed counts enter via a **masked** Beta-Binomial (the LRP55
    flattened-mask idiom): ``mu = sigmoid(x_m[i, t])`` is the logit mean and
    ``kappa_m`` the dispersion (measurement overdispersion, distinct from the
    dynamic ``sigma_proc``). Only the unmasked cells in ``panel.obs_mask`` are
    observed, so a child missing one score still contributes its other waves.

    The ``use_process_noise`` / ``shared_process_noise`` / ``*_prior_sigma``
    knobs implement the fallback ladder for sampling trouble (tighten priors;
    share one process-noise sd; drop process noise entirely).
    """
    OUT = tuple(panel.outcomes)
    if reading_symbol not in OUT:
        raise KeyError(
            f"reading_symbol {reading_symbol!r} not in panel.outcomes {OUT}"
        )
    K = len(OUT)
    N = panel.n_children
    T = panel.n_waves
    if T < 2:
        raise ValueError("LCSM needs at least two waves")
    cross = [s for s in OUT if s != reading_symbol]
    jidx = {s: i for i, s in enumerate(OUT)}

    # Observed counts / mask / denominators stacked as (N, T, K) in OUT order.
    counts_int = np.stack(
        [np.nan_to_num(panel.counts[s], nan=0.0).astype(np.int64) for s in OUT],
        axis=2,
    )
    mask = np.stack([panel.obs_mask[s] for s in OUT], axis=2)  # (N, T, K) bool
    n_trials_vec = np.array([panel.n_trials[s] for s in OUT], dtype=int)  # (K,)
    # Observed wave-1 mean logit anchors the initial-latent prior mean.
    w1_anchor = np.array(
        [np.nanmean(panel.logit[s][:, 0]) for s in OUT], dtype=float
    )

    coords = {
        "child": np.arange(N),
        "wave": panel.waves,
        "trans": panel.waves[1:],  # transitions into waves 2..T
        "outcome": list(OUT),
    }

    from dse_research_utils.math.constants import EPSILON  # local import

    with pm.Model(coords=coords) as model:
        age = pm.Data("age_std", panel.age_std, dims=("child", "wave"))
        dose = pm.Data("dose_std", panel.dose_std, dims=("child", "wave"))

        # Structural parameters (time-invariant, pooled over transitions).
        mu1 = pm.Normal("mu1", mu=w1_anchor, sigma=1.0, dims="outcome")
        sigma1 = pm.HalfNormal("sigma1", sigma=sigma_init_prior_sigma, dims="outcome")
        a_change = pm.Normal(
            "a_change", mu=0.0, sigma=intercept_prior_sigma, dims="outcome"
        )
        b_self = pm.Normal("b_self", mu=0.0, sigma=self_prior_sigma, dims="outcome")
        d_age = pm.Normal("d_age", mu=0.0, sigma=covariate_prior_sigma, dims="outcome")
        d_dose = pm.Normal(
            "d_dose", mu=0.0, sigma=covariate_prior_sigma, dims="outcome"
        )
        # Headline cross-couplings into the reading change (one per other measure).
        g_cross = {
            s: pm.Normal(f"g_{s}", mu=0.0, sigma=coupling_prior_sigma) for s in cross
        }
        kappa = pm.HalfNormal("kappa", sigma=kappa_prior_sigma, dims="outcome")

        sigma_proc: dict[str, pt.TensorVariable] = {}
        zproc: dict[str, pt.TensorVariable] = {}
        if use_process_noise:
            if shared_process_noise:
                sp = pm.HalfNormal("sigma_proc", sigma=sigma_proc_prior_sigma)
                sigma_proc = {s: sp for s in OUT}
            else:
                spv = pm.HalfNormal(
                    "sigma_proc", sigma=sigma_proc_prior_sigma, dims="outcome"
                )
                sigma_proc = {s: spv[jidx[s]] for s in OUT}
            zproc = {
                s: pm.Normal(f"zproc_{s}", 0.0, 1.0, dims=("child", "trans"))
                for s in OUT
            }

        # Initial latent (wave index 0), non-centred.
        x: dict[str, list[pt.TensorVariable]] = {}
        for s in OUT:
            z1 = pm.Normal(f"z1_{s}", 0.0, 1.0, dims="child")
            x[s] = [
                pm.Deterministic(
                    f"x1_{s}", mu1[jidx[s]] + sigma1[jidx[s]] * z1, dims="child"
                )
            ]

        # Latent change-score recursion over transitions (t = 1 .. T-1).
        for k in range(T - 1):
            t = k + 1
            prev = {s: x[s][t - 1] for s in OUT}
            for s in OUT:
                m = a_change[jidx[s]] + b_self[jidx[s]] * prev[s]
                m = m + d_age[jidx[s]] * age[:, t - 1] + d_dose[jidx[s]] * dose[:, t]
                if s == reading_symbol:
                    for cs in cross:
                        m = m + g_cross[cs] * prev[cs]
                delta = m
                if use_process_noise:
                    delta = delta + sigma_proc[s] * zproc[s][:, k]
                x[s].append(prev[s] + delta)

        # Stack latent to (child, wave, outcome) for reporting + likelihood.
        X = pt.stack([pt.stack(x[s], axis=1) for s in OUT], axis=2)
        X = pm.Deterministic("x_latent", X, dims=("child", "wave", "outcome"))

        # Masked Beta-Binomial observation (LRP55 flattened-mask idiom).
        mu = pm.math.sigmoid(X)
        mu_clip = pm.math.clip(mu, EPSILON, 1 - EPSILON)
        alpha_bb = (mu_clip * kappa[None, None, :]).reshape((-1,))
        beta_bb = ((1 - mu_clip) * kappa[None, None, :]).reshape((-1,))
        idx_i, idx_t, idx_k = np.nonzero(mask)
        lin = np.ravel_multi_index((idx_i, idx_t, idx_k), (N, T, K))
        pm.BetaBinomial(
            "y_obs",
            n=n_trials_vec[idx_k],
            alpha=alpha_bb[lin],
            beta=beta_bb[lin],
            observed=counts_int[idx_i, idx_t, idx_k],
        )

    return BuiltModel(model=model, variables=_variables_dict(model), prepared=panel)


def riclpm_structure_mask(
    structure: str,
    outcomes: tuple[str, ...],
    reading_symbol: str = "W",
    letter_symbol: str = "L",
) -> np.ndarray:
    """Build the ``(K, K)`` 0/1 mask selecting which entries of the RI-CLPM
    transition matrix ``A[target, source]`` are free.

    The diagonal (autoregressive, AR) is always on. Off-diagonals are toggled
    per competing structure:

    - ``"ar"``        : AR only (no cross-lagged paths).
    - ``"l_to_r"``    : AR + letter-sounds -> reading (``A[W, L]``).
    - ``"r_driven"``  : AR + reading -> each other measure (reverse / practice).
    - ``"reciprocal"``: AR + all cross-lagged paths.
    """
    j = {s: i for i, s in enumerate(outcomes)}
    K = len(outcomes)
    mask = np.eye(K)
    if structure == "ar":
        pass
    elif structure == "l_to_r":
        mask[j[reading_symbol], j[letter_symbol]] = 1.0
    elif structure == "r_driven":
        for s in outcomes:
            if s != reading_symbol:
                mask[j[s], j[reading_symbol]] = 1.0
    elif structure == "reciprocal":
        mask[:] = 1.0
    else:
        raise ValueError(
            f"structure must be one of ar/l_to_r/r_driven/reciprocal, got {structure!r}"
        )
    return mask


def build_riclpm_model(
    panel: WavePanel,
    *,
    structure: str = "reciprocal",
    reading_symbol: str = "W",
    letter_symbol: str = "L",
    cross_prior_sigma: float = 0.5,
    ar_prior_sigma: float = 0.5,
    sigma_u_prior_sigma: float = 1.0,
    sigma_w_prior_sigma: float = 1.0,
    covariate_prior_sigma: float = 0.5,
    kappa_prior_sigma: float = 50.0,
) -> BuiltModel:
    """Constrained random-intercept cross-lagged panel model (LRP68).

    The within-child triangulation of LRP67 / LRP65. The logit mean decomposes
    into a stable between-child trait and a within-child deviation::

        m_m[i, t] = u_m[i] + w_m[i, t] + d_age_m * age[i, t] + d_dose_m * dose[i, t]

    ``u_m[i]`` is a non-centred child random intercept per measure (the stable
    trait, independent across measures by default). The within-child deviations
    ``w`` follow a **time-invariant** VAR(1)::

        w[i, t] = A @ w[i, t-1] + innovation[i, t]

    ``A[target, source]`` carries the autoregressive (diagonal) and cross-lagged
    (off-diagonal) coefficients, pooled across the three transitions. Which
    off-diagonals are free is set by ``structure`` (see
    :func:`riclpm_structure_mask`) so the pipeline can fit the competing
    AR-only / L->R / R-driven / reciprocal models and compare them by LOO.

    The headline is ``A[reading, letter]`` (``A[W, L]``): when a child is
    temporarily above their own expected letter-sounds, do they make greater
    subsequent reading gains? Cross-lagged paths carry a regularising
    ``Normal(0, cross_prior_sigma)`` prior (default 0.5; refit at 0.3 / 0.7 for
    the prior-sensitivity check). Observed scores enter the **masked**
    Beta-Binomial directly — there is no measurement-error / latent-indicator
    layer (deliberately, at n~54).
    """
    OUT = tuple(panel.outcomes)
    if reading_symbol not in OUT or letter_symbol not in OUT:
        raise KeyError(
            f"reading_symbol / letter_symbol must be in panel.outcomes {OUT}"
        )
    K = len(OUT)
    N = panel.n_children
    T = panel.n_waves
    if T < 2:
        raise ValueError("RI-CLPM needs at least two waves")

    counts_int = np.stack(
        [np.nan_to_num(panel.counts[s], nan=0.0).astype(np.int64) for s in OUT],
        axis=2,
    )
    mask = np.stack([panel.obs_mask[s] for s in OUT], axis=2)  # (N, T, K)
    n_trials_vec = np.array([panel.n_trials[s] for s in OUT], dtype=int)
    # Grand-mean logit per measure anchors the trait-mean prior.
    grand_anchor = np.array(
        [np.nanmean(panel.logit[s]) for s in OUT], dtype=float
    )
    smask = riclpm_structure_mask(structure, OUT, reading_symbol, letter_symbol)
    sd_matrix = np.full((K, K), cross_prior_sigma, dtype=float)
    np.fill_diagonal(sd_matrix, ar_prior_sigma)

    coords = {
        "child": np.arange(N),
        "wave": panel.waves,
        "trans": panel.waves[1:],
        "outcome": list(OUT),
        "outcome2": list(OUT),  # source axis of the transition matrix
    }

    from dse_research_utils.math.constants import EPSILON  # local import

    with pm.Model(coords=coords) as model:
        age = pm.Data("age_std", panel.age_std, dims=("child", "wave"))
        dose = pm.Data("dose_std", panel.dose_std, dims=("child", "wave"))

        # Stable between-child trait (non-centred, independent across measures).
        mu = pm.Normal("mu", mu=grand_anchor, sigma=1.0, dims="outcome")
        sigma_u = pm.HalfNormal("sigma_u", sigma=sigma_u_prior_sigma, dims="outcome")
        zu = pm.Normal("zu", 0.0, 1.0, dims=("child", "outcome"))
        u = pm.Deterministic(
            "u_child", mu[None, :] + sigma_u[None, :] * zu, dims=("child", "outcome")
        )

        # Time-invariant transition matrix A[target, source]; off-diagonals
        # gated by the competing-structure mask.
        A_raw = pm.Normal("A_raw", mu=0.0, sigma=sd_matrix, dims=("outcome", "outcome2"))
        A = pm.Deterministic("A", A_raw * smask, dims=("outcome", "outcome2"))

        # Within-child deviations: wave-1 free, then VAR(1) with innovations.
        sigma_w1 = pm.HalfNormal("sigma_w1", sigma=sigma_w_prior_sigma, dims="outcome")
        zw1 = pm.Normal("zw1", 0.0, 1.0, dims=("child", "outcome"))
        sigma_inn = pm.HalfNormal("sigma_inn", sigma=sigma_w_prior_sigma, dims="outcome")
        zinn = pm.Normal("zinn", 0.0, 1.0, dims=("child", "trans", "outcome"))

        w = [sigma_w1[None, :] * zw1]  # (N, K)
        for k in range(T - 1):
            w_mean = pt.dot(w[k], A.T)  # (N, K): w_t[target] = sum_s A[target,s] w_prev[s]
            w.append(w_mean + sigma_inn[None, :] * zinn[:, k, :])
        Wdev = pt.stack(w, axis=1)  # (N, T, K)

        d_age = pm.Normal("d_age", 0.0, covariate_prior_sigma, dims="outcome")
        d_dose = pm.Normal("d_dose", 0.0, covariate_prior_sigma, dims="outcome")
        m_logit = (
            u[:, None, :]
            + Wdev
            + d_age[None, None, :] * age[:, :, None]
            + d_dose[None, None, :] * dose[:, :, None]
        )
        m_logit = pm.Deterministic(
            "m_logit", m_logit, dims=("child", "wave", "outcome")
        )

        kappa = pm.HalfNormal("kappa", sigma=kappa_prior_sigma, dims="outcome")
        mu_p = pm.math.sigmoid(m_logit)
        mu_clip = pm.math.clip(mu_p, EPSILON, 1 - EPSILON)
        alpha_bb = (mu_clip * kappa[None, None, :]).reshape((-1,))
        beta_bb = ((1 - mu_clip) * kappa[None, None, :]).reshape((-1,))
        idx_i, idx_t, idx_k = np.nonzero(mask)
        lin = np.ravel_multi_index((idx_i, idx_t, idx_k), (N, T, K))
        pm.BetaBinomial(
            "y_obs",
            n=n_trials_vec[idx_k],
            alpha=alpha_bb[lin],
            beta=beta_bb[lin],
            observed=counts_int[idx_i, idx_t, idx_k],
        )

    return BuiltModel(model=model, variables=_variables_dict(model), prepared=panel)
