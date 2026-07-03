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

from dse_research_utils.statistics.models.pymc_utils import (
    get_variables_dict as _variables_dict,
)

from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.hsgp import (
    build_hsgp_1d,
    build_tau_modifier,
)
from language_reading_predictors.statistical_models.likelihood import (
    beta_binomial_from_logit,
)
from language_reading_predictors.statistical_models.measures import (
    ITT_OUTCOMES,
    is_distal,
)
from language_reading_predictors.statistical_models.preprocessing import (
    LongitudinalPanel,
    PreparedData,
    WavePanel,
    standardise,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _scalar_prior(name: str, prior_ctor) -> pt.TensorVariable:
    return prior_ctor().to_pymc(name)


def _tau_sigma_for(outcome_symbol: str, override: float | None = None) -> float:
    """Treatment-effect prior SD for a single-outcome causal term (issue #141).

    Returns ``override`` when given (prior-sensitivity fits), else the outcome
    tier default: the tighter ``TAU_SIGMA_DISTAL`` for broad standardised-transfer
    outcomes (``measures.DISTAL_OUTCOMES``) and the wider ``TAU_SIGMA_PROXIMAL``
    for the directly-taught / decoding outcomes. Applied to the randomised
    treatment effect only (ITT ``tau``, gain-factors ``beta_trt``, level-factors
    group contrast, DiD ``delta``) — not to adjusted-association group terms
    (mechanism / dose-response ``beta_G``, aligned ``beta_cohort``).
    """
    if override is not None:
        return override
    return (
        _priors.TAU_SIGMA_DISTAL
        if is_distal(outcome_symbol)
        else _priors.TAU_SIGMA_PROXIMAL
    )


def _add_child_random_intercept(
    eta: pt.TensorVariable,
    child_idx: pt.TensorVariable,
    *,
    sigma_prior_sigma: float = 0.5,
) -> pt.TensorVariable:
    """Add a non-centred subject random intercept to ``eta`` (call inside a model).

    Creates ``sigma_child ~ HalfNormal(sigma_prior_sigma)``,
    ``u_child_raw ~ Normal(0, 1, dims="child")`` and the deterministic
    ``u_child = sigma_child * u_child_raw``, then returns ``eta + u_child[child_idx]``.
    Centralises the block previously copy-pasted across the mechanism,
    dose-response, DiD, gain-factors and level-factors factories so the random-
    intercept parameterisation cannot drift between them.
    """
    sigma_child = pm.HalfNormal("sigma_child", sigma=sigma_prior_sigma)
    u_child_raw = pm.Normal("u_child_raw", mu=0.0, sigma=1.0, dims="child")
    u_child = pm.Deterministic("u_child", sigma_child * u_child_raw, dims="child")
    return eta + u_child[child_idx]


@dataclass
class BuiltModel:
    model: pm.Model
    variables: dict[str, pt.TensorVariable]
    prepared: PreparedData | WavePanel | LongitudinalPanel
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
    tau_sigma: float | None = None,
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
        silently fits two unidentifiable GPs.
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
    tau_sigma
        Override the treatment-effect prior SD (issue #141). ``None`` (default)
        uses the outcome tier: ``TAU_SIGMA_DISTAL`` (0.3) for the broad
        standardised-transfer outcomes in ``measures.DISTAL_OUTCOMES``,
        ``TAU_SIGMA_PROXIMAL`` (0.5) otherwise. Pass an explicit value for a
        prior-sensitivity fit (``scripts/tau_prior_sensitivity.py``).
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
        tau0 = _priors.tau_prior(
            sigma=_tau_sigma_for(own, tau_sigma)
        ).to_pymc("tau")

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
    model with the GP included.

    ``use_residual_correlation`` (default False): when True, adds an
    ``u_i ~ MvNormal(0, Sigma)`` residual with ``Sigma = diag(sigma) Corr
    diag(sigma)`` and ``Corr ~ LKJCorr(eta=4)``, non-centred via
    ``pm.LKJCholeskyCov`` + ``z_raw``. Turned off by default after the
    2026-04-18 LRP55 fit showed the LKJ block was prior-dominated (all
    off-diagonal correlation CIs spanning zero, sigma_outcome CIs reaching
    zero).
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
    independent.

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
            eta = _add_child_random_intercept(
                eta, child_idx_d, sigma_prior_sigma=sigma_child_prior_sigma
            )

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
        # When age is the
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
# LRP77: period-resolved dose-response factory (#104 Phase 2)
# ---------------------------------------------------------------------------


def build_dose_response_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str = "W",
    adjust_baseline_symbol: str = "W",
    dose_covariate: str = "attend",
    dose_stage_covariate: str | None = "attend_cumul",
    period_varying_dose: bool = True,
    use_subject_random_intercept: bool = True,
    adjust_group: bool = True,
    adjust_age: bool = True,
    ability_adjust_symbols: Iterable[str] = (),
    sigma_child_prior_sigma: float = 0.5,
) -> BuiltModel:
    """Period-resolved dose-response on the outcome post-score (#104 Phase 2).

    Outcome-generic: the target is ``outcome_symbol`` (default ``"W"``, word
    reading — its lead use in LRP77; reused for letter sounds ``"L"`` in LRP86)
    and the autoregressive baseline is ``adjust_baseline_symbol`` (default the
    same measure).

    Estimand: how the intervention **dose** (per-period sessions attended)
    relates to the outcome's **conditional change** — the outcome post-count
    modelled Beta-Binomial conditional on its own baseline logit — and whether
    that dose-gain slope **varies by period**.

    Uses all three phase transitions (``prepared.phase_mode == "all"``) with
    phase-specific intercepts. The linear predictor is

        eta = alpha + alpha_phase[p]
            + beta_G * G                      # arm (G=1 = immediate-intervention, G = 2 - group)
            + gamma_own * logit(outcome_pre)  # autoregression / RTM
            + gamma_A * z(age)                # maturation precision covariate
            + u_child[child]                  # subject random intercept
            + dose_term                       # the estimand (see below)
            + gamma_dose_stage * z(attend_cumul)   # dose-stage control
            + sum_s gamma_s_pre * logit(s_pre)     # optional ability adjusters

    ``dose_term`` is ``beta_dose_phase[p] * z(attend)`` with partial-pooled
    per-period slopes ``beta_dose_phase = mu_dose + sigma_dose * z_p`` when
    ``period_varying_dose`` is True (the headline model), or a single pooled
    ``beta_dose * z(attend)`` when False (the nested comparator). The dose enters
    standardised, so the slope is the outcome-logit change per 1 SD of
    per-period dose.

    Causal note (DAG v5): for the dose -> outcome edge, ``G`` (intervention arm)
    is the sole backdoor confounder; the outcome's own baseline is the
    regression-to-the-mean control and ``age`` a precision covariate. The full
    sample (including the
    waitlist controls' zero-dose period-1 rows) anchors the slope at dose = 0.
    The model assumes **no ability -> dose** edge; ``ability_adjust_symbols``
    (the sensitivity fit) conditions on the baseline-skill cluster to probe it.

    Parameters mirror :func:`build_mechanism_model`'s backbone options
    (``use_subject_random_intercept``, ``adjust_baseline_symbol``). ``G`` and the
    age covariate are toggled by ``adjust_group`` / ``adjust_age``;
    ``dose_stage_covariate=None`` drops the cumulative-dose control.
    """
    if prepared.phase_mode != "all":
        raise ValueError("Dose-response factory requires phase_mode='all'")
    if outcome_symbol not in prepared.pre_logit:
        raise KeyError(f"Outcome {outcome_symbol!r} missing from prepared data")
    if adjust_baseline_symbol not in prepared.pre_logit:
        raise KeyError(
            f"Baseline {adjust_baseline_symbol!r} missing from prepared data"
        )
    if dose_covariate not in prepared.covariates:
        raise KeyError(
            f"Dose covariate {dose_covariate!r} missing from prepared.covariates; "
            "pass it via load_and_prepare(covariates=...)"
        )
    if dose_stage_covariate is not None and dose_stage_covariate not in prepared.covariates:
        raise KeyError(
            f"Dose-stage covariate {dose_stage_covariate!r} missing from "
            "prepared.covariates"
        )
    ability_adjust_symbols = tuple(ability_adjust_symbols)
    for s in ability_adjust_symbols:
        if s not in prepared.pre_logit:
            raise KeyError(
                f"Ability-adjuster {s!r} has no pre-score; add it to "
                "load_and_prepare(outcomes=...)"
            )

    outcome_post = prepared.post_counts[outcome_symbol]
    keep = ~np.isnan(outcome_post)
    prepared = _subset(prepared, keep)

    outcome_post = prepared.post_counts[outcome_symbol].astype(np.int64)
    N_outcome = prepared.n_trials[outcome_symbol]
    own_pre_logit = prepared.pre_logit[adjust_baseline_symbol]
    dose = prepared.covariates[dose_covariate]

    coords = {
        "obs_id": np.arange(prepared.n_obs),
        "phase": np.arange(prepared.n_phases),
        "child": np.arange(prepared.n_children),
    }

    with pm.Model(coords=coords) as model:
        A_std_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
        G_d = pm.Data("G", prepared.G.astype(float), dims="obs_id")
        own_pre_d = pm.Data("own_pre_logit", own_pre_logit, dims="obs_id")
        phase_d = pm.Data("phase_idx", prepared.phase.astype(np.int64), dims="obs_id")
        child_idx_d = pm.Data(
            "child_idx", prepared.child_idx.astype(np.int64), dims="obs_id"
        )
        dose_d = pm.Data(f"{dose_covariate}_std", dose, dims="obs_id")
        dose_stage_d = None
        if dose_stage_covariate is not None:
            dose_stage_d = pm.Data(
                f"{dose_stage_covariate}_std",
                prepared.covariates[dose_stage_covariate],
                dims="obs_id",
            )
        ability_data: dict[str, pt.TensorVariable] = {}
        for s in ability_adjust_symbols:
            ability_data[s] = pm.Data(
                f"{s}_pre_logit", prepared.pre_logit[s], dims="obs_id"
            )

        alpha = _scalar_prior("alpha", _priors.alpha_prior)
        alpha_phase = pm.Normal("alpha_phase", mu=0.0, sigma=0.5, dims="phase")
        gamma_own = _priors.gamma_own_prior().to_pymc("gamma_own")

        eta = alpha + alpha_phase[phase_d] + gamma_own * own_pre_d

        if adjust_group:
            beta_G = _priors.tau_prior().to_pymc("beta_G")
            eta = eta + beta_G * G_d
        if adjust_age:
            gamma_A = _priors.gamma_cross_prior().to_pymc("gamma_A")
            eta = eta + gamma_A * A_std_d

        if use_subject_random_intercept:
            eta = _add_child_random_intercept(
                eta, child_idx_d, sigma_prior_sigma=sigma_child_prior_sigma
            )

        # Dose effect (the estimand). Standardised dose -> slope is per-1-SD.
        if period_varying_dose:
            mu_dose = _priors.beta_mech_prior().to_pymc("mu_dose")
            sigma_dose = _priors.sigma_dose_phase_prior().to_pymc("sigma_dose")
            beta_dose_phase_raw = pm.Normal(
                "beta_dose_phase_raw", mu=0.0, sigma=1.0, dims="phase"
            )
            beta_dose_phase = pm.Deterministic(
                "beta_dose_phase", mu_dose + sigma_dose * beta_dose_phase_raw, dims="phase"
            )
            eta = eta + beta_dose_phase[phase_d] * dose_d
        else:
            beta_dose = _priors.beta_mech_prior().to_pymc("beta_dose")
            eta = eta + beta_dose * dose_d

        # Dose-stage control (prior cumulative dose), so a dose-stage effect is
        # not misread as a period effect.
        if dose_stage_d is not None:
            gamma_dose_stage = _priors.gamma_cross_prior().to_pymc("gamma_dose_stage")
            eta = eta + gamma_dose_stage * dose_stage_d

        # Baseline-skill (ability) adjusters - the no-g->dose sensitivity fit.
        for s in ability_adjust_symbols:
            gamma_s = _priors.gamma_cross_prior().to_pymc(f"gamma_{s}_pre")
            eta = eta + gamma_s * ability_data[s]

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
    period_varying_dose: bool = False,
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
    period_varying_dose
        With ``dose=True``, replace the single pooled ``beta_dose`` by
        partial-pooled per-period slopes ``beta_dose_phase[p] = mu_dose +
        sigma_dose * z_p`` (non-centred), so each period's dose-gain slope shrinks
        toward the shared mean. Mirrors :func:`build_dose_response_model`; the
        nested ``period_varying_dose=False`` model is its pooled comparator for a
        PSIS-LOO test of period variation (#135). Requires ``dose=True``.
    """
    if prepared.phase_mode != "all":
        raise ValueError("build_did_model requires phase_mode='all'")
    own = outcome_symbol
    if own not in prepared.post_counts or own not in prepared.pre_logit:
        raise KeyError(f"Outcome {own!r} missing pre/post in prepared data")
    periods = tuple(int(p) for p in periods)
    if dose and "attend" not in prepared.covariates:
        raise KeyError("dose=True requires the 'attend' covariate to be loaded")
    if period_varying_dose and not dose:
        raise ValueError("period_varying_dose=True requires dose=True")

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
    if period_varying_dose:
        coords["dose_phase"] = np.arange(len(periods))
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
            child_idx_d = pm.Data(
                "child_idx", prepared.child_idx.astype(np.int64), dims="obs_id"
            )
            eta = _add_child_random_intercept(eta, child_idx_d, sigma_prior_sigma=0.5)

        # Linear predictor without the treatment term, so the pipeline can read
        # the off-treatment baseline for the average-marginal-effect translation.
        eta_base = pm.Deterministic("eta_base", eta, dims="obs_id")

        if dose:
            z_attend = pm.Data("z_attend", prepared.covariates["attend"], dims="obs_id")
            if period_varying_dose:
                # Partial-pooled per-period dose slopes (non-centred): each
                # period's slope shrinks toward the shared mean mu_dose. The
                # nested pooled model (period_varying_dose=False) is its LOO
                # comparator (#135). Variable names match build_dose_response_model
                # so the shared dose-slope summary reads them.
                period_pos = np.searchsorted(
                    np.asarray(periods), prepared.phase
                ).astype(np.int64)
                dose_phase_idx = pm.Data("dose_phase_idx", period_pos, dims="obs_id")
                mu_dose = _priors.tau_prior().to_pymc("mu_dose")
                sigma_dose = _priors.sigma_dose_phase_prior().to_pymc("sigma_dose")
                beta_dose_phase = pm.Deterministic(
                    "beta_dose_phase",
                    mu_dose
                    + sigma_dose
                    * pm.Normal("beta_dose_phase_raw", 0.0, 1.0, dims="dose_phase"),
                    dims="dose_phase",
                )
                eta_full = eta_base + beta_dose_phase[dose_phase_idx] * z_attend
            else:
                beta_dose = _priors.tau_prior().to_pymc("beta_dose")
                eta_full = eta_base + beta_dose * z_attend
        else:
            treated_d = pm.Data("treated", treated, dims="obs_id")
            delta = _priors.tau_prior(sigma=_tau_sigma_for(own)).to_pymc("delta")
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


@dataclass
class TwoMediatorData:
    """Row-aligned phase-0 arrays + scalers for the two-mediator g-formula (LRP64).

    Two count mediators — letter-sound knowledge ``L`` and expressive vocabulary
    ``E`` — are each modelled with a Beta-Binomial leg conditioned on their own
    baseline; the outcome leg adds both standardised post-mediators and their
    treatment interactions. :func:`mediation.decompose_two_mediator` re-simulates
    each mediator under each treatment arm to compute the joint indirect effect,
    the direct effect, and the (ordering-dependent) path-specific indirect effects.
    """

    G: np.ndarray
    A_std: np.ndarray
    W1_logit: np.ndarray
    conf1_logit: dict[str, np.ndarray]
    n_trials_W: int
    # Mediator L (letter-sound).
    L1_logit: np.ndarray
    n_trials_L: int
    zL_mean: float
    zL_sd: float
    # Mediator E (expressive vocabulary).
    E1_logit: np.ndarray
    n_trials_E: int
    zE_mean: float
    zE_sd: float
    mediator_symbols: tuple[str, str] = ("L", "E")
    confounder_symbols: tuple[str, ...] = ("R",)


def build_two_mediator_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str = "W",
    mediator_symbols: tuple[str, str] = ("L", "E"),
    confounder_symbols: Iterable[str] = ("R",),
) -> tuple[BuiltModel, TwoMediatorData]:
    """Joint two-mediator + outcome model for the ITT-phase decomposition (LRP64).

    Generalises :func:`build_mediation_model` to **two named count mediators** so
    the word-reading effect can be split into a path via letter-sound knowledge, a
    path via expressive vocabulary, and a direct/residual path. Three Beta-Binomial
    legs share the randomised treatment ``G`` and a baseline-covariate adjustment::

        L_t2 ~ aL0 + aL_G·G + aL_L·logit(L_t1) + aL_A·A + sum aL_c·C_t1
        E_t2 ~ aE0 + aE_G·G + aE_E·logit(E_t1) + aE_A·A + sum aE_c·C_t1
        W_t2 ~ b0 + b_G·G + b_L·zL + b_E·zE + b_GL·G·zL + b_GE·G·zE
               + b_W·logit(W_t1) + b_A·A + sum b_c·C_t1

    where ``zL`` / ``zE`` are the standardised post-mediator logits. The two
    treatment×mediator interactions admit exposure-mediator interaction; the
    natural (in)direct effects are computed by counterfactual simulation in
    :func:`mediation.decompose_two_mediator`, **not** from coefficients.

    Confounders ``C`` (e.g. receptive vocab ``R``) are taken at **baseline (t1)**
    on the logit scale (cross-world assumption). Expressive vocab is a *mediator*
    here, not a confounder, so only its baseline enters (autoregressively in the
    ``E`` leg). Requires ``prepared.phase_mode == 'itt'`` (one row per child).
    """
    if prepared.phase_mode != "itt":
        raise ValueError("Two-mediator factory requires phase_mode='itt'")
    confounder_symbols = tuple(confounder_symbols)
    mL, mE = mediator_symbols
    # The PyMC node and coefficient names below are hard-coded to the L/E legs
    # (L_pre_logit, z_L, aL_*, b_L, ...), so only the ('L', 'E') pair is
    # supported; other symbols would silently mislabel the fitted variables.
    if (mL, mE) != ("L", "E"):
        raise NotImplementedError(
            "build_two_mediator_model hard-codes L/E variable names; "
            f"mediator_symbols must be ('L', 'E'), got {mediator_symbols!r}"
        )
    needed = {outcome_symbol, mL, mE, *confounder_symbols}
    for s in needed:
        if s not in prepared.pre_logit:
            raise KeyError(f"Symbol {s!r} missing from prepared data")

    from language_reading_predictors.statistical_models.preprocessing import (
        logit_safe,
        standardise,
    )

    keep = ~np.isnan(prepared.post_counts[outcome_symbol])
    for s in (mL, mE):
        keep = keep & ~np.isnan(prepared.post_counts[s])
    if not keep.all():
        prepared = _subset(prepared, keep)

    N_W = prepared.n_trials[outcome_symbol]
    N_L = prepared.n_trials[mL]
    N_E = prepared.n_trials[mE]
    L2 = prepared.post_counts[mL].astype(np.int64)
    E2 = prepared.post_counts[mE].astype(np.int64)
    W2 = prepared.post_counts[outcome_symbol].astype(np.int64)

    zL, zL_scaler = standardise(logit_safe(L2, N_L))
    zE, zE_scaler = standardise(logit_safe(E2, N_E))

    L1 = prepared.pre_logit[mL]
    E1 = prepared.pre_logit[mE]
    W1 = prepared.pre_logit[outcome_symbol]
    conf_logit = {s: prepared.pre_logit[s] for s in confounder_symbols}

    coords = {"obs_id": np.arange(prepared.n_obs)}
    G_f = prepared.G.astype(float)

    with pm.Model(coords=coords) as model:
        G_d = pm.Data("G", G_f, dims="obs_id")
        A_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
        L1_d = pm.Data("L_pre_logit", L1, dims="obs_id")
        E1_d = pm.Data("E_pre_logit", E1, dims="obs_id")
        W1_d = pm.Data("W_pre_logit", W1, dims="obs_id")
        conf_d = {
            s: pm.Data(f"{s}_pre_logit", conf_logit[s], dims="obs_id")
            for s in confounder_symbols
        }
        zL_d = pm.Data("z_L", zL, dims="obs_id")
        zE_d = pm.Data("z_E", zE, dims="obs_id")

        # --- Mediator L (letter-sound) ---
        aL0 = _priors.alpha_prior().to_pymc("aL0")
        aL_G = _priors.tau_prior().to_pymc("aL_G")
        aL_L = _priors.gamma_own_prior().to_pymc("aL_L")
        aL_A = _priors.gamma_cross_prior().to_pymc("aL_A")
        mu_L = aL0 + aL_G * G_d + aL_L * L1_d + aL_A * A_d
        for s in confounder_symbols:
            aL_c = _priors.gamma_cross_prior().to_pymc(f"aL_{s}")
            mu_L = mu_L + aL_c * conf_d[s]
        mu_L = pm.Deterministic("mu_L", mu_L, dims="obs_id")
        kappa_L = _priors.kappa_prior().to_pymc("kappa_L")
        beta_binomial_from_logit(
            "L_post", mu_L, n_trials=N_L, kappa=kappa_L, observed=L2, dims="obs_id"
        )

        # --- Mediator E (expressive vocabulary) ---
        aE0 = _priors.alpha_prior().to_pymc("aE0")
        aE_G = _priors.tau_prior().to_pymc("aE_G")
        aE_E = _priors.gamma_own_prior().to_pymc("aE_E")
        aE_A = _priors.gamma_cross_prior().to_pymc("aE_A")
        mu_E = aE0 + aE_G * G_d + aE_E * E1_d + aE_A * A_d
        for s in confounder_symbols:
            aE_c = _priors.gamma_cross_prior().to_pymc(f"aE_{s}")
            mu_E = mu_E + aE_c * conf_d[s]
        mu_E = pm.Deterministic("mu_E", mu_E, dims="obs_id")
        kappa_E = _priors.kappa_prior().to_pymc("kappa_E")
        beta_binomial_from_logit(
            "E_post", mu_E, n_trials=N_E, kappa=kappa_E, observed=E2, dims="obs_id"
        )

        # --- Outcome W ---
        b0 = _priors.alpha_prior().to_pymc("b0")
        b_G = _priors.tau_prior().to_pymc("b_G")
        b_L = _priors.b_path_prior().to_pymc("b_L")
        b_E = _priors.b_path_prior().to_pymc("b_E")
        b_GL = _priors.gamma_cross_prior().to_pymc("b_GL")
        b_GE = _priors.gamma_cross_prior().to_pymc("b_GE")
        b_W = _priors.gamma_own_prior().to_pymc("b_W")
        b_A = _priors.gamma_cross_prior().to_pymc("b_A")
        eta_Y = (
            b0
            + b_G * G_d
            + b_L * zL_d
            + b_E * zE_d
            + b_GL * (G_d * zL_d)
            + b_GE * (G_d * zE_d)
            + b_W * W1_d
            + b_A * A_d
        )
        for s in confounder_symbols:
            b_c = _priors.gamma_cross_prior().to_pymc(f"b_{s}")
            eta_Y = eta_Y + b_c * conf_d[s]
        eta_Y = pm.Deterministic("eta", eta_Y, dims="obs_id")
        kappa_Y = _priors.kappa_prior().to_pymc("kappa_Y")
        beta_binomial_from_logit(
            "y_post", eta_Y, n_trials=N_W, kappa=kappa_Y, observed=W2, dims="obs_id"
        )

    med_data = TwoMediatorData(
        G=prepared.G.astype(float),
        A_std=prepared.A_std,
        W1_logit=W1,
        conf1_logit={s: conf_logit[s] for s in confounder_symbols},
        n_trials_W=int(N_W),
        L1_logit=L1,
        n_trials_L=int(N_L),
        zL_mean=float(zL_scaler.mean),
        zL_sd=float(zL_scaler.sd),
        E1_logit=E1,
        n_trials_E=int(N_E),
        zE_mean=float(zE_scaler.mean),
        zE_sd=float(zE_scaler.sd),
        mediator_symbols=(mL, mE),
        confounder_symbols=confounder_symbols,
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
# LRPHS: regularized-horseshoe predictor-ranking models (#116 Phase E)
# ---------------------------------------------------------------------------


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
    z = pm.Normal("hs_z", mu=0.0, sigma=1.0, dims="predictor")
    return pm.Deterministic("beta", tau * lam_tilde * z, dims="predictor")


def _resolve_level_predictor(prepared: PreparedData, key: str) -> tuple[str, np.ndarray, str]:
    """Concurrent (same-wave) standardised predictor for the levels-mode horseshoe.

    ``age`` -> standardised age; a covariate column -> that standardised covariate;
    a measure symbol -> standardised concurrent logit of its post-count. Missing
    values are mean-imputed (0 on the standardised scale) since PyMC — unlike
    LightGBM — cannot take NaN inputs; the ranking is a sensitivity read, not a
    calibrated fit, so mean-imputation is acceptable and is noted in the report.
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
) -> BuiltModel:
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

    return BuiltModel(model=model, variables=_variables_dict(model), prepared=prepared)


# ---------------------------------------------------------------------------
# LRPMM01: correlated-domain-factor measurement model (#134)
# ---------------------------------------------------------------------------


def build_correlated_factor_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str = "W",
    domains: dict[str, tuple[str, ...]] | None = None,
    structural_covariates: Iterable[str] = ("blocks",),
    use_age: bool = True,
    loading_sigma: float = 1.0,
    predictor_slope_sigma: float = 0.5,
    lkj_eta: float = 2.0,
) -> BuiltModel:
    """Correlated-domain-factor measurement model (LRPMM01, #134).

    Replaces the single latent general ability ``g`` of the (closed) LRP66 with
    **correlated domain factors** - vocabulary / code / grammar - each measured by
    its standardised T1 skill indicators, with an LKJ prior on the factor
    correlation matrix. Factor variances are fixed to 1 and loadings are positive.
    Because the indicator residual variance ``sigma_indicator`` is free, a loading
    ``lambda`` is a coefficient on the unit-variance factor, **not** in general a
    correlation; the indicator-factor **correlation** is ``lambda / sqrt(lambda**2
    + sigma**2)`` (the standardised loading, equal to ``sqrt(communality)``) and
    the **communality** ``lambda**2 / (lambda**2 + sigma**2)`` is the share of the
    indicator explained by its domain factor. A structural Beta-Binomial leg
    regresses the outcome gain (``outcome`` post conditioned on its T1 baseline via
    ``gamma_own``) on the latent factors, giving **measurement-error-corrected**
    factor->gain slopes.

    Identification-neutral but a better measurement match than a single ``g`` for
    the observed same-construct clustering (the locked DAG's deferred option,
    #115). This is a **measurement / triangulation** model, not a causal one: per
    ID-2 each factor->gain slope is a latent-ability-confounded **adjusted
    association**. At n ~ 51 it is fragile and prior-dependent - read the wide
    intervals as the honest result, as the closed LRP66 did.

    ``domains`` maps each factor name to its indicator symbols (default vocabulary
    {R, E} / code {L, B} / grammar {F, T}); every domain needs >= 2 indicators to
    be identified. ``structural_covariates`` are observed adjusters in the
    structural leg (default non-verbal MA ``blocks``); ``use_age`` adds a linear
    age term.
    """
    from language_reading_predictors.statistical_models.preprocessing import standardise

    if prepared.phase_mode not in {"span", "itt"}:
        raise ValueError(
            "Correlated-factor (between-child) model requires phase_mode in "
            f"{{'span', 'itt'}} (one row per child); got {prepared.phase_mode!r}"
        )
    if domains is None:
        domains = {"vocabulary": ("R", "E"), "code": ("L", "B"), "grammar": ("F", "T")}
    domain_names = list(domains)
    for d in domain_names:
        if len(tuple(domains[d])) < 2:
            raise ValueError(
                f"Domain {d!r} has < 2 indicators ({tuple(domains[d])}); a "
                "correlated factor needs at least two indicators to be identified."
            )
    D = len(domain_names)

    # One row per child: drop children missing the outcome post-score.
    post = prepared.post_counts[outcome_symbol]
    keep = ~np.isnan(post)
    if not keep.all():
        prepared = _subset(prepared, keep)

    post = prepared.post_counts[outcome_symbol].astype(np.int64)
    N = prepared.n_trials[outcome_symbol]
    own_pre_logit = prepared.pre_logit[outcome_symbol]

    # Standardised indicator matrix Z (n_obs, J) + per-indicator domain index.
    ind_names: list[str] = []
    domain_of: list[int] = []
    cols: list[np.ndarray] = []
    for di, d in enumerate(domain_names):
        for s in domains[d]:
            if s not in prepared.pre_logit:
                raise KeyError(
                    f"Indicator {s!r} (domain {d!r}) missing from prepared data"
                )
            z, _ = standardise(prepared.pre_logit[s])
            cols.append(z)
            ind_names.append(s)
            domain_of.append(di)
    Z = np.stack(cols, axis=1)
    domain_idx = np.asarray(domain_of, dtype=np.int64)

    coords = {
        "obs_id": np.arange(prepared.n_obs),
        "indicator": ind_names,
        "domain": domain_names,
        "domain_b": domain_names,
    }
    with pm.Model(coords=coords) as model:
        Z_d = pm.Data("Z", Z, dims=("obs_id", "indicator"))
        own_pre_d = pm.Data("own_pre_logit", own_pre_logit, dims="obs_id")

        # --- Measurement: correlated unit-variance domain factors ---
        # LKJ correlation matrix; factor variances are fixed to 1 by transforming
        # standard normals through the correlation's Cholesky (the LKJCholeskyCov
        # sds are unused). The residual variance sigma_indicator is free, so a
        # loading is a coefficient on the unit-variance factor; the standardised
        # loading / indicator-factor correlation is reported as sqrt(communality).
        _, corr, _ = pm.LKJCholeskyCov(
            "factor_cov",
            n=D,
            eta=lkj_eta,
            sd_dist=pm.Exponential.dist(1.0, size=D),
            compute_corr=True,
        )
        pm.Deterministic("factor_corr", corr, dims=("domain", "domain_b"))
        L_corr = pt.linalg.cholesky(corr)
        z_factor = pm.Normal("factor_z", 0.0, 1.0, dims=("obs_id", "domain"))
        factors = pm.Deterministic(
            "factors", z_factor @ L_corr.T, dims=("obs_id", "domain")
        )

        lam = pm.HalfNormal("lambda_load", sigma=loading_sigma, dims="indicator")
        sigma_ind = pm.HalfNormal("sigma_indicator", sigma=1.0, dims="indicator")
        mu_Z = lam[None, :] * factors[:, domain_idx]
        pm.Normal(
            "Z_obs",
            mu=mu_Z,
            sigma=sigma_ind[None, :],
            observed=Z_d,
            dims=("obs_id", "indicator"),
        )
        pm.Deterministic(
            "communality", lam**2 / (lam**2 + sigma_ind**2), dims="indicator"
        )

        # --- Structural: outcome gain ~ factors (+ covariates), Beta-Binomial ---
        alpha = _scalar_prior("alpha", _priors.alpha_prior)
        gamma_own = _priors.gamma_own_prior().to_pymc("gamma_own")
        beta_factor = pm.Normal(
            "beta_factor", 0.0, predictor_slope_sigma, dims="domain"
        )
        eta = alpha + gamma_own * own_pre_d + pm.math.dot(factors, beta_factor)

        if use_age:
            A_std_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
            beta_age = _priors.predictor_slope_prior(predictor_slope_sigma).to_pymc(
                "beta_age"
            )
            eta = eta + beta_age * A_std_d
        for c in structural_covariates:
            if c not in prepared.covariates:
                raise KeyError(
                    f"Structural covariate {c!r} missing from prepared data"
                )
            x_d = pm.Data(
                f"x_{c}", np.asarray(prepared.covariates[c], dtype=float), dims="obs_id"
            )
            beta_c = _priors.predictor_slope_prior(predictor_slope_sigma).to_pymc(
                f"beta_{c}"
            )
            eta = eta + beta_c * x_d

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
            + u_child[i]                           # partial GA repair

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
            beta_trt = _priors.tau_prior(
                sigma=_tau_sigma_for(outcome_symbol)
            ).to_pymc("beta_trt")
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
            eta = _add_child_random_intercept(
                eta, child_idx_d, sigma_prior_sigma=sigma_child_prior_sigma
            )

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
            + u_child[i]                   # partial GA repair

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
        _tau_sigma = _tau_sigma_for(outcome_symbol)
        if group_by_time:
            b_grp = _priors.tau_prior(sigma=_tau_sigma).to_pymc(
                "b_grp_time", dims="phase"
            )
            eta = eta + b_grp[phase_d] * G_d
        else:
            beta_grp = _priors.tau_prior(sigma=_tau_sigma).to_pymc("beta_grp")
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
            eta = _add_child_random_intercept(
                eta, child_idx_d, sigma_prior_sigma=sigma_child_prior_sigma
            )

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
# Longitudinal dynamic factory (LRP67 LCSM)
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
                     + d_age_W * age[t-1]

    Non-reading measures get a self-proportional change plus age only. The
    intervention-dose covariate is **omitted**: it is the locked DAG's ``IS``
    collider, so conditioning on it would reopen the latent-``GA`` backdoor onto
    the letter-sound -> reading couplings (ID-3).
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
    # Observed wave-1 mean logit anchors the initial-latent prior mean. Guard the
    # all-NaN case loudly: an outcome with no observed wave-1 value would make
    # np.nanmean return NaN, which would silently poison mu1's prior mean and
    # surface only as an opaque sampler failure.
    missing_w1 = [s for s in OUT if not np.isfinite(panel.logit[s][:, 0]).any()]
    if missing_w1:
        raise ValueError(
            "LCSM wave-1 anchor is undefined (no observed first-wave value) for: "
            f"{', '.join(missing_w1)}. Drop the outcome or choose a panel with "
            "wave-1 observations."
        )
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

        # Structural parameters (time-invariant, pooled over transitions).
        mu1 = pm.Normal("mu1", mu=w1_anchor, sigma=1.0, dims="outcome")
        sigma1 = pm.HalfNormal("sigma1", sigma=sigma_init_prior_sigma, dims="outcome")
        a_change = pm.Normal(
            "a_change", mu=0.0, sigma=intercept_prior_sigma, dims="outcome"
        )
        b_self = pm.Normal("b_self", mu=0.0, sigma=self_prior_sigma, dims="outcome")
        d_age = pm.Normal("d_age", mu=0.0, sigma=covariate_prior_sigma, dims="outcome")
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
                m = m + d_age[jidx[s]] * age[:, t - 1]
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


def build_growth_model(
    panel: WavePanel,
    *,
    baseline_covariate: str = "blocks",
    use_shared_factor: bool = False,
    intercept_prior_sigma: float = 1.5,
    slope_prior_sigma: float = 0.5,
    assoc_prior_sigma: float = 0.5,
    re_intercept_prior_sigma: float = 1.0,
    re_slope_prior_sigma: float = 0.5,
    loading_prior_sigma: float = 0.5,
    kappa_prior_sigma: float = 50.0,
) -> BuiltModel:
    """Joint multivariate latent growth-curve model (LRP69/70) on the logit scale.

    Characterises each measure's within-child trajectory across the waves and asks
    whether a **baseline** covariate (``blocks``, the t1-only WPPSI Block Design
    non-verbal score) predicts trajectory *shape*. For measure ``k``, child ``i``,
    wave ``t`` (with ``a`` = standardised age)::

        theta[i,t,k]   = intercept[i,k] + slope[i,k] * a[i,t]
        intercept[i,k] = alpha_k + delta_k * z(blocks_i) + sigma0_k * z0[i,k]
        slope[i,k]     = beta_k  + gamma_k * z(blocks_i) + [loading_k * G_i]
                                 + sigma1_k * z1[i,k]
        y[i,t,k] ~ BetaBinomial(N_k, mu = sigmoid(theta[i,t,k]), kappa_k)

    Growth is **linear in standardised age** — the identifiable choice at four
    waves. ``gamma_k`` (baseline non-verbal ability -> growth *rate*) is the
    headline Q5 estimand; ``delta_k`` is the effect on baseline *level*. Both are
    **adjusted / GA-confounded associations, never causal** (block design is an
    off-DAG ability proxy; see ``notes/202606231600-dag-revision-consolidated.md``).

    The child-level random intercept and slope are **independent per measure** —
    the within-measure intercept-slope correlation is deliberately omitted at
    n~54, mirroring the joint ITT model's disabled LKJ residual correlation (found
    prior-dominated at this sample size, ``notes/202604181600-lrp52-58-findings.md``).
    Everything is non-centred for sampling.

    ``use_shared_factor`` adds a rank-1 shared child-level growth-tempo factor
    ``G_i ~ Normal(0, 1)`` loading (positively, for identification) on every
    measure's slope — the genuinely *joint* layer (LRP70): does a common
    developmental tempo couple the measures, and (read out post-hoc) does baseline
    non-verbal ability predict it? ``LOO(LRP69 vs LRP70)`` shows whether the factor
    earns its keep. The core LRP69 keeps ``use_shared_factor=False``.

    Observed counts enter via a **masked** Beta-Binomial (the LRP55 flattened-mask
    idiom): only the unmasked cells in ``panel.obs_mask`` are observed, so a child
    missing one score still contributes its other waves. The intervention-dose
    covariate is **omitted** (the locked DAG's ``IS`` collider, as in
    :func:`build_lcsm_model`).
    """
    OUT = tuple(panel.outcomes)
    K = len(OUT)
    N = panel.n_children
    T = panel.n_waves
    if T < 2:
        raise ValueError("growth model needs at least two waves")
    if baseline_covariate not in panel.baseline:
        raise KeyError(
            f"baseline_covariate {baseline_covariate!r} not loaded; pass "
            f"baseline_covariates=({baseline_covariate!r},) to load_wave_panel."
        )

    # Observed counts / mask / denominators stacked as (N, T, K) in OUT order.
    counts_int = np.stack(
        [np.nan_to_num(panel.counts[s], nan=0.0).astype(np.int64) for s in OUT],
        axis=2,
    )
    mask = np.stack([panel.obs_mask[s] for s in OUT], axis=2)  # (N, T, K) bool
    n_trials_vec = np.array([panel.n_trials[s] for s in OUT], dtype=int)  # (K,)
    zb = np.asarray(panel.baseline[baseline_covariate], dtype=float)  # (N,) standardised

    # Intercept anchor: grand-mean observed logit per measure (the intercept is the
    # logit level at mean age, age_std = 0). Guard the all-NaN case loudly.
    missing = [s for s in OUT if not np.isfinite(panel.logit[s]).any()]
    if missing:
        raise ValueError(
            "growth intercept anchor is undefined (no observed value) for: "
            f"{', '.join(missing)}."
        )
    intercept_anchor = np.array(
        [np.nanmean(panel.logit[s]) for s in OUT], dtype=float
    )

    coords = {"child": np.arange(N), "wave": panel.waves, "outcome": list(OUT)}

    from dse_research_utils.math.constants import EPSILON  # local import

    with pm.Model(coords=coords) as model:
        age = pm.Data("age_std", panel.age_std, dims=("child", "wave"))
        blocks = pm.Data("blocks_std", zb, dims="child")

        # Population growth parameters (per measure).
        alpha = pm.Normal(
            "alpha", mu=intercept_anchor, sigma=intercept_prior_sigma, dims="outcome"
        )
        beta = pm.Normal("beta", mu=0.0, sigma=slope_prior_sigma, dims="outcome")
        # Baseline non-verbal ability -> trajectory shape (the Q5 estimands):
        # delta on the baseline level, gamma on the growth rate (headline).
        delta = pm.Normal("delta", mu=0.0, sigma=assoc_prior_sigma, dims="outcome")
        gamma = pm.Normal("gamma", mu=0.0, sigma=assoc_prior_sigma, dims="outcome")
        # Child-level random intercept + slope (independent per measure).
        sigma_intercept = pm.HalfNormal(
            "sigma_intercept", sigma=re_intercept_prior_sigma, dims="outcome"
        )
        sigma_slope = pm.HalfNormal(
            "sigma_slope", sigma=re_slope_prior_sigma, dims="outcome"
        )
        z_intercept = pm.Normal("z_intercept", 0.0, 1.0, dims=("child", "outcome"))
        z_slope = pm.Normal("z_slope", 0.0, 1.0, dims=("child", "outcome"))
        kappa = pm.HalfNormal("kappa", sigma=kappa_prior_sigma, dims="outcome")

        # child x outcome intercepts and slopes (non-centred).
        intercept = pm.Deterministic(
            "intercept",
            alpha[None, :]
            + delta[None, :] * blocks[:, None]
            + sigma_intercept[None, :] * z_intercept,
            dims=("child", "outcome"),
        )
        slope_mean = beta[None, :] + gamma[None, :] * blocks[:, None]
        if use_shared_factor:
            # Rank-1 shared child-level growth-tempo factor: positive loadings so
            # G is a common "faster growth on every measure" tempo (identification).
            G = pm.Normal("G_tempo", 0.0, 1.0, dims="child")
            loading = pm.HalfNormal(
                "loading", sigma=loading_prior_sigma, dims="outcome"
            )
            slope_mean = slope_mean + loading[None, :] * G[:, None]
        slope = pm.Deterministic(
            "slope",
            slope_mean + sigma_slope[None, :] * z_slope,
            dims=("child", "outcome"),
        )

        # Latent logit trajectory (linear in standardised age).
        theta = pm.Deterministic(
            "theta",
            intercept[:, None, :] + slope[:, None, :] * age[:, :, None],
            dims=("child", "wave", "outcome"),
        )

        # Masked Beta-Binomial observation (LRP55 flattened-mask idiom).
        mu = pm.math.sigmoid(theta)
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


# ---------------------------------------------------------------------------
# Historical group-by-wave growth (RLMHG, #165 - first non-RLI dataset)
# ---------------------------------------------------------------------------


def build_historical_growth_model(
    panel: LongitudinalPanel,
    *,
    measure: str = "basread",
    eta_prior_sigma: float = 1.5,
    sigma_subject_prior_sigma: float = 1.0,
    kappa_prior_sigma: float = 50.0,
) -> BuiltModel:
    """Descriptive group-by-wave growth model for a historical cohort.

    Beta-Binomial on a bounded count with a population ``eta[group, wave]`` grid
    and a non-centred, group-centred per-subject random intercept::

        score_it ~ BetaBinomial(n, p_it, kappa)
        logit(p_it) = eta[group_i, wave_t] + subject_offset_i

    This is **descriptive natural-history** evidence, not an intervention-effect
    model: ``group`` carries no treatment semantics, there is no baseline-as-
    precision term and no adjustment set. Deterministics expose the group-by-wave
    expected item score, within-group interval growth, and pairwise total-growth
    contrasts. Ported from the standalone ``rlmhg01`` script (#163) onto the
    shared pipeline (#165).
    """
    df = panel.long
    dataset = panel.dataset
    subj, wave_c, grp = dataset.subject_col, dataset.wave_col, dataset.group_col
    if measure not in panel.n_trials:
        raise KeyError(f"measure {measure!r} not in panel (have {panel.measures}).")
    n_trials = int(panel.n_trials[measure])

    group_codes = panel.group_codes
    group_labels = panel.group_labels
    wave_codes = list(panel.waves)
    subject_ids = panel.subject_ids
    group_index = {code: i for i, code in enumerate(group_codes)}
    wave_index = {w: i for i, w in enumerate(wave_codes)}
    subject_index = {s: i for i, s in enumerate(subject_ids)}

    group_idx = df[grp].map(group_index).to_numpy(dtype=int)
    wave_idx = df[wave_c].map(wave_index).to_numpy(dtype=int)
    subject_idx = df[subj].map(subject_index).to_numpy(dtype=int)
    observed = df[measure].to_numpy(dtype=int)
    subject_group = (
        df.drop_duplicates(subj)
        .set_index(subj)
        .loc[subject_ids, grp]
        .map(group_index)
        .to_numpy(dtype=int)
    )

    coords = {
        "group": group_labels,
        "wave": wave_codes,
        "subject": [str(s) for s in subject_ids],
        "obs": np.arange(len(df)),
    }

    with pm.Model(coords=coords) as model:
        eta_group_wave = pm.Normal(
            "eta_group_wave", mu=0.0, sigma=eta_prior_sigma, dims=("group", "wave")
        )
        sigma_subject = pm.HalfNormal("sigma_subject", sigma=sigma_subject_prior_sigma)
        z_subject = pm.Normal("z_subject", mu=0.0, sigma=1.0, dims="subject")
        # Group-centre the subject offsets for identifiability against
        # ``eta_group_wave`` (the group-by-wave level absorbs the group mean).
        z_group_mean = pm.math.stack(
            [z_subject[subject_group == g].mean() for g in range(len(group_codes))]
        )
        subject_offset = pm.Deterministic(
            "subject_offset",
            (z_subject - z_group_mean[subject_group]) * sigma_subject,
            dims="subject",
        )
        kappa = pm.HalfNormal("kappa", sigma=kappa_prior_sigma)

        eta_obs = eta_group_wave[group_idx, wave_idx] + subject_offset[subject_idx]
        p_obs = pm.math.sigmoid(eta_obs)
        pm.BetaBinomial(
            "score",
            n=n_trials,
            alpha=p_obs * kappa,
            beta=(1.0 - p_obs) * kappa,
            observed=observed,
            dims="obs",
        )
        pm.Deterministic("fitted_mean_items_obs", n_trials * p_obs, dims="obs")

        mean_items = pm.Deterministic(
            "mean_items",
            n_trials * pm.math.sigmoid(eta_group_wave),
            dims=("group", "wave"),
        )
        # Within-group interval growth (items), first->second and second->third
        # wave, plus first->last, when at least three waves are modelled.
        if len(wave_codes) >= 2:
            pm.Deterministic(
                "growth_first_next_items",
                mean_items[:, 1] - mean_items[:, 0],
                dims="group",
            )
        if len(wave_codes) >= 3:
            pm.Deterministic(
                "growth_next_last_items",
                mean_items[:, 2] - mean_items[:, 1],
                dims="group",
            )
        if len(wave_codes) >= 2:
            pm.Deterministic(
                "growth_first_last_items",
                mean_items[:, -1] - mean_items[:, 0],
                dims="group",
            )

    return BuiltModel(
        model=model, variables=_variables_dict(model), prepared=panel
    )
