# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Joint (multi-outcome) ITT model construction.

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
from language_reading_predictors.statistical_models.hsgp import (
    build_hsgp_1d,
)
from language_reading_predictors.statistical_models.fitted_payloads import (
    JointPayload,
)
from language_reading_predictors.statistical_models.measures import (
    ITT_OUTCOMES,
)
from language_reading_predictors.statistical_models.preprocessing import (
    PreparedData,
)
from language_reading_predictors.statistical_models.factories.base import (
    BuiltModel,
)

def build_joint_model(
    prepared: PreparedData,
    *,
    outcomes: Iterable[str] = ITT_OUTCOMES,
    use_age_gp: bool = False,
    partial_pool_age_gp: bool = True,
    use_residual_correlation: bool = False,
    use_cross_baselines: bool = True,
    use_age_linear: bool = False,
) -> BuiltModel[JointPayload]:
    """
    Build the joint available-case modified ITT model (LRPITT12; LRPITT15/15b/16).

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
    so the joint mirrors the single-outcome suite — each assigned-arm coefficient
    identifies an available-case modified ITT estimate without an adjustment set,
    with own baseline + linear age as precision terms.

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

    With ``use_residual_correlation=False`` the likelihood and priors factorise
    by outcome: this is a product of outcome-specific marginal models fitted in
    one PyMC graph, not a dependence-aware joint posterior. Per-outcome effects
    remain valid, but paired cross-outcome contrasts require an explicit
    dependence sensitivity. The registered parent specifications (LRPITT12 and
    the LRPITT15/15b/16 contrast parents) use this stable factorised form; their
    #551 dependence companions (lrp-rli-itt-215/315/216) switch the residual
    block on explicitly, so nothing claims within-child outcome covariance it
    does not model.

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
    # Explicit flattened cells make the observed likelihood robust to
    # outcome-specific post-score missingness. Register the cell coordinate below
    # so predictive and log-likelihood arrays never fall back to an anonymous
    # ``y_post_dim_0`` axis.
    idx_row, idx_col = np.nonzero(mask)

    coords = {
        "obs_id": np.arange(N_obs),
        "outcome": list(outcomes),
        "baseline": list(outcomes),
        "cell": np.arange(idx_row.size),
        # Second outcome axis for outcome×outcome quantities (residual
        # correlation). Cannot reuse "outcome" because PyMC requires
        # distinct dim names per axis.
        "outcome2": list(outcomes),
    }
    if use_residual_correlation:
        # The strictly-lower-triangle pairs of the residual correlation matrix
        # (#551): the free correlations as clean scalars, so the summary, the
        # prior-vs-posterior overlay and power scaling can show them without the
        # constant unit diagonal of ``u_corr`` breaking the density plots.
        pair_i, pair_j = np.tril_indices(K, k=-1)
        coords["outcome_pair"] = [f"{outcomes[i]}|{outcomes[j]}" for i, j in zip(pair_i, pair_j, strict=True)]

    G_f = prepared.G.astype(float)

    with pm.Model(coords=coords) as model:
        A_std_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
        G_d = pm.Data("G", G_f, dims="obs_id")

        # Pre-score matrix (N_obs, K) - same order as ``outcomes``.
        pre_logit = np.stack([prepared.pre_logit[s] for s in outcomes], axis=1)
        pre_logit_data = pm.Data(
            "pre_logit", pre_logit, dims=("obs_id", "baseline")
        )

        # Per-outcome scalar parameters — shared constructors (priors.py) so
        # the joint model cannot drift from the ITT / mechanism factories (issue #79).
        # alpha and tau are kept **common** (untiered) across outcomes here — the
        # joint is the deliberately uniform-prior cross-check against the tiered
        # single-outcome ITT fits (the note keeps the common tau; the intercept
        # follows the same rationale). Per-outcome alpha-SD tiering (Finding 1) in
        # the joint is a documented follow-up.
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
            # Read age from the ``A_std`` Data node (not the raw array) so a future
            # ``pm.set_data({"A_std": ...})`` updates the linear age term, matching how
            # ``G_d`` is wired. (The age-GP path below builds its HSGP basis from the
            # array directly; set_data on GP inputs is a separate concern, out of scope.)
            eta_core = eta_core + gamma_A[None, :] * A_std_d[:, None]

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
            # The free correlations as scalars (one per outcome pair, #551).
            pm.Deterministic(
                "u_corr_pair", corr[pair_i, pair_j], dims="outcome_pair"
            )
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
        flat_alpha = alpha_bb[idx_row, idx_col]
        flat_beta = beta_bb[idx_row, idx_col]
        flat_n = n_trials_vec[idx_col]
        flat_obs = post_counts_int[idx_row, idx_col]

        # Record the flattened cell mapping as constant data. Diagnostics use both
        # arrays to select one outcome for predictive checks (so incompatible
        # denominators are never pooled) and to aggregate pointwise log likelihood
        # by child for leave-one-child-out PSIS-LOO.
        pm.Data("y_post_cell_row", idx_row.astype("int64"), dims="cell")
        pm.Data("y_post_cell_outcome", idx_col.astype("int64"), dims="cell")

        pm.BetaBinomial(
            "y_post",
            n=flat_n,
            alpha=flat_alpha,
            beta=flat_beta,
            observed=flat_obs,
            dims="cell",
        )

    dependence = (
        "residual_correlated"
        if use_residual_correlation
        else "factorised_outcome_marginals"
    )
    return BuiltModel(
        model=model,
        prepared=prepared,
        payload=JointPayload(
            joint_dependence=dependence,
            loo_unit="child",
            outcomes=outcomes,
        ),
    )
