# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Longitudinal correlated-factor model construction.

Carved out of the 8,506-line ``factories.py`` by #637 stage 3, which is why
every name here is still re-exported from ``factories``. Every family module
depends only on :mod:`factories.base`; nothing crosses between families.
"""

from __future__ import annotations


from typing import TYPE_CHECKING

import numpy as np
import pymc as pm
import pytensor.tensor as pt

if TYPE_CHECKING:
    pass


from language_reading_predictors.statistical_models.fitted_payloads import (
    LongCorrFactorPayload,
)
from language_reading_predictors.statistical_models.preprocessing import (
    WavePanel,
)
from language_reading_predictors.statistical_models.factories.base import (
    BuiltModel,
    _LCF_DOMAINS,
)
from language_reading_predictors.statistical_models import priors as _priors

def build_longitudinal_corr_factor_model(
    panel: WavePanel,
    *,
    domains: dict[str, tuple[str, ...]] | None = None,
    # Loading / residual geometry (#383 follow-up). The default is the
    # communality-scale parameterisation the cross-sectional builders adopted in
    # #383, adapted to this model's POOLED (all-wave) standardisation: the free
    # parameter is the within-wave communality ``c ~ Beta(comm_alpha, comm_beta)``
    # and the loading / residual pair is derived so the model-implied pooled
    # indicator variance is exactly 1 (see the docstring — the cross-sectional
    # ``lambda**2 + sigma**2 = 1`` budget is the special case of no wave-mean
    # movement, and enforcing it verbatim here would overstate the within-wave
    # variance by the 5-18% share that wave-to-wave growth carries in these
    # indicators). ``loading_prior="free"`` retains the legacy unconstrained
    # TruncatedNormal / HalfNormal pair (knobs ``loading_sigma`` /
    # ``residual_sigma``) so a sensitivity contrast can vary only the geometry.
    loading_prior: str = "communality",
    comm_alpha: float = 2.0,
    comm_beta: float = 2.0,
    loading_sigma: float = 1.0,
    residual_sigma: float = 1.0,
    lkj_eta: float = 2.0,
    factor_mean_sigma: float = 1.0,
    trait_share_a: float = 1.5,
    trait_share_b: float = 1.5,
) -> BuiltModel[LongCorrFactorPayload, WavePanel]:
    """Longitudinal correlated-domain-factor measurement model (LRP-RLI-LCF-001, #313).

    The four-wave extension of the cross-sectional ``corr_factor`` CFA (``mm-001``):
    correlated **vocabulary / code / grammar** domain factors measured at every
    timepoint, delivering the **per-wave latent skill correlation matrices** and the
    conditional latent slopes derived from them. Its correlation matrix is a
    symmetric, measurement-error-aware companion to the concurrent regression
    family (``ca-001``, #312), while the derived conditional slopes are directional:
    every quantity is a **descriptive association**, never causal, and the two
    families estimate different quantities with no required magnitude ordering.

    **Structure (wave-invariant longitudinal CFA, fully marginalised).** For
    indicator ``j`` of domain ``d`` at wave ``t`` the standardised logit indicator is
    ``z[i,j,t] = lambda[j] * f[i,d,t] + eps[i,j,t]`` with loadings ``lambda[j]`` and
    residual SDs ``sigma[j]`` held **invariant across waves** (the factors mean the
    same thing at every t), positive loadings, and per-wave factor means carried by a
    zero-sum-over-waves ``factor_mean[d,t]`` (indicators are pooled-standardised, so
    the grand mean is removed and only the wave deviations remain). Each factor is
    unit-variance at every wave; the **within-wave factor correlation**
    ``factor_corr[t]`` is the headline. Across-wave dependence uses a **trait/state
    decomposition** ``f = sqrt(pi_d) * trait + sqrt(1 - pi_d) * state`` — a stable
    per-child trait (cross-factor correlated, shared across waves) plus a wave-specific
    state (cross-factor correlated, independent across waves) — which is PSD by
    construction, keeps unit factor variance, and induces across-wave autocorrelation
    equal to the trait share ``pi_d``. LKJ priors are placed on the shared trait
    correlation and each wave's state correlation; their trait-share-weighted sum
    induces the reported within-wave matrix. The matrices can vary through their
    state components but share the trait component, so they are neither independent
    nor themselves LKJ-distributed. This gives compound symmetry across waves;
    genuine AR(1) decay is the first relaxation if the equal-lag assumption misfits.

    **Small-n geometry.** The measurement model is Gaussian in the factors, so the
    per-child factor scores are **marginalised out** (as in ``mm-001``): each child's
    observed indicator cells are an ``MvNormal`` whose covariance is the trait/state
    factor covariance folded through the loadings, ``Sigma_z = Lambda Sigma_f Lambda'
    + diag(sigma^2)``, sliced to that child's observed cells. There is **no** per-child
    latent RV and hence no funnel; the geometry is set by the marginalised likelihood.
    Missing cells are handled by grouping children into observed-cell **patterns** and
    fitting one ``MvNormal`` per pattern (masked, not dropped — a child missing one
    wave still contributes its other cells, which matters at n ~ 54).

    **Loading / residual priors (#383 follow-up).** The default
    ``loading_prior="communality"`` makes each indicator's **within-wave
    communality** the free parameter — ``c ~ Beta(comm_alpha, comm_beta)`` on
    (0, 1) — and derives the loading / residual pair from the budget this model's
    data pipeline actually implies. The indicators are **pooled-standardised across
    waves** (wave-to-wave level change is deliberately preserved, carried by
    ``factor_mean``), so each indicator's unit sample variance decomposes into the
    within-wave variance ``lambda**2 + sigma**2`` plus the between-wave mean
    variance ``lambda**2 * V``, where ``V`` is the observed-cell-weighted variance
    of the model's own wave means for that indicator's domain. Setting ``lambda =
    sqrt(c / (1 + c V))`` and ``sigma = sqrt((1 - c) / (1 + c V))`` makes the
    model-implied pooled variance **exactly 1** at every parameter value (prior and
    posterior alike), with ``communality = lambda**2 / (lambda**2 + sigma**2) = c``
    exact and zero prior mass on Heywood configurations. The cross-sectional
    builders' ``lambda**2 + sigma**2 = 1`` budget (#383) is the ``V = 0`` special
    case; enforcing it verbatim here would overstate the within-wave variance by
    the 5-18% share that between-wave growth carries in these indicators (the
    fitted free-pair posterior has ``lambda**2 + sigma**2 ~ 0.75-0.88``). The
    legacy pair (``loading_prior="free"``: ``TruncatedNormal(0, loading_sigma,
    lower=0)`` loadings, ``HalfNormal(residual_sigma)`` residuals) implies
    ``communality ~ Beta(1/2, 1/2)`` — the arcsine prior piling mass on both
    singular corners — and ~32% of loading mass above 1 at the default scales
    (prior-critical review 2026-07-21, #383); it is retained so a sensitivity
    contrast can vary only the geometry. Both modes expose ``within_share``
    (``lambda**2 + sigma**2``, the within-wave share of the pooled unit variance;
    ``1 / (1 + c V)`` under the default) as a Deterministic.

    ``domains`` maps each factor to its indicator symbols (default vocabulary
    {R,E,TR,TE} / code {L,B} / grammar {F,T}); every domain needs >= 2 indicators to
    be identified. This is a **measurement / triangulation** model with no structural
    outcome leg: the latent slopes are a post-processing of ``factor_corr``.
    """
    if domains is None:
        domains = {k: tuple(v) for k, v in _LCF_DOMAINS.items()}
    domain_names = list(domains)
    D = len(domain_names)
    if D < 2:
        raise ValueError(
            f"A correlated-domain-factor model needs >= 2 domains (got {D}: "
            f"{domain_names}); with a single factor there are no cross-domain "
            "correlations to estimate and the per-wave off-diagonal (factor_corr_pairs) "
            "and cross-check outputs would be empty."
        )
    for d in domain_names:
        if len(tuple(domains[d])) < 2:
            raise ValueError(
                f"Domain {d!r} has < 2 indicators ({tuple(domains[d])}); a correlated "
                "factor needs at least two indicators to be identified."
            )
    if loading_prior not in {"communality", "free"}:
        raise ValueError(
            f"loading_prior must be 'communality' or 'free'; got {loading_prior!r}"
        )
    if not (
        np.isfinite(comm_alpha)
        and np.isfinite(comm_beta)
        and comm_alpha > 0.0
        and comm_beta > 0.0
    ):
        raise ValueError(
            "comm_alpha and comm_beta must be finite and positive (Beta shape "
            f"parameters); got {comm_alpha}, {comm_beta}."
        )
    T = int(panel.n_waves)
    N = int(panel.n_children)
    if T < 2:
        raise ValueError("longitudinal correlated-factor model needs >= 2 waves")

    waves = list(panel.waves)

    # Indicator list + per-indicator domain index; pooled (all-wave) standardisation
    # of the Haldane-corrected logit, preserving wave-to-wave level change in the data
    # (carried by the factor means). Missing cells stay NaN and are masked out below.
    ind_names: list[str] = []
    domain_of: list[int] = []
    z_cols: list[np.ndarray] = []
    standardisers: dict[str, tuple[float, float]] = {}
    for di, d in enumerate(domain_names):
        for s in domains[d]:
            if s not in panel.logit:
                raise KeyError(f"Indicator {s!r} (domain {d!r}) missing from panel.logit")
            lg = np.asarray(panel.logit[s], dtype=float)  # (N, T), NaN where missing
            if not np.isfinite(lg).any():
                raise ValueError(f"Indicator {s!r} has no observed cell")
            mean = float(np.nanmean(lg))
            sd = float(np.nanstd(lg, ddof=1))
            if not np.isfinite(sd) or sd == 0.0:
                raise ValueError(f"Indicator {s!r} has zero/undefined pooled SD")
            z_cols.append((lg - mean) / sd)  # (N, T), NaN preserved
            ind_names.append(s)
            domain_of.append(di)
            standardisers[s] = (mean, sd)
    J = len(ind_names)
    domain_of_idx = np.asarray(domain_of, dtype=np.int64)

    # Z (N, T, J) and its mask, flattened per child in (t, j) order (t slow).
    Z = np.stack(z_cols, axis=2)  # (N, T, J)
    obs3 = np.isfinite(Z)  # (N, T, J)
    Z_flat = Z.reshape(N, T * J)
    mask_flat = obs3.reshape(N, T * J)
    cell_names = [f"{ind_names[j]}_t{waves[t]}" for t in range(T) for j in range(J)]

    # Per-indicator observed-cell wave weights (J, T), fixed data for the pooled
    # variance budget under the communality parameterisation: indicator j's pooled
    # sample variance averages its waves with these weights, and missingness
    # differs by indicator, so the wave-mean variance entering the budget must use
    # each indicator's own weights (not the domain's unweighted means).
    _counts_tj = obs3.sum(axis=0).astype(float)  # (T, J)
    wave_weights = (_counts_tj / _counts_tj.sum(axis=0, keepdims=True)).T  # (J, T)

    # Group children by observed-cell pattern (near-rectangular: one big complete
    # group + a few singletons). Sort deterministically: largest group first, ties by
    # first child index.
    pattern_children: dict[tuple[bool, ...], list[int]] = {}
    for i in range(N):
        key = tuple(bool(x) for x in mask_flat[i])
        if not any(key):
            # A child with no observed cell at all contributes nothing; drop it.
            continue
        pattern_children.setdefault(key, []).append(i)
    sorted_patterns = sorted(
        pattern_children.items(), key=lambda kv: (-len(kv[1]), min(kv[1]))
    )

    onehot = np.zeros((J, D), dtype=float)
    onehot[np.arange(J), domain_of_idx] = 1.0

    iu, ju = np.triu_indices(D, k=1)
    pair_names = [
        f"{domain_names[i]}~{domain_names[j]}" for i, j in zip(iu, ju, strict=True)
    ]

    coords = {
        "indicator": ind_names,
        "domain": domain_names,
        "domain_b": domain_names,
        "wave": waves,
        "cell": cell_names,
        "cell_b": cell_names,
    }
    if pair_names:
        coords["factor_pair"] = pair_names

    z_nodes: list[str] = []
    child_of_node: dict[str, list[int]] = {}
    cell_indices_of_node: dict[str, list[int]] = {}
    observed_z_of_node: dict[str, np.ndarray] = {}

    with pm.Model(coords=coords) as model:
        # Per-wave factor means, declared FIRST: under the pooled-budget
        # communality parameterisation below the derived loading / residual scales
        # depend on the wave-mean spread, so the means must exist before the
        # measurement parameters. (Declaration order has no statistical effect;
        # the legacy mode shares the ordering for a stable node layout.)
        factor_mean = _priors.declare(
                          pm.ZeroSumNormal(
                                      "factor_mean", sigma=factor_mean_sigma, dims=("domain", "wave")
                                  ),
                          role="nuisance",
                          rationale=(
                              "Exact-zero-sum domain-by-wave mean deviations (ZeroSumNormal(1, "
                              "<constant>)); represents wave shifts after pooled indicator "
                              "standardisation."
                          ),
                      )

        # --- Measurement parameters (wave-invariant loadings + residuals) ---
        # Loading / residual parameterisation (#383 follow-up; see the docstring).
        # Default: the within-wave communality is the free parameter — c ~
        # Beta(comm_alpha, comm_beta) — and lambda / sigma are derived under the
        # POOLED unit-variance budget: with V the observed-cell-weighted variance
        # of the (fitted) wave means for the indicator's domain, lambda =
        # sqrt(c / (1 + c V)) and sigma = sqrt((1 - c) / (1 + c V)) make the
        # model-implied pooled indicator variance exactly 1 while keeping
        # communality = c exact. The node names (lambda_load / sigma_indicator /
        # communality) are unchanged in both modes; only which is free differs.
        if loading_prior == "communality":
            comm = _priors.declare(
                       pm.Beta(
                                       "communality", alpha=comm_alpha, beta=comm_beta, dims="indicator"
                                   ),
                       role="association",
                       rationale=(
                           "Indicator communality (Beta(2, 2)); the share of a standardised "
                           "test's variance explained by its domain factor, with the loading / "
                           "residual pair derived from c under the family's unit-variance "
                           "budget: lambda**2 + sigma**2 = 1 exactly for cross-sectionally "
                           "standardised indicators, and lambda**2 + sigma**2 = 1 / (1 + c V) "
                           "in the longitudinal CFA (V the spread of the fitted wave means, so "
                           "the POOLED indicator variance is exactly 1). Either way the "
                           "loading-residual ridge is removed and Heywood configurations have "
                           "zero prior mass."
                       ),
                   )
            _W = pt.as_tensor_variable(wave_weights)  # (J, T)
            _m_ind = factor_mean[domain_of_idx, :]  # (J, T) domain means per indicator
            _mbar = pt.sum(_W * _m_ind, axis=1, keepdims=True)  # (J, 1)
            _V = pt.sum(_W * pt.sqr(_m_ind - _mbar), axis=1)  # (J,)
            _denom = 1.0 + comm * _V
            lam = pm.Deterministic("lambda_load", pt.sqrt(comm / _denom), dims="indicator")
            sigma_ind = pm.Deterministic(
                "sigma_indicator", pt.sqrt((1.0 - comm) / _denom), dims="indicator"
            )
            pm.Deterministic("within_share", 1.0 / _denom, dims="indicator")
        else:
            # Legacy free pair, retained so a sensitivity contrast can vary only
            # the geometry. Defaults reproduce the original HalfNormal(1) pair
            # (TruncatedNormal(0, 1, lower=0) IS HalfNormal(1)).
            lam = _priors.declare(
                pm.TruncatedNormal(
                    "lambda_load", mu=0.0, sigma=loading_sigma, lower=0.0, dims="indicator"
                ),
                role="association",
                rationale=(
                    "Free factor loading of a standardised indicator on its domain factor "
                    "(TruncatedNormal(mu, sigma, lower=0)); the legacy free "
                    "loading/residual pair retained for the prior-geometry "
                    "sensitivity companion, where communality is derived rather "
                    "than sampled."
                ),
            )
            sigma_ind = _priors.declare(
                pm.HalfNormal(
                    "sigma_indicator", sigma=residual_sigma, dims="indicator"
                ),
                role="nuisance",
                rationale=(
                    "Indicator residual SD of the legacy free pair (HalfNormal); unbounded "
                    "support, so it makes sigma > 1 unlikely rather than capping it."
                ),
            )
            pm.Deterministic(
                "communality", lam**2 / (lam**2 + sigma_ind**2), dims="indicator"
            )
            pm.Deterministic(
                "within_share", lam**2 + sigma_ind**2, dims="indicator"
            )

        # --- Trait / state factor structure ---
        # Trait share per factor (across-wave autocorrelation) + trait/state
        # correlation matrices. PyMC 6.1's LKJCorr value is the lower-triangular
        # Cholesky *factor* of the correlation (unit-norm rows), so the correlation is
        # ``L @ L.T``. This carries only the correlation's degrees of freedom — no
        # nuisance standard-deviation scales (LKJCholeskyCov would add an unidentified
        # ``sd_dist`` per matrix, since only the correlation enters the model, and
        # those pollute the convergence gate). Five matrices: one trait + one state
        # per wave.
        pi = _priors.declare(
                 pm.Beta("trait_share", alpha=trait_share_a, beta=trait_share_b, dims="domain"),
                 role="nuisance",
                 rationale=(
                     "Domain-specific stable-trait variance share (Beta(1.5, 1.5)); "
                     "governs same-domain persistence across waves."
                 ),
             )
        L_trait = _priors.declare(
                      pm.LKJCorr("trait_corr_chol", n=D, eta=lkj_eta),
                      role="association",
                      rationale=(
                          "LKJ prior on the shared trait-component correlation "
                          "(LKJCorrRV(<constant>, 2)); trait-share weighting carries it into "
                          "every within-wave matrix."
                      ),
                  )
        corr_trait = L_trait @ L_trait.T
        pm.Deterministic("trait_corr", corr_trait, dims=("domain", "domain_b"))
        corr_state = []
        for t in range(T):
            L_s = _priors.declare(
                      pm.LKJCorr(f"state_corr_chol_w{waves[t]}", n=D, eta=lkj_eta),
                      role="association",
                      rationale=(
                          "LKJ prior on one wave's state-component correlation "
                          "(LKJCorrRV(<constant>, 2)); together with the shared trait "
                          "component it induces that wave's reported factor correlation."
                      ),
                  )
            corr_state.append(L_s @ L_s.T)

        sqrt_pi = pt.sqrt(pi)
        sqrt_1mpi = pt.sqrt(1.0 - pi)
        # Trait block B = diag(sqrt_pi) Corr_trait diag(sqrt_pi); it fills every
        # (t, t') block of the factor covariance (shared across all waves).
        B = corr_trait * (sqrt_pi[:, None] * sqrt_pi[None, :])
        trait_full = pt.linalg.kron(pt.ones((T, T)), B)  # (T*D, T*D)

        state_full = pt.zeros((T * D, T * D))
        within_blocks = []
        for t in range(T):
            S_t = corr_state[t] * (sqrt_1mpi[:, None] * sqrt_1mpi[None, :])
            E = np.zeros((T, T), dtype=float)
            E[t, t] = 1.0
            state_full = state_full + pt.linalg.kron(pt.as_tensor_variable(E), S_t)
            # Within-wave factor correlation at wave t (unit diagonal: pi + (1-pi) = 1).
            within_blocks.append(B + S_t)
        Sigma_f = trait_full + state_full  # (T*D, T*D)

        factor_corr = pm.Deterministic(
            "factor_corr", pt.stack(within_blocks, axis=0), dims=("wave", "domain", "domain_b")
        )
        if pair_names:
            # Gate exactly the released off-diagonals (the full matrix's constant unit
            # diagonal has undefined R-hat and would silently pass); one vector of the
            # unique pairs per wave.
            pairs = pt.stack(
                [factor_corr[:, i, j] for i, j in zip(iu, ju, strict=True)], axis=1
            )  # (wave, factor_pair)
            pm.Deterministic("factor_corr_pairs", pairs, dims=("wave", "factor_pair"))

        # --- Marginal indicator covariance + mean over the (t, j) stack ---
        Lambda_wave = lam[:, None] * pt.as_tensor_variable(onehot)  # (J, D)
        Lambda_full = pt.linalg.kron(pt.eye(T), Lambda_wave)  # (T*J, T*D)
        # A small diagonal nugget guarantees a numerically PD covariance for the
        # Cholesky even when a factor's trait share -> 1 (its waves become near-
        # identical, so Sigma_f is rank-deficient) coincides with a tiny residual SD
        # draw; z is standardised (~unit scale), so 1e-6 is negligible.
        sig2_full = pt.tile(sigma_ind**2, T) + 1e-6  # (T*J,) in (t, j) order

        # mean of z[t, j] = lambda[j] * factor_mean[domain(j), t]; flatten to (t, j).
        mean_full = (lam[:, None] * factor_mean[domain_of_idx, :]).T.reshape((T * J,))
        # Full assembled quantities exposed for inspection (NOT sliced for the
        # likelihood — the per-pattern sub-covariances are built from row-sliced
        # loadings below, which keeps the graph free of the double-advanced-index
        # slice-of-write that trips a PyTensor rewrite at the incomplete patterns).
        pm.Deterministic(
            "Sigma_z",
            Lambda_full @ Sigma_f @ Lambda_full.T + pt.diag(sig2_full),
            dims=("cell", "cell_b"),
        )
        pm.Deterministic("mean_z", mean_full, dims="cell")

        # --- Per-pattern marginalised MvNormal likelihood (masked, not dropped) ---
        for tag, (key, children) in enumerate(sorted_patterns):
            obs_idx = np.where(np.asarray(key, dtype=bool))[0]
            data = Z_flat[np.ix_(children, obs_idx)]  # (n_p, k_p), no NaN
            row_coord = f"row{tag}"
            cell_coord = f"cell{tag}"
            model.add_coords(
                {
                    row_coord: np.asarray(children, dtype=int),
                    cell_coord: [cell_names[c] for c in obs_idx],
                }
            )
            # Build this pattern's sub-covariance from the observed rows of the loading
            # matrix (a single advanced read) rather than by double-slicing Sigma_z.
            Lam_p = Lambda_full[obs_idx]  # (k_p, T*D)
            Sig_p = Lam_p @ Sigma_f @ Lam_p.T + pt.diag(sig2_full[obs_idx])
            chol_p = pt.linalg.cholesky(Sig_p)
            node = f"z_obs_{tag}"
            pm.MvNormal(
                node,
                mu=mean_full[obs_idx],
                chol=chol_p,
                observed=data,
                dims=(row_coord, cell_coord),
            )
            z_nodes.append(node)
            child_of_node[node] = list(children)
            cell_indices_of_node[node] = obs_idx.tolist()
            observed_z_of_node[node] = data

    payload = LongCorrFactorPayload(
        z_nodes=tuple(z_nodes),
        child_of_node={
            key: np.asarray(value, dtype=int) for key, value in child_of_node.items()
        },
        # Preserve the exact pattern-specific inputs used by the MvNormal nodes.
        # The LOO post-processor evaluates the same density from posterior
        # ``mean_z`` / ``Sigma_z`` without asking PyMC to reconstruct it through
        # transformed LKJCorr value variables.
        cell_indices_of_node={
            key: np.asarray(value, dtype=int)
            for key, value in cell_indices_of_node.items()
        },
        observed_z_of_node=observed_z_of_node,
        domains={key: tuple(value) for key, value in domains.items()},
        domain_of={
            ind_names[j]: domain_names[domain_of_idx[j]] for j in range(J)
        },
        indicators=tuple(ind_names),
        cell_names=tuple(cell_names),
        standardisers=standardisers,
        waves=tuple(waves),
        n_children=N,
        n_used_children=sum(len(c) for _, c in sorted_patterns),
        invariance="wave-invariant loadings and residual scales",
    )
    return BuiltModel(model=model, prepared=panel, payload=payload)
