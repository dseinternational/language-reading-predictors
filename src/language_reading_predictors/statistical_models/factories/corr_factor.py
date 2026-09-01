# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Correlated-factor measurement model construction.

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
    _scalar_prior,
)

def build_correlated_factor_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str = "W",
    domains: dict[str, tuple[str, ...]] | None = None,
    structural_covariates: Iterable[str] = ("blocks",),
    structural_factors: tuple[str, ...] | None = None,
    use_group: bool = False,
    use_age: bool = True,
    # Loading / residual geometry (#383). The default is the communality-scale
    # parameterisation ported from ``build_rlm_corr_factor_model`` (#409 item B):
    # ``communality ~ Beta(comm_alpha, comm_beta)`` with the loading ``sqrt(c)``
    # and residual ``sqrt(1 - c)`` derived, enforcing the lambda**2 + sigma**2 = 1
    # budget the standardised indicators imply. ``loading_prior="free"`` retains
    # the legacy unconstrained pair for the prior-geometry sensitivity companion
    # (LRPMM101): TruncatedNormal(mu=0, sigma=1, lower=0) IS HalfNormal(1), so the
    # legacy knobs' defaults reproduce the pre-#261 prior exactly. (An earlier
    # revision of #261 recalibrated the free pair to (0.6, 0.5) / 0.5 alongside
    # the marginalisation; the 2x2 ablation — LRPMM101, see
    # notes/202607101638-mm-001-convergence-reparameterisation.md — showed that
    # recalibration is neither necessary nor sufficient for convergence while
    # moving prior-implied median communality 0.50 -> 0.79, so it was reverted.
    # The communality default keeps the ablation's defended median of 0.5.)
    loading_prior: str = "communality",
    comm_alpha: float = 2.0,
    comm_beta: float = 2.0,
    loading_mu: float = 0.0,
    loading_sigma: float = 1.0,
    residual_sigma: float = 1.0,
    # Reconciled 0.5 -> 0.3 to match the shared ``predictor_slope_prior`` default the
    # 2026-07-07 prior review settled on (#141); applies to beta_factor, beta_age and
    # the structural-covariate slopes. The factors are unit-variance, so a per-SD-of-
    # factor logit slope is on the same scale as a standardised observed predictor, and
    # 0.3 keeps the CFA's structural priors in step with the rest of the suite rather
    # than sitting looser without a documented rationale. This also aligns the built RV
    # with the report's prior_predictor_slope panel (drawn at the 0.3 constructor
    # default), which previously showed 0.3 against a 0.5 RV (review finding B4, 2026-07-13).
    predictor_slope_sigma: float = 0.3,
    # #382 item 1: when set, the structural FACTOR slopes (beta_factor) take
    # this SD instead of predictor_slope_sigma, while every other slope —
    # including the arm covariate beta_G, which the review's recommendation 1
    # explicitly keeps at the association scale — is unchanged. The
    # prior-critical review flagged the mm-002 code->word slope as the one place
    # a headline prior is plausibly under-scaled: N(0, 0.3) is the association
    # default, but the EiV code->W slope is the documented PRIMARY mechanism,
    # whose linear-factory scale is N(0, 1). LRPMM102 is the registered
    # sensitivity companion that widens exactly that term.
    focal_slope_sigma: float | None = None,
    lkj_eta: float = 2.0,
) -> BuiltModel[EmptyPayload]:
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

    **Small-n geometry.** The original build sampled a per-child latent score for
    every domain and conditioned both the indicators *and* the structural outcome
    on it; coupled to free ``HalfNormal(1)`` loading and residual scales this gave
    an energy funnel (the reporting fit failed BFMI on every chain with ~1%
    divergences at n ~ 51). Because the measurement model is Gaussian in the
    factors, the indicators are marginalised to an ``MvNormal`` with the factor
    scores integrated out, and the scores are reintroduced only for the structural
    leg via their conjugate Gaussian conditional (non-centred, so the standard-
    normal offset is decoupled from the loading / residual scales). That rewrite is
    **measure-preserving** -- by conjugacy the posterior over loadings, residuals,
    factor correlations, scores and slopes is unchanged; only the geometry is -- and
    it is what repairs the energy diagnostic (BFMI 0.21 -> ~0.87). The reporting fit
    additionally lifts ``target_accept`` (via the spec) to clear the residual
    boundary divergences, which the strict gate requires to be exactly zero.

    **Loading / residual priors (#383).** The default ``loading_prior="communality"``
    makes each indicator's communality the free parameter — ``c ~ Beta(comm_alpha,
    comm_beta)`` on (0, 1), with ``lambda = sqrt(c)`` and ``sigma = sqrt(1 - c)``
    derived — exactly as ``build_rlm_corr_factor_model`` already does (#409 item B).
    Standardised indicators have unit sample variance, so this enforces the
    ``lambda**2 + sigma**2 = 1`` budget the data pipeline implies. The legacy free
    pair (``lambda`` and ``sigma`` iid ``HalfNormal(1)``) implies ``communality ~
    Beta(1/2, 1/2)`` — both squares are chi-square with one degree of freedom — an
    arcsine prior piling mass on both singular corners (the ``lambda -> 0`` neck and
    the Heywood-adjacent ``c -> 1`` boundary) and putting ~32% of loading mass above
    1, which the unit-variance budget rules out (prior-critical review 2026-07-21,
    #383). The default ``Beta(2, 2)`` keeps the median communality of 0.5 that the
    LRPMM101 ablation defended against the 0.79-median recalibration, while placing
    zero density at both corners. ``loading_prior="free"`` retains the original
    ``TruncatedNormal`` / ``HalfNormal`` pair (knobs ``loading_mu`` /
    ``loading_sigma`` / ``residual_sigma``) so the sensitivity companion (LRPMM101)
    can vary only the geometry; on the ablation history see
    ``notes/202607101638-mm-001-convergence-reparameterisation.md``.

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
    if loading_prior not in {"communality", "free"}:
        raise ValueError(
            f"loading_prior must be 'communality' or 'free'; got {loading_prior!r}"
        )
    if focal_slope_sigma is not None and not (
        np.isfinite(focal_slope_sigma) and focal_slope_sigma > 0.0
    ):
        raise ValueError(
            f"focal_slope_sigma must be finite and positive when set; got "
            f"{focal_slope_sigma}"
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
    if structural_factors is not None:
        structural_factors = tuple(structural_factors)
        _bad = [d for d in structural_factors if d not in domain_names]
        if _bad:
            raise ValueError(
                f"structural_factors {_bad} not in domains {domain_names}; the full "
                "measurement model is kept for identification, but the structural leg "
                "may regress on a chosen subset of the fitted factors (#228 item 14)."
            )

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
    if structural_factors is not None:
        coords["struct_domain"] = list(structural_factors)
    with pm.Model(coords=coords) as model:
        Z_d = pm.Data("Z", Z, dims=("obs_id", "indicator"))
        own_pre_d = pm.Data("own_pre_logit", own_pre_logit, dims="obs_id")

        # --- Measurement: correlated unit-variance domain factors ---
        # The per-child factor scores are MARGINALISED OUT of the Gaussian
        # measurement likelihood. The original build sampled a latent
        # score for every child x domain and conditioned both the indicators and the
        # structural outcome on it; coupled to the free loading / residual scales
        # this gave an energy funnel (the reporting fit failed BFMI on every chain
        # with ~1% divergences at n ~ 51). Because the measurement model is Gaussian
        # in the factors, the indicators marginalise analytically to
        # ``Z_i ~ MVN(0, Lambda Corr Lambda' + diag(sigma^2))`` with no per-child
        # latent, and the factor scores are reintroduced ONLY for the (non-Gaussian)
        # structural leg via their conjugate Gaussian conditional -- non-centred
        # around the data-informed conditional mean, so the standard-normal offset
        # ``factor_z`` is decoupled from the loading / residual scales. This is a
        # measure-preserving reparameterisation: the posterior over loadings,
        # residuals, factor correlations, factor scores and slopes is unchanged;
        # only the sampler geometry is.
        # Correlation-only role, so use bare ``LKJCorr`` rather than
        # ``LKJCholeskyCov``. The previous build discarded both the Cholesky factor
        # and the sds (``_, corr, _``) and used only the correlation; but in a CFA the
        # factor scale is fixed by the loadings, so those D sd components are
        # **unidentified**. They wandered, mixed poorly, and — because the convergence
        # gate scans every free RV — failed R-hat/ESS on ``factor_cov`` (R-hat up to
        # 1.024, ESS down to 213 at reporting tier) while every quantity the model
        # actually reports converged cleanly (``factor_corr`` R-hat <= 1.003 with ESS
        # 2.2k-24k; ``lambda_load`` and ``communality`` better still). The gate was
        # therefore failing on a nuisance parameter nothing downstream reads.
        #
        # Bare ``LKJCorr`` has no sds to leave unidentified and gives the correlation
        # the same LKJ(eta) marginal, so this is measure-preserving for everything
        # reported. It follows the reasoning already applied in
        # ``build_longitudinal_corr_factor_model`` and the measure-correlation block
        # below, and ``lrp-rli-lcf-001`` — which already used bare ``LKJCorr`` — is the
        # natural experiment: it passed the gate where these four did not.
        #
        # The environment's ``LKJCorr`` returns the CHOLESKY FACTOR L, not R, so
        # R = L @ L.T. A single-domain model has no free correlation at all.
        if D > 1:
            factor_chol = _priors.declare(
                              pm.LKJCorr("factor_corr_chol", n=D, eta=lkj_eta),
                              role="association",
                              rationale=(
                                  "LKJ prior on the Cholesky factor of the cross-domain factor "
                                  "correlation (LKJCorr(eta)); R = chol @ chol.T is the reported "
                                  "between-domain correlation this measurement model exists to "
                                  "estimate."
                              ),
                          )
            corr = pm.Deterministic(
                "factor_corr", factor_chol @ factor_chol.T, dims=("domain", "domain_b")
            )
        else:
            corr = pm.Deterministic(
                "factor_corr", pt.eye(D), dims=("domain", "domain_b")
            )

        # The headline quantities of this model are the D*(D-1)/2 unique
        # off-diagonal factor correlations, but ``factor_corr`` cannot be used to
        # gate them: it carries a constant unit diagonal and a duplicated lower
        # triangle, and a constant has undefined R-hat / zero variance, so ESS and
        # R-hat computed over the full matrix are meaningless (they silently pass).
        # Expose the unique off-diagonals as their own 1-D vector so the strict
        # convergence gate evaluates exactly the numbers the report releases.
        # (A single-factor model has no off-diagonals, so the node is skipped: the
        # downstream gate treats a missing var_name as nothing to check.)
        iu, ju = np.triu_indices(D, k=1)
        if len(iu):
            corr_pair_names = [
                f"{domain_names[i]}~{domain_names[j]}"
                for i, j in zip(iu, ju, strict=True)
            ]
            model.add_coords({"factor_pair": corr_pair_names})
            pm.Deterministic(
                "factor_corr_pairs",
                pt.stack([corr[i, j] for i, j in zip(iu, ju, strict=True)]),
                dims="factor_pair",
            )

        # Loading / residual parameterisation (#383; see the docstring). Default:
        # the communality is the free parameter — c ~ Beta(comm_alpha, comm_beta)
        # on (0, 1), lambda = sqrt(c), sigma = sqrt(1 - c) — enforcing the
        # lambda**2 + sigma**2 = 1 budget that standardised indicators imply,
        # exactly as build_rlm_corr_factor_model already does. The node names
        # (lambda_load / sigma_indicator / communality) are unchanged in both
        # modes; only which of them is the free RV differs.
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
            lam = pm.Deterministic("lambda_load", pt.sqrt(comm), dims="indicator")
            sigma_ind = pm.Deterministic(
                "sigma_indicator", pt.sqrt(1.0 - comm), dims="indicator"
            )
        else:
            # Legacy free pair, retained for the prior-geometry sensitivity
            # companion (LRPMM101). Knob defaults reproduce the original
            # HalfNormal(1) priors; the TruncatedNormal form exists so a companion
            # can shift the loading mode off zero.
            #
            # NB the earlier claim that a HalfNormal(residual_sigma=0.5) "caps the
            # residual SD below the unit total variance of a standardised indicator"
            # was wrong: a HalfNormal has unbounded support and merely makes
            # sigma > 1 unlikely (~5% of prior mass). No cap is imposed, or needed.
            lam = _priors.declare(
                pm.TruncatedNormal(
                    "lambda_load",
                    mu=loading_mu,
                    sigma=loading_sigma,
                    lower=0.0,
                    dims="indicator",
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

        # Sparse loading matrix Lambda (J x D): indicator j loads on its domain only.
        onehot = np.zeros((len(ind_names), D), dtype=float)
        onehot[np.arange(len(ind_names)), domain_idx] = 1.0
        Lambda = lam[:, None] * pt.as_tensor_variable(onehot)  # (J, D)
        sig2 = sigma_ind**2  # (J,)

        # Marginal measurement likelihood (factor scores integrated out):
        # Sigma_Z = Lambda Corr Lambda' + diag(sigma^2), fed to the MVN via its
        # Cholesky for stability.
        Sigma_Z = Lambda @ corr @ Lambda.T + pt.diag(sig2)
        L_Z = pt.linalg.cholesky(Sigma_Z)
        pm.MvNormal(
            "Z_obs",
            mu=pt.zeros(len(ind_names)),
            chol=L_Z,
            observed=Z_d,
            dims=("obs_id", "indicator"),
        )

        # Conjugate Gaussian conditional p(factors | Z, params) = MVN(cond_mean, V):
        #   V        = (Corr^{-1} + Lambda' diag(sigma^-2) Lambda)^{-1}
        #   cond_mean_i = V Lambda' diag(sigma^-2) Z_i
        # Reintroduce the factor scores for the structural leg, non-centred around
        # the conditional mean so factor_z stays standard-normal (no funnel).
        # The two D×D inverses use an explicit ``inv`` rather than a ``solve`` with
        # an identity RHS: at D = 3 the conditioning difference is negligible (the
        # inverses agree with a Cholesky solve to machine precision), the
        # ``solve → cholesky`` path is unsupported by the Numba forward-sampling
        # backend used for the prior/posterior-predictive draws (it rejects a
        # ``cholesky`` on the read-only buffer a ``solve`` returns), and — decisive
        # here — the ``solve`` variant empirically produced a boundary divergence
        # that trips the strict zero-divergence gate, whereas ``inv`` clears it.
        # PREDICTIVE-SIMULATION CAVEAT (read before interpreting any PPC here).
        # ``cond_mean`` is built from the *data container* ``Z_d``, not from the
        # ``Z_obs`` random variable. That is correct for inference — the factor
        # scores should condition on the observed indicators — but it means the two
        # observed nodes are NOT jointly simulated in a forward pass:
        #
        #   * ``Z_obs`` replicates the indicators from the marginal MVN, and
        #   * ``factors`` (hence ``y_post``) stays conditioned on the OBSERVED Z.
        #
        # So a replicated indicator is statistically independent of the replicated
        # factor it nominally loads on, and drawing both nodes does *not* constitute
        # a draw from the joint model. Read them as two separate checks: ``Z_obs``
        # is a marginal check of the measurement covariance, and ``y_post`` is a
        # check of the structural leg CONDITIONAL on the observed indicators. The
        # same caveat applies to the prior predictive, and more sharply: the
        # ``y_post`` prior draws condition on the observed Z, so they are not a
        # prior predictive of the outcome in the usual (data-free) sense.
        #
        # A coherent joint simulation would require separate generative nodes
        # (factors ~ MVN(0, Corr); Z | factors; y | factors) alongside the
        # inferential ones. Not done here — the labelling above is the honest
        # description of what the pipeline currently emits.
        corr_inv = pt.linalg.inv(corr)  # (D, D)
        A = Lambda.T * (1.0 / sig2)[None, :]  # (D, J) = Lambda' diag(sigma^-2)
        V = pt.linalg.inv(corr_inv + A @ Lambda)  # (D, D)
        W = V @ A  # (D, J)
        L_V = pt.linalg.cholesky(V)
        cond_mean = Z_d @ W.T  # (n, D)
        z_factor = _priors.declare(
                       pm.Normal("factor_z", 0.0, 1.0, dims=("obs_id", "domain")),
                       role="nuisance",
                       rationale=(
                           "Non-centred standard-normal per-observation, per-domain factor "
                           "scores (Normal(0, 1)); the latent domain scores the loadings map "
                           "onto each standardised indicator."
                       ),
                   )
        factors = pm.Deterministic(
            "factors", cond_mean + z_factor @ L_V.T, dims=("obs_id", "domain")
        )

        # --- Structural: outcome gain ~ factors (+ covariates), Beta-Binomial ---
        # ``structural_factors`` (default None) regresses on ALL fitted domain factors
        # (mm-001). When set (e.g. ("code",), #228 item 14) the full measurement model
        # is kept for identification but the structural leg uses only the named
        # factor(s) — isolating one latent construct's measurement-error-corrected slope.
        alpha = _scalar_prior("alpha", _priors.alpha_prior)
        gamma_own = _priors.gamma_own_prior().to_pymc("gamma_own")
        # #382 item 1: the focal factor slopes may take a wider,
        # primary-mechanism-scale prior than the association-scale default.
        _focal_sigma = (
            predictor_slope_sigma if focal_slope_sigma is None else focal_slope_sigma
        )
        if structural_factors is None:
            beta_factor = _priors.declare(
                              pm.Normal(
                                              "beta_factor", 0.0, _focal_sigma, dims="domain"
                                          ),
                              role="association",
                              panel="predictor_slope",
                              rationale=(
                                  "Standardised predictor slope ~ Normal(0, 0.3) by default."
                              ),
                          )
            struct = pm.math.dot(factors, beta_factor)
        else:
            _sidx = [domain_names.index(d) for d in structural_factors]
            beta_factor = _priors.declare(
                              pm.Normal(
                                              "beta_factor", 0.0, _focal_sigma, dims="struct_domain"
                                          ),
                              role="association",
                              panel="predictor_slope",
                              rationale=(
                                  "Standardised predictor slope ~ Normal(0, 0.3) by default."
                              ),
                          )
            struct = pm.math.dot(factors[:, _sidx], beta_factor)
        eta = alpha + gamma_own * own_pre_d + struct

        if use_group:
            # Randomised arm as an adjusted-association covariate (NOT a randomised
            # effect here) on the association-scale predictor_slope prior — mirrors the
            # mech-058 adjustment set for the errors-in-variables mechanism (#228 item 14).
            G_d = pm.Data("G", np.asarray(prepared.G, dtype=float), dims="obs_id")
            beta_G = _priors.predictor_slope_prior(predictor_slope_sigma).to_pymc(
                "beta_G"
            )
            eta = eta + beta_G * G_d

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

    return BuiltModel(model=model, prepared=prepared, payload=EmptyPayload())


def build_rlm_corr_factor_model(
    battery,
    *,
    domains: dict[str, tuple[str, ...]],
    single_indicator_reliability: float = 0.8,
    comm_alpha: float = 2.0,
    comm_beta: float = 2.0,
    lkj_eta: float = 2.0,
) -> BuiltModel[EmptyPayload]:
    """Byrne correlated-domain-factor measurement model (#338 Phase B).

    The **measurement core** of the RLI ``build_correlated_factor_model`` on the
    Byrne one-wave battery (:class:`preprocessing.RlmWaveBattery`): correlated
    unit-variance domain factors over standardised Haldane-logit indicators,
    with the per-child factor scores **marginalised out** of the Gaussian
    likelihood (``Z_i ~ MVN(0, Lambda Corr Lambda' + diag(sigma^2))`` - the
    mm-001 funnel fix, measure-preserving). There is **no structural leg**: the
    deliverable is the loadings/communalities table and the factor correlation
    matrix - the modern analogue of the paper's correlation tables. Nothing is
    causal; every correlation is a descriptive association.

    **Single-indicator domains.** The Byrne memory domain has one indicator
    (``basdig``), which cannot identify a free loading and residual. Its
    loading and residual are **fixed** by an assumed reliability:
    ``lambda = sqrt(r)``, ``sigma = sqrt(1 - r)`` with
    ``r = single_indicator_reliability`` (default 0.8, a conventional
    test-retest figure for short-term memory span scales; the report states the
    assumption and the correlation involving that domain scales with it).
    **Communality parameterisation (#409 item B, the mm-001 gate rescue).**
    Standardised indicators have unit marginal variance, so a free indicator's
    ``lambda**2 + sigma**2 = 1``. The earlier free ``HalfNormal`` loading and
    residual left that sum unconstrained — an over-parameterised lambda-sigma ridge
    whose Heywood corner (``lambda -> 1``, ``sigma -> 0``) drove the gate failure
    (143 divergences, R-hat 1.03). Instead the **communality is the free parameter**:
    ``c ~ Beta(comm_alpha, comm_beta)`` on the open unit interval, then
    ``lambda = sqrt(c)`` and ``sigma = sqrt(1 - c)``. This enforces the unit variance
    the standardisation implies — as the fixed single-indicator domain already does —
    removes the ridge, and makes the reported communality a direct parameter.
    ``comm_alpha``/``comm_beta`` default to ``Beta(2, 2)``, a weakly-informative
    communality prior centred at 0.5 that (with both shapes > 1) also has zero density
    at the singular corners c = 0 and c = 1; ``comm_alpha, comm_beta`` must be
    positive.
    """
    domain_names = list(domains)
    D = len(domain_names)
    ind_names: list[str] = []
    domain_of: list[int] = []
    fixed_mask: list[bool] = []
    cols: list[np.ndarray] = []
    for di, d in enumerate(domain_names):
        syms = tuple(domains[d])
        if not syms:
            raise ValueError(f"Domain {d!r} has no indicators.")
        single = len(syms) == 1
        for s in syms:
            if s not in battery.indicators:
                raise KeyError(
                    f"Indicator {s!r} (domain {d!r}) missing from battery "
                    f"(have {list(battery.indicators)})."
                )
            cols.append(battery.indicators[s])
            ind_names.append(s)
            domain_of.append(di)
            fixed_mask.append(single)
    Z = np.stack(cols, axis=1)
    J = len(ind_names)
    domain_idx = np.asarray(domain_of, dtype=np.int64)
    fixed = np.asarray(fixed_mask)
    free_names = [n for n, f in zip(ind_names, fixed, strict=True) if not f]
    r = float(single_indicator_reliability)
    if not (0.0 < r < 1.0):
        raise ValueError(f"single_indicator_reliability must be in (0, 1); got {r}.")
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

    coords = {
        "obs_id": np.arange(battery.n_obs),
        "indicator": ind_names,
        "free_indicator": free_names,
        "domain": domain_names,
        "domain_b": domain_names,
    }
    with pm.Model(coords=coords) as model:
        # Factor correlation matrix, mirroring the RLI mm-001 build: the unique
        # off-diagonals are exposed as their own vector so the strict gate
        # evaluates exactly the numbers the report releases.
        # Correlation-only role, so use bare ``LKJCorr`` rather than
        # ``LKJCholeskyCov``. The previous build discarded both the Cholesky factor
        # and the sds (``_, corr, _``) and used only the correlation; but in a CFA the
        # factor scale is fixed by the loadings, so those D sd components are
        # **unidentified**. They wandered, mixed poorly, and — because the convergence
        # gate scans every free RV — failed R-hat/ESS on ``factor_cov`` (R-hat up to
        # 1.024, ESS down to 213 at reporting tier) while every quantity the model
        # actually reports converged cleanly (``factor_corr`` R-hat <= 1.003 with ESS
        # 2.2k-24k; ``lambda_load`` and ``communality`` better still). The gate was
        # therefore failing on a nuisance parameter nothing downstream reads.
        #
        # Bare ``LKJCorr`` has no sds to leave unidentified and gives the correlation
        # the same LKJ(eta) marginal, so this is measure-preserving for everything
        # reported. It follows the reasoning already applied in
        # ``build_longitudinal_corr_factor_model`` and the measure-correlation block
        # below, and ``lrp-rli-lcf-001`` — which already used bare ``LKJCorr`` — is the
        # natural experiment: it passed the gate where these four did not.
        #
        # The environment's ``LKJCorr`` returns the CHOLESKY FACTOR L, not R, so
        # R = L @ L.T. A single-domain model has no free correlation at all.
        if D > 1:
            factor_chol = _priors.declare(
                              pm.LKJCorr("factor_corr_chol", n=D, eta=lkj_eta),
                              role="association",
                              rationale=(
                                  "LKJ prior on the Cholesky factor of the cross-domain factor "
                                  "correlation (LKJCorr(eta)); R = chol @ chol.T is the reported "
                                  "between-domain correlation this measurement model exists to "
                                  "estimate."
                              ),
                          )
            corr = pm.Deterministic(
                "factor_corr", factor_chol @ factor_chol.T, dims=("domain", "domain_b")
            )
        else:
            corr = pm.Deterministic(
                "factor_corr", pt.eye(D), dims=("domain", "domain_b")
            )
        iu, ju = np.triu_indices(D, k=1)
        if len(iu):
            pm.Deterministic(
                "factor_corr_pairs",
                pt.stack([corr[i, j] for i, j in zip(iu, ju, strict=True)]),
            )

        # Communality parameterisation (see the docstring): the free parameter is the
        # communality c in (0, 1); the loading and residual are derived so that
        # lambda**2 + sigma**2 = 1 exactly. This enforces the unit variance the
        # standardised indicators imply and removes the over-parameterised
        # lambda-sigma ridge / Heywood corner that gate-failed the free build.
        comm_free = _priors.declare(
                        pm.Beta(
                                    "communality_free", alpha=comm_alpha, beta=comm_beta, dims="free_indicator"
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
        lam_free = pm.Deterministic(
            "lambda_free", pt.sqrt(comm_free), dims="free_indicator"
        )
        sigma_free = pm.Deterministic(
            "sigma_free", pt.sqrt(1.0 - comm_free), dims="free_indicator"
        )
        lam_full = pt.zeros(J)
        sig_full = pt.zeros(J)
        free_pos = np.flatnonzero(~fixed)
        fixed_pos = np.flatnonzero(fixed)
        lam_full = pt.set_subtensor(lam_full[free_pos], lam_free)
        sig_full = pt.set_subtensor(sig_full[free_pos], sigma_free)
        if len(fixed_pos):
            lam_full = pt.set_subtensor(lam_full[fixed_pos], np.sqrt(r))
            sig_full = pt.set_subtensor(sig_full[fixed_pos], np.sqrt(1.0 - r))
        lam = pm.Deterministic("loading", lam_full, dims="indicator")
        sig = pm.Deterministic("sigma_indicator", sig_full, dims="indicator")

        # Sparse J x D loading matrix: indicator j loads only on its domain.
        Lmat = pt.zeros((J, D))
        Lmat = pt.set_subtensor(Lmat[np.arange(J), domain_idx], lam)
        Sigma = Lmat @ corr @ Lmat.T + pt.diag(sig**2)

        pm.Deterministic(
            "indicator_factor_corr",
            lam / pt.sqrt(lam**2 + sig**2),
            dims="indicator",
        )
        pm.Deterministic(
            "communality", lam**2 / (lam**2 + sig**2), dims="indicator"
        )

        Z_d = pm.Data("Z", Z, dims=("obs_id", "indicator"))
        pm.MvNormal(
            "Z_obs",
            mu=pt.zeros(J),
            cov=Sigma,
            observed=Z_d,
            dims=("obs_id", "indicator"),
        )

    return BuiltModel(model=model, prepared=battery, payload=EmptyPayload())
