# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model factories for the statistical models.

One ``build_*`` factory per model family (keyed by :class:`ModelSpec.kind`), the
most-used being:

- :func:`build_itt_model` — the LRPITT ITT suite (one outcome, RCT phase) and its
  SES-adjusted companions; the floored outcomes use its ``bernoulli_offfloor``
  likelihood mode for the off-floor primary estimand.
- :func:`build_joint_model` — the joint model (LRPITT12) and the two-outcome
  generalisation contrasts (LRPITT15/15b), RCT phase, optional LKJ Σ.
- :func:`build_mechanism_model` — LRP56, LRP57, LRP58 (adjustment-set
  mechanism regressions on ``W_post`` using all phases).

The rest cover the remaining families: :func:`build_mediation_model`,
:func:`build_did_model`, :func:`build_gain_factors_model`,
:func:`build_level_factors_model`, :func:`build_aligned_model`,
:func:`build_adjusted_model`, :func:`build_correlated_factor_model`,
:func:`build_dose_response_model`, :func:`build_lcsm_model`,
:func:`build_two_mediator_model`, :func:`build_horseshoe_model`,
:func:`build_growth_model` and :func:`build_historical_growth_model`. See
``definitions.py`` for the authoritative family/kind registry.

Each returns a :class:`BuiltModel` carrying the PyMC ``model``, a ``variables``
dict mapping names to PyMC variables (used by the pipeline to extract posterior
draws and assemble report tables), the row-subset ``prepared`` data, and any
``extras`` the reporting layer needs (e.g. treatment-interaction moderators).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

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

# Basis count for the mechanism-curve HSGP (issue #265). Fewer functions than the
# generic default (20) shrink the parameter space feeding the boundary-geometry
# funnel; at n ~ 157 with a smooth curve, ~12 is more resolution than the data
# support. Scoped to f_mech so other GP-bearing models keep the default.
_MECH_HSGP_M = 10


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


def _alpha_sigma_for(outcome_symbol: str, override: float | None = None) -> float:
    """Intercept prior SD for a single-outcome ANCOVA (prior-critical-review 2026-07-07).

    Mirrors :func:`_tau_sigma_for`: returns ``override`` when given (prior-
    sensitivity fits), else the outcome tier — the tighter ``ALPHA_SIGMA_DISTAL``
    (1.0) for the broad high-denominator standardised-transfer outcomes
    (``measures.DISTAL_OUTCOMES``) and the wider ``ALPHA_SIGMA_PROXIMAL`` (1.5)
    otherwise. A no-op for proximal outcomes, so only distal-outcome fits tighten.
    Applies to the free ``alpha`` intercept of the ANCOVA families whose linear
    predictor already carries the outcome level in the ``gamma_own * logit(y_pre)``
    term (so ``alpha``'s mean is a ~0 deviation, tiered by SD — not re-anchored;
    see :func:`priors.alpha_prior`). The growth/LCSM *level* models instead anchor
    the intercept mean and do not use this.
    """
    if override is not None:
        return override
    return (
        _priors.ALPHA_SIGMA_DISTAL
        if is_distal(outcome_symbol)
        else _priors.ALPHA_SIGMA_PROXIMAL
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
    extras: dict[str, Any] = field(default_factory=dict)
    """Optional per-model artefacts the pipeline needs but that are not RVs.

    Used to carry the exact moderator vectors of any treatment×covariate
    interactions (aligned with ``prepared``'s ``obs_id`` order) so the
    average-marginal-effect report can net out the *full* per-row treatment
    contribution — not just the treatment main effect — without recomputing
    (and risking drift from) the standardisation the factory used. See
    ``reporting._itt_ame_draws``'s ``moderators`` argument.
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
    alpha_sigma: float | None = None,
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
    alpha_sigma
        Override the intercept prior SD (prior-critical-review 2026-07-07,
        Finding 1). ``None`` (default) uses the outcome tier via
        :func:`_alpha_sigma_for`: ``ALPHA_SIGMA_DISTAL`` (1.0) for the
        high-denominator broad-transfer outcomes, ``ALPHA_SIGMA_PROXIMAL`` (1.5)
        otherwise — a no-op for proximal outcomes. The intercept is a *deviation*
        (the level is carried by ``gamma_own * logit(y_pre)``), so this tiers its
        SD rather than re-anchoring its mean. Pass an explicit value for a
        prior-sensitivity fit.
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

        alpha = _priors.alpha_prior(
            sigma=_alpha_sigma_for(own, alpha_sigma)
        ).to_pymc("alpha")
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
            # Standardise the own-baseline logit before the GP so the lengthscale /
            # boundary / basis priors stay in their calibrated (unit-SD) regime
            # (issue #273 item 13 / #265); the age GP above already uses A_std.
            y_pre_std, _ = standardise(y_pre_logit)
            f_ypre = build_hsgp_1d("f_ypre", y_pre_std)
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
    # Expose the tau-moderator vector so the AME report can net out the full
    # per-row treatment contribution ``(tau + gamma_tau_int·z_M)·G`` when a
    # linear tau moderator with interaction is fitted (Part B; latent — no
    # registered spec sets ``tau_moderator_symbol`` today). ``gamma_tau_mod`` is
    # a main effect and cancels in the toggle, so only ``gamma_tau_int`` enters.
    tau_moderators: list[tuple[str, np.ndarray]] = []
    if z_M is not None and tau_moderator_interaction:
        tau_moderators.append(("gamma_tau_int", np.asarray(z_M, dtype=float)))
    return BuiltModel(
        model=model,
        variables=variables,
        prepared=prepared,
        extras={"tau_interaction_moderators": tau_moderators},
    )


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

        # Record each flattened cell's outcome index (0..K-1, matching the "outcome"
        # coord order) as constant data, so the prior/posterior-predictive plotter
        # can select one outcome's cells rather than pooling counts with
        # denominators 6..170 into one histogram (issue #271 item 2).
        pm.Data("y_post_cell_outcome", idx_col.astype("int64"))

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
    adjust_for: Iterable[str] = (),
    mechanism_is_covariate: bool = False,
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

    ``adjust_for`` (default ()): revised-DAG confounders that are not bounded-count
    measures and so cannot enter via ``confounder_symbols`` — hearing status
    (``hs`` / ``hs_missing``), speech production (``deapp_c``), phonological memory
    (``erbto``), session dose (``attend``). Each must be a key in
    ``prepared.covariates`` (the pipeline standardises the continuous ones and adds
    missing-indicators); they enter as linear ``gamma_{c}`` terms with the
    regularising cross-coupling prior, exactly as in ``build_itt_model`` (#245).
    Age and group need no entry here: age is absorbed by the phase-specific
    intercepts and group is always in ``beta_G``.

    ``mechanism_is_covariate`` (default False): treat the *exposure* as a
    standardised continuous covariate (a key of ``prepared.covariates``, e.g.
    phonological memory ``erbto``) rather than a bounded-count measure (#311's
    route (b): the ERB total's documented test maximum is recorded nowhere in the
    repo, so registering it as a ``Measure`` would fabricate a denominator). The
    exposure is re-standardised on the kept rows and enters as
    ``beta_mech * z(exposure)``; a ``mech_covariate`` Data node replaces
    ``mech_post_logit``. Requires ``linear_mechanism=True`` — the HSGP curve, its
    priors and the readiness-threshold post-processing all assume a bounded-count
    logit exposure. The caller is responsible for restricting to genuinely
    observed exposure rows (``require_observed`` in the loader): mean-imputation
    plus a missingness indicator is an *adjuster* policy and is not acceptable for
    the exposure itself.
    """
    # Materialise once: ``confounder_symbols`` is iterated several times below
    # (keep-mask, coefficient loop, the "A in confounders" check, and the
    # "every declared confounder reaches eta" invariant). A generator argument
    # would be exhausted after the first pass and silently drop every confounder
    # — the exact failure the invariant exists to catch.
    confounder_symbols = tuple(confounder_symbols)
    adjust_for = tuple(adjust_for)
    if prepared.phase_mode != "all":
        raise ValueError("Mechanism factory requires phase_mode='all'")
    if mechanism_is_covariate:
        if not linear_mechanism:
            raise ValueError(
                "mechanism_is_covariate=True requires linear_mechanism=True: the "
                "HSGP curve, its priors and the readiness-threshold post-processing "
                "assume a bounded-count logit exposure."
            )
        if mechanism_symbol not in prepared.covariates:
            raise KeyError(
                f"Covariate mechanism {mechanism_symbol!r} not in "
                "prepared.covariates (load it via the pipeline's covariate lists)."
            )
    elif mechanism_symbol not in prepared.pre_logit:
        raise KeyError(f"Mechanism {mechanism_symbol!r} missing from prepared data")
    if outcome_symbol not in prepared.pre_logit:
        raise KeyError(f"Outcome {outcome_symbol!r} missing from prepared data")
    if (
        moderator_symbol is not None
        and not moderator_is_covariate
        and moderator_symbol not in prepared.pre_logit
    ):
        raise KeyError(f"Moderator {moderator_symbol!r} missing from prepared data")

    # Outcome post (target) and mechanism exposure (predictor) are both needed.
    outcome_post = prepared.post_counts[outcome_symbol]
    if mechanism_is_covariate:
        mechanism_vals = prepared.covariates[mechanism_symbol]
    else:
        mechanism_vals = prepared.post_counts[mechanism_symbol]

    keep = ~(np.isnan(outcome_post) | np.isnan(mechanism_vals))
    if moderator_symbol is not None and not moderator_is_covariate:
        keep = keep & ~np.isnan(prepared.post_counts[moderator_symbol])
    for s in confounder_symbols:
        if s not in prepared.pre_logit and s not in {"G", "A"}:
            raise KeyError(f"Confounder {s!r} not recognised")
        if s in prepared.post_counts:
            keep = keep & ~np.isnan(prepared.post_counts[s])
    for c in adjust_for:
        if c not in prepared.covariates:
            raise KeyError(f"Adjuster covariate {c!r} not loaded in prepared data")
        keep = keep & ~np.isnan(prepared.covariates[c])
    prepared = _subset(prepared, keep)

    from language_reading_predictors.statistical_models.preprocessing import (
        logit_safe,
        standardise,
    )

    outcome_post = prepared.post_counts[outcome_symbol].astype(np.int64)
    N_outcome = prepared.n_trials[outcome_symbol]
    if mechanism_is_covariate:
        # Standardised-covariate exposure: the loader's z-values on the kept rows.
        # No n_trials / logit transform exists for it (that is the point of the
        # covariate route: no fabricated denominator).
        mech_input = prepared.covariates[mechanism_symbol]
    else:
        N_mechanism = prepared.n_trials[mechanism_symbol]
        mech_input = logit_safe(
            prepared.post_counts[mechanism_symbol], N_mechanism
        )

    own_pre_logit = prepared.pre_logit[adjust_baseline_symbol]

    # Standardised mechanism logit, computed on the kept rows so the mean/sd match
    # the fitted data. Used both for the LINEAR moderation term ``z_L`` (a centred
    # version, so gamma_mod reads as the moderator effect at the mean of L) and —
    # issue #265 / #273 item 13 — as the HSGP ``f_mech`` input. Feeding the GP the
    # *raw* logit (spread wider than unit SD) miscalibrated the lengthscale
    # (``InverseGamma(3, 1)``), boundary factor ``c`` and basis count ``m``, all of
    # which are set for standardised inputs — the boundary-geometry neck that left a
    # residual divergence at reporting tier. Standardising the input fixes the
    # geometry without moving the fitted curve (f_mech is still evaluated per-obs and
    # plotted against the raw logit). For a covariate exposure the input is the
    # loader's z-values, re-standardised here on the kept rows so beta_mech reads
    # per SD of the exposure on the fitted data.
    mech_logit_std, _ = standardise(mech_input)
    z_L: np.ndarray | None = None
    z_M: np.ndarray | None = None
    if moderator_symbol is not None or linear_mechanism:
        z_L = mech_logit_std
    if moderator_symbol is not None:
        if moderator_is_covariate:
            # Continuous covariate moderator: dispatch on ``moderator_symbol`` so
            # the label matches the vector (like the ITT factory). ``"A"`` is age
            # (``prepared.A_std``); any other symbol must be a ``prepared.covariates``
            # key. Re-standardised on the kept rows so gamma_mod reads at mean L and
            # gamma_int is unit-free. Raising on an unknown symbol prevents the old
            # silent behaviour of fitting age moderation regardless of the symbol.
            if moderator_symbol == "A":
                raw_M = prepared.A_std
            elif moderator_symbol in prepared.covariates:
                raw_M = prepared.covariates[moderator_symbol]
            else:
                raise KeyError(
                    f"Covariate moderator {moderator_symbol!r} not in "
                    "prepared.covariates (use 'A' for age)."
                )
            z_M, _ = standardise(raw_M)
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
        if mechanism_is_covariate:
            # The exposure is a standardised covariate, not a bounded-count logit;
            # register it under its own name so introspection cannot mistake it
            # for a logit-scale measure.
            pm.Data("mech_covariate", mech_input, dims="obs_id")
        else:
            pm.Data("mech_post_logit", mech_input, dims="obs_id")
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
        adjust_data: dict[str, pt.TensorVariable] = {}
        for c in adjust_for:
            adjust_data[c] = pm.Data(
                f"{c}_adj", prepared.covariates[c], dims="obs_id"
            )

        alpha = _priors.alpha_prior(
            sigma=_alpha_sigma_for(outcome_symbol)
        ).to_pymc("alpha")
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

        # Raw-covariate adjusters (revised-DAG confounders that are not bounded-count
        # measures): hearing (hs/hs_missing), speech (deapp_c), phonological memory
        # (erbto), session dose (attend). Linear gamma terms, mirroring the
        # build_itt_model adjust_for path (#245).
        for c in adjust_for:
            gamma_c = _priors.gamma_cross_prior().to_pymc(f"gamma_{c}")
            eta = eta + gamma_c * adjust_data[c]

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
                    build_hsgp_1d(
                        f"f_mech_phase{p}",
                        mech_logit_std,
                        m=_MECH_HSGP_M,
                        lengthscale_prior=_priors.ell_prior_mech(),
                    )
                )
            # Register the combined per-observation curve as ``f_mech`` (each row's
            # phase-specific value), so ``_write_mechanism_curve`` finds it and
            # writes ``mechanism_curve.csv`` / the plot instead of silently skipping
            # — the phase-specific ``f_mech_phase{p}`` builders above only register
            # the per-phase GP hyperparameters, not the selected per-obs curve
            # (issue #265 review; supersedes the warn-only #273 item 20).
            f_mech = pm.Deterministic(
                "f_mech",
                pt.stack(phase_specific, axis=1)[np.arange(prepared.n_obs), phase_d],
                dims="obs_id",
            )
            eta = eta + f_mech
        else:
            # Standardised input + a moderate-lengthscale prior + fewer basis
            # functions (issue #265 / #273 item 13): keeps the HSGP priors in their
            # calibrated regime and smooths the boundary geometry that left residual
            # divergences, without discarding the curve. Scoped to f_mech only. The
            # curve is still plotted against the raw logit downstream, so its
            # shape/location is unchanged where the old fit was trustworthy.
            f_mech = build_hsgp_1d(
                "f_mech",
                mech_logit_std,
                m=_MECH_HSGP_M,
                lengthscale_prior=_priors.ell_prior_mech(),
            )
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
    # Default OFF (issue #269): conditioning on cumulative prior dose (a running sum
    # of the IS collider) reopens the latent-GA backdoor, so the headline fits do not
    # adjust it. Set it explicitly only for the flagged collider-sensitivity fit.
    dose_stage_covariate: str | None = None,
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

        alpha = _priors.alpha_prior(
            sigma=_alpha_sigma_for(outcome_symbol)
        ).to_pymc("alpha")
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
    use_varying_delta: bool = False,
    likelihood: str = "beta_binomial",
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
                  [ + gamma_A * A_std_{i,p}    if use_age ]
                  [ + u_child_i                if use_child_re ]

    where ``Treated_{i,p} = 1`` when child *i* is receiving the intervention in
    period *p* (immediate: both periods; waitlist: P2 only). With ``dose=True`` the
    binary ``delta * Treated`` is replaced by ``beta_dose * z(attend)`` (the
    standardised intervention-session count for that period) — a dose-response
    sensitivity; the caller must have loaded ``covariates=("attend",)``.

    **No own-baseline term in either likelihood branch** (A2 / #257 review): for the
    immediate arm's P2 the period-start score is post-P1-treatment, so a ``gamma_own``
    term would adjust a treatment-affected variable and bias the differenced
    ``delta``. ``delta`` is identified by the period × treated structure plus the
    child random intercept (each child their own control) instead; the graded and
    off-floor branches differ only in likelihood.

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
    use_varying_delta
        Add a non-centred per-child random slope on the treated indicator,
        ``delta_i = delta + sigma_delta * z_i``, and report ``sigma_delta`` — the
        between-child SD of the crossover treatment effect — as the treatment-effect
        heterogeneity variance component (#230 §2/§4a). ``delta`` remains the
        population-mean effect. Requires ``dose=False`` and ``use_child_re=True``.
    likelihood
        ``"beta_binomial"`` (default) fits the graded post-count. For heavily
        floored outcomes (P, N) pass ``"bernoulli_offfloor"``: the observation is a
        Bernoulli on the binary off-floor indicator (period post > 0), so ``delta``
        is the within-person DiD on the log-odds of *being off the floor at period
        end* — off-floor PREVALENCE, ``Pr(post > 0)``, not the floor-exit transition
        ``Pr(post > 0 | pre = 0)``. Its items-scale marginal (``n_trials=1``) is a
        model-implied off-floor risk difference from toggling ``Treated``, not a
        probability-scale DiD cross-difference (parallel trends holds on the
        log-odds scale). No ``kappa`` under the Bernoulli (neither branch has an
        own-baseline ``gamma_own`` term — see above). Requires ``dose=False``.
    """
    if prepared.phase_mode != "all":
        raise ValueError("build_did_model requires phase_mode='all'")
    own = outcome_symbol
    if own not in prepared.post_counts or own not in prepared.pre_logit:
        raise KeyError(f"Outcome {own!r} missing pre/post in prepared data")
    periods = tuple(int(p) for p in periods)
    # The time / treated indicators below hard-code the waitlist-crossover 2×2
    # (P1 untreated-waitlist vs P2 crossover, ``is_p2 = phase >= 1``,
    # ``treated = (G==1) | (phase>=1)``). They are only correct for the
    # crossover pair ``(0, 1)``; any other window would silently mislabel the
    # time and treatment cells (e.g. ``(1, 2)`` makes both indicators constant,
    # leaving delta / beta_period unidentified). Fail loudly instead.
    if periods != (0, 1):
        raise ValueError(
            "build_did_model hard-codes the P1-vs-P2 crossover contrast and "
            f"requires periods=(0, 1); got {periods}."
        )
    if likelihood not in ("beta_binomial", "bernoulli_offfloor"):
        raise ValueError(
            "likelihood must be 'beta_binomial' or 'bernoulli_offfloor', "
            f"got {likelihood!r}"
        )
    if likelihood == "bernoulli_offfloor" and dose:
        raise ValueError(
            "bernoulli_offfloor is the floor-rule binary estimand; use dose=False"
        )
    if dose and "attend" not in prepared.covariates:
        raise KeyError("dose=True requires the 'attend' covariate to be loaded")
    if period_varying_dose and not dose:
        raise ValueError("period_varying_dose=True requires dose=True")
    if use_varying_delta and dose:
        raise ValueError(
            "use_varying_delta is the binary treatment-effect heterogeneity variant; "
            "use dose=False"
        )
    if use_varying_delta and not use_child_re:
        # A per-child random *slope* on the treatment without a per-child random
        # *intercept* is rarely sensible and would confound level with responsiveness;
        # require the intercept (it also provides the shared child index).
        raise ValueError("use_varying_delta=True requires use_child_re=True")

    post = prepared.post_counts[own]
    keep = np.isin(prepared.phase, periods) & ~np.isnan(post)
    if dose:
        keep = keep & np.isfinite(prepared.covariates["attend"])
    prepared = _subset(prepared, keep)

    post = prepared.post_counts[own].astype(np.int64)
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

        alpha = _priors.alpha_prior(
            sigma=_alpha_sigma_for(outcome_symbol)
        ).to_pymc("alpha")
        beta_period = _priors.tau_prior().to_pymc("beta_period")

        eta = alpha + beta_period * period_d

        # No own-baseline (autoregressive) conditioning in EITHER likelihood branch
        # (A2 / #257 review, team decision 2026-07-13). For the immediate arm's P2 the
        # period-start score is *post* that arm's P1 treatment, so a ``gamma_own`` term
        # would condition the differenced ``delta`` on a treatment-affected variable — a
        # lagged-DV/ANCOVA adjustment that biases the total-effect / ITT-replication
        # reading (Rosenbaum 1984, doi:10.2307/2981697), and a child random intercept
        # does not restore it. ``delta`` is identified by the period × treated DiD
        # structure plus the child intercept (each child their own control), so no
        # own-baseline term is needed; the graded and off-floor branches now differ only
        # in likelihood (Beta-Binomial vs Bernoulli).

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
            # Re-standardise the dose covariate on the kept rows: it was z-scored
            # over all stacked transitions (P1–P3) at load time, but this model is
            # fit on ``periods`` (P1–P2) only, so without this the dose slope would
            # be "per SD of dose pooled over P1–P3", not per SD of the fitted rows.
            attend_std, _ = standardise(prepared.covariates["attend"])
            z_attend = pm.Data("z_attend", attend_std, dims="obs_id")
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
                # Use the dose-response factory's dose-slope prior (beta_mech,
                # Normal(0, 1)) — NOT tau — so the shared dose-slope summary compares
                # like with like (the docstring says this model mirrors
                # build_dose_response_model). Previously these reused tau_prior,
                # silently differing from the model they are compared against.
                mu_dose = _priors.beta_mech_prior().to_pymc("mu_dose")
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
                # Match build_dose_response_model's dose-slope prior (beta_mech).
                beta_dose = _priors.beta_mech_prior().to_pymc("beta_dose")
                eta_full = eta_base + beta_dose * z_attend
        else:
            treated_d = pm.Data("treated", treated, dims="obs_id")
            delta = _priors.tau_prior(sigma=_tau_sigma_for(own)).to_pymc("delta")
            if use_varying_delta:
                # Treatment-effect heterogeneity as a variance component (#230 §2/§4a):
                # a per-child on-intervention slope delta_i = delta + sigma_delta * z_i
                # (non-centred — mandatory here to avoid a Neal's funnel), whose SD is
                # the reported estimand. ``delta`` stays the population-mean effect, so
                # the DiD summary and AME read it unchanged; ``sigma_delta`` answers
                # "is the treatment effect homogeneous across children?".
                sigma_delta = _priors.sigma_delta_prior().to_pymc("sigma_delta")
                v_delta = pm.Deterministic(
                    "v_delta",
                    sigma_delta * pm.Normal("v_delta_raw", 0.0, 1.0, dims="child"),
                    dims="child",
                )
                delta_i = pm.Deterministic("delta_i", delta + v_delta, dims="child")
                eta_full = eta_base + delta_i[child_idx_d] * treated_d
            else:
                eta_full = eta_base + delta * treated_d

        eta_full = pm.Deterministic("eta", eta_full, dims="obs_id")
        if likelihood == "beta_binomial":
            kappa = _scalar_prior("kappa", _priors.kappa_prior)
            beta_binomial_from_logit(
                "y_post",
                eta_full,
                n_trials=n_trials,
                kappa=kappa,
                observed=post,
                dims="obs_id",
            )
        else:  # bernoulli_offfloor: off-floor PREVALENCE Pr(post > 0); no kappa
            off_floor = (post > 0).astype(np.int64)
            pm.Bernoulli(
                "y_offfloor", logit_p=eta_full, observed=off_floor, dims="obs_id"
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
    #: The mediator's data symbol (L for LRP59, TE for LRP68, N for LRP74); the
    #: own-baseline coefficient node is ``a_{mediator_symbol}``.
    mediator_symbol: str = "L"
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
    #: Off-floor (Bernoulli) OUTCOME (#228 item 12, e.g. nonword N): the outcome leg
    #: is a Bernoulli on the off-floor indicator (post > 0) with no own-baseline term,
    #: so ``decompose`` reads no ``b_W`` and reports NIE/NDE on the off-floor
    #: risk-difference (probability) scale (``n_trials_W`` is set to 1, collapsing the
    #: ``words_*`` columns onto the risk difference). Default False = graded outcome.
    off_floor: bool = False


def _baseline_confounder_value(prepared: PreparedData, symbol: str) -> np.ndarray:
    """Baseline value column for a mediation-family confounder (#246).

    Bounded-count measures enter on their t1 logit scale (``prepared.pre_logit``);
    the revised-DAG raw confounders that are not measures — hearing (``hs`` /
    ``hs_missing``), speech (``deapp_c``), phonological memory (``erbto``) — enter
    from ``prepared.covariates`` (already standardised, taken from the t1 pre-row in
    the ITT phase, so treatment-unaffected as the cross-world assumption requires).
    """
    if symbol in prepared.pre_logit:
        return prepared.pre_logit[symbol]
    return prepared.covariates[symbol]


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
    outcome_kind: str = "beta_binomial",
):
    """Shared outcome leg for the single-mediator-design factories.

    Both LRP59 (:func:`build_mediation_model`) and LRP62
    (:func:`_build_route_composite_model`) regress ``logit(W_t2)`` on treatment,
    the standardised post mediator and its ``G`` interaction, baseline word
    reading, age, and the baseline confounders — identical save for
    ``mediator_node`` (``z_med`` for LRP59, the route composite for LRP62). Must
    be called inside an open ``pm.Model`` context so the nodes register.

    ``outcome_kind="beta_binomial"`` (default) fits the graded post count and is
    **byte-identical** to the original build. ``"bernoulli_offfloor"`` (#228 item 12,
    a heavily-floored outcome such as nonword N) instead fits a Bernoulli on the
    off-floor indicator ``post > 0`` (node ``y_offfloor``, no ``kappa_Y``) and
    **drops the own-baseline term** ``b_W * W1`` — mirroring the off-floor ITT / DiD /
    gain-factor convention (the ``Normal(1, ·)`` autoregressive prior does not
    transfer to a binary indicator, and a floored baseline logit is degenerate). In
    that case ``W1_d`` is unused (may be ``None``).
    """
    off_floor = outcome_kind == "bernoulli_offfloor"
    b0 = _priors.alpha_prior().to_pymc("b0")
    b_G = _priors.tau_prior().to_pymc("b_G")
    b_M = _priors.b_path_prior().to_pymc("b_M")
    b_GM = _priors.gamma_cross_prior().to_pymc("b_GM")
    if not off_floor:
        # Own-baseline coefficient — created before b_A so the graded path's free-RV
        # order (and therefore its sampling) is byte-identical to the original.
        b_W = _priors.gamma_own_prior().to_pymc("b_W")
    b_A = _priors.gamma_cross_prior().to_pymc("b_A")
    if off_floor:
        eta_Y = (
            b0
            + b_G * G_d
            + b_M * mediator_node
            + b_GM * (G_d * mediator_node)
            + b_A * A_d
        )
    else:
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
    if off_floor:
        off = (np.asarray(W2_count) > 0).astype(np.int64)
        return pm.Bernoulli("y_offfloor", logit_p=eta_Y, observed=off, dims="obs_id")
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
    outcome_kind: str = "beta_binomial",
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

    The resulting NDE/NIE are **not identified natural effects**: beyond the
    latent-general-ability confounding of the mediator->outcome path, dose ``IS``
    is a treatment-induced (exposure-induced) mediator-outcome confounder, so the
    decomposition is model-based under stated (cross-world) assumptions. An
    interventional estimand (``decompose(..., interventional=True)``; LRP78) drops
    the cross-world requirement and so escapes the ``IS`` obstacle, but it still
    assumes no unmeasured mediator-outcome confounding, which latent general
    ability violates here — a weaker-assumption target, not an identified one. See
    the :mod:`mediation` module docstring and the report assumptions sections.

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
    if outcome_kind not in ("beta_binomial", "bernoulli_offfloor"):
        raise ValueError(f"Unknown outcome_kind {outcome_kind!r}")
    off_floor = outcome_kind == "bernoulli_offfloor"
    # The mediator always needs its baseline (own-baseline coupling a_M). The outcome
    # needs its baseline only for the graded own-baseline term b_W — an off-floor
    # (Bernoulli) outcome drops it (see _build_outcome_leg), so its pre-score is not
    # required and a degenerate/floored baseline logit never enters.
    required_pre = (mediator_symbol,) if off_floor else (mediator_symbol, outcome_symbol)
    for s in required_pre:
        if s not in prepared.pre_logit:
            raise KeyError(f"Symbol {s!r} missing from prepared data")
    if outcome_symbol not in prepared.post_counts:
        raise KeyError(f"Outcome {outcome_symbol!r} post-count missing from prepared data")
    for s in confounder_symbols:
        if s not in prepared.pre_logit and s not in prepared.covariates:
            raise KeyError(f"Confounder {s!r} not in prepared pre_logit or covariates")

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
    # Off-floor outcome: no own-baseline term, so its pre-score is neither required
    # nor used (a floored baseline logit would be degenerate). Zeros are a safe,
    # never-referenced placeholder for the row-aligned MediationData field.
    W1 = np.zeros(prepared.n_obs) if off_floor else prepared.pre_logit[outcome_symbol]
    conf_logit = {
        s: _baseline_confounder_value(prepared, s) for s in confounder_symbols
    }

    coords = {"obs_id": np.arange(prepared.n_obs)}
    G_f = prepared.G.astype(float)

    with pm.Model(coords=coords) as model:
        G_d = pm.Data("G", G_f, dims="obs_id")
        # Mediator baseline / own-baseline coef / likelihood are parameterised by
        # ``mediator_symbol`` so a non-L mediator (LRP68 TE, LRP74 N) gets correctly
        # labelled nodes; when mediator_symbol == 'L' every name is byte-identical to
        # the original LRP59 build.
        L1_d = pm.Data(f"{mediator_symbol}_pre_logit", L1, dims="obs_id")
        # Outcome own-baseline data node only for the graded path; the off-floor
        # outcome leg drops the b_W term, so no W_pre_logit node is created.
        W1_d = None if off_floor else pm.Data("W_pre_logit", W1, dims="obs_id")
        A_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
        conf_d = {
            s: pm.Data(f"{s}_pre_logit", conf_logit[s], dims="obs_id")
            for s in confounder_symbols
        }
        z_med_d = pm.Data("z_med", z_med, dims="obs_id")

        # --- Mediator model: logit(mediator_t2) ---
        a0 = _priors.alpha_prior().to_pymc("a0")
        a_G = _priors.tau_prior().to_pymc("a_G")
        a_L = _priors.gamma_own_prior().to_pymc(f"a_{mediator_symbol}")
        a_A = _priors.gamma_cross_prior().to_pymc("a_A")
        mu_M = a0 + a_G * G_d + a_L * L1_d + a_A * A_d
        for s in confounder_symbols:
            a_c = _priors.gamma_cross_prior().to_pymc(f"a_{s}")
            mu_M = mu_M + a_c * conf_d[s]
        mu_M = pm.Deterministic("mu_M", mu_M, dims="obs_id")
        kappa_M = _priors.kappa_prior().to_pymc("kappa_M")
        beta_binomial_from_logit(
            f"{mediator_symbol}_post", mu_M, n_trials=N_med, kappa=kappa_M,
            observed=L2_count, dims="obs_id",
        )

        # --- Outcome model: logit(W_t2) (graded) or logit P(off-floor) (Bernoulli) ---
        _build_outcome_leg(
            mediator_node=z_med_d,
            G_d=G_d,
            W1_d=W1_d,
            A_d=A_d,
            conf_d=conf_d,
            confounder_symbols=confounder_symbols,
            N_out=N_out,
            W2_count=W2_count,
            outcome_kind=outcome_kind,
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
        # Off-floor outcome: n_trials_W = 1 so decompose's ``words_* = prob_* · N_W``
        # collapses onto the off-floor risk difference (the outcome is the binary
        # off-floor indicator, reported on the probability scale).
        n_trials_W=(1 if off_floor else int(N_out)),
        med_mean=float(med_scaler.mean),
        med_sd=float(med_scaler.sd),
        mediator_symbol=mediator_symbol,
        off_floor=off_floor,
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
    for s in (outcome_symbol, *route_symbols):
        if s not in prepared.pre_logit:
            raise KeyError(f"Symbol {s!r} missing from prepared data")
    for s in confounder_symbols:
        if s not in prepared.pre_logit and s not in prepared.covariates:
            raise KeyError(f"Confounder {s!r} not in prepared pre_logit or covariates")

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
    conf_logit = {
        s: _baseline_confounder_value(prepared, s) for s in confounder_symbols
    }

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
    #: Sequential code route (LRP75): the second mediator regresses on post-L, so
    #: the g-formula must draw it conditional on the simulated first mediator.
    chain: bool = False


def build_two_mediator_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str = "W",
    mediator_symbols: tuple[str, str] = ("L", "E"),
    confounder_symbols: Iterable[str] = ("R",),
    chain: bool = False,
) -> tuple[BuiltModel, TwoMediatorData]:
    """Joint two-mediator + outcome model for the ITT-phase decomposition (LRP64).

    Generalises :func:`build_mediation_model` to **two named count mediators** so
    the word-reading effect can be split into a path via letter-sound knowledge, a
    path via a second mediator (expressive vocabulary ``E`` in LRP64, phoneme
    blending ``B`` in LRP66), and a direct/residual path. The first leg is fixed to
    ``L``; the second is parameterised by ``mediator_symbols[1]``. Three
    Beta-Binomial legs share the randomised treatment ``G`` and a baseline-covariate
    adjustment::

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
    # The FIRST mediator leg's node/coefficient names are hard-coded to L
    # (L_pre_logit, z_L, aL_*, b_L, b_GL, L_post); the SECOND leg is
    # parameterised by its symbol ``mE`` ({mE}_pre_logit, z_{mE}, a{mE}_*,
    # b_{mE}, b_G{mE}, {mE}_post, kappa_{mE}). When mE == 'E' every generated
    # name is byte-identical to the original LRP64 build, so ('L', 'E') is
    # unchanged; ('L', 'B') etc. get correctly-labelled second-leg variables.
    if mL != "L":
        raise NotImplementedError(
            "build_two_mediator_model hard-codes the first leg to L; "
            f"mediator_symbols[0] must be 'L', got {mediator_symbols!r}"
        )
    for s in (outcome_symbol, mL, mE):
        if s not in prepared.pre_logit:
            raise KeyError(f"Symbol {s!r} missing from prepared data")
    for s in confounder_symbols:
        if s not in prepared.pre_logit and s not in prepared.covariates:
            raise KeyError(f"Confounder {s!r} not in prepared pre_logit or covariates")

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
    conf_logit = {
        s: _baseline_confounder_value(prepared, s) for s in confounder_symbols
    }

    coords = {"obs_id": np.arange(prepared.n_obs)}
    G_f = prepared.G.astype(float)

    with pm.Model(coords=coords) as model:
        G_d = pm.Data("G", G_f, dims="obs_id")
        A_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
        L1_d = pm.Data("L_pre_logit", L1, dims="obs_id")
        E1_d = pm.Data(f"{mE}_pre_logit", E1, dims="obs_id")
        W1_d = pm.Data("W_pre_logit", W1, dims="obs_id")
        conf_d = {
            s: pm.Data(f"{s}_pre_logit", conf_logit[s], dims="obs_id")
            for s in confounder_symbols
        }
        zL_d = pm.Data("z_L", zL, dims="obs_id")
        zE_d = pm.Data(f"z_{mE}", zE, dims="obs_id")

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

        # --- Mediator 2 (``mE``; expressive vocabulary in LRP64, blending in LRP66) ---
        aE0 = _priors.alpha_prior().to_pymc(f"a{mE}0")
        aE_G = _priors.tau_prior().to_pymc(f"a{mE}_G")
        aE_E = _priors.gamma_own_prior().to_pymc(f"a{mE}_{mE}")
        aE_A = _priors.gamma_cross_prior().to_pymc(f"a{mE}_A")
        mu_E = aE0 + aE_G * G_d + aE_E * E1_d + aE_A * A_d
        for s in confounder_symbols:
            aE_c = _priors.gamma_cross_prior().to_pymc(f"a{mE}_{s}")
            mu_E = mu_E + aE_c * conf_d[s]
        if chain:
            # Sequential code route (LRP75): the second mediator is downstream of
            # the first (L -> B), so post-L (``z_L``) enters the mE leg. The
            # coefficient a{mE}_L is the L->B coupling; the g-formula then draws the
            # second mediator conditional on the *simulated* L.
            aE_L = _priors.gamma_cross_prior().to_pymc(f"a{mE}_{mL}")
            mu_E = mu_E + aE_L * zL_d
        mu_E = pm.Deterministic(f"mu_{mE}", mu_E, dims="obs_id")
        kappa_E = _priors.kappa_prior().to_pymc(f"kappa_{mE}")
        beta_binomial_from_logit(
            f"{mE}_post", mu_E, n_trials=N_E, kappa=kappa_E, observed=E2, dims="obs_id"
        )

        # --- Outcome W ---
        b0 = _priors.alpha_prior().to_pymc("b0")
        b_G = _priors.tau_prior().to_pymc("b_G")
        b_L = _priors.b_path_prior().to_pymc("b_L")
        b_E = _priors.b_path_prior().to_pymc(f"b_{mE}")
        b_GL = _priors.gamma_cross_prior().to_pymc("b_GL")
        b_GE = _priors.gamma_cross_prior().to_pymc(f"b_G{mE}")
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
        chain=chain,
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
    predictor_slope_sigma: float = 0.3,
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
        alpha = _priors.alpha_prior(
            sigma=_alpha_sigma_for(outcome_symbol)
        ).to_pymc("alpha")
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
# LRP-CA: concurrent conditional-associations family (#312, workstream #314)
#
# Per-wave, cross-sectional, between-child Beta-Binomial regression of a focal
# outcome's LEVEL on the standardised same-wave logits of a set of predictor
# skills, plus age and a non-interpretable group nuisance term. One row per child
# at a single wave: no own-baseline (this is a level, not a gain), no child random
# intercept (one row per child, so the coefficients are genuinely between-child
# associations — a random intercept would tilt them toward the within-child
# question, as in the LRP65 note). The pipeline fits it once per wave and reports
# the four waves side by side. EVERY coefficient is an adjusted association — the
# family makes NO causal claim, so conditioning on post-treatment skill levels is
# intentional and licensed (contrast the level-factors family, which excludes
# cross-skill terms precisely to protect a causal group×time contrast).
# ---------------------------------------------------------------------------


def build_concurrent_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str = "W",
    predictor_symbols: Iterable[str] = ("L", "B", "TR", "TE", "R", "E"),
    include_age: bool = True,
    include_group: bool = True,
    predictor_slope_sigma: float = 0.3,
) -> BuiltModel:
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
    inputs and the associations are a descriptive read; a predictor's realised
    variance shrinks with its missingness, biasing that coefficient toward zero (the
    report flags this, as in the horseshoe level model). Rows missing the focal
    OUTCOME are dropped by the caller (an outcome cannot be imputed).

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
    structural_factors: tuple[str, ...] | None = None,
    use_group: bool = False,
    use_age: bool = True,
    # The ORIGINAL priors: TruncatedNormal(mu=0, sigma=1, lower=0) IS HalfNormal(1),
    # so these defaults reproduce the pre-#261 prior exactly while keeping the more
    # general TruncatedNormal parameterisation available to the sensitivity model.
    # An earlier revision of #261 recalibrated these to (0.6, 0.5) / 0.5 alongside
    # the marginalisation. The 2x2 ablation (LRPMM101; see
    # notes/202607101638-mm-001-convergence-reparameterisation.md) showed that the
    # recalibration is neither necessary nor sufficient for convergence — raising
    # target_accept is what clears the gate — while it does move the prior-implied
    # median communality from 0.50 to 0.79. With two indicators per factor at
    # n ~ 51 that is a real and unnecessary prior commitment, so the defaults revert.
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

    The priors are the model's **original** ones: ``lambda ~ HalfNormal(1)`` (written
    as ``TruncatedNormal(mu=0, sigma=1, lower=0)``, which is the same distribution)
    and ``sigma ~ HalfNormal(1)``. ``loading_mu`` / ``loading_sigma`` /
    ``residual_sigma`` exist so a prior-sensitivity companion (LRPMM101) can vary
    them; a 2x2 ablation over {old, recalibrated} priors x {0.95, 0.999}
    ``target_accept`` showed the priors are neither necessary nor sufficient for
    convergence and do not move the posterior, so they are not used to buy
    convergence here. See
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
        _, corr, _ = pm.LKJCholeskyCov(
            "factor_cov",
            n=D,
            eta=lkj_eta,
            sd_dist=pm.Exponential.dist(1.0, size=D),
            compute_corr=True,
        )
        pm.Deterministic("factor_corr", corr, dims=("domain", "domain_b"))

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

        # Free per-indicator loading and residual RVs. Defaults reproduce the
        # original HalfNormal(1) priors; the TruncatedNormal form only exists so a
        # prior-sensitivity companion can shift the loading mode off zero.
        #
        # NB the earlier claim that a HalfNormal(residual_sigma=0.5) "caps the
        # residual SD below the unit total variance of a standardised indicator" was
        # wrong: a HalfNormal has unbounded support and merely makes sigma > 1
        # unlikely (~5% of prior mass). No cap is imposed, or needed.
        lam = pm.TruncatedNormal(
            "lambda_load",
            mu=loading_mu,
            sigma=loading_sigma,
            lower=0.0,
            dims="indicator",
        )
        sigma_ind = pm.HalfNormal(
            "sigma_indicator", sigma=residual_sigma, dims="indicator"
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
        z_factor = pm.Normal("factor_z", 0.0, 1.0, dims=("obs_id", "domain"))
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
        if structural_factors is None:
            beta_factor = pm.Normal(
                "beta_factor", 0.0, predictor_slope_sigma, dims="domain"
            )
            struct = pm.math.dot(factors, beta_factor)
        else:
            _sidx = [domain_names.index(d) for d in structural_factors]
            beta_factor = pm.Normal(
                "beta_factor", 0.0, predictor_slope_sigma, dims="struct_domain"
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
# as for the ITT suite's lrp-rli-itt-013). Attendance / dose (``IS``) is a DAG collider
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
    adjust_for: Iterable[str] = (),
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

    ``adjust_for`` (default ()): revised-DAG confounders that are not bounded-count
    measures and so cannot enter via ``skill_symbols`` — hearing status (``hs`` /
    ``hs_missing``), speech production (``deapp_c`` / ``deapp_c_missing``) and
    phonological memory (``erbto`` / ``erbto_missing``) (#247). Each must be a key in
    ``prepared.covariates`` (the pipeline requests them via ``covariates=`` and
    standardises the continuous ones / adds missing-indicators). They enter as linear
    ``gamma_{c}`` terms with the regularising cross-coupling prior, exactly as in
    ``build_mechanism_model`` (#245/#258) — reused, not duplicated. These are
    exogenous, non-treatment-affected confounders (``IG`` has no edge to ``HS``,
    ``SP`` or ``RW``), so conditioning on them does not block the randomised
    ``beta_trt`` contrast; like every non-causal term they are adjusted associations.
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
    adjust_for = tuple(adjust_for)
    for c in adjust_for:
        if c not in prepared.covariates:
            raise KeyError(f"Adjuster covariate {c!r} not loaded in prepared data")

    valid_terms = {"trt", "age", "own", *skill_symbols}
    if ability_covariate is not None:
        valid_terms.add("ability")
    interactions = tuple(tuple(p) for p in interactions)
    for pair in interactions:
        for k in pair:
            if k not in valid_terms:
                raise KeyError(f"interaction term {k!r} not available; have {sorted(valid_terms)}")

    # Drop rows missing the outcome post, the own baseline, any skill baseline, or
    # any raw-covariate adjuster.
    keep = ~np.isnan(prepared.post_counts[own]) & ~np.isnan(prepared.pre_logit[own])
    for s in skill_symbols:
        keep = keep & ~np.isnan(prepared.pre_logit[s])
    for c in adjust_for:
        keep = keep & ~np.isnan(prepared.covariates[c])
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

    # Standardise the interaction-term components on the *kept* rows (used for the
    # interaction products and AME moderators). Main effects are entered on their
    # natural scales (raw logit baselines; age uses ``prepared.A_std``).
    # Re-standardise the ability covariate here too: treated_only (…b) variants drop
    # the untreated period-1 rows, so the load-time scaler (over all periods) would
    # otherwise mislabel the “per 1 SD” unit for the treated-only fit.
    # Age is intentionally NOT re-standardised: it is deliberately kept on the shared
    # load-time (all-period) scale so ``gamma_A`` and the age-moderation unit stay
    # directly comparable between each treated-only (…b) variant and its full
    # sibling — the small age-distribution shift from dropping period-1 rows is
    # accepted in exchange for that cross-variant comparability (issue #273).
    term_vecs: dict[str, np.ndarray] = {"trt": trt, "age": prepared.A_std}
    if ability_covariate is not None:
        term_vecs["ability"], _ = standardise(prepared.covariates[ability_covariate])
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
        # Own baseline is a precision term for the graded likelihood only; the off-floor
        # (Bernoulli) path drops it (A4 — see below), so its data node is not built.
        own_pre_d = (
            pm.Data("own_pre_logit", prepared.pre_logit[own], dims="obs_id")
            if likelihood != "bernoulli_offfloor"
            else None
        )
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
        adjust_d = {
            c: pm.Data(f"{c}_adj", prepared.covariates[c], dims="obs_id")
            for c in adjust_for
        }
        int_d = {
            pair: pm.Data(f"int_{pair[0]}_{pair[1]}", _interaction_product(term_vecs, *pair), dims="obs_id")
            for pair in active_interactions
        }

        alpha = _priors.alpha_prior(
            sigma=_alpha_sigma_for(outcome_symbol)
        ).to_pymc("alpha")
        alpha_phase = pm.Normal("alpha_phase", mu=0.0, sigma=0.5, dims="phase")
        gamma_A = _priors.gamma_age_prior().to_pymc("gamma_A")

        eta = alpha + alpha_phase[phase_d] + gamma_A * A_std_d
        # Own-baseline precision term — dropped on the off-floor (Bernoulli) path (A4,
        # 2026-07-13): its Normal(1, 0.25) "post tracks pre 1:1" prior is calibrated to
        # graded test-retest reliability and does not transfer to the binary off-floor
        # indicator, and at the period-1 post baseline it would move the reported
        # off-floor risk-difference operating point through a treatment-affected value.
        # The ITT floored specs (use_own_baseline=False) and the DiD off-floor branch
        # drop it for the same reason.
        if own_pre_d is not None:
            gamma_own = _priors.gamma_own_prior().to_pymc("gamma_own")
            eta = eta + gamma_own * own_pre_d

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
        # Raw-covariate adjusters (revised-DAG confounders that are not bounded-count
        # measures): hearing (hs/hs_missing), speech (deapp_c), phonological memory
        # (erbto). Linear gamma terms, mirroring build_mechanism_model's adjust_for
        # path (#245/#258, #247).
        for c in adjust_for:
            gamma_c = _priors.gamma_cross_prior().to_pymc(f"gamma_{c}")
            eta = eta + gamma_c * adjust_d[c]
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

    # Expose the treatment×covariate interaction moderators so the pipeline's
    # average-marginal-effect report can net out the *full* per-row treatment
    # contribution ``beta_trt + Σ_k gamma_int_trt_k · z_k`` — not just
    # ``beta_trt`` (issue: gain-family AME ignored the fitted trt interactions).
    # Each entry is ``(gamma_int coefficient name, standardised moderator vector)``
    # for a fitted interaction with ``trt`` as one member; the moderator is the
    # *other* member's term vector, exactly as multiplied into ``eta``. Only
    # populated when the treatment term is present (``include_trt``).
    trt_moderators: list[tuple[str, np.ndarray]] = []
    if include_trt:
        for pair in active_interactions:
            if "trt" not in pair:
                continue
            other = pair[0] if pair[1] == "trt" else pair[1]
            trt_moderators.append(
                (f"gamma_int_{pair[0]}_{pair[1]}", np.asarray(term_vecs[other], dtype=float))
            )

    return BuiltModel(
        model=model,
        variables=_variables_dict(model),
        prepared=prepared,
        extras={"trt_interaction_moderators": trt_moderators},
    )


def build_level_factors_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str,
    ability_covariate: str | None = None,
    adjust_for: Iterable[str] = (),
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

    ``adjust_for`` (default ()): revised-DAG confounders that are not bounded-count
    measures — hearing status (``hs`` / ``hs_missing``), speech production
    (``deapp_c`` / ``deapp_c_missing``) and phonological memory (``erbto`` /
    ``erbto_missing``) (#247). Each enters as a linear ``gamma_{c}`` term with the
    cross-coupling prior, reusing ``build_mechanism_model``'s idiom. These are
    exogenous, **non**-treatment-affected roots/upstream nodes (``IG`` has no edge to
    ``HS``/``SP``/``RW``), so they do not sit on the causal path from group and their
    adjustment does not block the randomised t2 contrast. Note the level model takes
    **no** measure-skill adjusters (unlike the gain factory's ``skill_symbols``): a
    levels model conditioning on another evolving skill's *contemporaneous* level
    would condition on a post-treatment mediator/collider and bias the group×time
    trajectory it exists to estimate.
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
    adjust_for = tuple(adjust_for)
    for c in adjust_for:
        if c not in prepared.covariates:
            raise KeyError(f"Adjuster covariate {c!r} not loaded in prepared data")

    keep = ~np.isnan(prepared.post_counts[own])
    for c in adjust_for:
        keep = keep & ~np.isnan(prepared.covariates[c])
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
        adjust_d = {
            c: pm.Data(f"{c}_adj", prepared.covariates[c], dims="obs_id")
            for c in adjust_for
        }

        # Level factors is own-baseline-free (a level model), so unlike the
        # growth/LCSM mean-anchor rationale ``alpha`` is deliberately kept on the
        # zero-centred prior: here the per-timepoint ``alpha_time`` vector carries
        # the absolute level at each wave, leaving ``alpha`` as a small global
        # offset for which the zero-centred prior is appropriate (issue #273).
        alpha = _priors.alpha_prior(
            sigma=_alpha_sigma_for(outcome_symbol)
        ).to_pymc("alpha")
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

        # Raw-covariate adjusters (revised-DAG exogenous confounders HS/SP/RW): linear
        # gamma terms, mirroring build_mechanism_model's adjust_for path (#247).
        for c in adjust_for:
            gamma_c = _priors.gamma_cross_prior().to_pymc(f"gamma_{c}")
            eta = eta + gamma_c * adjust_d[c]

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


def build_block_exposure_model(
    prepared: PreparedData,
    *,
    outcome_symbol: str,
    ability_covariate: str | None = None,
    adjust_for: Iterable[str] = (),
    use_child_re: bool = True,
    likelihood: str = "beta_binomial",
    sigma_child_prior_sigma: float = 0.5,
) -> BuiltModel:
    """Block-2 taught-vocabulary staggered block-active exposure model (LRPBX, #228 item 5).

    Block 2 of the taught-vocabulary programme (TE2/TR2 taught, UE2/UR2 not-taught)
    is introduced in phase 2, so it has **no t1 baseline and no randomised
    contrast**. Identification borrows the *staggered* block-2 teaching: the
    immediate arm is taught block 2 in phase 2 (measured at t3) while the wait-list
    arm is still on block 1; the wait-list arm reaches block 2 in phase 3 (measured
    at t4). This is a staggered-adoption / event-study design over the per-timepoint
    levels frame (``phase_mode="levels"``; t1 rows self-drop, block 2 being NaN
    there). Linear predictor (logit scale):

        eta = alpha + alpha_time[t]              # secular trend, per timepoint
            + delta * exposed_{i,t}              # <-- focal block-active exposure effect
            + gamma_A * A_std_t                  # age at t (precision)
            [ + gamma_ability * z(ability) ]     # cognitive-ability precision
            [ + sum_c gamma_c * z(c) ]           # revised-DAG adjusters (hs / erbto / deapp_c)
            [ + u_child_i ]                       # each child their own control

    where ``exposed_{i,t} = 1`` once child *i* has been taught block 2 by timepoint
    *t* — immediate arm (``G == 1``) from t3 (phase >= 2), wait-list arm (``G == 0``)
    from t4 (phase >= 3). The switch-on differs between arms only at t3 (immediate
    exposed, wait-list not), which is the identifying cell; both are exposed by t4
    and neither at t2.

    **Estimand and status.** ``delta`` is the shift in the logit of the block-2
    taught-vocabulary level attributable to block 2 being *actively taught*. It is
    an **association (parallel-trends)**, not a randomised effect: it is causal only
    if block-2 trajectories would have been parallel across arms absent block-2
    teaching. Over t2->t3 the wait-list arm is on **block 1**, not idle, so the
    contrast is "block-2-active *vs block-1-active*", never treated-vs-untreated.
    The child random intercept absorbs stable between-child (hence between-arm)
    differences and ``alpha_time`` absorbs the shared maturation trend; what remains
    confounded is age-at-block-2 (immediate children reach block 2 younger; linear
    ``gamma_A`` is only a precision term) and any arm-specific slope difference.
    Report ``delta`` accordingly (``Status.ASSOCIATION``). The taught outcomes
    (TE2/TR2) should show a positive ``delta`` and the not-taught comparators
    (UE2/UR2) should not — a within-family placebo / fidelity check.

    Structural siblings: the levels frame, ``alpha_time`` and the child intercept
    are the level-factors model's (``build_level_factors_model``); the ``eta_base``
    + ``delta * exposed`` split and the ``tau_prior(sigma=_tau_sigma_for(...))``
    effect prior are the DiD model's (``build_did_model``). ``delta`` carries the
    treatment-tier prior (it is the focal effect, mirroring the DiD ``delta`` /
    level-factors group term), not the tighter ``gamma_cross`` adjuster prior.

    ``adjust_for`` (default ()): revised-DAG confounders that are not bounded-count
    measures — hearing status (``hs`` / ``hs_missing``), speech production
    (``deapp_c`` / ``deapp_c_missing``, expressive outcomes) and phonological memory
    (``erbto`` / ``erbto_missing``) — each a linear ``gamma_{c}`` term with the
    cross-coupling prior, mirroring the gain/level-factor adjuster path (#247). No
    measure-skill adjusters and no own-baseline term: a levels model conditioning on
    a contemporaneous evolving skill would condition on a post-treatment
    mediator/collider, and a block-2 own-baseline does not exist (NaN at t1).
    """
    if prepared.phase_mode != "levels":
        raise ValueError("build_block_exposure_model requires phase_mode='levels'")
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
    adjust_for = tuple(adjust_for)
    for c in adjust_for:
        if c not in prepared.covariates:
            raise KeyError(f"Adjuster covariate {c!r} not loaded in prepared data")

    keep = ~np.isnan(prepared.post_counts[own])
    for c in adjust_for:
        keep = keep & ~np.isnan(prepared.covariates[c])
    prepared = _subset(prepared, keep)

    post = prepared.post_counts[own].astype(np.int64)
    # Staggered block-2 exposure: immediate arm (G==1) taught block 2 from t3
    # (phase >= 2), wait-list arm (G==0) from t4 (phase >= 3). Derived from the
    # design (G, phase), like build_did_model's ``treated``.
    exposed = (
        ((prepared.G == 1) & (prepared.phase >= 2))
        | ((prepared.G == 0) & (prepared.phase >= 3))
    ).astype(float)
    ability = prepared.covariates[ability_covariate] if ability_covariate is not None else None

    coords = {
        "obs_id": np.arange(prepared.n_obs),
        "phase": np.arange(prepared.n_phases),
        "child": np.arange(prepared.n_children),
    }
    with pm.Model(coords=coords) as model:
        phase_d = pm.Data("phase_idx", prepared.phase.astype(np.int64), dims="obs_id")
        A_std_d = pm.Data("A_std", prepared.A_std, dims="obs_id")
        exposed_d = pm.Data("exposed", exposed, dims="obs_id")
        adjust_d = {
            c: pm.Data(f"{c}_adj", prepared.covariates[c], dims="obs_id")
            for c in adjust_for
        }

        # Own-baseline-free level model: the per-timepoint ``alpha_time`` carries the
        # absolute level at each wave, leaving ``alpha`` a small zero-centred offset
        # (matches build_level_factors_model).
        alpha = _priors.alpha_prior(
            sigma=_alpha_sigma_for(outcome_symbol)
        ).to_pymc("alpha")
        alpha_time = pm.Normal("alpha_time", mu=0.0, sigma=0.5, dims="phase")
        gamma_A = _priors.gamma_age_prior().to_pymc("gamma_A")
        eta = alpha + alpha_time[phase_d] + gamma_A * A_std_d

        if ability is not None:
            ability_d = pm.Data(f"{ability_covariate}_std", ability, dims="obs_id")
            gamma_ability = _priors.gamma_cross_prior().to_pymc("gamma_ability")
            eta = eta + gamma_ability * ability_d

        # Raw-covariate adjusters (revised-DAG exogenous confounders HS/SP/RW): linear
        # gamma terms, mirroring the gain/level-factor adjuster path (#247).
        for c in adjust_for:
            gamma_c = _priors.gamma_cross_prior().to_pymc(f"gamma_{c}")
            eta = eta + gamma_c * adjust_d[c]

        if use_child_re:
            child_idx_d = pm.Data(
                "child_idx", prepared.child_idx.astype(np.int64), dims="obs_id"
            )
            eta = _add_child_random_intercept(
                eta, child_idx_d, sigma_prior_sigma=sigma_child_prior_sigma
            )

        # Linear predictor without the exposure term, so the pipeline can read the
        # un-exposed baseline for the average-marginal-effect translation (as DiD).
        eta_base = pm.Deterministic("eta_base", eta, dims="obs_id")
        delta = _priors.tau_prior(sigma=_tau_sigma_for(outcome_symbol)).to_pymc("delta")
        eta_full = pm.Deterministic("eta", eta_base + delta * exposed_d, dims="obs_id")

        if likelihood == "beta_binomial":
            kappa = _scalar_prior("kappa", _priors.kappa_prior)
            beta_binomial_from_logit(
                "y_post", eta_full, n_trials=prepared.n_trials[own], kappa=kappa,
                observed=post, dims="obs_id",
            )
        else:  # bernoulli_offfloor (not expected for block-2 taught; kept for parity)
            pm.Bernoulli(
                "y_offfloor", logit_p=eta_full,
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
    # Enter the own baseline on the *raw* logit scale, like every sibling factory
    # (ITT, mechanism, gain/level-factors, DiD): the ``gamma_own ~ Normal(1, 0.5)``
    # prior encodes "logit-post ≈ logit-pre" (a slope near 1 in logit units), which
    # only holds on the raw logit scale. Standardising the baseline here (as before)
    # left that prior mean of 1 meaning "1 logit per SD of baseline logit" — an
    # unintended, measure-dependent prior for this precision term.
    own_pre_logit = prepared.pre_logit[own]

    coords = {"obs_id": np.arange(prepared.n_obs)}
    with pm.Model(coords=coords) as model:
        own_pre_d = pm.Data("own_pre_logit", own_pre_logit, dims="obs_id")
        A_std_d = pm.Data("A_std", prepared.A_std, dims="obs_id")

        alpha = _priors.alpha_prior(
            sigma=_alpha_sigma_for(outcome_symbol)
        ).to_pymc("alpha")
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
    couplings: dict[str, tuple[str, ...]] | None = None,
    arm_window_intercepts: bool = False,
    covariate_block: tuple[str, ...] = (),
    covariate_targets: tuple[str, ...] = (),
    coupling_prior_sigma: float = 0.3,
    self_prior_mu: float = -0.3,
    self_prior_sigma: float = 0.2,
    intercept_prior_sigma: float = 1.5,
    covariate_prior_sigma: float = 0.3,
    use_process_noise: bool = True,
    shared_process_noise: bool = False,
    sigma_proc_prior_sigma: float = 0.5,
    sigma_init_prior_sigma: float = 1.0,
    kappa_prior_sigma: float = 50.0,
) -> BuiltModel:
    """Full coupled latent change-score model (LRP67 + the lagged suite) on the logit scale.

    A latent logit true-score ``x_m[i, t]`` is modelled for each measure ``m``
    (default ``W`` reading, ``L`` letter-sounds, ``E`` expressive vocabulary),
    child ``i`` and wave ``t``. The within-child trajectory follows a McArdle
    latent change-score recursion with **process noise**::

        x_m[i, 1] = mu1_m + sigma1_m * z1_m[i]                      (non-centred)
        x_m[i, t] = x_m[i, t-1] + Delta_m[i, t]
        Delta_m[i, t] = mean_Delta_m[i, t] + sigma_proc_m * zproc_m[i, t]

    ``couplings`` maps each **target** measure to the source measures whose
    prior-wave levels enter its change equation (prior *level* -> subsequent
    *change*, pooled over transitions). The default — ``None`` — reproduces
    LRP67 exactly: every non-``reading_symbol`` measure couples into the
    reading change::

        mean_Delta_W = a_W + b_W * x_W[t-1]
                     + sum_{c != W} g_c * x_c[t-1]
                     + d_age_W * age[t-1]

    With a single target the coupling parameters keep LRP67's ``g_{src}``
    names; with multiple targets they are ``g_{src}_{tgt}`` (the lagged
    reverse-coupling models LCSM-081/082, #250). Uncoupled measures get a
    self-proportional change plus age only.

    ``arm_window_intercepts`` replaces the pooled per-measure change intercept
    with **arm x window cells** ``a_change[arm, trans, outcome]`` — required
    for any model pooling couplings across transitions, because the waitlist
    crossover makes the randomised arm a confounder of every transition-2+
    coupling (verified d-separation, ``notes/202607141030-time-lagged-model-designs.md``).
    The window-1 cell contrast is exposed as the deterministic
    ``itt_w1_contrast`` (immediate - waitlist, per outcome): a randomised
    latent ITT contrast on the change scale, reported as a consistency check
    against the ITT suite. The intervention-dose covariate stays **omitted**:
    it is the locked DAG's ``IS`` collider, so conditioning on it would reopen
    the latent-``GA`` backdoor onto the couplings (ID-3); the arm x window
    cells derive from randomised ``IG`` + design timing instead.

    ``covariate_block`` names adjuster covariates on the panel — time-invariant
    (``panel.child_covariates``, e.g. ``hs``/``hs_missing``) or per-wave
    (``panel.wave_covariates``, e.g. ``erbto``/``deapp_c`` + indicators), the
    latter read at the transition's **prior** wave (the lagged DAG's
    parents-at-the-prior-wave rule). Each covariate gets one slope
    ``b_{name}`` **shared** across the ``covariate_targets`` change equations —
    the recommended parameter-sparing default at n~54.

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
    jidx = {s: i for i, s in enumerate(OUT)}

    if couplings is None:
        couplings = {reading_symbol: tuple(s for s in OUT if s != reading_symbol)}
    couplings = {tgt: tuple(srcs) for tgt, srcs in couplings.items()}
    for tgt, srcs in couplings.items():
        unknown = [s for s in (tgt, *srcs) if s not in OUT]
        if unknown:
            raise KeyError(
                f"coupling symbols {unknown} not in panel.outcomes {OUT}"
            )
        if tgt in srcs:
            raise ValueError(
                f"target {tgt!r} cannot couple to itself (that is b_self)"
            )

    if arm_window_intercepts and panel.group is None:
        raise ValueError(
            "arm_window_intercepts=True needs a panel with a group column"
        )
    covariate_block = tuple(covariate_block)
    covariate_targets = tuple(covariate_targets)
    if bool(covariate_block) != bool(covariate_targets):
        raise ValueError(
            "covariate_block and covariate_targets must be given together"
        )
    unknown_tgt = [s for s in covariate_targets if s not in OUT]
    if unknown_tgt:
        raise KeyError(
            f"covariate_targets {unknown_tgt} not in panel.outcomes {OUT}"
        )
    cov_arrays: dict[str, np.ndarray] = {}
    cov_is_wave: dict[str, bool] = {}
    for name in covariate_block:
        if name in panel.wave_covariates:
            cov_arrays[name] = panel.wave_covariates[name]
            cov_is_wave[name] = True
        elif name in panel.child_covariates:
            cov_arrays[name] = panel.child_covariates[name]
            cov_is_wave[name] = False
        else:
            raise KeyError(
                f"covariate {name!r} not on the panel; request it via "
                "load_wave_panel(wave_covariates=..., include_hearing=...)"
            )

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
    if arm_window_intercepts:
        coords["arm"] = ["immediate", "waitlist"]
        # Row index into the arm dimension per child (group 1 -> 0, group 2 -> 1).
        arm_idx = (np.asarray(panel.group) == 2).astype(int)

    from dse_research_utils.math.constants import EPSILON  # local import

    with pm.Model(coords=coords) as model:
        age = pm.Data("age_std", panel.age_std, dims=("child", "wave"))
        cov_data: dict[str, pt.TensorVariable] = {}
        for name in covariate_block:
            dims = ("child", "wave") if cov_is_wave[name] else ("child",)
            cov_data[name] = pm.Data(f"cov_{name}", cov_arrays[name], dims=dims)

        # Structural parameters (time-invariant, pooled over transitions).
        mu1 = pm.Normal("mu1", mu=w1_anchor, sigma=1.0, dims="outcome")
        sigma1 = pm.HalfNormal("sigma1", sigma=sigma_init_prior_sigma, dims="outcome")
        if arm_window_intercepts:
            a_change = pm.Normal(
                "a_change",
                mu=0.0,
                sigma=intercept_prior_sigma,
                dims=("arm", "trans", "outcome"),
            )
            # Window-1 randomised contrast on the latent change scale
            # (immediate - waitlist), the built-in ITT-suite consistency check.
            pm.Deterministic(
                "itt_w1_contrast",
                a_change[0, 0, :] - a_change[1, 0, :],
                dims="outcome",
            )
        else:
            a_change = pm.Normal(
                "a_change", mu=0.0, sigma=intercept_prior_sigma, dims="outcome"
            )
        # Self-feedback of the proportional change-score recursion: the level AR(1)
        # coefficient is phi = 1 + b_self, so the old ``mu=0`` centred phi on a unit
        # root (random walk) with ~50% prior mass on explosive phi > 1. A
        # proportional-change LCSM instead expects mean-reversion toward an asymptote
        # (negative self-feedback), so b_self is centred at -0.3 (phi ~ 0.7) with a
        # tighter sd: Normal(-0.3, 0.2) puts ~7% mass on explosive phi > 1 (vs ~50%),
        # taming the heavy-tailed geometry that drives divergences at n~54 (review
        # finding A3, 2026-07-13). Still weakly-informative — the data can pull b_self
        # back toward 0 given signal.
        b_self = pm.Normal(
            "b_self", mu=self_prior_mu, sigma=self_prior_sigma, dims="outcome"
        )
        d_age = pm.Normal("d_age", mu=0.0, sigma=covariate_prior_sigma, dims="outcome")
        # Headline cross-couplings (prior source level -> target change). With a
        # single target the parameters keep LRP67's ``g_{src}`` names; with
        # multiple targets the target joins the name (``g_{src}_{tgt}``).
        single_target = len(couplings) == 1
        g_par: dict[tuple[str, str], pt.TensorVariable] = {}
        for tgt, srcs in couplings.items():
            for src in srcs:
                pname = f"g_{src}" if single_target else f"g_{src}_{tgt}"
                g_par[(src, tgt)] = pm.Normal(
                    pname, mu=0.0, sigma=coupling_prior_sigma
                )
        # Adjuster-covariate slopes, shared across the covariate_targets
        # equations (the parameter-sparing default at n~54).
        b_cov = {
            name: pm.Normal(f"b_{name}", mu=0.0, sigma=covariate_prior_sigma)
            for name in covariate_block
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

        # Cross-process covariance is deliberately omitted: the initial statuses
        # (z1_*) and the per-transition process noises (zproc_*) are modelled as
        # independent across the W/L/E processes. An LKJ-correlated initial-status
        # block (as the growth model uses) is not reliably estimable at n ~ 54
        # here, so this is a small-n fallback — it may attenuate the coupling
        # coefficients g_L / g_E, which is accepted and flagged (issue #273).
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
                if arm_window_intercepts:
                    m = a_change[arm_idx, k, jidx[s]] + b_self[jidx[s]] * prev[s]
                else:
                    m = a_change[jidx[s]] + b_self[jidx[s]] * prev[s]
                m = m + d_age[jidx[s]] * age[:, t - 1]
                for src in couplings.get(s, ()):
                    m = m + g_par[(src, s)] * prev[src]
                if s in covariate_targets:
                    for name in covariate_block:
                        v = cov_data[name]
                        # Per-wave states are read at the prior wave (the lagged
                        # DAG's parents-at-the-prior-wave rule).
                        m = m + b_cov[name] * (v[:, t - 1] if cov_is_wave[name] else v)
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
    age_ability_interaction: bool = False,
    intercept_prior_sigma: float = 1.5,
    slope_prior_sigma: float = 0.5,
    assoc_prior_sigma: float = 0.3,
    re_intercept_prior_sigma: float = 0.5,
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

    ``age_ability_interaction`` (LRP85, #228 item 10) adds a child-level **baseline
    (t1) age** moderator ``age0`` (standardised across children — distinct from the
    within-child ``age_std`` time axis) to the slope, with its own main effect
    ``gamma_age`` and, headline, an ``age0 × ability`` interaction ``gamma_int``:
    positive ``gamma_int_k`` = older-and-more-able children grow faster on measure k
    than age and ability predict separately (the gain factors' ``gamma_int_A_ability``
    brought onto the growth rate). Default off, so LRP69/70 are unaffected. Still an
    adjusted, GA-confounded association.

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
        if age_ability_interaction:
            # Child-level baseline (t1) age, standardised ACROSS children — distinct
            # from the within-child ``age_std`` time axis the slope multiplies. Its
            # interaction with ability is the #228 item-10 estimand: older-and-more-
            # able children grow faster than age and ability predict separately,
            # bringing the gain factors' ``gamma_int_A_ability`` onto the growth rate.
            # ``gamma_int`` is on unit-scaled age0 × unit-scaled ability, matching the
            # gain-factor interaction's scale. Missing baseline age -> 0 (the mean).
            a0 = np.asarray(panel.age_std[:, 0], dtype=float)
            # Standardise across children with the shared helper (nanstd ddof=1,
            # matching every other standardised term; it raises on a degenerate
            # zero-variance axis rather than silently falling back to sd=1 and
            # fitting a flat interaction).
            age0_z, _ = standardise(a0)
            age0_np = np.where(np.isfinite(age0_z), age0_z, 0.0)
            age0 = pm.Data("age0_std", age0_np, dims="child")
            gamma_age = pm.Normal("gamma_age", 0.0, assoc_prior_sigma, dims="outcome")
            gamma_int = pm.Normal("gamma_int", 0.0, assoc_prior_sigma, dims="outcome")
            slope_mean = (
                slope_mean
                + gamma_age[None, :] * age0[:, None]
                + gamma_int[None, :] * age0[:, None] * blocks[:, None]
            )
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
# Longitudinal correlated-domain-factor model (LRP-RLI-LCF-001, #313)
# ---------------------------------------------------------------------------


_LCF_DOMAINS: dict[str, tuple[str, ...]] = {
    "vocabulary": ("R", "E", "TR", "TE"),
    "code": ("L", "B"),
    "grammar": ("F", "T"),
}


def build_longitudinal_corr_factor_model(
    panel: WavePanel,
    *,
    domains: dict[str, tuple[str, ...]] | None = None,
    loading_sigma: float = 1.0,
    residual_sigma: float = 1.0,
    lkj_eta: float = 2.0,
    factor_mean_sigma: float = 1.0,
    trait_share_a: float = 1.5,
    trait_share_b: float = 1.5,
) -> BuiltModel:
    """Longitudinal correlated-domain-factor measurement model (LRP-RLI-LCF-001, #313).

    The four-wave extension of the cross-sectional ``corr_factor`` CFA (``mm-001``):
    correlated **vocabulary / code / grammar** domain factors measured at every
    timepoint, delivering the **per-wave latent skill correlation matrices** and the
    disattenuated latent slopes derived from them. It is the symmetric, measurement-
    error-corrected counterpart to the concurrent regression family (``ca-001``,
    #312): every quantity is a **descriptive association**, never causal.

    **Structure (scalar-invariant longitudinal CFA, fully marginalised).** For
    indicator ``j`` of domain ``d`` at wave ``t`` the standardised logit indicator is
    ``z[i,j,t] = lambda[j] * f[i,d,t] + eps[i,j,t]`` with loadings ``lambda[j]`` and
    residual SDs ``sigma[j]`` held **invariant across waves** (the factors mean the
    same thing at every t), positive loadings, and per-wave factor means carried by a
    zero-sum-over-waves ``factor_mean[d,t]`` (indicators are pooled-standardised, so
    the grand mean is removed and only the wave deviations remain). Each factor is
    unit-variance at every wave; the **within-wave factor correlation** ``factor_corr[t]``
    (per wave, LKJ) is the headline. Across-wave dependence uses a **trait/state
    decomposition** ``f = sqrt(pi_d) * trait + sqrt(1 - pi_d) * state`` — a stable
    per-child trait (cross-factor correlated, shared across waves) plus a wave-specific
    state (cross-factor correlated, independent across waves) — which is PSD by
    construction, keeps unit factor variance, lets the within-wave correlation move
    freely per wave, and induces across-wave autocorrelation equal to the trait share
    ``pi_d``. This gives compound symmetry across waves; genuine AR(1) decay is the
    first relaxation if the equal-lag assumption misfits.

    **Small-n geometry.** The measurement model is Gaussian in the factors, so the
    per-child factor scores are **marginalised out** (as in ``mm-001``): each child's
    observed indicator cells are an ``MvNormal`` whose covariance is the trait/state
    factor covariance folded through the loadings, ``Sigma_z = Lambda Sigma_f Lambda'
    + diag(sigma^2)``, sliced to that child's observed cells. There is **no** per-child
    latent RV and hence no funnel; the geometry is set by the marginalised likelihood.
    Missing cells are handled by grouping children into observed-cell **patterns** and
    fitting one ``MvNormal`` per pattern (masked, not dropped — a child missing one
    wave still contributes its other cells, which matters at n ~ 54).

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

    with pm.Model(coords=coords) as model:
        # --- Measurement parameters (wave-invariant loadings + residuals) ---
        lam = pm.TruncatedNormal(
            "lambda_load", mu=0.0, sigma=loading_sigma, lower=0.0, dims="indicator"
        )
        sigma_ind = pm.HalfNormal("sigma_indicator", sigma=residual_sigma, dims="indicator")
        pm.Deterministic(
            "communality", lam**2 / (lam**2 + sigma_ind**2), dims="indicator"
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
        pi = pm.Beta("trait_share", alpha=trait_share_a, beta=trait_share_b, dims="domain")
        L_trait = pm.LKJCorr("trait_corr_chol", n=D, eta=lkj_eta)
        corr_trait = L_trait @ L_trait.T
        pm.Deterministic("trait_corr", corr_trait, dims=("domain", "domain_b"))
        corr_state = []
        for t in range(T):
            L_s = pm.LKJCorr(f"state_corr_chol_w{waves[t]}", n=D, eta=lkj_eta)
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

        factor_mean = pm.ZeroSumNormal(
            "factor_mean", sigma=factor_mean_sigma, dims=("domain", "wave")
        )
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

    extras = {
        "z_nodes": z_nodes,
        "child_of_node": child_of_node,
        "domains": {k: list(v) for k, v in domains.items()},
        "domain_of": {ind_names[j]: domain_names[domain_of_idx[j]] for j in range(J)},
        "indicators": list(ind_names),
        "cell_names": cell_names,
        "standardisers": standardisers,
        "waves": list(waves),
        "n_children": N,
        "n_used_children": sum(len(c) for _, c in sorted_patterns),
        "invariance": "scalar",
    }
    return BuiltModel(
        model=model, variables=_variables_dict(model), prepared=panel, extras=extras
    )


# ---------------------------------------------------------------------------
# Historical group-by-wave growth (RLMHG, #165 - first non-RLI dataset)
# ---------------------------------------------------------------------------


def build_historical_growth_model(
    panel: LongitudinalPanel,
    *,
    measure: str = "basread",
    eta_prior_sigma: float = 1.5,
    sigma_subject_prior_sigma: float = 0.5,
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
    contrasts. Ported from the standalone ``lrp-rlm-hg-001`` script (#163) onto the
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
