# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Available-case modified intention-to-treat model construction.

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
    build_tau_modifier,
)
from language_reading_predictors.statistical_models.fitted_payloads import (
    IttPayload,
)
from language_reading_predictors.statistical_models.likelihood import (
    SCORE_MEAN_LINKS,
    ScoreMeanLink,
    beta_binomial_from_score_mean_link,
)
from language_reading_predictors.statistical_models.measures import (
    ITT_OUTCOMES,
)
from language_reading_predictors.statistical_models.preprocessing import (
    _subset,
    PreparedData,
    standardise,
)
from language_reading_predictors.statistical_models.factories.base import (
    BuiltModel,
    _alpha_sigma_for,
    _rlm_dispersion_kappa,
    _tau_sigma_for,
)

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
    score_mean_link: ScoreMeanLink = "logit",
    tau_moderator_symbol: str | None = None,
    tau_moderator_is_covariate: bool = False,
    tau_moderator_interaction: bool = True,
    tau_sigma: float | None = None,
    alpha_sigma: float | None = None,
    gamma_own_sigma: float | None = None,
    kappa_sigma: float | None = None,
    kappa_prior_family: str = "halfnormal_concentration",
) -> BuiltModel[IttPayload]:
    """
    Build the single-outcome available-case modified ITT model used by the
    LRPITT suite and its companions.

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
    "beta_binomial"``), using either the ordinary logit score-mean link or the
    phoneme-blending three-choice guessing-floor link, or, for the floored-outcome
    floor rule, a Bernoulli on the binary "off-floor at t2" indicator ``post > 0``
    (``likelihood="bernoulli_offfloor"``).

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
        under the locked DAG the assigned-arm coefficient identifies the
        available-case modified ITT estimate without an adjustment set, so
        cross-baselines are dropped. Every requested symbol must be in
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
        rule (#119/#341) uses ``"bernoulli_offfloor"`` for its exploratory estimand: a
        Bernoulli/logistic ``tau`` on the binary off-floor indicator
        ``post > 0`` (no ``kappa``), which targets where the randomised signal
        verifiably lives for heavily-floored outcomes.
    score_mean_link
        Inverse link for the expected score proportion. ``"logit"`` is the suite
        default. ``"three_choice_guessing_floor"`` maps the inverse logit onto
        ``[1/3, 1]`` and is valid only for the ten-item, three-alternative phoneme-
        blending outcome ``B``. It changes the mean link, not the Beta-Binomial
        observation family.
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
    gamma_own_sigma
        Override the own-baseline prior SD. ``None`` (default) uses 0.25;
        pass 0.5 for the required weak-prior sensitivity. The prior remains
        centred at one, so this varies uncertainty rather than changing the
        baseline-coupling anchor.
    kappa_sigma
        Scale of the dispersion prior. Under the default
        ``kappa_prior_family="halfnormal_concentration"`` this is the HalfNormal SD
        on ``kappa`` itself (default 50); under ``"halfnormal_inverse_sqrt"`` it is
        the HalfNormal SD on ``1 / sqrt(kappa)`` (default 0.25).
    kappa_prior_family
        ``"halfnormal_concentration"`` (default, the registered suite prior) or
        ``"halfnormal_inverse_sqrt"``. The latter exists because a HalfNormal on
        the concentration cannot reach the near-Binomial limit ``kappa >> n``,
        which for a bounded count is the ordinary hypothesis "no extra-Binomial
        dispersion". At ``n_trials = 170`` that limit needs ``kappa > 1689`` for
        variance within 10% of Binomial, and ``HalfNormal(50)`` gives it
        effectively no mass — so the registered prior *enforces* a minimum
        overdispersion (about 5.9x at its own median). Used by the dispersion
        prior-family sweep (2026-08-22 ITT audit, finding 5); the registered fits
        keep the default.
        Override the Beta-Binomial concentration prior scale. ``None`` preserves
        the shared ``HalfNormal(50)`` default; larger values expose more of the
        near-Binomial region for the required likelihood-prior sensitivity. This
        argument is unused by the Bernoulli off-floor likelihood.
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
    if score_mean_link not in SCORE_MEAN_LINKS:
        raise ValueError(
            f"score_mean_link must be one of {SCORE_MEAN_LINKS}, "
            f"got {score_mean_link!r}"
        )
    if score_mean_link == "three_choice_guessing_floor" and outcome_symbol != "B":
        raise ValueError(
            "three_choice_guessing_floor is only valid for phoneme blending (B), "
            f"got {outcome_symbol!r}"
        )
    if likelihood != "beta_binomial" and score_mean_link != "logit":
        raise ValueError(
            "score_mean_link applies only to the Beta-Binomial likelihood; "
            f"got likelihood={likelihood!r}"
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
            gamma_own_spec = (
                _priors.gamma_own_prior()
                if gamma_own_sigma is None
                else _priors.gamma_own_prior(sigma=gamma_own_sigma)
            )
            gamma_own = gamma_own_spec.to_pymc("gamma_own")
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
            if kappa_prior_family == "halfnormal_inverse_sqrt":
                # Dispersion-scale parameterisation, so the near-Binomial limit is
                # reachable (2026-08-22 ITT audit, finding 5). ``HalfNormal`` on
                # the concentration cannot get there: at n = 170 coming within
                # 10% of Binomial variance needs kappa > 1689, and HalfNormal(50)
                # gives that effectively zero mass, so the prior *enforces* a
                # minimum overdispersion of roughly 5.9x at its own median. Same
                # constructor the RLM historical families use.
                kappa = _rlm_dispersion_kappa(
                    float(_priors.inv_sqrt_kappa_prior().sigma)
                    if kappa_sigma is None
                    else kappa_sigma
                )
            elif kappa_prior_family == "halfnormal_concentration":
                kappa_spec = (
                    _priors.kappa_prior()
                    if kappa_sigma is None
                    else _priors.kappa_prior(sigma=kappa_sigma)
                )
                kappa = kappa_spec.to_pymc("kappa")
            else:
                raise ValueError(
                    "kappa_prior_family must be 'halfnormal_concentration' or "
                    f"'halfnormal_inverse_sqrt', got {kappa_prior_family!r}"
                )
            beta_binomial_from_score_mean_link(
                "y_post",
                eta,
                n_trials=prepared.n_trials[own],
                kappa=kappa,
                score_mean_link=score_mean_link,
                observed=post,
                dims="obs_id",
            )
        else:  # bernoulli_offfloor: exploratory estimand for the floor rule
            off_floor = (post > 0).astype(np.int64)
            pm.Bernoulli("y_offfloor", logit_p=eta, observed=off_floor, dims="obs_id")

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
        prepared=prepared,
        payload=IttPayload(
            tau_interaction_moderators=tuple(tau_moderators),
            score_mean_link=score_mean_link,
        ),
    )
