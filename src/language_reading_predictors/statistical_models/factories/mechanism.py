# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Mechanism dose-response model construction.

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
    from preliz.distributions.distributions import Continuous


from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.hsgp import (
    build_hsgp_1d,
)
from language_reading_predictors.statistical_models.fitted_payloads import (
    MechanismDesign,
    MechanismPayload,
)
from language_reading_predictors.statistical_models.mechanism_design import (
    validate_mechanism_design,
)
from language_reading_predictors.statistical_models.likelihood import (
    beta_binomial_from_logit,
)
from language_reading_predictors.statistical_models.preprocessing import (
    _subset,
    PreparedData,
    Standardiser,
)
from language_reading_predictors.statistical_models.factories.base import (
    BuiltModel,
    _MECH_HSGP_C,
    _MECH_HSGP_M,
    _add_child_random_intercept,
    _alpha_sigma_for,
    _rlm_dispersion_kappa,
)

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
    decompose_between_within: bool = False,
    phase_varying_slope: bool = False,
    adjust_for: Iterable[str] = (),
    mechanism_is_covariate: bool = False,
    mechanism_at_pre: bool = False,
    mech_hsgp_m: int | None = None,
    mech_lengthscale_prior: Continuous | None = None,
    kappa_prior_family: str = "halfnormal_concentration",
    kappa_sigma: float | None = None,
    frozen_design: MechanismDesign | None = None,
) -> BuiltModel[MechanismPayload]:
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
    clean no-interaction baseline (e.g. LRP63base) that differs from the full
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
    ``mech_post_logit``. Works with ``linear_mechanism`` either way: True gives the
    single-slope estimand (e.g. LRP90 phonological memory); False fits the HSGP
    ``f_mech`` curve on the standardised covariate (LRP92 sessions -> word reading),
    whose readiness-threshold knee is then reported in the exposure's own raw units
    rather than back-transformed to a bounded count. The caller is responsible for
    restricting to genuinely
    observed exposure rows (``require_observed`` in the loader): mean-imputation
    plus a missingness indicator is an *adjuster* policy and is not acceptable for
    the exposure itself.

    ``mechanism_at_pre`` (default False): take the mechanism regressor from the
    *period-start* (pre) score of ``mechanism_symbol`` rather than its post score.
    The default post alignment is the concurrent form (mechanism and outcome both
    at period-end); setting this True gives the lagged / predictive form
    ``mechanism_pre -> outcome_post`` which, with ``adjust_baseline_symbol`` the
    outcome, conditions on the outcome's own period-start level and so estimates
    whether the period-start mechanism predicts the *change* in the outcome over
    that period (issue #405: does taught vocabulary predict letter-sound growth?).
    Only the mechanism's own alignment moves — the outcome stays at post and the
    autoregressive baseline stays at pre. The pre logit is on the same
    ``logit_safe`` scale as the post branch and is standardised identically, so the
    reported slope keeps its per-SD-of-mechanism reading. Incompatible with
    ``mechanism_is_covariate`` (a standardised covariate has no separate pre score);
    ``phase_specific_mechanism`` is likewise unaffected. Default-off, so the
    concurrent mechanism family is byte-identical.

    ``decompose_between_within`` (default False, #603): Mundlak split of the
    exposure. A single exposure coefficient over a child random intercept returns a
    precision-weighted **blend** of two associations that answer different
    questions — do children who generally score higher on the exposure generally
    score higher on the outcome (*between*), and does a child's outcome move when
    their own exposure moves (*within*)? The random intercept does not separate
    them: it models repeated-measures dependence under an independence assumption
    and is not permitted to correlate with the exposure. On these data the two
    differ substantially (the ``pooled_levels`` family measures r = 0.81 between
    against 0.45 within for letter sounds and word reading on the logit scale), so a
    blend is a poor answer to either. With this flag the standardised exposure is
    split into each child's fitted-row mean (``beta_between``) and their deviation
    from it (``beta_within``), both registered as ``pm.Data`` so the design is
    replayable. **Linear only** — a between/within split of a nonparametric curve is
    a larger design question. A within-child coefficient removes *stable*
    between-child confounding, including the stable part of latent general ability;
    it does not make the exposure temporally prior to the outcome (both are still
    same-wave), and it does not remove time-varying confounding or reverse
    causation. The result is a better-posed association, not an identified effect.

    ``phase_varying_slope`` (default False, #604): partially-pooled per-period
    exposure slopes ``beta_mech_phase = mu_mech + sigma_mech_phase * z_phase``
    instead of one pooled slope, so the family's stability-across-periods assumption
    can be checked rather than assumed. The family stacks the randomised t1->t2
    transition with the two post-crossover ones and gives each its own intercept but
    one common exposure slope; nothing in the published output tested that. Partial
    pooling rather than three free slopes, because at roughly 52 rows per period
    independent slopes are noisy — the same choice ``build_dose_response_model``
    makes for its period-varying dose. **Linear only**, and mutually exclusive with
    ``phase_specific_mechanism`` (which is itself rejected upstream), so a
    per-period *slope* cannot be confused with a per-period *curve*. When combined
    with ``decompose_between_within`` the per-period slopes carry the within-child
    deviation and ``beta_between`` stays pooled, again as in the dose family. A
    period difference is evidence against pooling, not evidence about mechanism
    change over time: a child's third transition differs from their first in age,
    treatment history and measurement position at once, and only the first is
    randomised-arm-clean — every per-period slope remains an adjusted association.

    ``kappa_prior_family`` / ``kappa_sigma`` (default ``"halfnormal_concentration"``
    / None, #605): the Beta-Binomial concentration prior, validated against the
    shared ``itt.KAPPA_PRIOR_FAMILIES`` tuple by the run plan. The default
    ``kappa ~ HalfNormal(50)`` is not weak on a high-denominator outcome: with
    ``alpha + beta = kappa`` the variance inflation over Binomial is
    ``(kappa + n) / (kappa + 1)``, so being within 10% of Binomial variance needs
    ``kappa >= 10 (n - 1) - 1`` — 779 at ``n = 79`` and 1689 at ``n = 170``, both of
    which ``HalfNormal(50)`` gives vanishing mass. The prior therefore *enforces* a
    floor on overdispersion (about 3.3x at the prior median for word reading, 5.9x
    for the vocabulary tests) rather than leaving the near-Binomial limit — the
    ordinary hypothesis "no extra-Binomial variation" — available.
    ``"halfnormal_inverse_sqrt"`` puts ``1 / sqrt(kappa) ~ HalfNormal(kappa_sigma)``
    instead, whose tail does reach that limit; it is the parameterisation the ITT
    family offers as a sensitivity and ``level_factors`` adopted as its default
    (#584 decision 4). ``kappa_sigma`` is the scale of whichever family is selected.

    ``mech_hsgp_m`` / ``mech_lengthscale_prior`` (both default None): thin-support
    reparameterisation of the ``f_mech`` HSGP (issue #430). ``None`` reproduces the
    shared defaults exactly — basis count ``_MECH_HSGP_M`` and ``ell_prior_mech()`` —
    so every existing mechanism model is byte-identical. A spec sets them when its
    exposure support is too thin for those defaults (e.g. mech-190 blending: 10 items,
    chance floor ≈ 3.3, ~19% at ceiling), where a smaller basis and a lengthscale prior
    with a thinner short-lengthscale tail smooth the boundary geometry that otherwise
    diverges even at target_accept 0.999. Applies to both the standard and the
    ``phase_specific_mechanism`` ``f_mech`` builds; ``linear_mechanism`` builds no
    HSGP at all and so **rejects** them rather than ignoring them (#637 stage 1 —
    the settings layer had always refused that combination, and a factory that
    accepted it built a different model than the one declared). Only ``None``
    selects the default — a non-positive ``mech_hsgp_m`` raises rather than falling
    back silently.
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
    # One design validator, shared with ``MechanismModelSettings`` (#637 stage 1).
    # The two lists had drifted apart in both directions, and the direct entry
    # point's gap was the damaging one: ``linear_mechanism`` with
    # ``phase_specific_mechanism`` selected the linear branch below, built one
    # pooled ``beta_mech`` and dropped the per-period request without a word.
    validate_mechanism_design(
        linear_mechanism=linear_mechanism,
        phase_specific_mechanism=phase_specific_mechanism,
        phase_varying_slope=phase_varying_slope,
        decompose_between_within=decompose_between_within,
        mechanism_is_covariate=mechanism_is_covariate,
        mechanism_at_pre=mechanism_at_pre,
        moderator_symbol=moderator_symbol,
        moderator_is_covariate=moderator_is_covariate,
        mech_hsgp_m=mech_hsgp_m,
        hsgp_lengthscale_declared=mech_lengthscale_prior is not None,
        kappa_prior_family=kappa_prior_family,
        default_hsgp_m=_MECH_HSGP_M,
    )
    if mechanism_is_covariate:
        # A covariate exposure may enter EITHER linearly (a single ``beta_mech``
        # slope) OR as a nonparametric HSGP curve (``linear_mechanism=False``). The
        # GP path (LRP92: intervention sessions -> word reading) builds ``f_mech`` on
        # the standardised covariate exactly as for a bounded-count measure; the only
        # difference downstream is that the readiness-threshold knee is reported in
        # the exposure's own (raw) units rather than back-transformed to a count.
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
    elif mechanism_at_pre:
        # Lagged form: the regressor is the period-start score, so the keep-mask
        # must require the *pre* value observed (a row may have TR_pre but no
        # TR_post, or vice versa).
        mechanism_vals = prepared.pre_logit[mechanism_symbol]
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
    elif mechanism_at_pre:
        # Period-start regressor, already on the loader's logit-safe scale (the
        # same transform ``logit_safe`` applies to the post branch), standardised
        # below exactly as the post logit is.
        mech_input = prepared.pre_logit[mechanism_symbol]
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
    # A leave-one-out refit must interpret its basis coefficients against the *same*
    # design as the fit that produced the point being scored, so ``frozen_design``
    # pins every data-derived design quantity (#438 review). Default None reproduces
    # the original behaviour exactly, so every existing caller is byte-identical.
    _mod_scaler: Standardiser | None = None
    _mech_c = _MECH_HSGP_C
    _hsgp_boundary_realised = False
    if frozen_design is None:
        mech_logit_std, _mech_scaler = standardise(mech_input)
    else:
        _mech_scaler = frozen_design.mech_scaler
        mech_logit_std = (np.asarray(mech_input, dtype=float) - _mech_scaler.mean) / (
            _mech_scaler.sd
        )
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
            if frozen_design is None:
                z_M, _mod_scaler = standardise(raw_M)
            else:
                _mod_scaler = frozen_design.require_moderator_scaler()
                z_M = (np.asarray(raw_M, dtype=float) - _mod_scaler.mean) / _mod_scaler.sd
        else:
            moderator_post_logit = logit_safe(
                prepared.post_counts[moderator_symbol],
                prepared.n_trials[moderator_symbol],
            )
            if frozen_design is None:
                z_M, _mod_scaler = standardise(moderator_post_logit)
            else:
                _mod_scaler = frozen_design.require_moderator_scaler()
                z_M = (
                    np.asarray(moderator_post_logit, dtype=float) - _mod_scaler.mean
                ) / _mod_scaler.sd

    # Mundlak split of the standardised exposure (#603). Built on the *fitted* rows —
    # after the keep-mask above — so the child mean is the mean of the rows this model
    # actually uses, and ``z_L == mech_child_mean + mech_within_dev`` exactly. Both
    # vectors are registered as ``pm.Data`` below so a refit replays this design
    # rather than re-deriving a slightly different one.
    mech_child_mean: np.ndarray | None = None
    mech_within_dev: np.ndarray | None = None
    if decompose_between_within:
        _child_np = np.asarray(prepared.child_idx, dtype=int)
        _z = np.asarray(z_L, dtype=float)
        mech_child_mean = np.zeros_like(_z)
        for _child in np.unique(_child_np):
            _rows = _child_np == _child
            mech_child_mean[_rows] = _z[_rows].mean()
        mech_within_dev = _z - mech_child_mean

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
        mech_between_d = mech_within_d = None
        if decompose_between_within:
            mech_between_d = pm.Data(
                "mech_child_mean", mech_child_mean, dims="obs_id"
            )
            mech_within_d = pm.Data(
                "mech_within_dev", mech_within_dev, dims="obs_id"
            )
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
            gamma_A = _priors.gamma_age_prior().to_pymc("gamma_A")
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
            if decompose_between_within:
                # Mundlak split (#603). The between term stays pooled across periods
                # — it is one number per child by construction — while the slope on
                # the within-child deviation is the one a period-varying sensitivity
                # would vary. Same arrangement as the dose family's
                # ``beta_dose_between`` beside its period slopes.
                beta_between = _priors.beta_mech_prior().to_pymc("beta_between")
                eta = eta + beta_between * mech_between_d
                slope_target = mech_within_d
            else:
                slope_target = z_L_d
            if phase_varying_slope:
                # Partially-pooled per-period slopes (#604): a shared mean plus
                # shrunk per-period deviations, non-centred for geometry.
                mu_mech = _priors.beta_mech_prior().to_pymc("mu_mech")
                sigma_mech_phase = _priors.sigma_mech_phase_prior().to_pymc(
                    "sigma_mech_phase"
                )
                beta_mech_phase_raw = pm.Normal(
                    "beta_mech_phase_raw", mu=0.0, sigma=1.0, dims="phase"
                )
                beta_mech_phase = pm.Deterministic(
                    "beta_mech_phase",
                    mu_mech + sigma_mech_phase * beta_mech_phase_raw,
                    dims="phase",
                )
                eta = eta + beta_mech_phase[phase_d] * slope_target
            elif decompose_between_within:
                beta_within = _priors.beta_mech_prior().to_pymc("beta_within")
                eta = eta + beta_within * slope_target
            else:
                beta_mech = _priors.beta_mech_prior().to_pymc("beta_mech")
                eta = eta + beta_mech * slope_target
        elif phase_specific_mechanism:
            phase_specific = []
            for p in range(prepared.n_phases):
                phase_specific.append(
                    build_hsgp_1d(
                        f"f_mech_phase{p}",
                        mech_logit_std,
                        m=_MECH_HSGP_M if mech_hsgp_m is None else mech_hsgp_m,
                        lengthscale_prior=(
                            _priors.ell_prior_mech()
                            if mech_lengthscale_prior is None
                            else mech_lengthscale_prior
                        ),
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
            # ``build_hsgp_1d`` derives its boundary as ``max(|X|) * c``, so a refit on
            # n-1 rows would silently move it and redefine what the basis weights mean.
            # Passing the equivalent ``c`` reproduces the *fit's* boundary exactly on
            # the subset's support (#438 review).
            if frozen_design is not None:
                _mech_c = frozen_design.hsgp_c_for(mech_logit_std)
            _hsgp_boundary_realised = True
            f_mech = build_hsgp_1d(
                "f_mech",
                mech_logit_std,
                m=_MECH_HSGP_M if mech_hsgp_m is None else mech_hsgp_m,
                c=_mech_c,
                lengthscale_prior=(
                    _priors.ell_prior_mech()
                    if mech_lengthscale_prior is None
                    else mech_lengthscale_prior
                ),
            )
            eta = eta + f_mech

        eta = pm.Deterministic("eta", eta, dims="obs_id")
        if kappa_prior_family == "halfnormal_inverse_sqrt":
            # Dispersion-scale parameterisation (#605), so the near-Binomial limit is
            # reachable. ``HalfNormal`` on the concentration cannot get there: at
            # n = 79 (word reading) coming within 10% of Binomial variance needs
            # kappa > 779 and at n = 170 (the vocabulary tests) kappa > 1689, both of
            # which HalfNormal(50) gives effectively zero mass — so the registered
            # prior enforces roughly threefold and sixfold overdispersion a priori.
            # Same constructor as the ITT sensitivity and the level-factors default.
            kappa = _rlm_dispersion_kappa(
                float(_priors.inv_sqrt_kappa_prior().sigma)
                if kappa_sigma is None
                else kappa_sigma
            )
        else:
            kappa = (
                _priors.kappa_prior()
                if kappa_sigma is None
                else _priors.kappa_prior(sigma=kappa_sigma)
            ).to_pymc("kappa")

        beta_binomial_from_logit(
            "y_post",
            eta,
            n_trials=N_outcome,
            kappa=kappa,
            observed=outcome_post,
            dims="obs_id",
        )

    # Publish the realised design so a leave-one-out refit can replay it rather than
    # re-deriving a slightly different one from n-1 rows (#438 review). ``hsgp_L`` is
    # None on the linear-mechanism and phase-specific paths, which is what
    # ``loo_refit`` checks before it will refit at all.
    realised_design = MechanismDesign(
        mech_scaler=_mech_scaler,
        hsgp_L=(
            float(max(abs(mech_logit_std.min()), abs(mech_logit_std.max())) * _mech_c)
            if _hsgp_boundary_realised
            else None
        ),
        moderator_scaler=_mod_scaler,
    )
    return BuiltModel(
        model=model,
        prepared=prepared,
        payload=MechanismPayload(design=realised_design),
    )
