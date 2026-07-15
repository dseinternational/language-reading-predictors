# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Named prior constructors shared across the statistical models.

Every factory calls the same function so priors cannot drift between models.
Priors are defined as ``preliz`` distributions; call ``.to_pymc(name)`` inside
a PyMC model block to register them, or ``plot_and_save`` to output an SVG/PNG
for the Quarto report.
"""

from __future__ import annotations

import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import preliz as pz
from preliz.distributions.distributions import Continuous

from language_reading_predictors.statistical_models.plotting import (
    save_styled_figure,
)


# ---------------------------------------------------------------------------
# Shared prior template (document these on the report)
# ---------------------------------------------------------------------------


# Intercept prior scales, tiered by outcome proximity (prior-critical-review
# 2026-07-07, Finding 1). In the ANCOVA families the linear predictor is
# ``alpha + gamma_own * logit(y_pre) + ...`` with ``gamma_own`` centred at 1, so
# the own-baseline term already carries the outcome *level* and ``alpha``'s mean
# is correctly ~0 (a deviation, not the level — anchoring its mean at the baseline
# would double-count). The intercept blow-out the review flagged is therefore an
# SD problem, not a mean problem: on a high-denominator distal test (e.g. the
# 170-item receptive-vocabulary R, ~36 items/logit near the operating point) the
# ``Normal(0, 1.5)`` intercept alone sweeps ~+/-54 items of pure intercept noise,
# past the plausible range for DS children. The distal tier tightens the *SD* to
# 1.0 (a ~+/-36-item band on R) exactly as ``tau`` was tiered in #141. The scale
# is set from the item count (a measurement fact) and the low-occupancy operating
# point established in the DS literature — NOT from this sample's realised
# post-scores (see notes/202607071400-prior-critical-review.md, "Prior provenance
# and anti-double-dipping"). Proximal outcomes keep the wider 1.5.
#
# The growth/LCSM *level* models instead anchor the intercept *mean* at the
# observed grand-mean logit, because they have no ``gamma_own * pre`` term — there
# the intercept *is* the level, so a mean anchor (not an SD tier) is the right
# tool. The two parameterisations are consistent in intent, not a drift.
ALPHA_SIGMA_PROXIMAL: float = 1.5
ALPHA_SIGMA_DISTAL: float = 1.0


def alpha_prior(sigma: float = ALPHA_SIGMA_PROXIMAL) -> Continuous:
    """Intercept alpha ~ Normal(0, 1.5).

    (Parametrised by ``sigma`` for the outcome tier and for prior-sensitivity
    fits; the docstring's numeric value is the proximal default that
    ``_dist_from_doc`` extracts for the name-only table path. When a built RV is
    available the table reads the *actual* registered scale, so a distal
    ``Normal(0, 1.0)`` intercept is reported correctly.)
    """
    return pz.Normal(mu=0.0, sigma=sigma)


def alpha_prior_distal() -> Continuous:
    """Distal-tier intercept alpha ~ Normal(0, 1.0)."""
    return alpha_prior(sigma=ALPHA_SIGMA_DISTAL)


# Treatment-effect prior scales, tiered by outcome proximity (issue #141). The
# proximal (directly-taught / decoding) outcomes keep the wider default; the
# distal broad-transfer outcomes (see ``measures.DISTAL_OUTCOMES``) take a tighter
# prior, because on the high-denominator standardised tests the wider logit prior
# implies implausibly large item-scale effects.
TAU_SIGMA_PROXIMAL: float = 0.5
TAU_SIGMA_DISTAL: float = 0.3


def tau_prior(sigma: float = TAU_SIGMA_PROXIMAL) -> Continuous:
    """Treatment effect tau ~ Normal(0, 0.5).

    (Parametrised by ``sigma`` for the outcome tier and for prior-sensitivity
    fits; the docstring's numeric value is the proximal default that
    ``_dist_from_doc`` extracts for the name-only table path. When a built RV is
    available the table reads the *actual* registered scale, so a distal
    ``Normal(0, 0.3)`` tau is reported correctly.)
    """
    return pz.Normal(mu=0.0, sigma=sigma)


def tau_prior_distal() -> Continuous:
    """Distal-tier treatment effect tau ~ Normal(0, 0.3)."""
    return tau_prior(sigma=TAU_SIGMA_DISTAL)


def gamma_own_prior() -> Continuous:
    """Own-baseline coupling gamma_own ~ Normal(1, 0.25).

    Centred at 1 (post-logit tracks pre-logit 1:1 — no regression to the mean
    asserted a priori) with the SD tightened from 0.5 to 0.25 (prior-critical-
    review 2026-07-07, Finding 2; #141 open decision 3). This is the one shared
    coefficient prior informative in its *mean*, and the tighter SD is anchored on
    published test-retest reliabilities for these measures (r ~ 0.8-0.95 at these
    ages, from the test manuals) — an admissible *external* source, not this
    trial's data: a retest r ~ 0.9 implies a tracking slope near 1 with little
    spread, so ``Normal(1, 0.25)`` (95% ~ 0.5-1.5) is well-calibrated where the
    old ``0.5`` (95% ~ 0-2, from no tracking to double) was not. A *precision*
    term, so it cannot bias ``tau``; the tighter SD also reduces the item-scale
    prior spread on the long tests (Finding 1).
    """
    return pz.Normal(mu=1.0, sigma=0.25)


def gamma_cross_prior() -> Continuous:
    """Cross-baseline coupling gamma_k ~ Normal(0, 0.3)."""
    return pz.Normal(mu=0.0, sigma=0.3)


def gamma_age_prior() -> Continuous:
    """Linear age main-effect coupling gamma_A ~ Normal(0, 0.3).

    Used by the LRPITT suite (``build_itt_model(use_age_linear=True)``) for the
    plain linear age term ``gamma_A * A_std``. Age is a *precision* covariate
    only — under the locked DAG the ITT effect ``tau`` is identified by the empty
    adjustment set, so age (like the own baseline) sharpens ``tau`` without
    licensing the causal claim. ``A_std`` is unit-SD standardised age, so the
    same weakly-regularising ``Normal(0, 0.3)`` scale as the cross-baseline
    couplings is appropriate; a dedicated constructor (rather than reusing
    ``gamma_cross_prior``) documents the term and surfaces it in the report
    prior panel.
    """
    return pz.Normal(mu=0.0, sigma=0.3)


def kappa_prior(sigma: float = 50.0) -> Continuous:
    """Beta-binomial concentration kappa ~ HalfNormal(50).

    (Parametrised by ``sigma`` so the dispersion prior can be anchored once
    against the tests' normative raw-score SDs — an admissible external source,
    #141's still-unused anchor — prior-critical-review 2026-07-07, recommendation
    5. The default ``50`` is only *partly* permissive: it is generous for the
    low-denominator outcomes, but on high-denominator outcomes (e.g. R / E at
    ``n_trials`` up to 170) it concentrates ``kappa`` well below ``n``, so it
    effectively **enforces a minimum over-dispersion** — the near-Binomial limit
    (``kappa >> n``) is off the table. That is deliberate here (the review found
    the prior-predictive over-dispersion to be location-driven, not
    ``kappa``-driven, so raising the cap is lower priority), and the default is
    unchanged pending the normative-SD calibration. The docstring's numeric value
    is what ``_dist_from_doc`` extracts for the name-only table path.)
    """
    return pz.HalfNormal(sigma=sigma)


def beta_mech_prior(sigma: float = 1.0) -> Continuous:
    """Linear-mechanism slope beta_mech ~ Normal(0, 1).

    Used by ``build_mechanism_model(linear_mechanism=True)`` in place of the
    HSGP ``f_mech`` on low / floored count outcomes (e.g. nonword decoding,
    LRP72). The input is the standardised ``z(logit L_post)``, so the slope is
    the change in the outcome logit per 1 SD of the mechanism on the logit
    scale; the weakly-informative unit scale lets the data speak for the
    primary effect while still regularising.

    (Parametrised by ``sigma`` for the prior-sensitivity sweep — prior-critical-
    review 2026-07-07, recommendation 4, and #141's remaining recommended sweep.
    ``Normal(0, 1)`` is the loosest coefficient prior in the suite: a per-family
    pushforward shows a +1 SD move shifts the outcome by dp ~ +/-0.38 at p95 on a
    mid-scale baseline, ~2.7x the shared ``gamma_cross`` association scale.
    ``sigma=0.5`` roughly halves that; kept at 1.0 by default pending the sweep,
    which is an *analysis*, not a default change. The docstring's numeric value
    is the default that ``_dist_from_doc`` extracts for the name-only table path.)
    """
    return pz.Normal(mu=0.0, sigma=sigma)


def sigma_dose_phase_prior() -> Continuous:
    """Between-period SD of the dose slope sigma_dose ~ HalfNormal(0.5).

    Used by ``build_dose_response_model(period_varying_dose=True)`` for the
    partial-pooled, period-specific dose slopes
    ``beta_dose_phase = mu_dose + sigma_dose * z_phase`` (#104 Phase 2). The
    slope is on the per-1-SD-of-dose logit scale (the dose mean ``mu_dose`` uses
    the unit-scale :func:`beta_mech_prior`), so a HalfNormal(0.5) keeps the
    three period slopes shrunk toward the pooled effect unless the data show
    real period variation — appropriate given the weak Phase-1 dose signal.
    """
    return pz.HalfNormal(sigma=0.5)


def sigma_delta_prior() -> Continuous:
    """Between-child SD of the on-intervention effect sigma_delta ~ HalfNormal(0.5).

    Used by ``build_did_model(use_varying_delta=True)`` for the treatment-effect
    heterogeneity variance component (#230 §2/§4a): the per-child on-intervention
    slope is ``delta_i = delta + sigma_delta * z_i`` (non-centred), so ``sigma_delta``
    is the between-child SD of the crossover treatment effect on the logit scale — the
    reported estimand for "is the treatment effect homogeneous?". The 0.5 scale matches
    the population effect prior (``tau``/``delta`` ~ Normal(0, 0.5)) and the child
    random-intercept SD (``sigma_child`` ~ HalfNormal(0.5)), so it is weakly-informative
    and lets the data pull the SD toward zero — the whole point of the estimand (a
    near-zero posterior is the clean "no reliable between-child variation" result).
    """
    return pz.HalfNormal(sigma=0.5)


def b_path_prior(sigma: float = 1.0) -> Continuous:
    """Mediator -> outcome slope (b-path) ~ Normal(0, 1).

    Used by the LRP59 mediation outcome model for ``b_M``, the coefficient on the
    standardised mediator ``z(logit L_t2)``. Weakly-informative on the unit
    (per-SD) scale so the data identify the key b-path of the decomposition,
    while still regularising; the treatment and confounder couplings use the
    tighter ``tau_prior`` / ``gamma_cross_prior``.

    (Parametrised by ``sigma`` for the same prior-sensitivity sweep as
    :func:`beta_mech_prior` — the two share the loosest ``Normal(0, 1)`` scale in
    the suite; ``sigma=0.5`` roughly halves the p95 outcome shift. Default kept at
    1.0 pending the sweep. prior-critical-review 2026-07-07, recommendation 4.)
    """
    return pz.Normal(mu=0.0, sigma=sigma)


def sigma_mediator_prior() -> Continuous:
    """Gaussian-mediator residual SD sigma_M ~ HalfNormal(1.0).

    Used by the LRP62 reading-route model, where the mediator is a continuous
    standardised code-based-route composite modelled as ``Normal(mu_M, sigma_M)``.
    The composite-post is standardised (SD 1), so after conditioning on the
    baseline composite and covariates the residual SD is below 1; HalfNormal(1.0)
    is weakly-informative on that scale.
    """
    return pz.HalfNormal(sigma=1.0)


def eta_main_prior() -> Continuous:
    """GP amplitude (main effect) eta ~ HalfNormal(0.3).

    Tightened from HalfNormal(1.0) after LRP52 showed the GP amplitudes had
    posterior mass at zero, creating a Neal's funnel with the basis weights
    that caused ~2.5% divergences at target_accept 0.95 and 0.98. With 50-60
    children per ITT run and only 1 post-score per child, the data cannot
    identify a 20-basis HSGP; a tighter prior keeps the flexibility available
    while pushing the funnel neck away from zero.
    """
    return pz.HalfNormal(sigma=0.3)


def eta_tau_prior() -> Continuous:
    """GP amplitude (tau modifier) eta_tau ~ HalfNormal(0.3) - deliberately tight."""
    return pz.HalfNormal(sigma=0.3)


def ell_prior() -> Continuous:
    """GP lengthscale ell ~ InverseGamma(3, 1) on standardised inputs."""
    return pz.InverseGamma(alpha=3.0, beta=1.0)


def ell_prior_mech() -> Continuous:
    """Mechanism-curve GP lengthscale ell ~ InverseGamma(5, 5) (issue #265).

    On standardised inputs the default ``ell_prior`` (``InverseGamma(3, 1)``, mode
    0.25 / mean 0.5) keeps non-negligible mass on very short lengthscales, which at
    n ≈ 157 forces the ``f_mech`` HSGP toward a wiggly, weakly-identified fit — the
    boundary-geometry neck that leaves residual divergences on mech-058/071/158.
    ``InverseGamma(5, 5)`` (mode 0.83 / mean 1.25) shifts the mass to
    moderate-to-long lengthscales — a smoother curve — while still allowing genuine
    curvature. Scoped to the mechanism GPs so the ITT age-GP and dose-response GP,
    which converge cleanly under the default, do not regress.
    """
    return pz.InverseGamma(alpha=5.0, beta=5.0)


def eta_partial_pool_prior() -> Continuous:
    """Joint-model outcome-specific age-GP amplitude ~ HalfNormal(0.3)."""
    return pz.HalfNormal(sigma=0.3)


def predictor_slope_prior(sigma: float = 0.3) -> Continuous:
    """Standardised predictor slope ~ Normal(0, 0.3) by default.

    (Parametrised by ``sigma``; the prior table reports this default scale — the
    docstring's numeric value is what ``_dist_from_doc`` extracts, so the table
    shows ``Normal(0, 0.3)`` rather than the literal token ``sigma``.)

    Per-SD coefficient on a standardised predictor in a between-child adjusted or
    concurrent-association model. Fixed weakly-informative and regularising, given
    the collinear skill/general-ability cluster and n ~ 51-53.

    The default was reconciled from 0.5 to 0.3 (prior-critical-review 2026-07-07,
    recommendation 3): it is an *association* scale and now matches the shared
    ``gamma_cross_prior`` (``Normal(0, 0.3)``) rather than sitting looser without
    a documented rationale — one of the three cross-family scale drifts the review
    flagged. 0.3 is within the ``{0.3, 0.7}`` sensitivity band the LRP65
    which-predictors-clear-zero conclusion was already checked against, so the
    substantive finding is unchanged.
    """
    return pz.Normal(mu=0.0, sigma=sigma)


# ---------------------------------------------------------------------------
# Regularized ("Finnish") horseshoe (LRPHS, #116 Phase E)
# ---------------------------------------------------------------------------
# Global-local shrinkage prior for the standardised-predictor coefficients of the
# horseshoe ranking models (Piironen & Vehtari 2017, doi:10.1214/17-EJS1337SI):
#
#   tau       ~ HalfCauchy(tau0)                         # global scale
#   lambda_k  ~ HalfCauchy(1)                            # per-predictor local scale
#   c^2       ~ InverseGamma(slab_df/2, slab_df*slab_scale^2/2)   # slab (heavy-tail cap)
#   lambda_tilde_k = sqrt(c^2 * lambda_k^2 / (c^2 + tau^2 * lambda_k^2))
#   beta_k    ~ Normal(0, tau * lambda_tilde_k)          # non-centred (see factories)
#
# The scalar components are panelled/tabled here; the vector lambda + the
# non-centred coefficients are assembled in ``factories._build_horseshoe_betas``.


def horseshoe_tau_prior(tau0: float = 0.1) -> Continuous:
    """Horseshoe global shrinkage tau ~ HalfCauchy(0.1) by default.

    (Parametrised by ``tau0``; the table reports the default scale.) The global
    scale pulls all coefficients toward zero; the expected number of relevant
    predictors sets it via ``tau0 ~ p0/(D-p0) * 1/sqrt(N)``. For the pilot
    (~11 constructs, n~54, a handful expected relevant) this is ~0.1.
    """
    return pz.HalfCauchy(beta=tau0)


def horseshoe_local_prior() -> Continuous:
    """Horseshoe per-predictor local shrinkage lambda ~ HalfCauchy(1).

    The heavy Cauchy tail lets a genuinely large coefficient escape the global
    pull toward zero (the "spike-and-slab" behaviour), so relevant predictors are
    not over-shrunk while noise predictors collapse to ~0.
    """
    return pz.HalfCauchy(beta=1.0)


def horseshoe_slab_prior(slab_scale: float = 2.0, slab_df: float = 4.0) -> Continuous:
    """Regularized-horseshoe slab c^2 ~ InverseGamma(2, 8) by default.

    (``InverseGamma(slab_df/2, slab_df*slab_scale^2/2)``; defaults slab_scale=2,
    slab_df=4 give ``InverseGamma(2, 8)``.) Caps the effective scale of the
    largest coefficients at ~``slab_scale`` on the logit scale, regularising the
    Cauchy tail so the sampler does not chase implausibly large logit effects.
    """
    return pz.InverseGamma(alpha=slab_df / 2.0, beta=slab_df * slab_scale**2 / 2.0)


# ---------------------------------------------------------------------------
# Registry - used to render the prior panel in every report
# ---------------------------------------------------------------------------


SHARED_PRIORS: dict[str, "callable[[], Continuous]"] = {
    "alpha": alpha_prior,
    "tau": tau_prior,
    "gamma_own": gamma_own_prior,
    "gamma_cross": gamma_cross_prior,
    "gamma_age": gamma_age_prior,
    "kappa": kappa_prior,
    "predictor_slope": predictor_slope_prior,
    "eta_main": eta_main_prior,
    "eta_tau": eta_tau_prior,
    "ell": ell_prior,
}


# Constructors used by some factories but absent from :data:`SHARED_PRIORS`
# (so they were never panelled before #125). Added here so a model that uses them
# gets a panel and a priors-table row.
_EXTRA_PRIORS: dict[str, "callable[[], Continuous]"] = {
    "tau_distal": tau_prior_distal,
    "alpha_distal": alpha_prior_distal,
    "beta_mech": beta_mech_prior,
    "sigma_dose": sigma_dose_phase_prior,
    "sigma_delta": sigma_delta_prior,
    "b_path": b_path_prior,
    "sigma_mediator": sigma_mediator_prior,
    "eta_partial_pool": eta_partial_pool_prior,
    "hs_tau": horseshoe_tau_prior,
    "hs_lambda": horseshoe_local_prior,
    "hs_c2": horseshoe_slab_prior,
}

ALL_PRIORS: dict[str, "callable[[], Continuous]"] = {**SHARED_PRIORS, **_EXTRA_PRIORS}


# Role of each named prior in the DAG-faithful workflow (issue #125 Area 1): only
# the *causal* prior backs an effect identified by randomisation; *precision*
# priors sharpen it without licensing a causal claim; *association* priors back
# adjusted (confounded) couplings; *nuisance* priors are the intercept /
# dispersion / random-intercept scale; *gp* priors parameterise the optional
# Gaussian-process terms.
_ROLE_BY_CTOR: dict[str, str] = {
    "alpha": "nuisance",
    "alpha_distal": "nuisance",
    "tau": "causal",
    "tau_distal": "causal",
    "gamma_own": "precision",
    "gamma_cross": "association",
    "gamma_age": "precision",
    "kappa": "nuisance",
    "predictor_slope": "association",
    "beta_mech": "association",
    "sigma_dose": "nuisance",
    "sigma_delta": "nuisance",
    "b_path": "association",
    "sigma_mediator": "nuisance",
    "eta_main": "gp",
    "eta_tau": "gp",
    "ell": "gp",
    "eta_partial_pool": "gp",
    "hs_tau": "nuisance",
    "hs_lambda": "nuisance",
    "hs_c2": "nuisance",
}

# Map a registered PyMC RV name to the shared-prior constructor that built it.
# Several RVs share one constructor (``tau_prior`` backs every randomised effect
# term; ``gamma_cross_prior`` backs every adjusted coupling), so role assignment
# is by constructor, not RV name. Names not listed fall back by prefix.
_RV_TO_CTOR: dict[str, str] = {
    "alpha": "alpha",
    "tau": "tau",
    "beta_G": "tau",
    "beta_period": "tau",
    "delta": "tau",
    # ``beta_dose`` is built from ``beta_mech_prior`` (Normal(0, 1)) in both the
    # dose-response and DiD-dose factories — it is a dose slope (association), not a
    # tau-scaled randomised effect. Map it to ``beta_mech`` so the prior table is
    # correct by default rather than relying on the per-family override (the "tau"
    # entry was a leftover from when the slope reused tau_prior).
    "beta_dose": "beta_mech",
    "beta_trt": "tau",
    "b_grp_time": "tau",
    "beta_grp": "tau",
    "a_G": "tau",
    "b_G": "tau",
    "gamma_own": "gamma_own",
    "a_L": "gamma_own",
    "a_comp": "gamma_own",
    "b_W": "gamma_own",
    "gamma_A": "gamma_age",
    "kappa": "kappa",
    "kappa_M": "kappa",
    "kappa_Y": "kappa",
    "beta_mech": "beta_mech",
    "mu_dose": "beta_mech",
    "sigma_dose": "sigma_dose",
    "sigma_delta": "sigma_delta",
    "b_M": "b_path",
    # Two-mediator (mediation_multi / LRP64) mediator -> outcome b-paths. Without
    # these explicit entries they fall through the ``b_`` prefix to gamma_cross
    # and the prior table would misreport their Normal(0, 1) scale as Normal(0, 0.3).
    "b_L": "b_path",
    "b_E": "b_path",
    "b_B": "b_path",  # LRP66 blending mediator -> reading b-path (same Normal(0, 1))
    "sigma_M": "sigma_mediator",
    "eta_main": "eta_main",
    "eta_tau": "eta_tau",
    "ell": "ell",
    "eta_partial_pool": "eta_partial_pool",
    # Regularized-horseshoe shrinkage hyperparameters (LRPHS, #116 Phase E). The
    # coefficient vector ``beta`` is a Deterministic (tau * lambda_tilde * hs_z),
    # not a free RV, so it is not a table row; ``hs_z`` is documented inline below.
    "hs_tau": "hs_tau",
    "hs_lambda": "hs_lambda",
    "hs_c2": "hs_c2",
}

# Inline priors created directly in the factories (not via a named constructor),
# so they would be invisible to a SHARED_PRIORS-only table. Documented here.
_INLINE_PRIORS: dict[str, dict[str, str]] = {
    "alpha_phase": {
        "role": "nuisance",
        "distribution": "Normal(0, 0.5)",
        "rationale": "Per-phase intercept offset alpha_phase ~ Normal(0, 0.5).",
    },
    "alpha_time": {
        "role": "nuisance",
        "distribution": "Normal(0, 0.5)",
        "rationale": "Per-timepoint intercept offset alpha_time ~ Normal(0, 0.5).",
    },
    "sigma_child": {
        "role": "nuisance",
        "distribution": "HalfNormal(0.5)",
        "rationale": "Child random-intercept SD sigma_child ~ HalfNormal(0.5).",
    },
    "beta_dose_phase_raw": {
        "role": "nuisance",
        "distribution": "Normal(0, 1)",
        "rationale": (
            "Standard-normal non-centred period-dose offset; scaled by sigma_dose."
        ),
    },
    "beta_group_nuisance": {
        "role": "nuisance",
        "distribution": "Normal(0, 1)",
        "rationale": (
            "Concurrent-family group-composition nuisance term; not a reported "
            "association."
        ),
    },
    "hs_z": {
        "role": "nuisance",
        "distribution": "Normal(0, 1)",
        "rationale": (
            "Standard-normal non-centred horseshoe coefficient offset; scaled by "
            "tau * lambda_tilde to give beta (LRPHS)."
        ),
    },
}


def _first_docline(ctor) -> str:
    """First line of a constructor's docstring (the prior's rationale)."""
    return (ctor.__doc__ or "").strip().split("\n")[0].strip()


def _dist_from_doc(ctor) -> str:
    """Extract the distribution signature (e.g. ``Normal(0, 0.5)``) from the doc."""
    line = _first_docline(ctor)
    m = re.search(r"~\s*([A-Za-z]+\([^)]*\))", line)
    return m.group(1) if m else ""


def _ctor_key_for_rv(
    rv_name: str,
    *,
    ctor_overrides: dict[str, str] | None = None,
) -> str | None:
    """Constructor key backing an RV, by exact name, HSGP suffix, then prefix."""
    base = rv_name.split("[")[0]
    if base in _INLINE_PRIORS:
        return None
    if ctor_overrides is not None and base in ctor_overrides:
        return ctor_overrides[base]
    if base in _RV_TO_CTOR:
        return _RV_TO_CTOR[base]
    # HSGP amplitude / lengthscale RVs are named ``{term}__eta`` / ``{term}__ell``
    # by the dse_research_utils HSGP builder (build_hsgp_1d / build_tau_modifier).
    # They carry the shared eta / ell priors but never matched a constructor key,
    # so the GP amplitude (HalfNormal(0.3)) and lengthscale (InverseGamma(3, 1))
    # priors were silently absent from priors_table / the report panels (#141).
    if base.endswith("__eta"):
        return "eta_tau" if base.startswith("g_tau") else "eta_main"
    if base.endswith("__ell"):
        return "ell"
    # LRP65-style adjusted predictor slopes are named beta_{predictor}. Exact
    # beta_* entries above win first (beta_G, beta_trt, beta_period, ...).
    if base.startswith("beta_"):
        return "predictor_slope"
    # gamma_* couplings share the weakly-informative cross prior. (The ``b_`` /
    # ``a_`` prefix was DROPPED here: it over-matched bespoke inline priors such
    # as the LCSM ``a_change`` / ``b_self``, mislabelling their Normal(0, 1.5) /
    # Normal(0, 0.5) as gamma_cross's Normal(0, 0.3). Genuine b_ / a_ cross
    # couplings are captured by :func:`_classify_fallback` via the RV's own
    # distribution, so their gamma_cross panel and role are preserved.)
    if base.startswith("gamma"):
        return "gamma_cross"
    return None


def _normalise_dist_str(s: str) -> str:
    """Tidy a ``pymc.printing.str_for_dist`` string to the docstring house style.

    PyMC prints ``HalfNormal(0, x)`` (with the loc) and ``_lkjcholeskycov(...)``;
    the shared constructors document ``HalfNormal(x)`` and ``LKJCholeskyCov(...)``.
    Matching the two keeps the report's prior table visually consistent whether a
    row came from a constructor docstring or was read off the built RV.
    """
    m = re.match(r"HalfNormal\(0(?:\.0)?,\s*([^)]+)\)$", s)
    if m:
        return f"HalfNormal({m.group(1)})"
    if s.startswith("_lkjcholeskycov"):
        return "LKJCholeskyCov" + s[len("_lkjcholeskycov"):]
    return s


def _dist_from_rv(rv) -> str | None:
    """Authoritative distribution string read off a *built* PyMC RV.

    Uses ``pymc.printing.str_for_dist`` (the same machinery behind
    ``print(model)``), so the reported distribution is the one actually
    registered — no constructor guessing. Returns ``None`` for anything without a
    readable distribution (a plain name, a deterministic, an unexpected op), so
    the caller can fall back to the docstring / ``(model prior)``.
    """
    if rv is None or not hasattr(rv, "owner"):
        return None
    try:
        from pymc.printing import str_for_dist

        s = str_for_dist(rv, formatting="plain")
    except Exception:
        return None
    if "~" in s:
        s = s.split("~", 1)[1]
    return _normalise_dist_str(s.strip())


def _classify_fallback(rv_name: str, distribution: str | None) -> tuple[str, str]:
    """(role, panel) for an RV with no shared-constructor mapping.

    Covers the bespoke inline priors of the LCSM / correlated-factor / two-mediator
    / joint-LKJ families and the non-centred offsets, so no *actively-used* prior
    is reported with role ``other`` (issue #141, "cannot silently omit"). The
    distribution has already been read off the RV, so a Normal(0, 0.3) coupling is
    routed to the ``gamma_cross`` panel/role by its actual scale rather than by a
    name prefix (which would misfire on e.g. ``a_change``). Returns ``("other",
    "")`` only for a genuinely unrecognised RV — the guard test then fails, which
    is the intended signal to document the new prior here.
    """
    base = rv_name.split("[")[0]
    # HSGP basis weights (Normal(0, 1) unit coefficients) — a GP nuisance.
    if base.endswith("hsgp_coeffs") or base.endswith("_coeffs"):
        return ("gp", "")
    # Non-centred standard-normal offsets (paired with a scale prior that carries
    # the meaning): child / factor / process / initial-latent z's and *_raw.
    if (
        base.endswith("_raw")
        or base.startswith(("u_z", "z1_", "zproc_", "factor_z"))
    ):
        return ("nuisance", "")
    # Correlation / covariance priors (joint LKJ residual, factor covariance).
    if base.startswith(("u_chol", "u_corr", "factor_cov", "chol")):
        return ("nuisance", "")
    # Scales / dispersions / random-effect SDs.
    if base.startswith(("sigma", "kappa", "tau_")) or base == "sigma1":
        return ("nuisance", "")
    # Measurement loadings (correlated-factor CFA).
    if base.startswith(("lambda", "load")):
        return ("association", "")
    # Latent-mean anchors and per-leg intercepts (…0, mu1, a_change).
    if base in {"mu1", "a_change"} or re.search(r"0$", base):
        return ("nuisance", "")
    # Age couplings are a precision covariate (named …_A / d_age in these models).
    # Checked before the Normal(0, 0.3) signature below: age shares the
    # cross-coupling scale but is a precision term, not an association.
    if base == "d_age" or re.search(r"_A$", base):
        return ("precision", "")
    # Own-baseline couplings reuse the gamma_own signature (Normal(1, 0.25)) — a
    # precision term (e.g. the mediator legs' aL_L / aE_E autoregression). The SD
    # was tightened from 0.5 to 0.25 (Finding 2); keep this signature in step.
    if distribution == "Normal(1, 0.25)":
        return ("precision", "gamma_own")
    # Everything else that is coupling-shaped is an adjusted association; route to
    # the gamma_cross panel when it shares that prior's scale.
    if distribution == "Normal(0, 0.3)":
        return ("association", "gamma_cross")
    if base.startswith(("g_", "b_", "a_", "aL", "aE", "d_")) or base.endswith("_self"):
        return ("association", "")
    return ("other", "")


def prior_info_for_rv(
    rv_name: str,
    *,
    rv=None,
    ctor_overrides: dict[str, str] | None = None,
    role_overrides: dict[str, str] | None = None,
    rationale_overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """``{parameter, distribution, role, rationale, panel}`` for a registered RV.

    ``panel`` is the basename (without ``prior_`` / extension) of the prior-PDF
    panel for this parameter, or ``""`` for inline / bespoke priors that have no
    panel — the report maps a table row to ``prior_{panel}.png``.

    Pass the built PyMC variable as ``rv`` (``priors_table`` does) so bespoke
    inline priors — LCSM, correlated-factor, two-mediator, joint-LKJ — are
    documented from their *actual* distribution rather than dropped to
    ``(model prior)`` or mislabelled by a name prefix (issue #141).
    """
    base = rv_name.split("[")[0]
    rationale_overrides = rationale_overrides or {}
    rationale = rationale_overrides.get(rv_name, rationale_overrides.get(base))
    if base in _INLINE_PRIORS:
        info = _INLINE_PRIORS[base]
        if rationale is not None:
            info = {**info, "rationale": rationale}
        return {"parameter": rv_name, **info, "panel": ""}
    key = _ctor_key_for_rv(rv_name, ctor_overrides=ctor_overrides)
    if key is None:
        distribution = _dist_from_rv(rv) or "(model prior)"
        role, panel = _classify_fallback(rv_name, distribution)
        return {
            "parameter": rv_name,
            "distribution": distribution,
            "role": (role_overrides or {}).get(base, role),
            "rationale": rationale or "",
            "panel": panel,
        }
    ctor = ALL_PRIORS[key]
    # Prefer the built RV's own scale over the constructor docstring: a
    # constructor may be called with a non-default ``sigma`` (the distal-tier tau,
    # or a prior-sensitivity fit), so the docstring's default would misreport it.
    # Falls back to the docstring for the name-only path (no RV).
    return {
        "parameter": rv_name,
        "distribution": _dist_from_rv(rv) or _dist_from_doc(ctor),
        "role": (role_overrides or {}).get(base, _ROLE_BY_CTOR[key]),
        "rationale": rationale or _first_docline(ctor),
        "panel": key,
    }


def used_prior_keys(
    model,
    *,
    ctor_overrides: dict[str, str] | None = None,
) -> list[str]:
    """Constructor keys with a panel to render for ``model`` (for panel pruning).

    Derived from :func:`prior_info_for_rv` so a panel reached only via the RV-
    distribution fallback — e.g. the ``gamma_cross`` panel behind an inline
    Normal(0, 0.3) coupling like ``b_R``, or an HSGP ``eta_main`` / ``ell``
    amplitude/lengthscale — is not dropped (issue #141).
    """
    keys: list[str] = []
    for rv in list(model.free_RVs) + list(getattr(model, "deterministics", [])):
        panel = prior_info_for_rv(rv.name, rv=rv, ctor_overrides=ctor_overrides)[
            "panel"
        ]
        if panel and panel in ALL_PRIORS and panel not in keys:
            keys.append(panel)
    return keys


def priors_table(
    model,
    *,
    ctor_overrides: dict[str, str] | None = None,
    role_overrides: dict[str, str] | None = None,
    rationale_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Per-model prior table with a ``role`` column (issue #125 Area 1).

    One row per registered free RV (vector coefficients collapse to one row),
    driven by the *actual* model so it never lists priors the model did not use
    and captures the inline ``alpha_phase`` / ``alpha_time`` / ``sigma_child``
    priors a SHARED_PRIORS-only table would miss. Columns: ``parameter``,
    ``distribution``, ``role``, ``rationale``.
    """
    rows = [
        prior_info_for_rv(
            rv.name,
            rv=rv,
            ctor_overrides=ctor_overrides,
            role_overrides=role_overrides,
            rationale_overrides=rationale_overrides,
        )
        for rv in model.free_RVs
    ]
    return pd.DataFrame(
        rows, columns=["parameter", "distribution", "role", "rationale", "panel"]
    )


def _prior_title(name: str) -> str:
    """Human-readable title for a prior-PDF panel.

    Turns the raw file stem (``"prior_gamma_own"``) into ``"gamma_own · precision
    prior"`` using the role registry, so the panel says what the parameter *is*
    rather than repeating the filename.
    """
    key = name[len("prior_"):] if name.startswith("prior_") else name
    role = _ROLE_BY_CTOR.get(key)
    return f"{key} · {role} prior" if role else f"{key} prior"


def plot_and_save(dist: Continuous, output_dir: str, name: str) -> str:
    """Plot a prior PDF and save as ``{name}.png`` (+ an SVG sibling per #208).

    Reports reference the PNG (raster keeps model-output pages quick to browse);
    the small vector SVG is emitted alongside as a downloadable artifact.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(5, 3))
    try:
        dist.plot_pdf()
    except Exception:
        # Some preliz distributions (e.g. InverseGamma) need an explicit axis.
        ax = plt.gca()
        dist.plot_pdf(pointinterval=False, ax=ax)
    plt.title(_prior_title(name))
    return save_styled_figure(output_dir, name, fig=fig)


def save_shared_prior_panel(
    output_dir: str, used: list[str] | None = None
) -> list[str]:
    """Plot the priors the model uses and return the generated files.

    ``used`` is a list of constructor keys (from :func:`used_prior_keys`); when
    given, only those panels are written (pruning the 4–6 dead panels per model
    that the old all-of-:data:`SHARED_PRIORS` behaviour produced, and adding
    panels for the previously-unpanelled ``beta_mech`` / ``b_path`` /
    ``sigma_mediator`` / ``eta_partial_pool``). When ``None``, every shared prior
    is plotted (back-compatible default).
    """
    keys = list(SHARED_PRIORS) if used is None else used
    paths: list[str] = []
    for name in keys:
        ctor = ALL_PRIORS.get(name)
        if ctor is None:
            continue
        paths.append(plot_and_save(ctor(), output_dir, f"prior_{name}"))
    return paths
