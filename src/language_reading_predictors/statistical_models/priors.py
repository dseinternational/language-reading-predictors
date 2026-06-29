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


# ---------------------------------------------------------------------------
# Shared prior template (document these on the report)
# ---------------------------------------------------------------------------


def alpha_prior() -> Continuous:
    """Intercept alpha ~ Normal(0, 1.5)."""
    return pz.Normal(mu=0.0, sigma=1.5)


def tau_prior() -> Continuous:
    """Treatment effect tau ~ Normal(0, 0.5)."""
    return pz.Normal(mu=0.0, sigma=0.5)


def gamma_own_prior() -> Continuous:
    """Own-baseline coupling gamma_own ~ Normal(1, 0.5)."""
    return pz.Normal(mu=1.0, sigma=0.5)


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


def kappa_prior() -> Continuous:
    """Beta-binomial concentration kappa ~ HalfNormal(50)."""
    return pz.HalfNormal(sigma=50.0)


def beta_mech_prior() -> Continuous:
    """Linear-mechanism slope beta_mech ~ Normal(0, 1).

    Used by ``build_mechanism_model(linear_mechanism=True)`` in place of the
    HSGP ``f_mech`` on low / floored count outcomes (e.g. nonword decoding,
    LRP72). The input is the standardised ``z(logit L_post)``, so the slope is
    the change in the outcome logit per 1 SD of the mechanism on the logit
    scale; the weakly-informative unit scale lets the data speak for the
    primary effect while still regularising.
    """
    return pz.Normal(mu=0.0, sigma=1.0)


def b_path_prior() -> Continuous:
    """Mediator -> outcome slope (b-path) ~ Normal(0, 1).

    Used by the LRP59 mediation outcome model for ``b_M``, the coefficient on the
    standardised mediator ``z(logit L_t2)``. Weakly-informative on the unit
    (per-SD) scale so the data identify the key b-path of the decomposition,
    while still regularising; the treatment and confounder couplings use the
    tighter ``tau_prior`` / ``gamma_cross_prior``.
    """
    return pz.Normal(mu=0.0, sigma=1.0)


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


def eta_partial_pool_prior() -> Continuous:
    """Joint-model outcome-specific age-GP amplitude ~ HalfNormal(0.3)."""
    return pz.HalfNormal(sigma=0.3)


def predictor_slope_prior(sigma: float = 0.5) -> Continuous:
    """LRP65 standardised-predictor slope ~ Normal(0, sigma) (default 0.5).

    Per-SD coefficient on a standardised baseline predictor in the between-child
    adjusted model (letter sounds, language composite, blending, age, and the
    tested covariates). Fixed weakly-informative and regularising, given the
    collinear general-ability cluster and n ~ 54; the which-predictors-clear-zero
    conclusion is checked against ``sigma`` in {0.3, 0.7}.
    """
    return pz.Normal(mu=0.0, sigma=sigma)


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
    "beta_mech": beta_mech_prior,
    "b_path": b_path_prior,
    "sigma_mediator": sigma_mediator_prior,
    "eta_partial_pool": eta_partial_pool_prior,
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
    "tau": "causal",
    "gamma_own": "precision",
    "gamma_cross": "association",
    "gamma_age": "precision",
    "kappa": "nuisance",
    "beta_mech": "association",
    "b_path": "association",
    "sigma_mediator": "nuisance",
    "eta_main": "gp",
    "eta_tau": "gp",
    "ell": "gp",
    "eta_partial_pool": "gp",
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
    "beta_dose": "tau",
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
    "b_M": "b_path",
    "sigma_M": "sigma_mediator",
    "eta_main": "eta_main",
    "eta_tau": "eta_tau",
    "ell": "ell",
    "eta_partial_pool": "eta_partial_pool",
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
}


def _first_docline(ctor) -> str:
    """First line of a constructor's docstring (the prior's rationale)."""
    return (ctor.__doc__ or "").strip().split("\n")[0].strip()


def _dist_from_doc(ctor) -> str:
    """Extract the distribution signature (e.g. ``Normal(0, 0.5)``) from the doc."""
    line = _first_docline(ctor)
    m = re.search(r"~\s*([A-Za-z]+\([^)]*\))", line)
    return m.group(1) if m else ""


def _ctor_key_for_rv(rv_name: str) -> str | None:
    """Constructor key backing an RV, by exact name then by prefix fallback."""
    base = rv_name.split("[")[0]
    if base in _RV_TO_CTOR:
        return _RV_TO_CTOR[base]
    # Adjusted couplings are all gamma_cross-scaled (covariate, moderator,
    # skill, mediator b-paths share the weakly-informative cross prior).
    if base.startswith(("gamma", "b_", "a_")):
        return "gamma_cross"
    return None


def prior_info_for_rv(rv_name: str) -> dict[str, str]:
    """``{parameter, distribution, role, rationale, panel}`` for a registered RV.

    ``panel`` is the basename (without ``prior_`` / extension) of the prior-PDF
    panel for this parameter, or ``""`` for inline / unmapped priors that have no
    panel — the report maps a table row to ``prior_{panel}.svg``.
    """
    base = rv_name.split("[")[0]
    if base in _INLINE_PRIORS:
        info = _INLINE_PRIORS[base]
        return {"parameter": rv_name, **info, "panel": ""}
    key = _ctor_key_for_rv(rv_name)
    if key is None:
        return {
            "parameter": rv_name,
            "distribution": "(model prior)",
            "role": "other",
            "rationale": "",
            "panel": "",
        }
    ctor = ALL_PRIORS[key]
    return {
        "parameter": rv_name,
        "distribution": _dist_from_doc(ctor),
        "role": _ROLE_BY_CTOR[key],
        "rationale": _first_docline(ctor),
        "panel": key,
    }


def used_prior_keys(model) -> list[str]:
    """Constructor keys actually registered by ``model`` (for panel pruning)."""
    keys: list[str] = []
    for rv in list(model.free_RVs) + list(getattr(model, "deterministics", [])):
        key = _ctor_key_for_rv(rv.name)
        if key is not None and key in ALL_PRIORS and key not in keys:
            keys.append(key)
    return keys


def priors_table(model) -> pd.DataFrame:
    """Per-model prior table with a ``role`` column (issue #125 Area 1).

    One row per registered free RV (vector coefficients collapse to one row),
    driven by the *actual* model so it never lists priors the model did not use
    and captures the inline ``alpha_phase`` / ``alpha_time`` / ``sigma_child``
    priors a SHARED_PRIORS-only table would miss. Columns: ``parameter``,
    ``distribution``, ``role``, ``rationale``.
    """
    rows = [prior_info_for_rv(rv.name) for rv in model.free_RVs]
    return pd.DataFrame(
        rows, columns=["parameter", "distribution", "role", "rationale", "panel"]
    )


def plot_and_save(dist: Continuous, output_dir: str, name: str) -> str:
    """Plot a prior PDF and save as ``{name}.png``.

    PNG only: the reports prefer raster images so model-output pages stay quick
    to browse (no large SVGs to render in the viewer).
    """
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(5, 3))
    try:
        dist.plot_pdf()
    except Exception:
        # Some preliz distributions (e.g. InverseGamma) need an explicit axis.
        ax = plt.gca()
        dist.plot_pdf(pointinterval=False, ax=ax)
    png = os.path.join(output_dir, f"{name}.png")
    plt.title(name)
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png


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
