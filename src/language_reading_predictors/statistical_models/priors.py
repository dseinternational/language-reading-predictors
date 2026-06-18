# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Named prior constructors shared across LRP52-LRP58.

Every factory calls the same function so priors cannot drift between models.
Priors are defined as ``preliz`` distributions; call ``.to_pymc(name)`` inside
a PyMC model block to register them, or ``plot_and_save`` to output an SVG/PNG
for the Quarto report.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
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
    standardised phonics-route composite modelled as ``Normal(mu_M, sigma_M)``.
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
    """LRP55 outcome-specific age-GP amplitude ~ HalfNormal(0.3)."""
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
    "kappa": kappa_prior,
    "predictor_slope": predictor_slope_prior,
    "eta_main": eta_main_prior,
    "eta_tau": eta_tau_prior,
    "ell": ell_prior,
}


def plot_and_save(dist: Continuous, output_dir: str, name: str) -> str:
    """Plot a prior PDF and save as ``{name}.png`` + ``{name}.svg``."""
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(5, 3))
    try:
        dist.plot_pdf()
    except Exception:
        # Some preliz distributions (e.g. InverseGamma) need an explicit axis.
        ax = plt.gca()
        dist.plot_pdf(pointinterval=False, ax=ax)
    png = os.path.join(output_dir, f"{name}.png")
    svg = os.path.join(output_dir, f"{name}.svg")
    plt.title(name)
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png


def save_shared_prior_panel(output_dir: str) -> list[str]:
    """Plot every prior in :data:`SHARED_PRIORS` and return the generated files."""
    paths: list[str] = []
    for name, ctor in SHARED_PRIORS.items():
        paths.append(plot_and_save(ctor(), output_dir, f"prior_{name}"))
    return paths
