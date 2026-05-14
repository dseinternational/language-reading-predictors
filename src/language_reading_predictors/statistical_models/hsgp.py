# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Hilbert-space Gaussian-process (HSGP) helpers.

Two constructors are exposed:

- :func:`build_hsgp_1d` — 1D HSGP with default amplitude ~HalfNormal(1.0) and
  InverseGamma(3, 1) lengthscale. Used for age main effects and own-baseline
  main effects.
- :func:`build_tau_modifier` — 1D HSGP with tighter amplitude ~HalfNormal(0.3).
  Used for effect-modification GPs (e.g. age-varying treatment effect).

Lengthscale calibration follows ``pm.gp.hsgp_approx.approx_hsgp_hyperparams``
so that the basis size ``m`` and the boundary factor ``c`` cover the observed
input range at the chosen lengthscale floor.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from preliz.distributions.distributions import Continuous

from language_reading_predictors.statistical_models import priors as _priors


def _approx_hsgp_params(
    x: np.ndarray,
    ls_range: tuple[float, float],
    c_floor: float | None = None,
) -> tuple[list[int], list[float]]:
    """Return ``(m, L)`` sized to cover ``x`` for the given lengthscale range.

    ``c_floor`` (optional) is a minimum boundary factor enforced after
    :func:`pm.gp.hsgp_approx.approx_hsgp_hyperparams` proposes its own
    ``c``. PyMC recommends ``c ≥ 1.2``; pass a tighter floor here when
    callers know the lengthscale prior has a long upper tail.
    """
    x = np.asarray(x, dtype=float)
    x_min, x_max = float(x.min()), float(x.max())
    m, c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
        x_range=[x_min, x_max],
        lengthscale_range=list(ls_range),
        cov_func="expquad",
    )
    if c_floor is not None:
        c = max(float(c), float(c_floor))
    S = max(abs(x_min), abs(x_max))
    return [int(m)], [float(S * c)]


def build_hsgp_1d(
    name: str,
    X: np.ndarray,
    *,
    m: int = 20,
    c: float = 1.5,
    amplitude_prior: Continuous | None = None,
    lengthscale_prior: Continuous | None = None,
    ls_range: tuple[float, float] | None = None,
) -> pt.TensorVariable:
    """
    Construct a 1D HSGP prior evaluated at ``X`` (standardised input).

    Parameters
    ----------
    name
        Variable-name prefix. The eta, ell, and GP basis weights are named
        ``"{name}__eta"``, ``"{name}__ell"`` and ``"{name}__g_unit"``.
    X
        Standardised 1D inputs, shape ``(n,)`` or ``(n, 1)``.
    m, c
        HSGP basis size and boundary factor. If ``ls_range`` is supplied they
        are recomputed via :func:`pm.gp.hsgp_approx.approx_hsgp_hyperparams`.
    amplitude_prior
        preliz distribution for eta. Defaults to ``HalfNormal(1.0)``.
    lengthscale_prior
        preliz distribution for ell. Defaults to ``InverseGamma(3, 1)``.
    ls_range
        Optional ``(ls_low, ls_high)`` on the standardised scale. When
        provided, ``m`` and ``c`` are calibrated from it.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if ls_range is not None:
        # When the caller supplies an explicit ``c``, treat it as a
        # minimum floor on the calibrated boundary factor — otherwise
        # the kwarg was silently ignored.
        m_list, L = _approx_hsgp_params(X[:, 0], ls_range, c_floor=c)
        m_val = m_list[0]
        L_val = L
    else:
        S = float(max(abs(X.min()), abs(X.max())))
        m_val = int(m)
        L_val = [S * float(c)]

    amplitude = (amplitude_prior or _priors.eta_main_prior()).to_pymc(f"{name}__eta")
    lengthscale = (lengthscale_prior or _priors.ell_prior()).to_pymc(f"{name}__ell")

    cov = pm.gp.cov.ExpQuad(1, ls=lengthscale)
    hsgp = pm.gp.HSGP(cov_func=cov, m=[m_val], L=L_val)
    g_unit = hsgp.prior(f"{name}__g_unit", X=X)
    return pm.Deterministic(name, amplitude * g_unit)


def build_tau_modifier(
    name: str,
    X: np.ndarray,
    *,
    m: int = 15,
    c: float = 1.5,
    amplitude_prior: Continuous | None = None,
    lengthscale_prior: Continuous | None = None,
    ls_range: tuple[float, float] | None = None,
) -> pt.TensorVariable:
    """Same API as :func:`build_hsgp_1d` but with tighter default eta prior."""
    return build_hsgp_1d(
        name,
        X,
        m=m,
        c=c,
        amplitude_prior=amplitude_prior or _priors.eta_tau_prior(),
        lengthscale_prior=lengthscale_prior,
        ls_range=ls_range,
    )
