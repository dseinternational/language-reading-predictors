# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Hilbert-space Gaussian-process (HSGP) helpers.

The HSGP machinery now lives in :mod:`dse_research_utils.statistics.models.hsgp`;
these wrappers preserve the project's prior defaults and forward explicit basis
designs to the shared constructor:

- :func:`build_hsgp_1d` — amplitude ``eta_main_prior`` (``HalfNormal(0.3)``),
  lengthscale ``ell_prior`` (``InverseGamma(3, 1)``). Age / own-baseline main effects.
- :func:`build_tau_modifier` — tighter amplitude ``eta_tau_prior`` for
  effect-modification GPs (e.g. age-varying treatment effect).
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from dse_research_utils.statistics.models.hsgp import build_hsgp_1d as _build_hsgp_1d
from preliz.distributions.distributions import Continuous

from language_reading_predictors.statistical_models import priors as _priors
from language_reading_predictors.statistical_models.hsgp_migration import HSGP_BASIS_VERSION


def build_hsgp_1d(
    name: str,
    X: np.ndarray,
    *,
    m: int = 20,
    c: float = 1.5,
    L: float | None = None,
    center: float | None = None,
    amplitude_prior: Continuous | None = None,
    lengthscale_prior: Continuous | None = None,
    ls_range: tuple[float, float] | None = None,
) -> pt.TensorVariable:
    """Project 1D HSGP: defaults to ``eta_main_prior`` / ``ell_prior``."""
    curve = _build_hsgp_1d(
        name,
        X,
        m=m,
        c=c,
        L=L,
        center=center,
        amplitude_prior=amplitude_prior or _priors.eta_main_prior(),
        lengthscale_prior=lengthscale_prior or _priors.ell_prior(),
        ls_range=ls_range,
    )
    pm.modelcontext(None)._lrp_hsgp_basis_version = HSGP_BASIS_VERSION
    return curve


def build_tau_modifier(
    name: str,
    X: np.ndarray,
    *,
    m: int = 15,
    c: float = 1.5,
    L: float | None = None,
    center: float | None = None,
    amplitude_prior: Continuous | None = None,
    lengthscale_prior: Continuous | None = None,
    ls_range: tuple[float, float] | None = None,
) -> pt.TensorVariable:
    """Project effect-modification HSGP: defaults to the tight ``eta_tau_prior``."""
    curve = _build_hsgp_1d(
        name,
        X,
        m=m,
        c=c,
        L=L,
        center=center,
        amplitude_prior=amplitude_prior or _priors.eta_tau_prior(),
        lengthscale_prior=lengthscale_prior or _priors.ell_prior(),
        ls_range=ls_range,
    )
    pm.modelcontext(None)._lrp_hsgp_basis_version = HSGP_BASIS_VERSION
    return curve
