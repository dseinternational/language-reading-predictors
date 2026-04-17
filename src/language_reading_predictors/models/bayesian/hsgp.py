# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
HSGP (Hilbert Space GP) helpers for DAG edges.

Each GP edge contributes ``eta * f(x_std)`` to its child's linear
predictor, where ``f`` is a zero-mean HSGP with ``ExpQuad`` covariance.
Parents also contribute a linear term ``β·x_std`` — the GP captures
variation from the central linear trend.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from language_reading_predictors.models.bayesian.definition import ParentSpec


def standardise(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return (z, mean, sd). Unit-variance, zero-mean standardisation.

    ``sd`` is floored at ``1e-6`` so downstream division is safe on
    degenerate inputs.
    """
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x, ddof=0))
    if sd < 1e-6:
        sd = 1.0
    return (x - mu) / sd, mu, sd


def build_gp_edge(
    name: str,
    x_std: np.ndarray,
    parent: ParentSpec,
) -> pt.TensorVariable:
    """Construct an HSGP contribution for a single edge.

    Parameters
    ----------
    name
        Edge label (e.g. ``"A_to_W"``). Used to namespace PyMC variables.
    x_std
        Standardised 1-D parent values (zero mean, unit SD) at the
        observed data points.
    parent
        The :class:`ParentSpec` providing GP hyperparameters.

    Returns
    -------
    pt.TensorVariable
        A 1-D tensor of length ``len(x_std)`` giving the GP's contribution
        to the linear predictor at each observation.
    """
    x_std = np.asarray(x_std, dtype=float)
    x_range = float(np.max(x_std) - np.min(x_std))
    # Choose L generously so boundary artefacts don't bite.
    L = max(1.5, 1.5 * float(np.max(np.abs(x_std))), 1.5 * x_range / 2.0)

    log_ell = pm.Normal(
        f"log_ell__{name}",
        mu=parent.ell_log_mu,
        sigma=parent.ell_log_sigma,
    )
    ell = pm.Deterministic(f"ell__{name}", pm.math.exp(log_ell))
    eta = pm.HalfNormal(f"eta__{name}", sigma=parent.eta_sigma)

    cov = pm.gp.cov.ExpQuad(1, ls=ell)
    hsgp = pm.gp.HSGP(cov_func=cov, m=[parent.n_basis], L=[L])
    f_unit = hsgp.prior(f"f__{name}", X=x_std.reshape(-1, 1))
    return eta * f_unit
