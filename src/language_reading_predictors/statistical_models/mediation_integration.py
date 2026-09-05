# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Integrate mediator uncertainty within each posterior draw.

The resulting draws contain posterior uncertainty alone. ESS and the MCSE of
their median describe MCMC precision; they cannot measure inner simulation error.
Count mediators have finite support and are summed exactly. Normal mediators use
successively finer Gauss-Hermite quadrature and fail if the cells do not stabilise.
"""

from collections.abc import Callable

import numpy as np
from scipy.special import roots_hermitenorm
from scipy.stats import betabinom

from language_reading_predictors.statistical_models.preprocessing import logit_safe

NORMAL_INTEGRATION_TOLERANCE = 1e-10
NORMAL_INTEGRATION_ORDERS = (32, 64, 128, 256, 512)
NORMAL_EFFECT_DISTRIBUTION_TOLERANCE = 0.001


def count_mass(k: int | np.ndarray, n: int, p: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """Beta-binomial mass with dispersion broadcast over the observation rows."""
    return np.asarray(betabinom.pmf(k, n, p * kappa[:, None], (1 - p) * kappa[:, None]))


def count_cells(
    outcome_p: Callable[[float, np.ndarray | float], np.ndarray],
    p_treat: np.ndarray,
    p_ctrl: np.ndarray,
    kappa: np.ndarray,
    n: int,
    mean: float,
    sd: float,
) -> np.ndarray:
    """Return E[Y(1,M(1))], E[Y(0,M(0))], E[Y(1,M(0))] per draw."""
    cells = np.zeros((3, p_treat.shape[0]))
    for k in range(n + 1):
        z = (logit_safe(np.asarray(k), n) - mean) / sd
        weight_t = count_mass(k, n, p_treat, kappa)
        weight_c = count_mass(k, n, p_ctrl, kappa)
        y_t, y_c = outcome_p(1.0, z), outcome_p(0.0, z)
        cells[0] += (y_t * weight_t).mean(axis=1)
        cells[1] += (y_c * weight_c).mean(axis=1)
        cells[2] += (y_t * weight_c).mean(axis=1)
    return cells


def normal_cells(
    outcome_p: Callable[[float, np.ndarray], np.ndarray],
    mu_treat: np.ndarray,
    mu_ctrl: np.ndarray,
    sigma: np.ndarray,
    *,
    ci_prob: float = 0.89,
) -> np.ndarray:
    """Adapt quadrature until every draw's averaged counterfactual cell agrees.

    The comparison measures numerical integration error independently of posterior
    sampling. It is a convergence check, not a mathematical error bound. A failed
    check stops the decomposition rather than publishing unqualified intervals.
    Effect quantiles must change by no more than 0.1% of the reported interval's
    half-width (with a floating-point floor), and P(effect > 0) by at most 0.001.
    """
    previous = None
    for order in NORMAL_INTEGRATION_ORDERS:
        nodes, weights = roots_hermitenorm(order)
        cells = np.zeros((3, mu_treat.shape[0]))
        for node, weight in zip(nodes, weights / np.sqrt(2 * np.pi), strict=True):
            z_t = mu_treat + sigma[:, None] * node
            z_c = mu_ctrl + sigma[:, None] * node
            cells[0] += weight * outcome_p(1.0, z_t).mean(axis=1)
            cells[1] += weight * outcome_p(0.0, z_c).mean(axis=1)
            cells[2] += weight * outcome_p(1.0, z_c).mean(axis=1)
        if previous is not None and np.all(np.isfinite(cells)):
            effects = cells[[0, 2, 0]] - cells[[1, 1, 2]]
            old_effects = previous[[0, 2, 0]] - previous[[1, 1, 2]]
            tail = (1 - ci_prob) / 2
            probabilities = [tail, 0.25, 0.5, 0.75, 1 - tail]
            quantiles = np.quantile(effects, probabilities, axis=1)
            old_quantiles = np.quantile(old_effects, probabilities, axis=1)
            half_width = (quantiles[-1] - quantiles[0]) / 2
            allowed_change = np.maximum(
                half_width * NORMAL_EFFECT_DISTRIBUTION_TOLERANCE,
                32 * np.finfo(float).eps,
            )
            direction_change = np.abs((effects > 0).mean(axis=1) - (old_effects > 0).mean(axis=1))
            if (
                np.max(np.abs(cells - previous)) <= NORMAL_INTEGRATION_TOLERANCE
                and np.all(np.abs(quantiles - old_quantiles) <= allowed_change)
                and np.all(direction_change <= NORMAL_EFFECT_DISTRIBUTION_TOLERANCE)
            ):
                return cells
        previous = cells
    raise ValueError(
        f"Normal mediator integration did not converge at {NORMAL_INTEGRATION_ORDERS[-1]} quadrature nodes"
    )
