# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Beta-binomial likelihood helper used by every factory."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from dse_research_utils.math.constants import EPSILON


def beta_binomial_from_logit(
    name: str,
    eta: pt.TensorVariable,
    n_trials: int | np.ndarray,
    kappa: pt.TensorVariable,
    observed: np.ndarray | None = None,
    dims: tuple[str, ...] | str | None = None,
) -> pt.TensorVariable:
    """
    Register a Beta-Binomial observation node parameterised by ``(logit(mu), kappa)``.

    ``alpha = mu * kappa``, ``beta = (1 - mu) * kappa``, with ``mu`` clipped
    away from 0/1 to avoid NaNs during sampling.
    """
    mu = pm.math.sigmoid(eta)
    mu_clip = pm.math.clip(mu, EPSILON, 1 - EPSILON)
    alpha = mu_clip * kappa
    beta = (1 - mu_clip) * kappa
    return pm.BetaBinomial(
        name,
        n=n_trials,
        alpha=alpha,
        beta=beta,
        observed=observed,
        dims=dims,
    )
