# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Beta-binomial likelihood helpers used by the statistical-model factories.

The ordinary logit implementation lives in the shared package and remains
re-exported here.  The RLI phoneme-blending sensitivity additionally needs a
mechanically justified three-choice guessing floor while retaining the same
Beta-binomial observation family.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from dse_research_utils.statistics.models.likelihood import beta_binomial_from_logit
from dse_research_utils.math.constants import EPSILON

ScoreMeanLink = Literal["logit", "three_choice_guessing_floor"]
SCORE_MEAN_LINKS: tuple[ScoreMeanLink, ...] = (
    "logit",
    "three_choice_guessing_floor",
)

def apply_score_mean_link(
    unit_probability: Any,
    score_mean_link: ScoreMeanLink,
) -> Any:
    """Map an inverse-logit probability to the declared score-mean scale.

    ``unit_probability`` may be a NumPy array or a PyTensor expression.  The
    ordinary link returns it unchanged.  The phoneme-blending sensitivity maps
    it onto ``[1/3, 1]`` because each item has three response alternatives.
    """

    if score_mean_link == "logit":
        return unit_probability
    if score_mean_link == "three_choice_guessing_floor":
        return (1.0 / 3.0) + (2.0 / 3.0) * unit_probability
    raise ValueError(
        f"score_mean_link must be one of {SCORE_MEAN_LINKS}, "
        f"got {score_mean_link!r}"
    )


def beta_binomial_from_score_mean_link(
    name: str,
    eta: pt.TensorVariable,
    n_trials: int | np.ndarray,
    kappa: pt.TensorVariable,
    *,
    score_mean_link: ScoreMeanLink = "logit",
    observed: np.ndarray | None = None,
    dims: tuple[str, ...] | str | None = None,
) -> pt.TensorVariable:
    """Register a Beta-Binomial node under the selected score-mean link."""

    if score_mean_link == "logit":
        return beta_binomial_from_logit(
            name,
            eta,
            n_trials=n_trials,
            kappa=kappa,
            observed=observed,
            dims=dims,
        )

    mu = apply_score_mean_link(pm.math.sigmoid(eta), score_mean_link)
    mu_clip = pm.math.clip(mu, EPSILON, 1.0 - EPSILON)
    return pm.BetaBinomial(
        name,
        n=n_trials,
        alpha=mu_clip * kappa,
        beta=(1.0 - mu_clip) * kappa,
        observed=observed,
        dims=dims,
    )


__all__ = [
    "SCORE_MEAN_LINKS",
    "ScoreMeanLink",
    "apply_score_mean_link",
    "beta_binomial_from_logit",
    "beta_binomial_from_score_mean_link",
]
