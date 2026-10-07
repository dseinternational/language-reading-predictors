# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Beta-binomial likelihood helpers used by the statistical-model factories.

The ordinary logit implementation is re-exported from the shared package.
The RLI phoneme-blending companion adds a floor under the assumed random-guessing
mechanism while retaining the Beta-Binomial observation family.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from dse_research_utils.statistics.models.likelihood import (
    beta_binomial_from_logit,
    beta_binomial_from_p,
)

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
    raise ValueError(f"score_mean_link must be one of {SCORE_MEAN_LINKS}, got {score_mean_link!r}")


def invert_score_mean_link(
    score_mean: Any,
    score_mean_link: ScoreMeanLink,
) -> Any:
    """Map a score mean back onto the inverse-logit (unit) scale.

    Invert :func:`apply_score_mean_link` before converting an observed score
    location to a linear-predictor anchor. For the guessing-floor link, require
    a mean strictly between 1/3 and 1 so its logit is finite. The ordinary link
    returns its input without range validation.
    """

    if score_mean_link == "logit":
        return score_mean
    if score_mean_link == "three_choice_guessing_floor":
        unit = (np.asarray(score_mean, dtype=float) - (1.0 / 3.0)) / (2.0 / 3.0)
        if np.any(unit <= 0.0) or np.any(unit >= 1.0):
            raise ValueError(
                "score mean is outside the three-choice guessing floor's range "
                "(1/3, 1), so it has no inverse-logit representation: "
                f"{np.asarray(score_mean, dtype=float)}"
            )
        return unit if np.ndim(score_mean) else float(unit)
    raise ValueError(f"score_mean_link must be one of {SCORE_MEAN_LINKS}, got {score_mean_link!r}")


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

    # The clip -> (alpha, beta) -> BetaBinomial construction is the shared
    # beta_binomial_from_p; only the link applied to the mean is ours.
    return beta_binomial_from_p(
        name,
        apply_score_mean_link(pm.math.sigmoid(eta), score_mean_link),
        n_trials,
        kappa,
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
