# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Concurrent calculations and summaries."""

from __future__ import annotations

from language_reading_predictors.statistical_models.posteriors import REPORTING_CI_PROB

from collections.abc import Sequence
from dataclasses import dataclass
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import expit
from language_reading_predictors.statistical_models.likelihood import (
    ScoreMeanLink,
    apply_score_mean_link,
)
from language_reading_predictors.statistical_models.posteriors import (
    band50,
)


@dataclass
class ConcurrentTerm:
    """One standardised predictor for the concurrent-associations items-scale marginals (#312).

    The concurrent family (``kind="concurrent"``) fits, per wave, a between-child
    Beta-Binomial regression of the focal outcome's *level* on the standardised
    same-wave logits of a set of predictor skills (main effects only — no
    interactions, unlike the gain family). Each predictor's coefficient
    ``beta_{label}`` is therefore per-SD-of-the-raw-logit, and a ``+1 SD`` (or, for
    a bounded-count predictor, a ``+k items``) perturbation maps to a *scalar*
    linear-predictor shift per posterior draw — so :func:`concurrent_marginals`
    needs none of the per-observation interaction machinery of
    :func:`association_marginals`.

    Attributes
    ----------
    label
        Predictor name for the report row (e.g. ``"L"``, ``"TR"``, ``"age"``).
    coef
        Posterior variable name of the predictor's standardised main-effect
        coefficient (``"beta_L"`` etc.).
    sd_logit
        SD of the predictor's raw same-wave logit on the fitted rows — the data-scale
        size of ``+1 SD``. A ``+k items`` increment at the mean operating point is
        ``Δz = (logit_safe(ȳ + k, N) − logit_safe(ȳ, N)) / sd_logit``
        standardised units, using the fitted Haldane-corrected transformation.
    n_items
        Denominator of the predictor when it is a bounded-count measure; enables the
        ``+k items`` row. ``None`` for age / continuous predictors.
    mean_items
        Mean bounded-count predictor score on the fitted rows — the operating point at
        which the ``+k items`` perturbation is evaluated with the same
        Haldane-corrected logit used to fit the model.
    k_items
        The per-predictor items increment for the ``+k items`` row (the pipeline sets
        it per measure, e.g. ``max(1, round(n_items / 10))``, so a fixed ``+5`` does
        not span 3 %–50 % of scales that differ tenfold — the #310/#325 caveat).
    """

    label: str
    coef: str
    sd_logit: float
    n_items: int | None = None
    mean_items: float | None = None
    k_items: int | None = None


def concurrent_marginals(
    trace: xr.DataTree,
    *,
    terms: Sequence[ConcurrentTerm],
    n_trials: int,
    eta_name: str = "eta",
    ci_prob: float = REPORTING_CI_PROB,
    group: str = "posterior",
    score_mean_link: ScoreMeanLink = "logit",
) -> pd.DataFrame:
    """Per-predictor items-scale marginals for the concurrent family (#312).

    For each predictor in ``terms`` it forms the per-draw change in the linear
    predictor from a ``+1 SD`` perturbation of that predictor (and, for a
    bounded-count predictor, a ``+k items`` perturbation at the mean operating
    point), holding every other predictor at its observed value, and averages the
    response-scale change ``m(η + Δη) − m(η)`` over the fitted rows, where ``m`` is
    the fitted score mean ``score_mean_link ∘ expit``.
    Reported on the probability and items scales (``n_trials`` = the *focal
    outcome's* denominator × probability), with an equal-tailed ``ci_prob`` interval
    and an inner 50 % band.

    ``score_mean_link`` must be the link the model was **built** with: under the
    phoneme-blending guessing floor the same ``Δη`` maps to a smaller response-scale
    change, so summarising a floor-link posterior at the default would overstate
    every association in items (#619).

    Because the concurrent model has **no interaction terms**, the shift is a scalar
    per draw: ``Δη_s = β_s · Δz`` where ``Δz = 1`` for ``+1 SD`` and
    ``Δz = (logit_safe(ȳ + k, N) − logit_safe(ȳ, N)) / sd_logit`` for ``+k
    items``, where ``logit_safe`` is the Haldane-corrected transformation used in the
    factory. This helper applies equally to adjusted and bivariate traces; callers
    label that fit distinction in the output. Every row carries
    ``role = "association"``; no term here is causal (post-treatment conditioning is
    intentional, per the family's documented estimand).
    """
    posterior = getattr(trace, group)
    eta = (
        posterior[eta_name]
        .stack(sample=("chain", "draw"))
        .transpose("obs_id", "sample")
        .values
    )  # (n_obs, S)

    lo_q = (1 - ci_prob) / 2
    hi_q = 1 - lo_q
    rows: list[dict[str, float | str]] = []

    for term in terms:
        beta = posterior[term.coef].stack(sample=("chain", "draw")).values.ravel()  # (S,)

        perturbations: list[tuple[str, float]] = [("+1 SD", 1.0)]
        if (
            term.n_items
            and term.mean_items is not None
            and np.isfinite(term.mean_items)
            and term.k_items
            and term.sd_logit > 0
            and np.isfinite(term.sd_logit)
        ):
            from language_reading_predictors.statistical_models.preprocessing import (
                logit_safe,
            )

            y = float(np.clip(term.mean_items, 0.0, term.n_items))
            # Cap the increment to the largest whole-item shift that reaches no farther
            # than the instrument ceiling. The Haldane correction is finite at both
            # boundaries, so a shift that lands exactly on the ceiling is valid.
            max_k = int(np.floor(term.n_items - y))
            k_eff = min(int(term.k_items), max_k)
            if k_eff >= 1:
                raw = logit_safe(np.asarray([y]), term.n_items)[0]
                raw_k = logit_safe(np.asarray([y + k_eff]), term.n_items)[0]
                dz = (raw_k - raw) / term.sd_logit
                perturbations.append((f"+{k_eff} items", dz))

        for scale_label, dz in perturbations:
            delta_eta = beta * dz  # (S,), scalar shift per draw (no interactions)
            # Map both operating points through the fitted score mean before
            # differencing: under a non-identity link the response-scale change is
            # not the logit-scale one rescaled (#619).
            ame_prob = (
                apply_score_mean_link(
                    expit(eta + delta_eta[None, :]), score_mean_link
                )
                - apply_score_mean_link(expit(eta), score_mean_link)
            ).mean(axis=0)  # (S,)
            ame_items = float(n_trials) * ame_prob
            prob_lo50, prob_hi50 = band50(ame_prob)
            items_lo50, items_hi50 = band50(ame_items)
            rows.append(
                {
                    "term": term.label,
                    "role": "association",
                    "scale": scale_label,
                    "prob_median": float(np.median(ame_prob)),
                    "prob_lo": float(np.quantile(ame_prob, lo_q)),
                    "prob_hi": float(np.quantile(ame_prob, hi_q)),
                    "prob_lo50": prob_lo50,
                    "prob_hi50": prob_hi50,
                    "items_median": float(np.median(ame_items)),
                    "items_lo": float(np.quantile(ame_items, lo_q)),
                    "items_hi": float(np.quantile(ame_items, hi_q)),
                    "items_lo50": items_lo50,
                    "items_hi50": items_hi50,
                    "prob_pos": float(np.mean(ame_items > 0)),
                }
            )
    return pd.DataFrame(rows)
