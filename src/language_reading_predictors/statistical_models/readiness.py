# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Readiness-threshold post-processing for mechanism curves (#230 §2).

Given the posterior of a mechanism model's nonparametric curve ``f_mech`` (the
outcome-logit contribution as a function of a predictor's level), summarise the
**knee**: the predictor level at which the outcome starts to move. For the
letter-sound -> word-reading mechanism (``lrp-rli-mech-058``) this answers the
instructional-sequencing question "does reading only move above ~k letter
sounds?".

Two per-draw summaries are computed, both reported in the predictor's raw units
(via ``count = n_trials * sigmoid(logit)``):

- ``knee (max slope)`` - the level of steepest ascent of the curve (argmax of the
  first derivative). This is "where the outcome accelerates fastest with the
  predictor".
- ``half rise`` - the level at which the curve first reaches the midpoint between
  its lowest and highest value over the observed range. More robust, complementary.

Both are computed per posterior draw and returned as a median + credible interval,
so the output is a genuine posterior for the threshold, not a point read off the
mean curve. ``monotone_frac`` (the share of draws whose curve rises overall) flags
how well-defined the knee is: a flat/non-monotone curve makes the knee noise, and
that shows up as a low ``monotone_frac`` and a wide interval.

This is post-processing of an already-fitted mechanism posterior; it fits nothing.
As with the mechanism curve itself it is an **adjusted association** (the mechanism
model is observational, ID-3), read as a feature of the fitted curve over the
*observed* predictor range - not extrapolated, and not a manipulation effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit


def readiness_threshold(
    mech_logit: np.ndarray,
    f_draws: np.ndarray,
    n_trials: int,
    *,
    n_grid: int = 200,
    ci_prob: float = 0.95,
) -> pd.DataFrame:
    """Posterior for the knee of a mechanism curve.

    Parameters
    ----------
    mech_logit:
        ``(n_obs,)`` logit of the predictor proportion - the curve's x-axis, as
        written by ``_write_mechanism_curve``.
    f_draws:
        ``(n_obs, n_sample)`` per-draw ``f_mech`` (outcome-logit contribution),
        aligned to ``mech_logit`` by observation.
    n_trials:
        Predictor denominator, mapping the logit x-axis to a raw count
        (``count = n_trials * sigmoid(logit)``).
    n_grid:
        Regular-grid resolution used to interpolate each draw's curve.
    ci_prob:
        Central credible-interval mass (equal-tailed), matching the mechanism
        curve's band.

    Returns
    -------
    A tidy frame with one row per estimand (``knee (max slope)``, ``half rise``),
    giving the posterior median and interval in raw predictor units and on the
    logit x-scale, plus ``monotone_frac``, ``n_draws`` and the observed count
    range.
    """
    x = np.asarray(mech_logit, dtype=float)
    f = np.asarray(f_draws, dtype=float)
    if f.ndim != 2 or f.shape[0] != x.shape[0]:
        raise ValueError(
            f"f_draws must be (n_obs, n_sample) aligned to mech_logit; "
            f"got {f.shape} vs n_obs={x.shape[0]}"
        )

    # f_mech is deterministic in x, so collapse duplicate x to one point (sorted).
    xu, idx = np.unique(x, return_index=True)
    fu = f[idx]  # (n_unique, n_sample)
    if xu.size < 4:
        raise ValueError(
            f"need >=4 distinct predictor values to locate a knee; got {xu.size}"
        )

    grid = np.linspace(xu.min(), xu.max(), n_grid)
    n_sample = fu.shape[1]
    knee_x = np.empty(n_sample)
    half_x = np.full(n_sample, np.nan)
    increasing = np.empty(n_sample, dtype=bool)

    for s in range(n_sample):
        fs = np.interp(grid, xu, fu[:, s])
        deriv = np.gradient(fs, grid)
        knee_x[s] = grid[int(np.argmax(deriv))]
        lo_f, hi_f = fs.min(), fs.max()
        target = 0.5 * (lo_f + hi_f)
        crossings = np.flatnonzero(fs >= target)
        if crossings.size:
            half_x[s] = grid[int(crossings[0])]
        increasing[s] = fs[-1] > fs[0]

    q_lo, q_hi = (1.0 - ci_prob) / 2.0, 1.0 - (1.0 - ci_prob) / 2.0

    def _summary(name: str, logit_draws: np.ndarray) -> dict:
        d = logit_draws[np.isfinite(logit_draws)]
        counts = n_trials * expit(d)
        return {
            "estimand": name,
            "count_median": float(np.median(counts)),
            "count_lo": float(np.quantile(counts, q_lo)),
            "count_hi": float(np.quantile(counts, q_hi)),
            "logit_median": float(np.median(d)),
            "logit_lo": float(np.quantile(d, q_lo)),
            "logit_hi": float(np.quantile(d, q_hi)),
            "ci_prob": ci_prob,
            "monotone_frac": float(increasing.mean()),
            "n_draws": int(n_sample),
            "obs_count_min": float(n_trials * expit(xu.min())),
            "obs_count_max": float(n_trials * expit(xu.max())),
        }

    return pd.DataFrame(
        [_summary("knee (max slope)", knee_x), _summary("half rise", half_x)]
    )
