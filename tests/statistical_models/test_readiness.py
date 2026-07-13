# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the readiness-threshold post-processing (#230 §2).

The estimand is a posterior for the "knee" of a mechanism curve. These build a
synthetic per-draw curve with a *known* inflection and check the recovered knee,
that a flat/noise curve is flagged as ill-defined, and the input guards.
"""

from __future__ import annotations

import numpy as np
import pytest

from language_reading_predictors.statistical_models.readiness import (
    readiness_threshold,
)


def _logistic_ramp(x: np.ndarray, x0: float, *, n_sample: int, seed: int = 0):
    """Increasing logistic curves in ``x`` with inflection (steepest ascent) at x0."""
    rng = np.random.default_rng(seed)
    base = 1.5 / (1.0 + np.exp(-3.0 * (x - x0)))  # inflection at x = x0
    amp = rng.normal(1.0, 0.1, size=n_sample).clip(0.5)
    noise = rng.normal(0.0, 0.02, size=(x.size, n_sample))
    return base[:, None] * amp[None, :] + noise


def test_recovers_known_knee():
    N = 32
    x = np.linspace(-3.0, 3.0, 60)
    x0 = 0.4  # true steepest-ascent point (logit scale)
    df = readiness_threshold(x, _logistic_ramp(x, x0, n_sample=300), N)

    assert set(df["estimand"]) == {"knee (max slope)", "half rise"}
    knee = df[df["estimand"] == "knee (max slope)"].iloc[0]
    # recovered inflection close to x0, on both the logit and raw-count scales
    assert abs(knee.logit_median - x0) < 0.6
    assert abs(knee.count_median - N / (1.0 + np.exp(-x0))) < 4.0
    assert knee.monotone_frac > 0.9
    assert knee.obs_count_min <= knee.count_median <= knee.obs_count_max
    assert knee.n_draws == 300
    # half-rise of a symmetric logistic sits at the same inflection
    half = df[df["estimand"] == "half rise"].iloc[0]
    assert abs(half.logit_median - x0) < 0.7


def test_flags_flat_noise_curve():
    N = 32
    x = np.linspace(-3.0, 3.0, 60)
    rng = np.random.default_rng(1)
    f = rng.normal(0.0, 0.3, size=(x.size, 400))  # no trend
    knee = readiness_threshold(x, f, N)
    knee = knee[knee["estimand"] == "knee (max slope)"].iloc[0]
    # a trendless curve rises about half the time, so the knee is not well-defined
    assert 0.2 < knee.monotone_frac < 0.8


def test_input_guards():
    N = 32
    with pytest.raises(ValueError):  # < 4 distinct predictor values
        readiness_threshold(np.array([0.0, 1.0, 2.0]), np.zeros((3, 10)), N)
    with pytest.raises(ValueError):  # f_draws not aligned to mech_logit
        readiness_threshold(np.linspace(0.0, 1.0, 5), np.zeros((4, 10)), N)
