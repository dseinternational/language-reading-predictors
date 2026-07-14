# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the readiness-threshold post-processing (#230 §2/§5).

The estimand is a posterior for the "knee" (steepest rise) of a mechanism curve,
computed by ``reporting._readiness_knee`` on per-observation ``f_mech`` draws.
These build synthetic curves with a *known* steepest-rise point and check the
recovered knee and half-rise, that flat/falling curves are flagged as ill-defined
via ``increasing_frac``, and the input guards.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from language_reading_predictors.statistical_models import reporting


def _haldane_logit(counts: np.ndarray, n_trials: int) -> np.ndarray:
    return np.log((counts + 0.5) / (n_trials - counts + 0.5))


def _logistic_draws(
    L: np.ndarray, l0: float, *, n_sample: int, seed: int = 0
) -> np.ndarray:
    """Increasing logistic curves in the count L with steepest rise at ``l0``."""
    rng = np.random.default_rng(seed)
    base = 1.5 / (1.0 + np.exp(-0.4 * (L - l0)))
    amp = rng.normal(1.0, 0.1, size=n_sample).clip(0.5)
    noise = rng.normal(0.0, 0.02, size=(L.size, n_sample))
    return base[:, None] * amp[None, :] + noise


def test_recovers_known_knee_and_half_rise():
    n_trials, l0 = 32, 16.0
    counts = np.linspace(0.0, 32.0, 120)
    ell = _haldane_logit(counts, n_trials)
    out = reporting._readiness_knee(
        _logistic_draws(counts, l0, n_sample=300), ell, n_trials=n_trials
    )

    # The knee is quantised to between-bin midpoints, so allow ~a bin width.
    assert abs(out["knee_count_median"] - l0) < 5.5
    assert out["knee_count_ci_low"] <= out["knee_count_median"] <= out["knee_count_ci_high"]
    # A symmetric logistic reaches its mid-rise at the steepest point.
    assert abs(out["half_rise_count_median"] - l0) < 4.0
    assert out["increasing_frac"] > 0.9
    assert out["slope_above_knee_median"] >= out["slope_below_knee_median"]
    assert out["obs_count_min"] <= out["knee_count_median"] <= out["obs_count_max"]
    assert out["n_draws"] == 300


def test_flags_flat_noise_curve():
    n_trials = 32
    counts = np.linspace(0.0, 32.0, 120)
    ell = _haldane_logit(counts, n_trials)
    rng = np.random.default_rng(1)
    f = rng.normal(0.0, 0.3, size=(counts.size, 400))  # no trend
    out = reporting._readiness_knee(f, ell, n_trials=n_trials)
    # A trendless curve rises about half the time, so the knee is not well-defined.
    assert 0.2 < out["increasing_frac"] < 0.8


def test_falling_curve_yields_no_knee():
    n_trials = 32
    counts = np.linspace(0.0, 32.0, 120)
    ell = _haldane_logit(counts, n_trials)
    f = -_logistic_draws(counts, 16.0, n_sample=100)  # strictly falling
    out = reporting._readiness_knee(f, ell, n_trials=n_trials)
    # No increasing draws: the estimand summaries are undefined, not misleading.
    assert out["increasing_frac"] == 0.0
    assert math.isnan(out["knee_count_median"])
    assert math.isnan(out["half_rise_count_median"])


def test_input_guard_too_few_bins():
    n_trials = 32
    ell = np.zeros(50)  # one distinct predictor value -> one bin
    with pytest.raises(ValueError, match="bins"):
        reporting._readiness_knee(np.zeros((50, 10)), ell, n_trials=n_trials)
