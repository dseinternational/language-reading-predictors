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
    # A genuine interior S-curve is the one case that *should* qualify.
    assert out["knee_well_defined"] is True
    assert out["boundary_pinned"] is False
    assert out["prob_slope_above_gt_below"] >= 0.91
    assert out["scale"] == "latent_logit"


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


# ---------------------------------------------------------------------------
# Qualification: a net rise is not a threshold (#586 finding 1).
#
# Before this batch the only gate was ``increasing_frac > 0.9``, so a perfectly
# straight increasing line and a curve that accelerates to the edge of its support
# were both reported as a "well defined" knee. The letter-sound fits were the live
# case: 73% of draws put the argmax in the top interval and the knee median equalled
# its own 89% upper limit, i.e. the location was censored by the end of the data.
# ---------------------------------------------------------------------------


def test_linear_increasing_curve_is_not_a_well_defined_knee():
    """A straight line has no knee, whatever its net rise."""
    n_trials = 32
    counts = np.linspace(0.0, 32.0, 120)
    ell = _haldane_logit(counts, n_trials)
    rng = np.random.default_rng(3)
    slope = rng.normal(0.05, 0.004, size=400).clip(0.02)
    f = counts[:, None] * slope[None, :] + rng.normal(0.0, 0.01, size=(counts.size, 400))

    out = reporting._readiness_knee(f, ell, n_trials=n_trials)

    # Every draw rises end to end, so the *old* criterion would have passed it.
    assert out["increasing_frac"] > 0.9
    # The local slope contrast is the discriminator: for a line it is a coin flip.
    assert out["prob_slope_above_gt_below"] < 0.91
    assert out["knee_well_defined"] is False


def test_curve_accelerating_to_the_edge_is_boundary_pinned():
    """An argmax on the last interval is a bound, not a located threshold."""
    n_trials = 32
    counts = np.linspace(0.0, 32.0, 120)
    ell = _haldane_logit(counts, n_trials)
    rng = np.random.default_rng(4)
    # Convex and still steepening where the data stop — the letter-sound shape.
    amp = rng.normal(1.0, 0.05, size=400).clip(0.5)
    f = ((counts / 32.0) ** 3)[:, None] * amp[None, :] + rng.normal(
        0.0, 0.01, size=(counts.size, 400)
    )

    out = reporting._readiness_knee(f, ell, n_trials=n_trials)

    assert out["increasing_frac"] > 0.9
    # The curvature check passes — it really is bending — but the location is not
    # identified, because the observed range ends before the curve stops steepening.
    assert out["prob_slope_above_gt_below"] >= 0.91
    assert out["boundary_pinned"] is True
    assert out["steepest_interval_index"] == out["n_bins"] - 2
    assert out["knee_well_defined"] is False


def test_items_scale_maximum_can_differ_from_the_latent_logit_maximum():
    """The reported derivative is on the logit scale, and that choice matters.

    ``f_mech`` is a logit contribution, so ``d E[y] / dx`` carries an extra
    ``p * (1 - p)`` inverse-link factor. Under a baseline that pushes the fitted
    probability past the midpoint, the expected-items derivative peaks at a *lower*
    exposure than the latent-logit derivative — so the two are different estimands
    and the report must name which one it shows.
    """
    x = np.linspace(-3.0, 3.0, 241)
    # A logit curve whose own steepest rise is late in the range.
    f_curve = 3.0 / (1.0 + np.exp(-2.5 * (x - 1.2)))
    baseline = 1.5  # other terms hold the linear predictor above the logit midpoint

    d_logit = np.gradient(f_curve, x)
    p = 1.0 / (1.0 + np.exp(-(baseline + f_curve)))
    d_items = p * (1.0 - p) * d_logit

    assert x[int(np.argmax(d_items))] < x[int(np.argmax(d_logit))] - 0.25

    # And the implementation labels its own scale, so a renderer cannot confuse them.
    counts = np.linspace(0.0, 32.0, 120)
    out = reporting._readiness_knee(
        _logistic_draws(counts, 16.0, n_sample=120),
        _haldane_logit(counts, 32),
        n_trials=32,
    )
    assert out["scale"] == "latent_logit"


def test_low_end_steepest_interval_reports_no_below_slope():
    """When the argmax is the lowest interval there is no 'below' set to average.

    mech-191 shipped exactly this: ``slope_below_knee_median`` was all-NaN and the
    rendered report printed the literal string "nan logit units per item".
    """
    n_trials = 32
    counts = np.linspace(0.0, 32.0, 120)
    ell = _haldane_logit(counts, n_trials)
    rng = np.random.default_rng(5)
    # Saturating: steepest at the very start, flat thereafter.
    amp = rng.normal(1.0, 0.05, size=300).clip(0.5)
    f = (1.0 - np.exp(-0.6 * counts))[:, None] * amp[None, :] + rng.normal(
        0.0, 0.01, size=(counts.size, 300)
    )

    out = reporting._readiness_knee(f, ell, n_trials=n_trials)

    assert out["steepest_interval_index"] == 0
    assert out["boundary_pinned"] is True
    assert math.isnan(out["slope_below_knee_median"])
    assert out["knee_well_defined"] is False
