# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for :mod:`ml_utils`.

Covers the cross-validation score formatting (regression test for the bug
where ``abs()`` of the fold mean turned a negative R² into a deceptively
positive score) and the Gaussian-process kernel helpers.
"""

from __future__ import annotations

import numpy as np

from language_reading_predictors.ml_utils import (
    cross_validation_score_rows,
    ornstein_uhlenbeck_kernel,
    periodic_kernel,
    quadratic_distance_kernel,
)


def _by_metric(rows):
    return {r["metric"]: r for r in rows}


# ── cross_validation_score_rows ──────────────────────────────────────────


def test_negative_r2_reported_with_true_sign():
    # The whole point: a worse-than-mean model (negative R²) must stay negative.
    rows = _by_metric(
        cross_validation_score_rows({"test_r2": np.array([-0.5, -0.3, -0.25])})
    )
    assert rows["r2"]["mean"] < 0
    assert np.isclose(rows["r2"]["mean"], -0.35)


def test_neg_error_scorers_flipped_to_positive():
    scores = {
        "test_mae": np.array([-2.0, -3.0, -4.0]),
        "test_rmse": np.array([-5.0, -5.0, -5.0]),
        "test_medae": np.array([-1.0, -2.0, -3.0]),
    }
    rows = _by_metric(cross_validation_score_rows(scores))
    assert np.isclose(rows["mae"]["mean"], 3.0)
    assert np.isclose(rows["rmse"]["mean"], 5.0)
    assert np.isclose(rows["medae"]["mean"], 2.0)


def test_positive_r2_unchanged():
    rows = _by_metric(
        cross_validation_score_rows({"test_r2": np.array([0.4, 0.6, 0.5])})
    )
    assert np.isclose(rows["r2"]["mean"], 0.5)


def test_non_test_keys_are_ignored():
    scores = {
        "train_r2": np.array([0.9]),
        "fit_time": np.array([1.0]),
        "test_r2": np.array([0.5]),
    }
    assert [r["metric"] for r in cross_validation_score_rows(scores)] == ["r2"]


def test_sd_is_sign_agnostic():
    rows = _by_metric(
        cross_validation_score_rows({"test_mae": np.array([-2.0, -4.0])})
    )
    assert np.isclose(rows["mae"]["sd"], np.std([-2.0, -4.0]))


# ── GP kernel helpers ────────────────────────────────────────────────────


def test_quadratic_distance_kernel_shape_diagonal_symmetry():
    X = np.linspace(-2, 2, 10)[:, None]
    K = quadratic_distance_kernel(X, X, eta=1.0, sigma=0.5)
    assert K.shape == (10, 10)
    assert np.allclose(np.diag(K), 1.0)  # zero distance -> eta**2
    assert np.allclose(K, K.T)
    assert np.all(K >= 0.0) and np.all(K <= 1.0 + 1e-9)


def test_ornstein_uhlenbeck_kernel_diagonal_one():
    X = np.linspace(0, 3, 8)[:, None]
    K = ornstein_uhlenbeck_kernel(X, X, eta_squared=1.0, rho=4.0)
    assert K.shape == (8, 8)
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T)


def test_periodic_kernel_diagonal_one():
    X = np.linspace(0, 5, 6)[:, None]
    K = periodic_kernel(X, X)
    assert K.shape == (6, 6)
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T)
