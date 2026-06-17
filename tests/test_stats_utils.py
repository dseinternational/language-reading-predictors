# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for :mod:`stats_utils`.

``CLAUDE.md`` documents this file as the home of the stats-utility tests; it
had gone missing. These cover the descriptive-statistics helpers and the
feature-selection dependency matrices used by the model pipeline's
``feature_selection_diagnostics`` step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from language_reading_predictors.stats_utils import (
    convert_to_categorical,
    describe,
    differential_entropy_standardized,
    invlogit,
    logit,
    mutual_info_dissimilarity,
    spearman_distance_matrix,
    standardize,
)


# ── standardize / logit ──────────────────────────────────────────────────


def test_standardize_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    z = standardize(rng.normal(5, 3, size=200))
    assert np.isclose(z.mean(), 0.0, atol=1e-9)
    assert np.isclose(z.std(), 1.0, atol=1e-9)


def test_standardize_constant_input_returns_centred_zeros():
    # std < EPSILON path: must not divide by ~0.
    z = standardize(np.full(10, 4.2))
    assert np.allclose(z, 0.0)


def test_logit_invlogit_roundtrip():
    for p in (0.1, 0.5, 0.9):
        assert np.isclose(invlogit(logit(p)), p)


# ── differential entropy edge cases ──────────────────────────────────────


def test_differential_entropy_constant_is_neg_inf():
    assert differential_entropy_standardized(np.full(20, 3.0)) == -np.inf


def test_differential_entropy_too_few_points_is_nan():
    assert np.isnan(differential_entropy_standardized(np.array([1.0])))


def test_differential_entropy_normal_near_gaussian_value():
    rng = np.random.default_rng(1)
    h = differential_entropy_standardized(rng.normal(size=4000))
    # Standard-normal differential entropy is ~1.4189 nats; KDE estimate is
    # close but biased low, so allow a generous band.
    assert np.isfinite(h)
    assert 1.0 < h < 1.8


# ── spearman_distance_matrix (regression test for the NaN-policy bug) ─────


def test_spearman_distance_matrix_dataframe_ndarray_agree_under_nan():
    rng = np.random.default_rng(2)
    a = rng.normal(size=60)
    b = a * 0.8 + rng.normal(size=60) * 0.4
    c = rng.normal(size=60)
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    df.loc[0, "a"] = np.nan
    d_df, c_df = spearman_distance_matrix(df)
    d_nd, c_nd = spearman_distance_matrix(df.to_numpy())
    assert np.allclose(c_df, c_nd, equal_nan=True)
    assert np.allclose(d_df, d_nd, equal_nan=True)


def test_spearman_distance_matrix_shape_symmetry_diagonal():
    rng = np.random.default_rng(3)
    dist, corr = spearman_distance_matrix(rng.normal(size=(40, 4)))
    assert dist.shape == corr.shape == (4, 4)
    assert np.allclose(corr, corr.T)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(np.diag(dist), 0.0)
    assert np.all(dist >= -1e-12) and np.all(dist <= 1.0 + 1e-12)


def test_spearman_distance_matrix_two_columns_returns_matrix():
    # scipy.stats.spearmanr returns a bare scalar for exactly two columns;
    # the function must still return a 2x2 matrix.
    rng = np.random.default_rng(4)
    _dist, corr = spearman_distance_matrix(rng.normal(size=(30, 2)))
    assert corr.shape == (2, 2)


# ── mutual information dissimilarity ─────────────────────────────────────


def test_mutual_info_dissimilarity_symmetric_zero_diagonal():
    rng = np.random.default_rng(5)
    d = mutual_info_dissimilarity(rng.normal(size=(80, 4)), random_state=0)
    assert d.shape == (4, 4)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)


# ── misc helpers ─────────────────────────────────────────────────────────


def test_convert_to_categorical_returns_codes():
    codes = convert_to_categorical(pd.Series(["a", "b", "a", "c"]))
    assert list(codes) == [0, 1, 0, 2]


def test_describe_contains_expected_keys():
    rng = np.random.default_rng(6)
    out = describe(rng.normal(size=50), alpha=0.05)
    for key in ("mean", "std", "range", "skew", "kurtosis", "shapiro_pvalue", "entropy"):
        assert key in out.index


# ── distance correlation (optional dcor dependency) ──────────────────────


def test_distance_corr_matrix_basic_properties():
    pytest.importorskip("dcor")
    from language_reading_predictors.stats_utils import distance_corr_matrix

    rng = np.random.default_rng(7)
    M = distance_corr_matrix(rng.normal(size=(50, 3)))
    assert M.shape == (3, 3)
    assert np.allclose(np.diag(M), 1.0)
    assert np.allclose(M, M.T)
    assert np.all(M >= 0.0) and np.all(M <= 1.0)
