# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the pooled out-of-fold, subject-block permutation importance (#631).

The regression test here is the one that would have caught #631 finding 1: under
near-leave-one-subject-out ``GroupKFold`` (every held-out fold is one child), the
old per-fold permutation left child-constant columns literally unchanged and
scored them exactly zero. The pooled subject-block scheme must give a
child-constant predictor carrying real signal a clearly positive importance,
while a pure-noise predictor stays near zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from language_reading_predictors.models.permutation import (
    pooled_permutation_deltas,
    subject_block_permutation_indices,
)


def _one_child_folds(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray):
    """Fit one estimator per GroupKFold fold with as many splits as children."""
    n_children = len(np.unique(groups))
    estimators, test_indices = [], []
    for tr, te in GroupKFold(n_splits=n_children).split(X, y, groups=groups):
        estimators.append(LinearRegression().fit(X.iloc[tr], y[tr]))
        test_indices.append(te)
    return estimators, test_indices


def _child_constant_data(seed: int = 0):
    """24 children x 3 rows; column 0 is child-constant and carries the signal."""
    rng = np.random.default_rng(seed)
    n_children, rows_per_child = 24, 3
    child_vals = rng.normal(size=n_children)
    groups = np.repeat(np.arange(n_children), rows_per_child)
    const_feat = np.repeat(child_vals, rows_per_child)
    noise_feat = rng.normal(size=n_children * rows_per_child)
    X = pd.DataFrame({"const_signal": const_feat, "row_noise": noise_feat})
    y = 3.0 * const_feat + 0.1 * rng.normal(size=len(const_feat))
    return X, y, groups


def test_child_constant_signal_feature_gets_positive_importance():
    """The #631 finding-1 regression test: a child-constant predictor with real
    signal must receive clearly positive importance under GroupKFold with
    one-child held-out folds (the old per-fold permutation mechanically scored
    it exactly zero), and a pure-noise feature must stay near zero.
    """
    X, y, groups = _child_constant_data()
    estimators, test_indices = _one_child_folds(X, y, groups)

    deltas = pooled_permutation_deltas(
        estimators,
        X,
        y,
        test_indices,
        groups,
        {0: [0], 1: [1]},
        n_repeats=10,
        seed=47,
    )
    signal_mean = float(deltas[0].mean())
    noise_mean = float(deltas[1].mean())

    # Permuting the child-constant signal column must clearly raise the pooled
    # out-of-fold RMSE; permuting the noise column must barely move it.
    assert signal_mean > 1.0
    assert abs(noise_mean) < 0.1 * signal_mean


def test_within_fold_constant_column_would_zero_under_old_scheme():
    """Sanity check on the failure mode being fixed: within any single held-out
    fold the child-constant column really is constant, so a within-fold
    permutation is the identity and the old scheme's delta was exactly zero.
    """
    X, y, groups = _child_constant_data()
    _estimators, test_indices = _one_child_folds(X, y, groups)
    for val_idx in test_indices:
        assert X.iloc[val_idx]["const_signal"].nunique() == 1


def test_subject_block_permutation_preserves_child_structure():
    groups = np.array(["a", "a", "a", "b", "b", "c", "c", "c", "c"])
    rng = np.random.default_rng(3)
    donor_index = subject_block_permutation_indices(groups, rng)

    assert donor_index.shape == (len(groups),)
    # Each recipient child's rows must all come from exactly ONE donor child,
    # and the recipient -> donor map must be a permutation of the children.
    donor_of = {}
    for subject in np.unique(groups):
        rows = np.flatnonzero(groups == subject)
        donors = set(groups[donor_index[rows]])
        assert len(donors) == 1
        donor_of[subject] = donors.pop()
    assert sorted(donor_of.values()) == sorted(np.unique(groups))
    # Within-child alignment: recipient row t takes the donor's row t (modulo
    # the donor's row count), in the donor's original row order.
    for subject, donor in donor_of.items():
        rows = np.flatnonzero(groups == subject)
        donor_rows = np.flatnonzero(groups == donor)
        expected = donor_rows[np.arange(len(rows)) % len(donor_rows)]
        assert np.array_equal(donor_index[rows], expected)


def test_pooled_permutation_deltas_deterministic_per_seed():
    """Same inputs + seed -> identical deltas (per-repeat seeding contract)."""
    X, y, groups = _child_constant_data(seed=1)
    estimators, test_indices = _one_child_folds(X, y, groups)
    blocks = {0: [0], 1: [1]}

    d1 = pooled_permutation_deltas(
        estimators, X, y, test_indices, groups, blocks, n_repeats=4, seed=47
    )
    d2 = pooled_permutation_deltas(
        estimators, X, y, test_indices, groups, blocks, n_repeats=4, seed=47
    )
    for key in blocks:
        assert np.allclose(d1[key], d2[key])

    # A different seed draws different subject permutations.
    d3 = pooled_permutation_deltas(
        estimators, X, y, test_indices, groups, blocks, n_repeats=4, seed=48
    )
    assert any(not np.allclose(d1[key], d3[key]) for key in blocks)
