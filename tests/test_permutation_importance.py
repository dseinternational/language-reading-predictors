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
import pytest
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


# --- Shared-evaluator adoption (#662) ----------------------------------------


def _reference_pooled_deltas(
    estimators, X, y, test_indices, groups, col_blocks, *, n_repeats, seed
):
    """The pre-#662 local scoring loop, kept as the numerical reference.

    ``pooled_permutation_deltas`` now delegates its baseline, its permuted
    frames and its pooled score to ``ml.permutation.pooled_oof_permutation_deltas``
    while keeping this project's donor design. The published importance columns
    must not move, so the retired loop is retained here to compare against.
    """
    y = np.asarray(y, dtype=float)
    estimators = list(estimators)
    test_indices = [np.asarray(t) for t in test_indices]
    oof = np.full(len(y), np.nan, dtype=float)
    for estimator, rows in zip(estimators, test_indices, strict=True):
        oof[rows] = estimator.predict(X.iloc[rows])
    covered = ~np.isnan(oof)
    baseline = float(np.sqrt(np.mean((y[covered] - oof[covered]) ** 2)))
    deltas: dict = {key: [] for key in col_blocks}
    for repeat in range(n_repeats):
        donor = subject_block_permutation_indices(
            groups, np.random.default_rng([seed, repeat])
        )
        for key, columns in col_blocks.items():
            columns = list(columns)
            permuted = X.copy()
            permuted.iloc[:, columns] = X.iloc[donor, columns].to_numpy()
            prediction = np.full(len(y), np.nan, dtype=float)
            for estimator, rows in zip(estimators, test_indices, strict=True):
                prediction[rows] = estimator.predict(permuted.iloc[rows])
            permuted_score = float(
                np.sqrt(np.mean((y[covered] - prediction[covered]) ** 2))
            )
            deltas[key].append(permuted_score - baseline)
    return {key: np.asarray(values) for key, values in deltas.items()}


def test_shared_evaluator_reproduces_the_local_deltas_exactly():
    """Bit-for-bit agreement, for singleton and multi-column (cluster) blocks."""
    X, y, groups = _child_constant_data(seed=5)
    estimators, test_indices = _one_child_folds(X, y, groups)
    blocks = {0: [0], 1: [1], "cluster": [0, 1]}

    new = pooled_permutation_deltas(
        estimators, X, y, test_indices, groups, blocks, n_repeats=3, seed=47
    )
    old = _reference_pooled_deltas(
        estimators, X, y, test_indices, groups, blocks, n_repeats=3, seed=47
    )

    assert new.keys() == old.keys()
    for key in old:
        np.testing.assert_array_equal(new[key], old[key])


def test_every_block_shares_one_donor_plan_per_repeat(monkeypatch):
    """One subject-block permutation per repeat, handed to every block.

    The shared evaluator takes a donor plan *per block*; drawing a separate
    permutation for each would make block importances incomparable and would
    make the result depend on block iteration order. The adapter passes one
    ``(n_repeats, n_rows)`` array under every key, and the plan is exactly what
    ``subject_block_permutation_indices`` draws from ``default_rng([seed, r])``.
    """
    from language_reading_predictors.models import permutation as module

    X, y, groups = _child_constant_data(seed=6)
    estimators, test_indices = _one_child_folds(X, y, groups)
    captured: dict = {}
    real = module.pooled_oof_permutation_deltas

    def _spy(*args, **kwargs):
        captured["donor_indices"] = kwargs["donor_indices"]
        captured["column_blocks"] = args[4]
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "pooled_oof_permutation_deltas", _spy)
    module.pooled_permutation_deltas(
        estimators, X, y, test_indices, groups, {0: [0], "both": [0, 1]},
        n_repeats=4, seed=47,
    )

    plans = list(captured["donor_indices"].values())
    assert captured["donor_indices"].keys() == {0, "both"}
    assert all(plan is plans[0] for plan in plans)
    assert plans[0].shape == (4, len(X))
    for repeat in range(4):
        np.testing.assert_array_equal(
            plans[0][repeat],
            subject_block_permutation_indices(
                groups, np.random.default_rng([47, repeat])
            ),
        )
    # Column positions are translated to the labels the shared API blocks by.
    assert captured["column_blocks"] == {
        0: ["const_signal"],
        "both": ["const_signal", "row_noise"],
    }


def test_pooled_rmse_is_pooled_not_an_average_of_fold_scores():
    """Unequal fold sizes make the two summaries differ; the pooled one is ours."""
    from language_reading_predictors.models.permutation import pooled_rmse

    target = np.array([0.0, 0.0, 0.0, 0.0])
    prediction = np.array([2.0, 0.0, 0.0, 0.0])
    pooled = pooled_rmse(target, prediction)
    fold_mean = np.mean(
        [pooled_rmse(target[:1], prediction[:1]), pooled_rmse(target[1:], prediction[1:])]
    )
    assert pooled == pytest.approx(1.0)
    assert fold_mean == pytest.approx(1.0)
    # A one-row fold and a three-row fold weight differently once both err.
    prediction = np.array([2.0, 1.0, 1.0, 1.0])
    assert pooled_rmse(target, prediction) != pytest.approx(
        np.mean(
            [
                pooled_rmse(target[:1], prediction[:1]),
                pooled_rmse(target[1:], prediction[1:]),
            ]
        )
    )
