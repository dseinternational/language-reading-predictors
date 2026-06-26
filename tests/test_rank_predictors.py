# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit test for the grouped permutation-importance core of ``scripts/rank_predictors.py``.

``cluster_permutation_importance`` is the one piece of genuinely new numeric logic in
the GB ranking script (issue #116). Scripts aren't on the import path in this repo, so
the module is loaded by file path. The check is behavioural: with two clusters where
only one carries the signal, the grouped (joint) block shuffle must rank the signal
cluster well above the noise cluster, and the noise cluster's delta must sit near zero.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "rank_predictors.py"


@pytest.fixture(scope="module")
def rank_predictors():
    """Load ``scripts/rank_predictors.py`` as a module (not on the import path)."""
    spec = importlib.util.spec_from_file_location("rank_predictors", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fit_folds(X: pd.DataFrame, y: np.ndarray, n_splits: int):
    estimators, test_indices = [], []
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=0).split(X):
        estimators.append(LinearRegression().fit(X.iloc[tr], y[tr]))
        test_indices.append(te)
    return estimators, test_indices


def test_grouped_perm_deltas_ranks_signal_cluster_above_noise(rank_predictors):
    rng = np.random.default_rng(0)
    n = 200
    # cluster A: two correlated columns carrying all the signal.
    a0 = rng.normal(size=n)
    a1 = a0 + 0.05 * rng.normal(size=n)
    # cluster B: two columns of pure noise, unrelated to y.
    X = pd.DataFrame({"a0": a0, "a1": a1, "b0": rng.normal(size=n), "b1": rng.normal(size=n)})
    y = 5.0 * a0 + 0.01 * rng.normal(size=n)

    estimators, test_indices = _fit_folds(X, y, n_splits=4)
    cluster_cols = {0: [0, 1], 1: [2, 3]}  # 0 = signal cluster A, 1 = noise cluster B

    deltas = rank_predictors._grouped_perm_deltas(
        estimators, X, y, test_indices, cluster_cols, n_repeats=5, seed=47
    )
    signal_mean = float(deltas[0].mean())
    noise_mean = float(deltas[1].mean())

    # Permuting the signal cluster must clearly raise held-out RMSE; the noise cluster
    # must barely move it, so the signal cluster ranks first.
    assert signal_mean > 0.0
    assert signal_mean > noise_mean
    assert abs(noise_mean) < 0.25 * signal_mean


def test_grouped_perm_deltas_is_deterministic_per_seed(rank_predictors):
    """Same inputs + seed → identical deltas (the per-fold RNG reset is reproducible)."""
    rng = np.random.default_rng(1)
    n = 120
    X = pd.DataFrame(
        {"a0": rng.normal(size=n), "a1": rng.normal(size=n), "b0": rng.normal(size=n)}
    )
    y = 2.0 * X["a0"].to_numpy() + 0.01 * rng.normal(size=n)
    estimators, test_indices = _fit_folds(X, y, n_splits=3)
    cluster_cols = {0: [0, 1], 1: [2]}

    d1 = rank_predictors._grouped_perm_deltas(
        estimators, X, y, test_indices, cluster_cols, n_repeats=4, seed=47
    )
    d2 = rank_predictors._grouped_perm_deltas(
        estimators, X, y, test_indices, cluster_cols, n_repeats=4, seed=47
    )
    for c in cluster_cols:
        assert np.allclose(d1[c], d2[c])
