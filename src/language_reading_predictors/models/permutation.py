# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pooled out-of-fold, subject-block permutation importance (issue #631).

Why this exists
---------------
The previous scheme computed sklearn ``permutation_importance`` *per fold* on
each fold's held-out rows. Under the project's near-leave-one-subject-out
cross-validation (``cv_splits`` ≈ number of children) a held-out fold contains a
single child, so any predictor that is constant within a child (``gender``,
``group``, ``hearing``, …) is *unchanged* by a within-fold permutation — the
permuted matrix equals the original and the importance is mechanically exactly
zero, regardless of how much signal the predictor carries.

The scheme here instead permutes **globally, once per repeat, across all rows**,
and scores the change in the *pooled* out-of-fold RMSE:

1. Per repeat, draw ONE **subject-block permutation** over all rows: the
   children are permuted, and each child's rows are remapped to a donor child's
   values (aligned by within-child row order, wrapping when the donor has fewer
   rows). This both mixes values *between* children — so child-constant columns
   really change — and respects the *within-child* longitudinal dependence, so
   the permutation null does not also destroy the repeated-measures structure.
2. Each fold's already-fitted estimator predicts its own held-out rows of the
   permuted matrix; the predictions are pooled into one out-of-fold RMSE.
3. The importance delta is that pooled permuted RMSE minus the unpermuted
   pooled out-of-fold RMSE — positive means the column (block) was useful.

The routines are pure-numeric and estimator-agnostic so they can be reused by
``EstimatorPipeline.permutation_importance_analysis`` (per-feature blocks) and
``scripts/rank_predictors.py`` (per-cluster blocks) and unit-tested on their
own.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def subject_block_permutation_indices(
    groups: pd.Series | np.ndarray | Sequence[Hashable],
    rng: np.random.Generator,
) -> np.ndarray:
    """Row-donor index array for one subject-block permutation.

    Draws a permutation of the *subjects* (children) and maps every row of a
    recipient subject to the corresponding row of its donor subject, aligned by
    within-subject row order. When the donor has fewer rows than the recipient
    the donor's rows are recycled (index modulo the donor's row count); extra
    donor rows are dropped. A subject may map to itself (an ordinary
    permutation, not a derangement).

    Parameters
    ----------
    groups
        Per-row subject labels (any hashable values), length ``n_rows``.
    rng
        A ``numpy.random.Generator``; one subject permutation is drawn from it.

    Returns
    -------
    numpy.ndarray
        Integer array ``donor_index`` of length ``n_rows`` such that the
        permuted value of any column ``x`` is ``x[donor_index]``.
    """
    groups_arr = np.asarray(groups)
    subjects = np.unique(groups_arr)
    rows_by_subject = {s: np.flatnonzero(groups_arr == s) for s in subjects}

    perm = rng.permutation(len(subjects))
    donor_index = np.empty(len(groups_arr), dtype=np.intp)
    for i, subject in enumerate(subjects):
        recipient_rows = rows_by_subject[subject]
        donor_rows = rows_by_subject[subjects[perm[i]]]
        take = donor_rows[np.arange(len(recipient_rows)) % len(donor_rows)]
        donor_index[recipient_rows] = take
    return donor_index


def _pooled_oof_rmse(
    estimators: Sequence[Any],
    X: pd.DataFrame,
    y: np.ndarray,
    test_indices: Sequence[np.ndarray],
) -> tuple[float, np.ndarray]:
    """Pooled out-of-fold RMSE plus the row mask the folds cover."""
    oof_pred = np.full(len(y), np.nan, dtype=float)
    for est, val_idx in zip(estimators, test_indices, strict=True):
        oof_pred[val_idx] = est.predict(X.iloc[val_idx])
    covered = ~np.isnan(oof_pred)
    resid = y[covered] - oof_pred[covered]
    return float(np.sqrt(np.mean(resid**2))), covered


def pooled_permutation_deltas(
    estimators: Iterable[Any],
    X: pd.DataFrame,
    y: np.ndarray | Sequence[float],
    test_indices: Iterable[np.ndarray],
    groups: pd.Series | np.ndarray | Sequence[Hashable],
    col_blocks: Mapping[Hashable, Sequence[int]],
    *,
    n_repeats: int,
    seed: int,
) -> dict[Hashable, np.ndarray]:
    """Pooled out-of-fold subject-block permutation deltas, one entry per block.

    Per repeat, ONE subject-block permutation (see
    :func:`subject_block_permutation_indices`) is drawn over ALL rows and shared
    by every block. For each block, only that block's columns are replaced by
    their permuted values; every fold's already-fitted estimator predicts its
    own held-out rows of the permuted matrix, and the pooled out-of-fold RMSE is
    compared with the unpermuted pooled out-of-fold RMSE. Because the
    permutation spans all rows (not one near-singleton fold), predictors that
    are constant within a child receive a genuine permutation null instead of a
    mechanical zero.

    Determinism: repeat ``r`` uses ``np.random.default_rng([seed, r])``, so the
    same inputs and seed always reproduce the same deltas, independently of
    block iteration order.

    Parameters
    ----------
    estimators
        Per-fold fitted estimators (e.g. ``cross_validate(..., return_estimator=True)``).
    X : pandas.DataFrame
        The full design matrix; folds are selected positionally via ``.iloc``.
    y : array-like
        The full target vector.
    test_indices
        Per-fold held-out row positions (aligned with ``estimators``).
    groups
        Per-row subject labels used for the subject-block permutation.
    col_blocks : mapping
        Maps block key -> list of column *positions* in ``X``. Use singleton
        blocks (``{i: [i]}``) for per-feature importance, or cluster blocks for
        grouped importance.
    n_repeats : int
        Number of subject-block permutations.
    seed : int
        Base RNG seed; repeat ``r`` is seeded with ``[seed, r]``.

    Returns
    -------
    dict
        Block key -> array of ``n_repeats`` deltas (rise in pooled out-of-fold
        RMSE when the block is permuted; positive = the block was useful).
    """
    y = np.asarray(y, dtype=float)
    estimators = list(estimators)
    test_indices = [np.asarray(t) for t in test_indices]

    base_rmse, covered = _pooled_oof_rmse(estimators, X, y, test_indices)

    deltas: dict[Hashable, list[float]] = {key: [] for key in col_blocks}
    for r in range(n_repeats):
        rng = np.random.default_rng([seed, r])
        donor_index = subject_block_permutation_indices(groups, rng)
        for key, cols in col_blocks.items():
            cols = list(cols)
            Xp = X.copy()
            Xp.iloc[:, cols] = X.iloc[donor_index, cols].to_numpy()
            oof_pred = np.full(len(y), np.nan, dtype=float)
            for est, val_idx in zip(estimators, test_indices, strict=True):
                oof_pred[val_idx] = est.predict(Xp.iloc[val_idx])
            resid = y[covered] - oof_pred[covered]
            perm_rmse = float(np.sqrt(np.mean(resid**2)))
            deltas[key].append(perm_rmse - base_rmse)

    return {key: np.asarray(v) for key, v in deltas.items()}
