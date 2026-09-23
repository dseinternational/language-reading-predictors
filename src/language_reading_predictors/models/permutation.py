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
   values at the same assessment waves. Donors must have the same observed
   wave schedule. A schedule represented by only one child cannot be permuted.
   This defines importance conditional on the observed assessment schedule.
   A predictor determined by wave (including ``time``), or otherwise constant
   among eligible donors at each wave, cannot be assessed under this design.
   Its deltas are missing, not zero importance.
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
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from dse_research_utils.ml.permutation import pooled_oof_permutation_deltas

PERMUTATION_DESIGN_VERSION = "subject_blocks_same_wave_schedule_v2"


@dataclass(frozen=True)
class SubjectPermutationDesign:
    """Build schedule blocks once, then reuse them for support and each repeat."""

    blocks: dict[tuple[Hashable, ...], list[np.ndarray]]
    n_rows: int

    @classmethod
    def from_labels(cls, groups: Any, waves: Any = None) -> SubjectPermutationDesign:
        return cls(_schedule_blocks(groups, waves), len(groups))

    def indices(self, rng: np.random.Generator) -> np.ndarray:
        donor_index = np.arange(self.n_rows, dtype=np.intp)
        for subjects in self.blocks.values():
            perm = rng.permutation(len(subjects))
            for index, recipient_rows in enumerate(subjects):
                donor_index[recipient_rows] = subjects[perm[index]]
        return donor_index

    def support(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"wave_schedule": ",".join(map(str, schedule)), "n_subjects": len(subjects),
             "n_rows": sum(len(rows) for rows in subjects),
             "n_movable_subjects": len(subjects) if len(subjects) > 1 else 0}
            for schedule, subjects in self.blocks.items()
        ], columns=["wave_schedule", "n_subjects", "n_rows", "n_movable_subjects"])

    def assessable(self, X: pd.DataFrame, columns: Sequence[int]) -> bool:
        """Whether at least one permitted donor can change a value in this block."""
        if len(X) != self.n_rows:
            raise ValueError("permutation design must match the predictor rows")
        for subjects in self.blocks.values():
            for wave_rows in zip(*subjects, strict=True):
                if (X.iloc[list(wave_rows), list(columns)].nunique(dropna=False) > 1).any():
                    return True
        return False


def subject_block_permutation_indices(
    groups: pd.Series | np.ndarray | Sequence[Hashable],
    rng: np.random.Generator,
    *,
    waves: pd.Series | np.ndarray | Sequence[Hashable] | None = None,
) -> np.ndarray:
    """Row-donor index array for one subject-block permutation.

    Draws a permutation of the *subjects* (children) and maps every row of a
    recipient subject to its donor's row at the same assessment wave. Subjects
    are permuted only within identical observed wave schedules. The mapping is
    invariant to row order for fixed labels and seed. A subject may map to
    itself; a singleton schedule necessarily does so.

    Parameters
    ----------
    groups
        Per-row subject labels (any hashable values), length ``n_rows``.
    rng
        A ``numpy.random.Generator``; one subject permutation is drawn from it.
    waves
        Per-row assessment labels. Required for repeated observations. Omit
        only for cross-sectional data with one row per subject.

    Returns
    -------
    numpy.ndarray
        Integer array ``donor_index`` of length ``n_rows`` such that the
        permuted value of any column ``x`` is ``x[donor_index]``.
    """
    return SubjectPermutationDesign.from_labels(groups, waves).indices(rng)


def _schedule_blocks(
    groups: pd.Series | np.ndarray | Sequence[Hashable],
    waves: pd.Series | np.ndarray | Sequence[Hashable] | None,
) -> dict[tuple[Hashable, ...], list[np.ndarray]]:
    """Canonical subject rows partitioned by their observed assessment schedule."""
    labels = np.asarray(groups)
    if labels.ndim != 1 or pd.isna(labels).any():
        raise ValueError("subject labels must be a one-dimensional array without missing values")
    if waves is None:
        if len(set(labels)) != len(labels):
            raise ValueError("assessment waves are required for repeated observations")
        wave_labels = np.zeros(len(labels), dtype=int)
    else:
        wave_labels = np.asarray(waves)
    if wave_labels.shape != labels.shape or pd.isna(wave_labels).any():
        raise ValueError("assessment waves must match the subject rows and contain no missing values")

    def key(value: Hashable) -> tuple[str, str]:
        return type(value).__name__, repr(value)

    blocks: dict[tuple[Hashable, ...], list[np.ndarray]] = {}
    for subject in sorted(set(labels), key=key):
        rows = np.flatnonzero(labels == subject)
        ordered = np.asarray(sorted(rows, key=lambda index: key(wave_labels[index])), dtype=np.intp)
        schedule = tuple(wave_labels[ordered])
        if len(set(schedule)) != len(schedule):
            raise ValueError("each subject must have at most one row per assessment wave")
        blocks.setdefault(schedule, []).append(ordered)
    return blocks


def permutation_schedule_support(
    groups: pd.Series | np.ndarray | Sequence[Hashable],
    waves: pd.Series | np.ndarray | Sequence[Hashable] | None,
) -> pd.DataFrame:
    """Report the rows and children for which an alternative donor is available."""
    return SubjectPermutationDesign.from_labels(groups, waves).support()


def _predict(estimator: Any, frame: pd.DataFrame) -> np.ndarray:
    """The prediction callback the shared evaluator calls per fold."""
    return estimator.predict(frame)


def pooled_rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    """Pooled RMSE over every scored row at once.

    Deliberately *not* an average of per-fold scores: with near
    leave-one-subject-out folds a per-fold RMSE is one child's error, and
    averaging those weights a one-row child like a four-row one. The shared
    evaluator scores the pooled out-of-fold predictions once, in original row
    order, which is what this project has always reported.
    """
    residual = target - prediction
    return float(np.sqrt(np.mean(residual**2)))


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
    waves: pd.Series | np.ndarray | Sequence[Hashable] | None = None,
    design: SubjectPermutationDesign | None = None,
) -> dict[Hashable, np.ndarray]:
    """Pooled out-of-fold subject-block permutation deltas, one entry per block.

    Per repeat, ONE subject-block permutation (see
    :func:`subject_block_permutation_indices`) is drawn over ALL rows and shared
    by every block. For each block, only that block's columns are replaced by
    their permuted values; every fold's already-fitted estimator predicts its
    own held-out rows of the permuted matrix, and the pooled out-of-fold RMSE is
    compared with the unpermuted pooled out-of-fold RMSE. Because the
    permutation spans all rows (not one near-singleton fold), predictors that
    are constant within a child can move among children sharing a schedule.
    Singleton schedules remain fixed and limit the scope of the ranking.

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
    waves : array-like, optional
        Assessment labels aligned with the rows. Required for repeated observations.
    design : SubjectPermutationDesign, optional
        Schedule blocks already built for these rows, for reuse with support tables.

    Returns
    -------
    dict
        Block key -> array of ``n_repeats`` deltas (rise in pooled out-of-fold
        RMSE when the block is permuted; positive = the block was useful).
        Blocks that cannot change under any permitted donor mapping contain NaN.
    """
    # The donor design is this project's (#631) and stays here: one subject-block
    # permutation per repeat, drawn from ``default_rng([seed, r])`` and SHARED by
    # every block, so block iteration order cannot move a delta. The scoring loop
    # — baseline, per-block permuted frames, pooled out-of-fold score, delta sign
    # — is ``ml.permutation.pooled_oof_permutation_deltas`` (#662).
    design = design or SubjectPermutationDesign.from_labels(groups, waves)
    assessable = {key: design.assessable(X, cols) for key, cols in col_blocks.items()}
    missing = {key: np.full(n_repeats, np.nan) for key, movable in assessable.items() if not movable}
    if len(missing) == len(col_blocks):
        return missing
    donor_plan = np.stack(
        [
            design.indices(np.random.default_rng([seed, r]))
            for r in range(n_repeats)
        ]
    )
    # The shared API blocks by column *label*; this one has always blocked by
    # column position, which the two callers build from ``X.columns`` order.
    labels = list(X.columns)
    column_blocks = {
        key: [labels[position] for position in cols] for key, cols in col_blocks.items() if assessable[key]
    }
    result = pooled_oof_permutation_deltas(
        estimators,
        X,
        # Positional, exactly as before: ``y`` may be a Series whose index does
        # not label ``X``'s rows, and the fold positions are what align them.
        np.asarray(y, dtype=float),
        [np.asarray(t) for t in test_indices],
        column_blocks,
        donor_indices=dict.fromkeys(column_blocks, donor_plan),
        predict=_predict,
        score=pooled_rmse,
        score_direction="lower_is_better",
    )
    return {key: np.asarray(value) for key, value in result.deltas.items()} | missing
