# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Preprocessing helpers shared across LRP52-LRP58.

- ``logit_safe`` applies a Haldane-Anscombe corrected logit to a count/total pair.
- ``standardise`` z-scores a vector and returns the scaler for inverse transforms.
- ``load_and_prepare`` reads ``rli_data_long.csv`` and returns a
  :class:`PreparedData` container of numpy arrays ready for PyMC.

Conventions
-----------
- RCT (randomised) phase is ``time in {1, 2}`` — that is, the pre-score is
  ``time == 1`` and the post-score is ``time == 2``. LRP52-LRP55 use this
  phase only.
- Mechanism models (LRP56-LRP58) stack all three phase transitions
  ``(t1 -> t2, t2 -> t3, t3 -> t4)`` with a phase indicator.
- Missing pre-scores in the chosen phase are dropped with a printed warning.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.measures import (
    ITT_OUTCOMES,
    MEASURES,
)


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------


def logit_safe(y: np.ndarray | pd.Series, n: int) -> np.ndarray:
    """Haldane-Anscombe corrected logit: ``log((y + 0.5) / (n - y + 0.5))``."""
    y = np.asarray(y, dtype=float)
    return np.log((y + 0.5) / (n - y + 0.5))


@dataclass
class Standardiser:
    mean: float
    sd: float

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.mean) / self.sd

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=float) * self.sd + self.mean


def standardise(x: np.ndarray | pd.Series) -> tuple[np.ndarray, Standardiser]:
    arr = np.asarray(x, dtype=float)
    mu = float(np.nanmean(arr))
    sd = float(np.nanstd(arr, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Standard deviation of x must be positive.")
    return (arr - mu) / sd, Standardiser(mean=mu, sd=sd)


# ---------------------------------------------------------------------------
# Prepared-data container
# ---------------------------------------------------------------------------


@dataclass
class PreparedData:
    """Arrays and metadata consumed by the PyMC factories."""

    subject_ids: np.ndarray
    """Subject identifier for each observation row. shape (n_obs,)."""
    child_idx: np.ndarray
    """Integer child index in ``0..n_children-1``. shape (n_obs,)."""
    phase: np.ndarray
    """Phase index (0 = t1->t2, 1 = t2->t3, 2 = t3->t4). shape (n_obs,)."""
    G: np.ndarray
    """Group indicator (0 = control arm, 1 = intervention arm). shape (n_obs,)."""
    A_months: np.ndarray
    """Age in months at the pre-timepoint of each phase. shape (n_obs,)."""
    A_std: np.ndarray
    """Standardised age. shape (n_obs,)."""
    age_scaler: Standardiser
    pre_logit: dict[str, np.ndarray]
    """For each symbol in ``ITT_OUTCOMES``: Haldane-corrected logit of pre-score."""
    post_counts: dict[str, np.ndarray]
    """For each symbol in ``ITT_OUTCOMES``: integer post-score count."""
    n_trials: dict[str, int]
    """Binomial denominator for each outcome (copied from ``MEASURES``)."""
    n_obs: int
    n_children: int
    n_phases: int
    dropped_rows: int
    """Number of rows dropped due to missing pre-scores or group."""
    phase_mode: str
    """``"itt"`` (RCT phase only) or ``"all"`` (three stacked phases)."""
    column_map: dict[str, str] = field(default_factory=dict)
    """Symbol -> column-name map for the subset of outcomes prepared."""
    covariates: dict[str, np.ndarray] = field(default_factory=dict)
    """Standardised non-outcome covariates keyed by source column name."""
    covariate_scalers: dict[str, Standardiser] = field(default_factory=dict)
    """Mean / SD scalers for entries in :attr:`covariates`."""


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------


def _default_data_path() -> Path:
    from language_reading_predictors.statistical_models import environment as env

    return Path(env.DATA_DIR) / "rli_data_long.csv"


def load_and_prepare(
    path: str | Path | None = None,
    phase_mode: str = "itt",
    outcomes: tuple[str, ...] = ITT_OUTCOMES,
    covariates: tuple[str, ...] = (),
    drop_missing_pre: bool = True,
) -> PreparedData:
    """
    Load ``rli_data_long.csv`` and build arrays for the model factories.

    Parameters
    ----------
    path
        Location of the long-format CSV. Defaults to ``data/rli_data_long.csv``.
    phase_mode
        ``"itt"`` keeps only the randomised phase (t1 -> t2). ``"all"`` stacks
        all three adjacent-time transitions (t1->t2, t2->t3, t3->t4) and adds
        a phase index.
    outcomes
        Symbols (from :data:`measures.ITT_OUTCOMES`) to include as
        pre/post variables.
    covariates
        Additional baseline/pre-timepoint columns to include as standardised
        linear covariates. Rows with missing requested covariates are dropped
        when ``drop_missing_pre`` is true.
    drop_missing_pre
        If True (default), rows with any missing pre-score or missing group
        are dropped and a warning is printed with the dropped-row count.

    Returns
    -------
    PreparedData
    """
    if phase_mode not in {"itt", "all"}:
        raise ValueError(f"phase_mode must be 'itt' or 'all', got {phase_mode!r}")

    csv_path = Path(path) if path is not None else _default_data_path()
    df = pd.read_csv(csv_path)

    phase_pairs: list[tuple[int, int]]
    if phase_mode == "itt":
        phase_pairs = [(1, 2)]
    else:
        phase_pairs = [(1, 2), (2, 3), (3, 4)]

    covariates = tuple(covariates)
    out_cols = [MEASURES[s].column for s in outcomes]

    per_phase_frames: list[pd.DataFrame] = []

    for phase_idx, (t_pre, t_post) in enumerate(phase_pairs):
        pre = df.loc[
            df[V.TIME] == t_pre,
            [V.SUBJECT_ID, V.GROUP, V.AGE] + out_cols + list(covariates),
        ].copy()
        post = df.loc[df[V.TIME] == t_post, [V.SUBJECT_ID] + out_cols].copy()
        pre = pre.rename(columns={c: f"{c}_pre" for c in out_cols})
        post = post.rename(columns={c: f"{c}_post" for c in out_cols})
        merged = pre.merge(post, on=V.SUBJECT_ID, how="inner")
        merged["phase"] = phase_idx
        per_phase_frames.append(merged)

    merged = pd.concat(per_phase_frames, axis=0, ignore_index=True)

    required_pre = [f"{MEASURES[s].column}_pre" for s in outcomes]
    required_post = [f"{MEASURES[s].column}_post" for s in outcomes]

    n_before = len(merged)

    if drop_missing_pre:
        required = [V.GROUP, V.AGE] + required_pre + list(covariates)
        mask_complete = merged[required].notna().all(axis=1)
        # Also require at least one post outcome to be present.
        mask_any_post = merged[required_post].notna().any(axis=1)
        merged = merged[mask_complete & mask_any_post].reset_index(drop=True)

    dropped = n_before - len(merged)
    if dropped > 0:
        warnings.warn(
            f"load_and_prepare: dropped {dropped} of {n_before} rows with missing "
            "pre-score, covariate, or group assignment.",
            stacklevel=2,
        )

    subject_ids = merged[V.SUBJECT_ID].to_numpy()
    _, child_idx = np.unique(subject_ids, return_inverse=True)

    # Group: dataset uses 1 = control, 2 = intervention; map to 0/1.
    G = (merged[V.GROUP].to_numpy(dtype=int) - 1).astype(np.int64)
    if not set(np.unique(G)).issubset({0, 1}):
        raise ValueError(
            f"Group codes outside {{1, 2}} after prep: found {np.unique(merged[V.GROUP])}"
        )

    A_months = merged[V.AGE].to_numpy(dtype=float)
    A_std, age_scaler = standardise(A_months)

    pre_logit: dict[str, np.ndarray] = {}
    post_counts: dict[str, np.ndarray] = {}
    n_trials_dict: dict[str, int] = {}
    column_map: dict[str, str] = {}
    for s in outcomes:
        m = MEASURES[s]
        pre_logit[s] = logit_safe(merged[f"{m.column}_pre"], m.n_trials)
        post_counts[s] = merged[f"{m.column}_post"].to_numpy()  # may contain NaN
        n_trials_dict[s] = m.n_trials
        column_map[s] = m.column

    covariate_values: dict[str, np.ndarray] = {}
    covariate_scalers: dict[str, Standardiser] = {}
    for c in covariates:
        z, scaler = standardise(merged[c])
        covariate_values[c] = z
        covariate_scalers[c] = scaler

    phase_arr = merged["phase"].to_numpy(dtype=np.int64)

    return PreparedData(
        subject_ids=subject_ids,
        child_idx=child_idx.astype(np.int64),
        phase=phase_arr,
        G=G,
        A_months=A_months,
        A_std=A_std,
        age_scaler=age_scaler,
        pre_logit=pre_logit,
        post_counts=post_counts,
        n_trials=n_trials_dict,
        covariates=covariate_values,
        covariate_scalers=covariate_scalers,
        n_obs=int(len(merged)),
        n_children=int(len(np.unique(child_idx))),
        n_phases=len(phase_pairs),
        dropped_rows=dropped,
        phase_mode=phase_mode,
        column_map=column_map,
    )
