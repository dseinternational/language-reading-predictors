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
from dataclasses import dataclass, field, replace
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
    """Group indicator. shape (n_obs,). Dataset group 1 (*Initial intervention*,
    the immediate arm) maps to ``1`` and group 2 (*Wait for intervention*, the
    waitlist control) maps to ``0`` (see :func:`load_and_prepare`). So a model's
    group coefficient is on the **intervention** indicator: e.g. an ITT ``tau > 0``
    means the immediate-intervention arm scores higher, and ``P(tau > 0)`` is
    ``P(treatment helps)``."""
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
    """``"itt"`` (RCT phase only), ``"all"`` (three stacked phases), or
    ``"span"`` (one row per child: t1 baseline paired with a single later wave)."""
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
    restrict_complete: tuple[str, ...] = (),
    post_time: int = 4,
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
        a phase index. ``"span"`` pairs the wave-1 baseline with a single later
        wave ``post_time`` (default t4), giving **one row per child** — the
        between-child design used by LRP65 (T1 baselines -> full-study gain).
        Because the pre-timepoint is t1, baseline-only covariates such as block
        design (administered at t1) are available without any broadcast.
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
    restrict_complete
        Columns that must be non-missing for a row to be kept (they join the
        complete-case mask exactly like ``covariates``), but which are **not**
        added to ``prepared.covariates`` and so receive no model coefficient.
        Use this to fit a model on the complete-case subset of some covariates
        *without* adjusting for them — e.g. a matched unadjusted comparator to a
        covariate-adjusted run (LRP60a vs LRP60).
    post_time
        The post wave for ``phase_mode="span"`` (default 4 = last wave). Ignored
        for the other modes.

    Returns
    -------
    PreparedData
    """
    if phase_mode not in {"itt", "all", "span"}:
        raise ValueError(
            f"phase_mode must be 'itt', 'all', or 'span', got {phase_mode!r}"
        )

    csv_path = Path(path) if path is not None else _default_data_path()
    df = pd.read_csv(csv_path)

    phase_pairs: list[tuple[int, int]]
    if phase_mode == "itt":
        phase_pairs = [(1, 2)]
    elif phase_mode == "span":
        if int(post_time) <= 1:
            raise ValueError(
                "phase_mode='span' pairs t1 with a later wave, so post_time must "
                f"be > 1; got {post_time!r}"
            )
        phase_pairs = [(1, int(post_time))]
    else:
        phase_pairs = [(1, 2), (2, 3), (3, 4)]

    covariates = tuple(covariates)
    restrict_complete = tuple(restrict_complete)
    # Columns pulled into the frame: adjusted covariates plus complete-case-only
    # restrictors. Deduplicated, preserving order; a column in both is adjusted.
    extra_cols = list(dict.fromkeys([*covariates, *restrict_complete]))
    out_cols = [MEASURES[s].column for s in outcomes]

    per_phase_frames: list[pd.DataFrame] = []

    for phase_idx, (t_pre, t_post) in enumerate(phase_pairs):
        pre = df.loc[
            df[V.TIME] == t_pre,
            [V.SUBJECT_ID, V.GROUP, V.AGE] + out_cols + extra_cols,
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
        required = [V.GROUP, V.AGE] + required_pre + extra_cols
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

    # Group: dataset uses 1 = immediate-intervention, 2 = wait-list control.
    # Recode so G = 1 is the intervention arm and G = 0 the control arm. This
    # gives the "positive = intervention benefit" sign convention for every
    # coefficient on G (tau, tau_i/tau_k, beta_G, b_G, b_GM, a_G). See the
    # "Sign convention" section of METHODS.md.
    G = (2 - merged[V.GROUP].to_numpy(dtype=int)).astype(np.int64)
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
        pre_raw = merged[f"{m.column}_pre"].to_numpy(dtype=float)
        post_counts[s] = merged[f"{m.column}_post"].to_numpy()  # may contain NaN
        # Beta-Binomial ceiling guard (#80): a pre/post count above n_trials
        # would silently produce a NaN/-inf log-likelihood (and an invalid
        # logit for the pre covariate). Fail loudly, naming the measure.
        for which, arr in (
            ("pre", pre_raw),
            ("post", np.asarray(post_counts[s], dtype=float)),
        ):
            finite = arr[np.isfinite(arr)]
            if finite.size and finite.max() > m.n_trials:
                raise ValueError(
                    f"Measure {s!r} ({m.column}_{which}) has value "
                    f"{finite.max():g} above its n_trials ceiling {m.n_trials}; "
                    "fix measures.py or check the source data."
                )
        pre_logit[s] = logit_safe(merged[f"{m.column}_pre"], m.n_trials)
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


def load_and_prepare_lagged_outcome(
    outcome_symbol: str,
    *,
    outcome_time: int,
    path: str | Path | None = None,
    outcomes: tuple[str, ...] = ITT_OUTCOMES,
    covariates: tuple[str, ...] = (),
) -> PreparedData:
    """ITT prep, but with ``outcome_symbol``'s post-counts taken from a later wave.

    Built on :func:`load_and_prepare` with ``phase_mode="itt"`` — so baselines are
    at t1 and the mediator's post is at t2 — but the **outcome** symbol's post
    counts are replaced with that symbol's scores at ``outcome_time`` (e.g. t3),
    aligned by subject. This is the temporal-ordering *sensitivity* design for the
    LRP59/LRP62 mediation models (issue #84): mediator at t2, outcome at a later
    wave, so the mediator precedes the outcome in time.

    Caveat (carried by the caller into the report): the t2 -> ``outcome_time``
    increment is **not randomised** — both arms are treated after t2 — so the
    treatment effect on the later outcome is no longer a clean randomised
    contrast. Use only as a triangulation point, not a headline estimate.

    Subjects in the ITT base who have no ``outcome_symbol`` score at
    ``outcome_time`` get ``NaN`` post-counts; the mediation factory drops those
    rows (its existing missing-outcome mask), so the effective ``n`` may shrink.
    """
    outcomes = tuple(outcomes)
    if outcome_time <= 2:
        raise ValueError(
            "outcome_time must be a post-RCT wave (>2); use load_and_prepare for "
            "the randomised t2 outcome"
        )
    if outcome_symbol not in outcomes:
        raise ValueError(
            f"outcome_symbol {outcome_symbol!r} must be included in outcomes={outcomes!r}"
        )
    base = load_and_prepare(
        path=path, phase_mode="itt", outcomes=outcomes, covariates=covariates
    )
    csv_path = Path(path) if path is not None else _default_data_path()
    df = pd.read_csv(csv_path)
    col = MEASURES[outcome_symbol].column
    later = (
        df.loc[df[V.TIME] == outcome_time, [V.SUBJECT_ID, col]]
        .drop_duplicates(V.SUBJECT_ID)
        .set_index(V.SUBJECT_ID)[col]
    )
    # Align the later-wave outcome to the ITT base rows by subject; missing -> NaN.
    new_post = later.reindex(base.subject_ids).to_numpy(dtype=float)
    post_counts = dict(base.post_counts)
    post_counts[outcome_symbol] = new_post
    return replace(base, post_counts=post_counts)
