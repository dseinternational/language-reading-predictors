# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Preprocessing helpers shared across the statistical models.

- ``logit_safe`` applies a Haldane-Anscombe corrected logit to a count/total pair.
- ``standardise`` z-scores a vector and returns the scaler for inverse transforms.
- ``load_and_prepare`` reads ``rli_data_long.csv`` and returns a
  :class:`PreparedData` container of numpy arrays ready for PyMC.

Conventions
-----------
- RCT (randomised) phase is ``time in {1, 2}`` — that is, the pre-score is
  ``time == 1`` and the post-score is ``time == 2``. The ITT models use this
  phase only.
- Mechanism models (LRP56-LRP58) stack all three phase transitions
  ``(t1 -> t2, t2 -> t3, t3 -> t4)`` with a phase indicator.
- Missing pre-scores in the chosen phase are dropped with a printed warning.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.statistical_models.datasets import (
    DatasetSpec,
    StudyMeasure,
)
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
    """Phase index. For ``itt``/``all`` the transition (0 = t1->t2, 1 = t2->t3,
    2 = t3->t4); for ``levels`` the timepoint index (0 = t1 ... 3 = t4).
    shape (n_obs,)."""
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
    """``"itt"`` (RCT phase only), ``"all"`` (three stacked transitions),
    ``"levels"`` (four per-timepoint score rows), or ``"span"`` (one row per
    child: t1 baseline paired with a single later wave)."""
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


#: Missing-indicator hearing-status covariates derived by :func:`add_hearing_status`
#: (revised DAG 2026-07-10, #244): ``hs`` = impaired (1) vs clear (0, the reference,
#: with unknown filled to it); ``hs_missing`` = 1 when hearing status is unknown.
#: Both are complete, so adjusting for them never triggers NaN-driven complete-case dropping.
HEARING_STATUS_COVARIATES: tuple[str, str] = ("hs", "hs_missing")


def add_hearing_status(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the missing-indicator hearing-status (HS) covariates.

    The revised DAG (2026-07-10, #233/#244) makes hearing status a common cause of
    the vocabulary and code skills, so it enters the observational adjustment sets.
    ``hearing_c`` (impaired hearing OR repeated ear infections) is missing for some
    children; per the #244 team decision HS enters by the **missing-indicator
    method** so no child is dropped for unknown hearing status - see
    ``notes/202607101100-dag-revision-team-decisions.md``. Adds two complete columns:

    - ``hs`` - 1 = impaired, 0 = clear (unknown filled to the clear reference);
    - ``hs_missing`` - 1 = hearing status unknown (carries the unknown group as its
      own adjustment level).

    Both are NaN-free, so requesting them as adjusters never triggers complete-case
    dropping. A no-op when ``hearing_c`` is absent (e.g. other datasets).
    """
    if V.HEARING_C not in df.columns:
        return df
    out = df.copy()
    hc = pd.to_numeric(out[V.HEARING_C], errors="coerce")
    out["hs"] = hc.fillna(0.0)
    out["hs_missing"] = hc.isna().astype(float)
    return out


#: Continuous revised-DAG confounder covariates that enter the observational
#: adjustment sets (#245): speech production (SP = ``deapp_c``) and word/nonword
#: repetition / phonological memory (RW = ``erbto``). Handled by
#: :func:`add_missing_indicator_covariates` with the hearing-status policy.
MISSING_INDICATOR_COVARIATES: tuple[str, ...] = ("deapp_c", "erbto")


def add_missing_indicator_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Fill + flag the continuous DAG-confounder covariates SP / RW (#245).

    ``deapp_c`` (speech production, SP) and ``erbto`` (word/nonword repetition =
    phonological memory, RW) are common causes in the revised DAG and so enter the
    mechanism / factor adjustment sets. Both are ~94-96% complete; to keep the
    within-child panel intact they take the same **missing-indicator** policy as
    hearing status (:func:`add_hearing_status`): the value is filled to its column
    mean (arm-blind; becomes 0 after standardisation) and a ``{col}_missing``
    indicator carries the unknown group as its own adjustment level. Both derived
    columns are NaN-free, so adjusting for them never drops rows. A no-op for any
    column absent from the frame (e.g. other datasets).
    """
    out = df.copy()
    for col in MISSING_INDICATOR_COVARIATES:
        if col not in out.columns:
            continue
        v = pd.to_numeric(out[col], errors="coerce")
        out[col] = v.fillna(v.mean())
        out[f"{col}_missing"] = v.isna().astype(float)
    return out


def load_and_prepare(
    path: str | Path | None = None,
    phase_mode: str = "itt",
    outcomes: tuple[str, ...] = ITT_OUTCOMES,
    covariates: tuple[str, ...] = (),
    baseline_covariates: tuple[str, ...] = (),
    drop_missing_pre: bool = True,
    restrict_complete: tuple[str, ...] = (),
    post_time: int = 4,
    pre_required: tuple[str, ...] | None = None,
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
        a phase index. ``"levels"`` returns one row per (child, timepoint t1-t4)
        with the score at each timepoint as the (post) outcome and no own
        baseline; group + the t1 baselines broadcast across the four rows, age is
        the per-timepoint age, and ``phase`` carries the timepoint index.
        ``"span"`` pairs the wave-1 baseline with a single later wave
        ``post_time`` (default t4), giving **one row per child** — the
        between-child design used by LRP65 (T1 baselines -> full-study gain).
        Because the pre-timepoint is t1, baseline-only covariates such as block
        design (administered at t1) are available without any broadcast.
    outcomes
        Symbols (from :data:`measures.ITT_OUTCOMES`) to include as
        pre/post variables.
    covariates
        Additional pre-timepoint columns to include as standardised linear
        covariates, taken from the per-transition pre row (``itt``/``all``) or
        the per-timepoint row (``levels``). Rows with missing requested
        covariates are dropped when ``drop_missing_pre`` is true.
    baseline_covariates
        Time-invariant baseline columns recorded once at t1 (e.g. cognitive
        ability, SES) to include as standardised linear covariates, **broadcast
        from t1 across every row** so they apply to all transitions / timepoints.
        Use this for t1-only baselines in ``"all"``/``"levels"`` mode (where a
        per-row pull would be NaN after t1). Missing values trigger complete-case
        dropping like ``covariates``.
    drop_missing_pre
        If True (default), rows with any missing pre-score (for the symbols in
        ``pre_required``) or missing group are dropped and a warning is printed
        with the dropped-row count.
    pre_required
        Symbols whose pre-score must be non-missing for a row to be kept. ``None``
        (default) requires every symbol in ``outcomes`` (the historical
        behaviour). Pass a subset — possibly ``()`` — to exempt an outcome whose
        baseline the model never uses, so its missing pre-scores do not silently
        drop rows. Used by the floored / post-only outcomes (e.g. nonword ``N``,
        whose age-only LRPITT model carries no own baseline): load with
        ``outcomes=("N",), pre_required=()`` so its missing ``nonword`` t1
        values are kept, while the GROUP/AGE and post-presence checks still
        apply. Every symbol listed must also be in ``outcomes``.
    restrict_complete
        Columns that must be non-missing for a row to be kept (they join the
        complete-case mask exactly like ``covariates``), but which are **not**
        added to ``prepared.covariates`` and so receive no model coefficient.
        Use this to fit a model on the complete-case subset of some covariates
        *without* adjusting for them — e.g. a matched unadjusted comparator to a
        covariate-adjusted run (LRPITT14 vs LRPITT13).
    post_time
        The post wave for ``phase_mode="span"`` (default 4 = last wave). Ignored
        for the other modes.

    Returns
    -------
    PreparedData
    """
    if phase_mode not in {"itt", "all", "levels", "span"}:
        raise ValueError(
            f"phase_mode must be 'itt', 'all', 'levels', or 'span', got {phase_mode!r}"
        )

    # A column can be a per-row covariate/restrict_complete OR a time-invariant t1
    # baseline_covariate, not both: the baseline merge below joins on subject and
    # would suffix any duplicate column (``_x``/``_y``), after which the later
    # standardise / complete-case lookups by bare name would raise ``KeyError``.
    # Fail fast with a clear message instead.
    both = set(baseline_covariates) & (set(covariates) | set(restrict_complete))
    if both:
        raise ValueError(
            "columns cannot be both a time-invariant baseline_covariate and a "
            f"per-row covariate/restrict_complete: {sorted(both)}; list each "
            "column in exactly one role."
        )

    csv_path = Path(path) if path is not None else _default_data_path()
    df = pd.read_csv(csv_path)
    # Derive the missing-indicator hearing-status covariates (HS; #244) up front so
    # ``hs`` / ``hs_missing`` are available as complete adjusters (no row dropping).
    df = add_hearing_status(df)
    # Same for the continuous DAG-confounder covariates SP (deapp_c) and RW (erbto)
    # (#245): fill + ``{col}_missing`` so they can be adjusted for without dropping
    # within-child rows.
    df = add_missing_indicator_covariates(df)

    covariates = tuple(covariates)
    baseline_covariates = tuple(baseline_covariates)
    restrict_complete = tuple(restrict_complete)
    # Columns pulled into the frame: adjusted covariates plus complete-case-only
    # restrictors. Deduplicated, preserving order; a column in both is adjusted.
    extra_cols = list(dict.fromkeys([*covariates, *restrict_complete]))
    out_cols = [MEASURES[s].column for s in outcomes]

    # ``itt``/``all`` are autoregressive (pre -> post over a transition); ``levels``
    # is not (the score at each timepoint is the outcome, no own baseline).
    has_pre = phase_mode in {"itt", "all", "span"}

    if has_pre:
        phase_pairs: list[tuple[int, int]]
        if phase_mode == "itt":
            phase_pairs = [(1, 2)]
        elif phase_mode == "span":
            if int(post_time) <= 1:
                raise ValueError(
                    "phase_mode='span' pairs t1 with a later wave, so post_time "
                    f"must be > 1; got {post_time!r}"
                )
            phase_pairs = [(1, int(post_time))]
        else:
            phase_pairs = [(1, 2), (2, 3), (3, 4)]
        per_phase_frames: list[pd.DataFrame] = []
        for phase_idx, (t_pre, t_post) in enumerate(phase_pairs):
            pre = df.loc[
                df[V.TIME] == t_pre,
                [V.SUBJECT_ID, V.GROUP, V.AGE] + out_cols + extra_cols,
            ].copy()
            post = df.loc[df[V.TIME] == t_post, [V.SUBJECT_ID] + out_cols].copy()
            pre = pre.rename(columns={c: f"{c}_pre" for c in out_cols})
            post = post.rename(columns={c: f"{c}_post" for c in out_cols})
            m_frame = pre.merge(post, on=V.SUBJECT_ID, how="inner")
            m_frame["phase"] = phase_idx
            per_phase_frames.append(m_frame)
        merged = pd.concat(per_phase_frames, axis=0, ignore_index=True)
        n_phases = len(phase_pairs)
    else:
        # ``levels``: one row per (child, timepoint t1-t4); the score at each
        # timepoint is the outcome "level". Age + any per-timepoint covariates
        # come from the timepoint row; group and the t1-only baselines enter via
        # ``baseline_covariates`` (broadcast below). ``phase`` is the timepoint index.
        timepoints = [1, 2, 3, 4]
        per_tp_frames: list[pd.DataFrame] = []
        for tp_idx, t in enumerate(timepoints):
            frame = df.loc[
                df[V.TIME] == t,
                [V.SUBJECT_ID, V.GROUP, V.AGE] + out_cols + extra_cols,
            ].copy()
            frame = frame.rename(columns={c: f"{c}_post" for c in out_cols})
            frame["phase"] = tp_idx
            per_tp_frames.append(frame)
        merged = pd.concat(per_tp_frames, axis=0, ignore_index=True)
        n_phases = len(timepoints)

    # Time-invariant t1 baselines (e.g. cognitive ability, SES) are recorded once
    # and broadcast from t1 across every row, so they apply to all transitions /
    # timepoints rather than only the pre-t1 row.
    if baseline_covariates:
        bc = df.loc[
            df[V.TIME] == 1, [V.SUBJECT_ID, *baseline_covariates]
        ].drop_duplicates(V.SUBJECT_ID)
        merged = merged.merge(bc, on=V.SUBJECT_ID, how="left")

    if has_pre:
        if pre_required is None:
            pre_required_syms: tuple[str, ...] = tuple(outcomes)
        else:
            pre_required_syms = tuple(pre_required)
            unknown = [s for s in pre_required_syms if s not in outcomes]
            if unknown:
                raise ValueError(
                    f"pre_required symbols must be a subset of outcomes; "
                    f"{unknown} not in {outcomes!r}"
                )
        required_pre = [f"{MEASURES[s].column}_pre" for s in pre_required_syms]
    else:
        if pre_required:
            raise ValueError("pre_required is not supported for phase_mode='levels'")
        required_pre = []
    required_post = [f"{MEASURES[s].column}_post" for s in outcomes]

    n_before = len(merged)

    if drop_missing_pre:
        required = [V.GROUP, V.AGE] + required_pre + extra_cols + list(baseline_covariates)
        mask_complete = merged[required].notna().all(axis=1)
        # Also require at least one post outcome to be present.
        mask_any_post = merged[required_post].notna().any(axis=1)
        merged = merged[mask_complete & mask_any_post].reset_index(drop=True)

    dropped = n_before - len(merged)
    if dropped > 0:
        warnings.warn(
            f"load_and_prepare: dropped {dropped} of {n_before} rows with missing "
            "score, covariate, or group assignment.",
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
        post_counts[s] = merged[f"{m.column}_post"].to_numpy()  # may contain NaN
        # Beta-Binomial ceiling guard (#80): a pre/post count above n_trials
        # would silently produce a NaN/-inf log-likelihood (and an invalid
        # logit for the pre covariate). Fail loudly, naming the measure.
        checks: list[tuple[str, np.ndarray]] = [
            ("post", np.asarray(post_counts[s], dtype=float))
        ]
        if has_pre:
            checks.append(("pre", merged[f"{m.column}_pre"].to_numpy(dtype=float)))
        for which, arr in checks:
            finite = arr[np.isfinite(arr)]
            if finite.size and finite.max() > m.n_trials:
                raise ValueError(
                    f"Measure {s!r} ({m.column}_{which}) has value "
                    f"{finite.max():g} above its n_trials ceiling {m.n_trials}; "
                    "fix measures.py or check the source data."
                )
        if has_pre:
            pre_logit[s] = logit_safe(merged[f"{m.column}_pre"], m.n_trials)
        n_trials_dict[s] = m.n_trials
        column_map[s] = m.column

    covariate_values: dict[str, np.ndarray] = {}
    covariate_scalers: dict[str, Standardiser] = {}
    for c in (*covariates, *baseline_covariates):
        if merged[c].nunique(dropna=True) <= 1:
            # Constant on the loaded rows (e.g. a missing-indicator with no missing
            # values in this subset): carries no information and cannot be
            # standardised, so drop it rather than divide by zero. It receives no
            # model coefficient; callers that iterate covariates must tolerate a
            # requested covariate being absent.
            warnings.warn(
                f"load_and_prepare: covariate {c!r} is constant on the loaded rows; "
                "dropping it (no coefficient).",
                stacklevel=2,
            )
            continue
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
        n_phases=n_phases,
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


def load_and_prepare_aligned(
    *,
    path: str | Path | None = None,
    outcomes: tuple[str, ...] = ITT_OUTCOMES,
    ability_covariate: str | None = None,
    include_dose: bool = False,
) -> PreparedData:
    """Per-protocol onset-aligned single-gain frame: one row per child.

    Aligns both arms by intervention onset across ~two periods (~the 40-week RLI
    program): the immediate arm (group 1) onsets at t1 -> aligned window t1->t3;
    the wait-list arm (group 2) onsets at t2 (post-crossover) -> aligned window
    t2->t4. Each child contributes ONE row -- ``pre`` = score at onset, ``post``
    = score at onset + two periods, ``A`` = age-at-onset, ``G`` = cohort
    (immediate vs wait-list) -- with cognitive ability (block design, recorded at
    t1) and, optionally, the cumulative sessions delivered over the window as a
    dose covariate.

    The cohort contrast is **not randomised**: it compares the two arms at their
    own aligned endpoints, so it is confounded by age-at-onset and cohort/timing
    (see the LRPAL design note). ``phase`` is all-zero and ``n_phases == 1``; with
    one row per child the aligned factory uses no child random intercept.

    Ability (block design) is a t1-only baseline, so it is merged from t1 for
    *both* arms -- not read from the wait-list arm's t2 onset row.
    """
    outcomes = tuple(outcomes)
    csv_path = Path(path) if path is not None else _default_data_path()
    df = pd.read_csv(csv_path)

    windows = {1: (1, 3), 2: (2, 4)}  # group -> (onset pre_t, aligned post_t)
    out_cols = [MEASURES[s].column for s in outcomes]

    rows: list[dict] = []
    for sid, sd in df.groupby(V.SUBJECT_ID):
        grp = int(sd[V.GROUP].iloc[0])
        if grp not in windows:
            continue
        pre_t, post_t = windows[grp]
        pre = sd.loc[sd[V.TIME] == pre_t]
        post = sd.loc[sd[V.TIME] == post_t]
        if pre.empty or post.empty:
            continue
        pre_r, post_r = pre.iloc[0], post.iloc[0]
        row = {
            V.SUBJECT_ID: sid,
            V.GROUP: grp,
            V.AGE: pre_r[V.AGE],  # age-at-onset
            "phase": 0,
        }
        for col in out_cols:
            row[f"{col}_pre"] = pre_r.get(col, np.nan)
            row[f"{col}_post"] = post_r.get(col, np.nan)
        if include_dose:
            row["dose"] = post_r.get(V.ATTEND_CUMUL, np.nan)
        rows.append(row)
    merged = pd.DataFrame(rows)

    # Ability (block design) is t1-only -> merge from t1 for every child.
    if ability_covariate is not None:
        ability_t1 = (
            df.loc[df[V.TIME] == 1, [V.SUBJECT_ID, ability_covariate]]
            .drop_duplicates(V.SUBJECT_ID)
        )
        merged = merged.merge(ability_t1, on=V.SUBJECT_ID, how="left")

    required = [V.GROUP, V.AGE]
    if ability_covariate is not None:
        required.append(ability_covariate)
    if include_dose:
        required.append("dose")
    required_post = [f"{c}_post" for c in out_cols]
    n_before = len(merged)
    mask = merged[required].notna().all(axis=1) & merged[required_post].notna().any(axis=1)
    merged = merged[mask].reset_index(drop=True)
    dropped = n_before - len(merged)
    if dropped > 0:
        warnings.warn(
            f"load_and_prepare_aligned: dropped {dropped} of {n_before} children "
            "with missing onset/aligned score, ability, or group.",
            stacklevel=2,
        )

    subject_ids = merged[V.SUBJECT_ID].to_numpy()
    _, child_idx = np.unique(subject_ids, return_inverse=True)
    G = (2 - merged[V.GROUP].to_numpy(dtype=int)).astype(np.int64)
    A_months = merged[V.AGE].to_numpy(dtype=float)
    A_std, age_scaler = standardise(A_months)

    pre_logit: dict[str, np.ndarray] = {}
    post_counts: dict[str, np.ndarray] = {}
    n_trials_dict: dict[str, int] = {}
    column_map: dict[str, str] = {}
    for s in outcomes:
        m = MEASURES[s]
        post_counts[s] = merged[f"{m.column}_post"].to_numpy()
        pre_arr = merged[f"{m.column}_pre"].to_numpy(dtype=float)
        for which, arr in (("post", post_counts[s].astype(float)), ("pre", pre_arr)):
            finite = arr[np.isfinite(arr)]
            if finite.size and finite.max() > m.n_trials:
                raise ValueError(
                    f"Measure {s!r} ({m.column}_{which}) has value {finite.max():g} "
                    f"above its n_trials ceiling {m.n_trials}; fix measures.py or data."
                )
        pre_logit[s] = logit_safe(merged[f"{m.column}_pre"], m.n_trials)
        n_trials_dict[s] = m.n_trials
        column_map[s] = m.column

    covariate_values: dict[str, np.ndarray] = {}
    covariate_scalers: dict[str, Standardiser] = {}
    cov_cols = []
    if ability_covariate is not None:
        cov_cols.append(ability_covariate)
    if include_dose:
        cov_cols.append("dose")
    for c in cov_cols:
        z, scaler = standardise(merged[c])
        covariate_values[c] = z
        covariate_scalers[c] = scaler

    return PreparedData(
        subject_ids=subject_ids,
        child_idx=child_idx.astype(np.int64),
        phase=merged["phase"].to_numpy(dtype=np.int64),
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
        n_phases=1,
        dropped_rows=dropped,
        phase_mode="aligned",
        column_map=column_map,
    )


# ---------------------------------------------------------------------------
# Wave-panel container + loader (LRP67 LCSM)
# ---------------------------------------------------------------------------


@dataclass
class WavePanel:
    """Rectangular child x wave panel for the longitudinal dynamic models.

    Unlike :class:`PreparedData` (stacked adjacent-wave transition *pairs*), this
    is a ``(n_children, n_waves)`` panel: one cell per child per wave for each
    measure, with an explicit boolean observation mask for the scattered missing
    cells. The latent change-score model (LRP67) needs all four waves per child
    laid out as a panel rather than as pre/post pairs.

    Symbols follow :data:`measures.MEASURES`: ``W`` = word reading (ewrswr),
    ``L`` = letter-sound knowledge (yarclet), ``E`` = expressive vocabulary
    (eowpvt). (The handoff's R / L / V map to W / L / E here; ``R`` is already
    receptive vocabulary in ``MEASURES`` and is *not* reused.)
    """

    subject_ids: np.ndarray
    """Subject identifier per child row. shape (n_children,)."""
    n_children: int
    n_waves: int
    waves: np.ndarray
    """Wave labels, ascending (e.g. ``[1, 2, 3, 4]``). shape (n_waves,)."""
    outcomes: tuple[str, ...]
    """Measure symbols included, in the column order used by the factories."""
    counts: dict[str, np.ndarray]
    """Symbol -> observed count, NaN where missing. shape (n_children, n_waves)."""
    obs_mask: dict[str, np.ndarray]
    """Symbol -> True where a count is observed. shape (n_children, n_waves)."""
    logit: dict[str, np.ndarray]
    """Symbol -> Haldane-corrected logit of the count, NaN where missing."""
    n_trials: dict[str, int]
    """Binomial denominator per symbol (copied from ``MEASURES``)."""
    age_months: np.ndarray
    """Age in months per cell, linearly interpolated within child over waves to
    fill the occasional missing cell. shape (n_children, n_waves)."""
    age_std: np.ndarray
    """Standardised ``age_months`` (mean / sd over all cells)."""
    age_scaler: Standardiser
    dose: np.ndarray
    """Intervention sessions (``attend``) per wave, missing -> 0 (no recorded
    sessions). shape (n_children, n_waves)."""
    dose_std: np.ndarray
    """Standardised ``dose``."""
    dose_scaler: Standardiser
    column_map: dict[str, str] = field(default_factory=dict)
    """Symbol -> source column name."""
    baseline: dict[str, np.ndarray] = field(default_factory=dict)
    """Time-invariant baseline covariate -> per-child *standardised* value, aligned
    to ``subject_ids``. shape (n_children,). Populated from the ``baseline_covariates``
    argument to :func:`load_wave_panel` (e.g. ``blocks``, the t1-only WPPSI Block
    Design score entering LRP69/70 as a predictor of trajectory shape)."""
    baseline_raw: dict[str, np.ndarray] = field(default_factory=dict)
    """As :attr:`baseline` but the raw (unstandardised) per-child value."""
    baseline_scaler: dict[str, Standardiser] = field(default_factory=dict)
    """Covariate -> :class:`Standardiser` for inverse transforms / reporting."""

    @property
    def n_obs(self) -> int:
        """Total child-wave cells (``n_children * n_waves``).

        Named to match :class:`PreparedData` so the shared pipeline header and
        ``reporting.write_run_metadata`` treat a panel as a drop-in container.
        """
        return self.n_children * self.n_waves

    @property
    def n_phases(self) -> int:
        """Number of wave-to-wave transitions (``n_waves - 1``)."""
        return self.n_waves - 1

    @property
    def dropped_rows(self) -> int:
        """Always 0 — missing cells are masked in the factory, not dropped."""
        return 0


def load_wave_panel(
    path: str | Path | None = None,
    outcomes: tuple[str, ...] = ("W", "L", "E"),
    baseline_covariates: tuple[str, ...] = (),
) -> WavePanel:
    """Pivot ``rli_data_long.csv`` into a child x wave :class:`WavePanel`.

    For each measure symbol in ``outcomes`` a ``(n_children, n_waves)`` array of
    observed counts is built (NaN where a child is missing that wave) together
    with a boolean ``obs_mask`` and the Haldane-corrected logit. ``age`` is
    interpolated within child across waves to fill the occasional missing cell;
    ``attend`` (dose) missings are treated as zero recorded sessions. Age and
    dose are standardised over all cells.

    Masked cells are **not** dropped — the dynamic-model factories observe only
    the unmasked cells (see :func:`factories.build_lcsm_model`), so a child with
    one missing score still contributes its other waves. This matters at n~54.

    ``baseline_covariates`` names *time-invariant* per-child columns (e.g.
    ``blocks``, the t1-only WPPSI Block Design score): each is taken from the
    first wave at which the child has a value and standardised over children,
    exposed on :attr:`WavePanel.baseline`. A baseline covariate must be observed
    for every child (it enters every child's trajectory), so a missing value
    raises rather than being masked.
    """
    csv_path = Path(path) if path is not None else _default_data_path()
    df = pd.read_csv(csv_path)

    waves = np.sort(df[V.TIME].dropna().unique()).astype(int)
    subject_ids = np.sort(df[V.SUBJECT_ID].unique())
    n_children = int(subject_ids.size)
    n_waves = int(waves.size)

    def _pivot(column: str) -> np.ndarray:
        wide = df.pivot_table(
            index=V.SUBJECT_ID, columns=V.TIME, values=column, aggfunc="first"
        ).reindex(index=subject_ids, columns=waves)
        return wide.to_numpy(dtype=float)

    counts: dict[str, np.ndarray] = {}
    obs_mask: dict[str, np.ndarray] = {}
    logit: dict[str, np.ndarray] = {}
    n_trials: dict[str, int] = {}
    column_map: dict[str, str] = {}
    for s in outcomes:
        m = MEASURES[s]
        c = _pivot(m.column)
        present = ~np.isnan(c)
        lg = np.full_like(c, np.nan)
        lg[present] = logit_safe(c[present], m.n_trials)
        counts[s] = c
        obs_mask[s] = present
        logit[s] = lg
        n_trials[s] = m.n_trials
        column_map[s] = m.column

    # Age: interpolate within child across waves (monotonic ~6-month steps) so
    # the rare missing cell is filled in both directions, then standardise.
    age_months = pd.DataFrame(_pivot(V.AGE), columns=waves).interpolate(
        axis=1, limit_direction="both"
    ).to_numpy(dtype=float)
    age_std, age_scaler = standardise(age_months)

    # Dose: intervention sessions per wave; missing -> 0 recorded sessions.
    dose = np.nan_to_num(_pivot(V.ATTEND), nan=0.0)
    dose_std, dose_scaler = standardise(dose)

    # Time-invariant baseline covariates (e.g. blocks, recorded at t1 only): one
    # value per child, taken from the first wave at which it is observed, then
    # standardised over children. They enter the growth models (LRP69/70) as a
    # per-child predictor of trajectory shape, not as a panel outcome.
    baseline: dict[str, np.ndarray] = {}
    baseline_raw: dict[str, np.ndarray] = {}
    baseline_scaler: dict[str, Standardiser] = {}
    for col in baseline_covariates:
        wide = _pivot(col)
        first_idx = np.argmax(~np.isnan(wide), axis=1)
        raw = wide[np.arange(n_children), first_idx]
        if np.isnan(raw).any():
            missing = subject_ids[np.isnan(raw)]
            raise ValueError(
                f"Baseline covariate {col!r} is missing for children "
                f"{missing.tolist()}; a time-invariant baseline covariate must be "
                "observed for every child."
            )
        z, scaler = standardise(raw)
        baseline_raw[col] = raw
        baseline[col] = z
        baseline_scaler[col] = scaler

    return WavePanel(
        subject_ids=subject_ids,
        n_children=n_children,
        n_waves=n_waves,
        waves=waves,
        outcomes=tuple(outcomes),
        counts=counts,
        obs_mask=obs_mask,
        logit=logit,
        n_trials=n_trials,
        age_months=age_months,
        age_std=age_std,
        age_scaler=age_scaler,
        dose=dose,
        dose_std=dose_std,
        dose_scaler=dose_scaler,
        column_map=column_map,
        baseline=baseline,
        baseline_raw=baseline_raw,
        baseline_scaler=baseline_scaler,
    )


# ---------------------------------------------------------------------------
# Longitudinal panel container + loader (multi-dataset, #165)
# ---------------------------------------------------------------------------


@dataclass
class LongitudinalPanel:
    """Descriptive repeated-measures panel for a non-intervention study.

    A sibling of :class:`WavePanel` for datasets that carry a **group** factor and
    no treatment / randomised-phase semantics (the Byrne historical cohort, #165).
    Rows in ``long`` are the tidy complete-case observations (one per subject x
    wave); the historical-growth factory indexes them by group / wave / subject.
    Exposes the same ``n_obs`` / ``n_children`` / ``n_phases`` / ``dropped_rows``
    accessors as :class:`WavePanel` so the shared pipeline header and
    ``reporting.write_run_metadata`` treat it as a drop-in container.
    """

    dataset: DatasetSpec
    measures: tuple[str, ...]
    """Study-local measure symbols included, in the requested order."""
    long: pd.DataFrame
    """Tidy complete-case frame, sorted by (group, subject, wave). Columns: the
    dataset subject / wave / group columns, ``<group_col>_label``, and one column
    per measure symbol."""
    subject_ids: list
    group_codes: list[int]
    group_labels: list[str]
    """Group labels aligned to ``group_codes``."""
    waves: tuple[int, ...]
    counts: dict[str, np.ndarray]
    """Symbol -> (n_subjects, n_waves) observed counts, NaN where missing."""
    obs_mask: dict[str, np.ndarray]
    n_trials: dict[str, int]
    n_subjects: int
    n_waves: int
    dropped_subjects: int
    group_label_col: str

    @property
    def n_obs(self) -> int:
        return len(self.long)

    @property
    def n_children(self) -> int:
        return self.n_subjects

    @property
    def n_phases(self) -> int:
        return self.n_waves - 1

    @property
    def dropped_rows(self) -> int:
        # Complete-case selection drops whole subjects; in the complete-case grid
        # each occupies one row per requested wave, so report the count in the same
        # observation-row units as ``n_obs`` (subjects x waves) — matching how the
        # pipeline header / ``write_run_metadata`` treat ``dropped_rows``.
        return self.dropped_subjects * self.n_waves


def load_longitudinal_panel(
    dataset: DatasetSpec,
    measures: Sequence[StudyMeasure],
    *,
    waves: tuple[int, ...],
    complete_case: bool = True,
    path: str | Path | None = None,
) -> LongitudinalPanel:
    """Load a study's long-format CSV into a :class:`LongitudinalPanel`.

    Keeps the ``waves`` requested and (when ``complete_case``) only subjects with
    every ``measures`` value observed at every wave - the per-measure complete-case
    subset the Byrne Table 2 reproduction uses. Validates that no observed count
    exceeds its measure ceiling and that each kept subject has exactly
    ``len(waves)`` rows.
    """
    if not measures:
        raise ValueError(
            "load_longitudinal_panel requires at least one measure "
            "(complete-case selection, counts and n_trials are all measure-keyed)."
        )
    csv_path = Path(path) if path is not None else Path(dataset.path)
    df = pd.read_csv(csv_path)

    subj, wave_c, grp = dataset.subject_col, dataset.wave_col, dataset.group_col
    label_col = f"{grp}_label"
    measure_syms = list(dict.fromkeys(m.symbol for m in measures))
    required = [subj, wave_c, grp, *[m.column for m in measures]]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{csv_path} missing required columns: {missing_cols}")

    df = df.copy()
    df[wave_c] = df[wave_c].astype(int)
    df[grp] = df[grp].astype(int)
    unknown_groups = sorted(set(df[grp].unique()) - set(dataset.group_labels))
    if unknown_groups:
        raise ValueError(
            f"{csv_path}: group codes {unknown_groups} have no label in "
            f"dataset.group_labels (known: {sorted(dataset.group_labels)})."
        )
    df[label_col] = df[grp].map(dataset.group_labels)

    waves = tuple(int(w) for w in waves)
    in_waves = df[df[wave_c].isin(waves)].copy()
    n_subjects_before = int(in_waves[subj].nunique())

    if complete_case:
        keep: pd.Index | None = None
        for m in measures:
            wide = in_waves.pivot_table(
                index=subj, columns=wave_c, values=m.column, aggfunc="first"
            ).reindex(columns=list(waves))
            complete = wide.dropna(subset=list(waves)).index
            keep = complete if keep is None else keep.intersection(complete)
        panel_df = in_waves[in_waves[subj].isin(keep)].copy()
    else:
        panel_df = in_waves

    # Expose each measure under its study-local symbol (symbol == column for the
    # Byrne measures, but keep the mapping explicit for future studies).
    for m in measures:
        if m.symbol != m.column:
            panel_df[m.symbol] = panel_df[m.column]

    keep_cols = [subj, wave_c, grp, label_col, *measure_syms]
    panel_df = (
        panel_df[keep_cols]
        .dropna(subset=[wave_c, grp])
        .sort_values([grp, subj, wave_c])
        .reset_index(drop=True)
    )

    # Ceiling guard (always) + complete-case row-count check.
    for m in measures:
        observed = panel_df[m.symbol].dropna()
        if len(observed) and float(observed.max()) > m.n_trials:
            raise ValueError(
                f"Observed {m.symbol!r} exceeds measure ceiling {m.n_trials} "
                f"(max {float(observed.max()):g})."
            )
    if complete_case and measure_syms:
        per_subject = panel_df.groupby(subj)[measure_syms[0]].size()
        bad = per_subject[per_subject != len(waves)]
        if not bad.empty:
            raise ValueError(
                f"Complete-case panel has subjects without all {len(waves)} "
                f"waves: {bad.to_dict()}"
            )

    subject_ids = panel_df[subj].drop_duplicates().tolist()
    group_codes = sorted(int(c) for c in panel_df[grp].unique())
    group_labels = [dataset.group_labels[c] for c in group_codes]

    counts: dict[str, np.ndarray] = {}
    obs_mask: dict[str, np.ndarray] = {}
    n_trials: dict[str, int] = {}
    for m in measures:
        wide = panel_df.pivot_table(
            index=subj, columns=wave_c, values=m.symbol, aggfunc="first"
        ).reindex(index=subject_ids, columns=list(waves))
        arr = wide.to_numpy(dtype=float)
        counts[m.symbol] = arr
        obs_mask[m.symbol] = ~np.isnan(arr)
        n_trials[m.symbol] = m.n_trials

    return LongitudinalPanel(
        dataset=dataset,
        measures=tuple(measure_syms),
        long=panel_df,
        subject_ids=subject_ids,
        group_codes=group_codes,
        group_labels=group_labels,
        waves=waves,
        counts=counts,
        obs_mask=obs_mask,
        n_trials=n_trials,
        n_subjects=len(subject_ids),
        n_waves=len(waves),
        dropped_subjects=n_subjects_before - len(subject_ids),
        group_label_col=label_col,
    )
