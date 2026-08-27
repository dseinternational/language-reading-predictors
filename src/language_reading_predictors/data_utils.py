# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later


import warnings

import numpy as np
import pandas as pd

from pathlib import Path

from language_reading_predictors.data_variables import Variables as vars
from language_reading_predictors.data_variables import Categories as cats

DEFAULT_GROUPKFOLD_SPLITS = 5

# Known-corrupt source cells quarantined to missing at load (#631 finding 3).
# Provenance: notes/202608262120-erb-word-repetition-quarantine-631.md.
#
# The t4 ERB word-repetition row for ID_FDCBDCF29AC0BF03 records erbword=28,
# erbnw=14, erbto=14 — the sole violation of the additivity identity
# erbto == erbword + erbnw, which holds exactly on the other 201 of 202
# complete rows, and erbword's maximum on every other row is 18. A word/total
# transposition (erbword=14, erbto=28) is plausible but cannot be verified from
# this checkout, so per the #631 decision rule the affected cells are recorded
# as missing pending source-archive verification — never guessed. erbnw=14 is
# consistent under both readings and is kept. The child's t3 row repeats the
# corrupt value through the derived erbword/erbto ``_next`` and ``_gain``
# columns, so those cells are quarantined with it.
#
# Each entry is (subject_id, time, columns-to-set-missing). This list is the
# single sanctioned bypass of :func:`validate_erb_consistency`.
KNOWN_BAD_CELLS: tuple[tuple[str, int, tuple[str, ...]], ...] = (
    ("ID_FDCBDCF29AC0BF03", 4, (vars.ERBWORD, vars.ERBTO)),
    (
        "ID_FDCBDCF29AC0BF03",
        3,
        (vars.ERBWORD_NEXT, vars.ERBWORD_GAIN, vars.ERBTO_NEXT, vars.ERBTO_GAIN),
    ),
)


def quarantine_known_bad_cells(df: pd.DataFrame) -> None:
    """Set the :data:`KNOWN_BAD_CELLS` entries to missing, in place.

    Emits a single warning naming the quarantined cells and the provenance note
    (notes/202608262120-erb-word-repetition-quarantine-631.md) so no analysis
    consumes the corrupt values silently.
    """
    quarantined: list[str] = []
    for subject, time, columns in KNOWN_BAD_CELLS:
        row_mask = (df[vars.SUBJECT_ID] == subject) & (df[vars.TIME] == time)
        for col in columns:
            if col not in df.columns:
                continue
            hit = row_mask & df[col].notna()
            if hit.any():
                df.loc[hit, col] = pd.NA
                quarantined.append(f"{col}@t{time}")
    if quarantined:
        warnings.warn(
            "quarantined known-bad cell(s) to missing pending "
            f"source-archive verification (#631 finding 3): {', '.join(quarantined)} "
            f"for {KNOWN_BAD_CELLS[0][0]}. See "
            "notes/202608262120-erb-word-repetition-quarantine-631.md.",
            stacklevel=3,
        )


def validate_erb_consistency(df: pd.DataFrame) -> None:
    """Fail loud on any NEW ERB word/nonword-repetition inconsistency.

    On rows where all three ERB columns are present, require the additivity
    identity ``erbword + erbnw == erbto``, non-negative values, and the
    observed-maximum soft caps ``erbword <= 18``, ``erbnw <= 18``,
    ``erbto <= 36`` (the caps are observed maxima pending documentation of the
    ERB test ceilings — not confirmed denominators). :data:`KNOWN_BAD_CELLS` is
    the single sanctioned bypass: cells named there are set to missing by the
    quarantine before this check runs, and any surviving violation raises,
    naming the subject and time.
    """
    cols = (vars.ERBWORD, vars.ERBNW, vars.ERBTO)
    if any(c not in df.columns for c in cols):
        return
    word, nw, total = (
        pd.to_numeric(df[c], errors="coerce").astype("float64") for c in cols
    )
    present = word.notna() & nw.notna() & total.notna()
    violates = present & (
        (word + nw != total)
        | (word < 0)
        | (nw < 0)
        | (total < 0)
        | (word > 18)
        | (nw > 18)
        | (total > 36)
    )
    known = {(subject, time) for subject, time, _ in KNOWN_BAD_CELLS}
    is_known = pd.Series(
        [
            (subject, time) in known
            for subject, time in zip(
                df[vars.SUBJECT_ID], df[vars.TIME], strict=True
            )
        ],
        index=df.index,
    )
    new_violations = violates & ~is_known
    if new_violations.any():
        rows = df.loc[new_violations, [vars.SUBJECT_ID, vars.TIME]]
        pairs = ", ".join(f"{s}@t{t}" for s, t in rows.itertuples(index=False))
        raise ValueError(
            "ERB word/nonword repetition inconsistency (erbword + erbnw != "
            "erbto, a negative value, or a value above the observed-maximum "
            f"caps 18/18/36) at: {pairs}. Verify the cell(s) against the source "
            "archive; the single sanctioned bypass is data_utils.KNOWN_BAD_CELLS "
            "(see notes/202608262120-erb-word-repetition-quarantine-631.md)."
        )


def load_data() -> pd.DataFrame:
    data_path = (
        Path(__file__).resolve().parent.parent.parent / "data" / "rli_data_long.csv"
    )
    df = pd.read_csv(data_path).convert_dtypes()
    # Known-corrupt cells go to missing before any dtype/derivation step, and the
    # ERB integrity check then guards against any new violation slipping in
    # (#631 finding 3; notes/202608262120-erb-word-repetition-quarantine-631.md).
    quarantine_known_bad_cells(df)
    validate_erb_consistency(df)
    configure_data_types(df)
    add_intervention_schema(df)
    _broadcast_baseline_blocks(df)
    return df


def _broadcast_baseline_blocks(df: pd.DataFrame) -> None:
    """Broadcast block design (recorded only at wave 1) across each child's rows.

    In the long format ``blocks`` is present only on the t1 row, so broadcasting a
    child's single value to all their rows makes it usable as a time-invariant
    baseline covariate (issue #186). ``blocks`` is in ``Variables.DEFAULT_EXCLUDED``,
    so this changes no default predictor set — it only affects models that opt it in
    via ``include`` (it realises the intent documented by
    ``Variables.TIME_INVARIANT_BASELINES`` for block design).
    """
    if vars.BLOCKS in df.columns:
        # blocks is recorded once per child (t1); map that single value to all of the
        # child's rows. Taking the first non-null is independent of the frame's row
        # order and well-defined even if a future extract ever carried more than one
        # non-null value for a child — unlike ``ffill().bfill()``, which would knit
        # together neighbouring values in that case.
        first_block = (
            df.dropna(subset=[vars.BLOCKS])
            .groupby(vars.SUBJECT_ID)[vars.BLOCKS]
            .first()
        )
        df[vars.BLOCKS] = df[vars.SUBJECT_ID].map(first_block)


def add_intervention_schema(df: pd.DataFrame) -> None:
    """Derive the ``period`` index and ``on_intervention`` indicator in place.

    Single source of truth for the period-resolved / intervention-aligned
    analyses (#104). Both columns are derived from ``group`` and ``time`` so
    nothing downstream hard-codes the group x period mapping.

    - ``period`` indexes the gain interval starting at the current wave: a
      gain recorded at baseline wave ``t`` covers period ``t``
      (``t`` in {1, 2, 3}; ``*_gain`` columns are NaN at t4). It equals
      ``time``.
    - ``on_intervention`` is ``True`` when the child was receiving the
      intervention during that period. The immediate group (``group == 1``)
      is on from period 1; the waitlist group (``group == 2``) is off in
      period 1 only and on from period 2 (after crossover). So a row is off
      intervention iff ``(group == 2) & (period == 1)``.

    Must run after :func:`configure_data_types` (which casts ``group`` /
    ``time`` to integers) and before any call to
    :func:`configure_data_categories` (which would remap ``group`` to label
    strings).
    """
    # NOTE: ``period`` is set equal to ``time`` for every row, so t4 rows carry
    # ``period == 4`` and (via the rule below) ``on_intervention == True`` as a
    # byproduct — even though a gain interval only spans periods {1, 2, 3} and the
    # ``*_gain`` columns are NaN at t4. Those values are only meaningful on gain
    # rows (where t4 is NaN and so drops out anyway). Do NOT use ``period`` /
    # ``on_intervention`` to filter a *level* (per-timepoint) analysis without also
    # excluding ``time == 4``; otherwise the t4 wave would be mislabelled as a
    # fourth on-intervention period.
    df[vars.PERIOD] = df[vars.TIME]
    df[vars.ON_INTERVENTION] = (
        (df[vars.GROUP] == 1) | (df[vars.PERIOD] >= 2)
    ).astype("boolean")


def load_and_filter(
    target_var: str,
    predictor_vars: list[str],
    outlier_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load data, filter to rows with non-null target, and optionally exclude outliers.

    Returns (df, X, y, groups) where *X* has ``pd.NA`` replaced with
    ``np.nan`` and cast to ``float64``.  Missing values are left in place
    because LightGBM handles NaN natively.
    """
    df = load_data()
    df = df[df[target_var].notna()].copy()
    if outlier_threshold is not None:
        df = df[df[target_var] < outlier_threshold]
    X = df[predictor_vars].replace({pd.NA: np.nan}).astype("float64")
    y = df[target_var].astype("float64")
    groups = df[vars.SUBJECT_ID]
    return df, X, y, groups


def configure_data_types(df: pd.DataFrame) -> None:
    numeric = list(vars.NUMERIC)
    gains = list(vars.GAINS)
    nexts = list(vars.NEXTS)
    categorical = list(vars.CATEGORICAL)

    df[numeric] = df[numeric].astype("Float64")
    df[gains] = df[gains].astype("Float64")
    df[nexts] = df[nexts].astype("Float64")
    df[categorical] = df[categorical].astype("UInt8")


def configure_data_categories(df: pd.DataFrame):
    df[vars.TIME] = df[vars.TIME].map(cats.TIME).astype("category")
    df[vars.AREA] = df[vars.AREA].map(cats.AREA).astype("category")
    df[vars.GROUP] = df[vars.GROUP].map(cats.GROUP).astype("category")
    df[vars.GENDER] = df[vars.GENDER].map(cats.GENDER).astype("category")
    df[vars.HEARING] = df[vars.HEARING].map(cats.IMPAIRED).astype("category")
    df[vars.VISION] = df[vars.VISION].map(cats.IMPAIRED).astype("category")
    df[vars.EARINF] = df[vars.EARINF].map(cats.NO_YES).astype("category")
    df[vars.MUMOCC] = df[vars.MUMOCC].astype("category")
    df[vars.DADOCC] = df[vars.DADOCC].astype("category")
    df[vars.BEDTIMEREAD] = (
        df[vars.BEDTIMEREAD].map(cats.WEEKLY_READING).astype("category")
    )
    df[vars.OTHERTIMEREAD] = (
        df[vars.OTHERTIMEREAD].map(cats.WEEKLY_READING).astype("category")
    )
