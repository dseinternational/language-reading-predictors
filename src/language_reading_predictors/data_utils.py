# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later


import numpy as np
import pandas as pd

from pathlib import Path

from language_reading_predictors.data_variables import Variables as vars
from language_reading_predictors.data_variables import Categories as cats

DEFAULT_GROUPKFOLD_SPLITS = 5


def load_data() -> pd.DataFrame:
    data_path = (
        Path(__file__).resolve().parent.parent.parent / "data" / "rli_data_long.csv"
    )
    df = pd.read_csv(data_path).convert_dtypes()
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
        df[vars.BLOCKS] = df.groupby(vars.SUBJECT_ID)[vars.BLOCKS].transform(
            lambda s: s.ffill().bfill()
        )


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


def configure_data_types(df: pd.DataFrame):
    df[vars.NUMERIC] = df[vars.NUMERIC].astype("Float64")
    df[vars.GAINS] = df[vars.GAINS].astype("Float64")
    df[vars.NEXTS] = df[vars.NEXTS].astype("Float64")
    df[vars.CATEGORICAL] = df[vars.CATEGORICAL].astype("UInt8")


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
