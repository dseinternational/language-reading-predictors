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
    return df


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
