# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the derived intervention schema in :mod:`data_utils`.

``period`` and ``on_intervention`` are derived once, in the load path, from
``group`` x ``time`` so the period-resolved / intervention-aligned analyses
(#104) share a single definition. These tests pin that mapping and the
expected per-stratum gain-row counts the diagnostic relies on.
"""

from __future__ import annotations

import pandas as pd

from language_reading_predictors.data_utils import load_data
from language_reading_predictors.data_variables import Variables as V


def _df() -> pd.DataFrame:
    return load_data()


# ── column presence + basic shape ────────────────────────────────────────


def test_period_and_on_intervention_columns_exist():
    df = _df()
    assert V.PERIOD in df.columns
    assert V.ON_INTERVENTION in df.columns


def test_period_equals_time():
    df = _df()
    assert bool((df[V.PERIOD] == df[V.TIME]).all())


def test_on_intervention_is_complete_boolean():
    df = _df()
    # group and time are always present, so the indicator never has gaps.
    assert int(df[V.ON_INTERVENTION].isna().sum()) == 0
    assert df[V.ON_INTERVENTION].dropna().isin([True, False]).all()


def test_derived_columns_stay_out_of_default_predictor_sets():
    # period / on_intervention are deliberately absent from ALL so they do
    # not leak into DEFAULT_GAIN / DEFAULT_LEVEL.
    assert V.PERIOD not in V.ALL
    assert V.ON_INTERVENTION not in V.ALL


# ── intervention mapping (group x period -> on/off) ──────────────────────


def test_group1_rows_are_all_on_intervention():
    df = _df()
    g1 = df[df[V.GROUP] == 1]
    assert bool(g1[V.ON_INTERVENTION].all())


def test_group2_period1_rows_are_all_off_intervention():
    df = _df()
    g2_p1 = df[(df[V.GROUP] == 2) & (df[V.PERIOD] == 1)]
    assert len(g2_p1) > 0
    assert not bool(g2_p1[V.ON_INTERVENTION].any())


def test_group2_period2_and_3_rows_are_all_on_intervention():
    df = _df()
    g2_late = df[(df[V.GROUP] == 2) & (df[V.PERIOD] >= 2)]
    assert len(g2_late) > 0
    assert bool(g2_late[V.ON_INTERVENTION].all())


def test_off_intervention_iff_group2_period1():
    df = _df()
    off = df[~df[V.ON_INTERVENTION].astype("bool")]
    # the only off rows are the waitlist group's first (pre-crossover) period
    assert bool(((off[V.GROUP] == 2) & (off[V.PERIOD] == 1)).all())


# ── per-stratum gain-row counts (anchored on ewrswr_gain = LRP-RLI-GBG-012) ──


def test_gain_row_counts_per_stratum():
    df = _df()
    gain = df[df[V.EWRSWR_GAIN].notna()]

    all_n = len(gain)
    assert 148 <= all_n <= 165, f"all-periods gain n out of range: {all_n}"

    for period in (1, 2, 3):
        n = int((gain[V.PERIOD] == period).sum())
        assert 45 <= n <= 56, f"period-{period} gain n out of range: {n}"

    n_on = int(gain[V.ON_INTERVENTION].astype("bool").sum())
    assert 122 <= n_on <= 140, f"intervention-aligned gain n out of range: {n_on}"

    # intervention-aligned drops exactly the waitlist period-1 (off) gains.
    n_off = int((~gain[V.ON_INTERVENTION].astype("bool")).sum())
    assert all_n == n_on + n_off
    assert n_off > 0


def test_per_period_counts_sum_to_all_periods():
    df = _df()
    gain = df[df[V.EWRSWR_GAIN].notna()]
    per_period = sum(int((gain[V.PERIOD] == p).sum()) for p in (1, 2, 3))
    # every gain-bearing row sits in exactly one of periods 1-3 (none at t4).
    assert per_period == len(gain)
