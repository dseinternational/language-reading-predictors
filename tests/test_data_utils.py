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
import pytest

from language_reading_predictors.data_utils import (
    KNOWN_BAD_CELLS,
    load_data,
    validate_erb_consistency,
)
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


# ── ERB word-repetition quarantine + integrity check (#631 finding 3) ────
# See notes/202608262120-erb-word-repetition-quarantine-631.md.

_ERB_BAD_SUBJECT = KNOWN_BAD_CELLS[0][0]  # ID_FDCBDCF29AC0BF03


def test_known_bad_erb_cells_load_as_missing():
    """The corrupt t4 ERB cells and their t3 derived cells are quarantined."""
    with pytest.warns(UserWarning, match="202608262120-erb-word-repetition"):
        df = load_data()
    child = df[df[V.SUBJECT_ID] == _ERB_BAD_SUBJECT]
    t4 = child[child[V.TIME] == 4]
    t3 = child[child[V.TIME] == 3]
    assert bool(t4[V.ERBWORD].isna().all())
    assert bool(t4[V.ERBTO].isna().all())
    # erbnw = 14 is consistent under both readings of the anomaly and is kept.
    assert bool((t4[V.ERBNW] == 14).all())
    for col in (V.ERBWORD_NEXT, V.ERBWORD_GAIN, V.ERBTO_NEXT, V.ERBTO_GAIN):
        assert bool(t3[col].isna().all()), col


def test_real_csv_satisfies_erb_identity_after_quarantine():
    """After the quarantine every complete row obeys erbword + erbnw == erbto
    (the identity held on 201 of 202 complete rows in the raw archive; the
    single violator is the quarantined cell)."""
    df = _df()
    complete = df[[V.ERBWORD, V.ERBNW, V.ERBTO]].dropna()
    assert len(complete) > 0
    assert bool(
        ((complete[V.ERBWORD] + complete[V.ERBNW]) == complete[V.ERBTO]).all()
    )
    # load_data itself ran validate_erb_consistency without raising; re-run it
    # on the loaded frame as an explicit regression check.
    validate_erb_consistency(df)


def test_validate_erb_consistency_rejects_new_identity_violation():
    frame = pd.DataFrame(
        {
            V.SUBJECT_ID: ["A", "B"],
            V.TIME: [1, 2],
            V.ERBWORD: [10.0, 12.0],
            V.ERBNW: [5.0, 6.0],
            V.ERBTO: [15.0, 20.0],  # B@t2 violates erbword + erbnw == erbto
        }
    )
    with pytest.raises(ValueError, match="B@t2"):
        validate_erb_consistency(frame)


def test_validate_erb_consistency_rejects_values_above_observed_maximum_caps():
    frame = pd.DataFrame(
        {
            V.SUBJECT_ID: ["C"],
            V.TIME: [1],
            V.ERBWORD: [19.0],  # above the observed-maximum soft cap of 18
            V.ERBNW: [5.0],
            V.ERBTO: [24.0],
        }
    )
    with pytest.raises(ValueError, match="C@t1"):
        validate_erb_consistency(frame)


def test_validate_erb_consistency_skips_incomplete_rows():
    """Rows with any of the three ERB columns missing are not checked — missing
    remains the documented data state, not an inconsistency."""
    frame = pd.DataFrame(
        {
            V.SUBJECT_ID: ["D"],
            V.TIME: [1],
            V.ERBWORD: [pd.NA],
            V.ERBNW: [5.0],
            V.ERBTO: [99.0],
        }
    )
    validate_erb_consistency(frame)  # must not raise


def test_known_bad_cells_are_the_single_sanctioned_bypass():
    """A frame reproducing the quarantined cell's values at its recorded
    (subject, time) key passes; the same values anywhere else raise."""
    values = {V.ERBWORD: [28.0], V.ERBNW: [14.0], V.ERBTO: [14.0]}
    sanctioned = pd.DataFrame(
        {V.SUBJECT_ID: [_ERB_BAD_SUBJECT], V.TIME: [4], **values}
    )
    validate_erb_consistency(sanctioned)  # must not raise
    unsanctioned = pd.DataFrame(
        {V.SUBJECT_ID: ["SOMEONE_ELSE"], V.TIME: [4], **values}
    )
    with pytest.raises(ValueError, match="SOMEONE_ELSE@t4"):
        validate_erb_consistency(unsanctioned)
