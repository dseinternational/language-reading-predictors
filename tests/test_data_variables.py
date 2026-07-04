# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Invariant tests for :mod:`data_variables`.

``data_variables`` is the single source of truth for every variable name,
grouping and default predictor set used across the model suite, so a typo or
copy-paste slip here silently propagates into every model. These tests pin the
structural invariants the rest of the codebase relies on.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from language_reading_predictors.data_variables import (
    Categories,
    Predictors,
    Variables,
)

V = Variables


def _has_no_duplicates(seq):
    return len(seq) == len(set(seq))


# ── grouping lists have no duplicates ────────────────────────────────────


def test_variable_group_lists_have_no_duplicates():
    for name in ("ALL", "GAINS", "NEXTS", "NUMERIC", "CATEGORICAL"):
        seq = getattr(V, name)
        assert _has_no_duplicates(seq), f"Variables.{name} contains duplicates"


def test_numeric_and_categorical_are_disjoint():
    overlap = set(V.NUMERIC) & set(V.CATEGORICAL)
    assert not overlap, f"NUMERIC and CATEGORICAL overlap: {sorted(overlap)}"


def test_gains_and_nexts_use_consistent_suffixes():
    assert all(g.endswith("_gain") for g in V.GAINS)
    assert all(n.endswith("_next") for n in V.NEXTS)


def test_numeric_and_categorical_are_subsets_of_all():
    all_set = set(V.ALL)
    # ATTEND_CUMUL is a derived column not listed in ALL; allow it explicitly.
    derived = {V.ATTEND_CUMUL}
    for name in ("NUMERIC", "CATEGORICAL"):
        members = set(getattr(V, name)) - derived
        missing = members - all_set
        assert not missing, f"Variables.{name} has members not in ALL: {sorted(missing)}"


# ── default predictor sets ───────────────────────────────────────────────


def test_default_gain_excludes_structural_groups():
    gain = set(Predictors.DEFAULT_GAIN)
    assert V.SUBJECT_ID not in gain
    assert not (gain & set(V.GAINS)), "DEFAULT_GAIN must not contain gain columns"
    assert not (gain & set(V.NEXTS)), "DEFAULT_GAIN must not contain next columns"
    assert not (gain & set(V.DEFAULT_EXCLUDED)), "DEFAULT_GAIN must honour DEFAULT_EXCLUDED"


def test_default_level_excludes_period_related():
    level = set(Predictors.DEFAULT_LEVEL)
    assert not (level & set(V.PERIOD_RELATED)), (
        "DEFAULT_LEVEL must exclude period-related measures"
    )
    assert not (level & set(V.GAINS))
    assert not (level & set(V.NEXTS))
    assert V.SUBJECT_ID not in level


def test_default_predictor_sets_are_nonempty_and_unique():
    for name in (
        "DEFAULT_GAIN",
        "DEFAULT_LEVEL",
        "DEFAULT_GAIN_NUMERIC",
        "DEFAULT_GAIN_CATEGORICAL",
        "DEFAULT_LEVEL_NUMERIC",
        "DEFAULT_LEVEL_CATEGORICAL",
    ):
        seq = getattr(Predictors, name)
        assert seq, f"Predictors.{name} is empty"
        assert _has_no_duplicates(seq), f"Predictors.{name} has duplicates"


def test_default_gain_numeric_categorical_partition():
    # Every DEFAULT_GAIN predictor is classified as numeric XOR categorical.
    gain = set(Predictors.DEFAULT_GAIN)
    numeric = set(Predictors.DEFAULT_GAIN_NUMERIC)
    categorical = set(Predictors.DEFAULT_GAIN_CATEGORICAL)
    assert numeric.issubset(gain)
    assert categorical.issubset(gain)
    assert not (numeric & categorical)
    assert numeric | categorical == gain


# ── construct mapping ────────────────────────────────────────────────────


def test_construct_membership_is_unique():
    seen: dict[str, str] = {}
    for construct, members in V.CONSTRUCTS.items():
        for m in members:
            assert m not in seen, (
                f"{m!r} appears in both {seen.get(m)!r} and {construct!r}"
            )
            seen[m] = construct


def test_construct_of_round_trips_every_member():
    for construct, members in V.CONSTRUCTS.items():
        for m in members:
            assert V.construct_of(m) == construct


def test_construct_of_strips_gain_and_next_suffixes():
    # ewrswr is in reading_word; its _gain/_next derivatives map identically.
    assert V.construct_of(V.EWRSWR) == "reading_word"
    assert V.construct_of(V.EWRSWR_GAIN) == "reading_word"
    assert V.construct_of(V.EWRSWR_NEXT) == "reading_word"


def test_construct_of_unknown_returns_other():
    assert V.construct_of("definitely_not_a_variable") == "other"


# ── name lookup + categories ─────────────────────────────────────────────


def test_get_variable_name_falls_back_to_identifier():
    assert V.get_variable_name(V.AGE) == "Age"
    assert V.get_variable_name("no_such_var") == "no_such_var"


def test_category_maps_are_nonempty_mappings():
    for name in ("GENDER", "AREA", "GROUP", "TIME", "IMPAIRED", "NO_YES"):
        mapping = getattr(Categories, name)
        assert isinstance(mapping, Mapping) and mapping


def test_schema_groups_are_read_only():
    assert isinstance(V.NUMERIC, tuple)
    assert isinstance(Predictors.DEFAULT_GAIN, tuple)
    with pytest.raises(AttributeError):
        V.NUMERIC.append("new_column")
    with pytest.raises(TypeError):
        Categories.GENDER[3] = "Other"
