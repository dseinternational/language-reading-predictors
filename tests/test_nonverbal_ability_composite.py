# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The Object Assembly column and the two-subtest non-verbal ability composite.

The column is derived from the committed deposit rather than collected here, so
the load-bearing test is that what is committed still matches what the deposit
says. The composite is derived rather than stored, so the load-bearing test
there is that its definition is the sum and that it is complete wherever Block
Design is — otherwise a fit naming it would silently analyse fewer children than
its Block Design parent, and the comparison the companions exist for would not
be like-for-like.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from language_reading_predictors import paths as _paths
from language_reading_predictors.data_variables import Predictors, Variables as V
from language_reading_predictors.statistical_models.preprocessing import (
    add_nonverbal_ability_composite,
    derive_nonverbal_ability_composite,
)

REPO = Path(__file__).resolve().parent.parent


def _script():
    spec = importlib.util.spec_from_file_location(
        "derive_object_assembly", REPO / "scripts/derive_object_assembly.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_column_still_matches_the_deposit():
    """``--check`` is the guarantee that the derivation was not hand-edited."""
    assert _script()._apply(write=False) == 0


def test_object_assembly_is_complete_for_every_analysed_child():
    wide = pd.read_csv(_paths.DATA_DIR / "rli_data_wide.csv")
    column = f"{V.OBJASS}1"

    assert column in wide.columns
    assert len(wide) == 54
    assert wide[column].notna().all()
    # The published maximum is 37 (Burgoyne et al. 2012, Table 1).
    assert wide[column].between(0, 37).all()


def test_the_long_file_carries_the_subtest_at_t1_only_like_block_design():
    long = pd.read_csv(_paths.DATA_DIR / "rli_data_long.csv")

    by_wave = long.groupby(V.TIME)[V.OBJASS].count().to_dict()
    blocks_by_wave = long.groupby(V.TIME)[V.BLOCKS].count().to_dict()
    assert by_wave == blocks_by_wave == {1: 54, 2: 0, 3: 0, 4: 0}


def test_load_data_broadcasts_both_subtests_identically():
    """A t1-only column that is broadcast and one that is not is a silent trap.

    ``load_data`` broadcasts ``blocks`` from t1 across a child's rows so it is
    usable as a time-invariant baseline. Object Assembly is recorded the same way
    and must behave the same way, or a model opting it in via ``include`` would
    get a column that is null on three of four rows while its sibling is not.
    Both stay out of the default predictor sets either way.
    """
    from language_reading_predictors.data_utils import load_data

    with pytest.warns(UserWarning):
        frame = load_data()

    for column in (V.BLOCKS, V.OBJASS):
        assert frame[column].notna().all(), f"{column} was not broadcast from t1"
        assert str(frame[column].dtype) == "Float64", f"{column} has the wrong dtype"
        per_child = frame.groupby(V.SUBJECT_ID)[column].nunique()
        assert per_child.max() <= 1, f"{column} is not constant within child"


def test_the_schema_knows_object_assembly_as_a_source_column():
    """Every comparable source measurement is in ALL and NUMERIC.

    ``NUMERIC`` is what ``data_utils.configure_data_types`` keys on, so omitting
    the column there left it inferred as ``Int64`` where ``blocks`` is
    ``Float64``. ``TIME_INVARIANT_BASELINES`` documents the once-measured
    variables whose replication across a child's rows can bias tree splits and
    permutation importance, which is exactly what this subtest is.
    """
    for group in (V.ALL, V.NUMERIC, V.TIME_INVARIANT_BASELINES):
        assert V.BLOCKS in group
        assert V.OBJASS in group
    # The composite is derived at load time, not a source column, so it belongs
    # in none of them -- the same treatment as ``hs`` and ``aptinfo_x2``.
    assert V.OBJASS_C not in V.ALL
    assert V.OBJASS_C not in V.NUMERIC


def test_the_composite_is_the_sum_of_the_two_subtests():
    df = pd.DataFrame({V.BLOCKS: [10.0, 0.0, np.nan], V.OBJASS: [5.0, 3.0, 2.0]})

    composite = derive_nonverbal_ability_composite(df)

    assert composite is not None
    assert composite.iloc[0] == 15.0
    assert composite.iloc[1] == 3.0
    # A missing component makes the composite missing rather than a single subtest.
    assert pd.isna(composite.iloc[2])


@pytest.mark.parametrize("absent", [V.BLOCKS, V.OBJASS])
def test_a_dataset_without_both_subtests_is_a_no_op(absent):
    """Other datasets (the historical cohort) must not get a silent fallback."""
    df = pd.DataFrame({V.BLOCKS: [1.0], V.OBJASS: [2.0]}).drop(columns=[absent])

    assert derive_nonverbal_ability_composite(df) is None
    assert V.OBJASS_C not in add_nonverbal_ability_composite(df).columns


def test_the_composite_is_complete_wherever_block_design_is():
    """A composite-adjusted fit must not analyse fewer children than its parent.

    Both subtests are complete for all 54 analysed children, so this holds by
    construction today; the test exists because a future extract that lost one
    subtest for one child would otherwise silently change the analysis set of
    every companion that names the composite.
    """
    long = pd.read_csv(_paths.DATA_DIR / "rli_data_long.csv")

    composite = derive_nonverbal_ability_composite(long)

    assert composite is not None
    assert composite.notna().equals(long[V.BLOCKS].notna())


def test_the_composite_reaches_a_prepared_fit_without_dropping_rows():
    """The whole point is a like-for-like comparison with the Block Design parent."""
    import dataclasses

    from language_reading_predictors.statistical_models.lrp_rli_mech_201 import (
        SPEC as PARENT,
    )
    from language_reading_predictors.statistical_models.mechanism import (
        resolve_mechanism_run_plan,
    )
    from language_reading_predictors.statistical_models.preprocessing import (
        load_and_prepare,
    )

    parent = resolve_mechanism_run_plan(PARENT)
    composite_spec = dataclasses.replace(
        PARENT,
        model_id="lrp-rli-mech-311",
        model_settings=dataclasses.replace(
            PARENT.model_settings, ability_covariate=V.OBJASS_C
        ),
    )
    composite = resolve_mechanism_run_plan(composite_spec)

    assert parent.ability_covariate == V.BLOCKS
    assert composite.ability_covariate == V.OBJASS_C
    with pytest.warns(UserWarning):
        parent_rows = load_and_prepare(**parent.prepare_kwargs()).n_obs
    with pytest.warns(UserWarning):
        composite_rows = load_and_prepare(**composite.prepare_kwargs()).n_obs
    assert composite_rows == parent_rows


def test_neither_subtest_enters_the_boosting_predictor_sets():
    """Adding the schema entries must not silently change 50 fitted GB models."""
    for name in (V.OBJASS, V.OBJASS_C):
        assert name in V.DEFAULT_EXCLUDED
        assert name not in Predictors.DEFAULT_GAIN
        assert name not in Predictors.DEFAULT_LEVEL


def test_the_composite_companions_are_registered_against_their_parents():
    from language_reading_predictors.statistical_models.definitions import (
        MODEL_REGISTRY,
    )
    from language_reading_predictors.statistical_models.mechanism import (
        resolve_mechanism_run_plan,
    )
    from language_reading_predictors.statistical_models.registry import (
        discover_models,
    )

    pairs = {
        "lrp-rli-mech-306": "lrp-rli-mech-196",
        "lrp-rli-mech-307": "lrp-rli-mech-197",
        "lrp-rli-mech-308": "lrp-rli-mech-198",
        "lrp-rli-mech-309": "lrp-rli-mech-199",
        "lrp-rli-mech-310": "lrp-rli-mech-200",
        "lrp-rli-mech-311": "lrp-rli-mech-201",
        "lrp-rli-mech-312": "lrp-rli-mech-258",
    }
    discovered = set(discover_models())

    for model_id, parent_id in pairs.items():
        assert model_id in discovered
        assert model_id in MODEL_REGISTRY
        module = importlib.import_module(
            "language_reading_predictors.statistical_models."
            f"lrp_rli_mech_{model_id[-3:]}"
        )
        parent = importlib.import_module(
            "language_reading_predictors.statistical_models."
            f"lrp_rli_mech_{parent_id[-3:]}"
        )
        plan = resolve_mechanism_run_plan(module.SPEC)
        parent_plan = resolve_mechanism_run_plan(parent.SPEC)

        # One knob, and one only: everything that defines the comparison must match.
        assert plan.ability_covariate == V.OBJASS_C
        assert parent_plan.ability_covariate == V.BLOCKS
        assert module.SPEC.outcome_symbol == parent.SPEC.outcome_symbol
        assert module.SPEC.mechanism_symbol == parent.SPEC.mechanism_symbol
        assert module.SPEC.adjustment == parent.SPEC.adjustment
        assert plan.adjust_for == parent_plan.adjust_for
        assert plan.adjust_baseline_symbol == parent_plan.adjust_baseline_symbol
        assert plan.outcomes == parent_plan.outcomes
        assert plan.linear_mechanism == parent_plan.linear_mechanism
        assert (
            plan.use_subject_random_intercept
            == parent_plan.use_subject_random_intercept
        )


def test_every_composite_companion_has_a_report_template():
    for number in range(306, 313):
        template = REPO / f"docs/models/lrp-rli-mech-{number}/index.qmd"
        assert template.is_file(), f"missing report template: {template}"
        text = template.read_text(encoding="utf-8")
        # The ceiling on interpretation is the point of these fits, so it must be
        # in the template rather than left to a reader's memory.
        assert "objass_c" in text
        assert "visuospatial" in text
        assert "adjusted association" in text
