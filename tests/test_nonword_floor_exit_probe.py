# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Contract tests for the pre-registered #433 Bernoulli promotion probe."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "notes/assets/202608101700-nonword-floor-exit-probe.py"
)
_SPEC = importlib.util.spec_from_file_location("nonword_floor_exit_probe", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
PROBE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = PROBE
_SPEC.loader.exec_module(PROBE)


def test_locked_row_policy_reproduces_planning_counts_and_missingness():
    frame = PROBE.transition_frame()
    design = PROBE.reference_design(frame)
    primary = PROBE.prepare_probe(
        frame, design, label="all_words", max_words=None
    )
    tail = PROBE.prepare_probe(
        frame, design, label="words_le_25", max_words=25
    )

    assert (len(primary.y), len(primary.subject_labels), int(primary.y.sum())) == (
        95,
        48,
        36,
    )
    assert (len(tail.y), len(tail.subject_labels), int(tail.y.sum())) == (
        92,
        47,
        33,
    )
    assert int(primary.frame["hearing_c"].isna().sum()) == 16
    assert int(primary.frame["deapp_c"].isna().sum()) == 3
    assert primary.X.shape == (95, len(PROBE.COVARIATE_NAMES))
    assert np.isfinite(primary.X).all()


def test_tail_sensitivity_reuses_full_population_exposure_scaling():
    frame = PROBE.transition_frame()
    design = PROBE.reference_design(frame)
    tail = PROBE.prepare_probe(
        frame, design, label="words_le_25", max_words=25
    )
    expected = (
        np.log1p(tail.frame["wr_pre"].to_numpy(float)) - design.log_wr_mean
    ) / design.log_wr_sd
    assert np.allclose(tail.wr_z, expected)
    assert tail.row_sha256 == PROBE.prepare_probe(
        frame, design, label="again", max_words=25
    ).row_sha256


def test_full_and_null_models_differ_only_by_word_reading_coefficient():
    frame = PROBE.transition_frame()
    design = PROBE.reference_design(frame)
    prepared = PROBE.prepare_probe(
        frame, design, label="all_words", max_words=None
    )
    null = PROBE.build_model(
        prepared, include_word_reading=False, slope_prior_sd=0.3
    )
    full = PROBE.build_model(
        prepared, include_word_reading=True, slope_prior_sd=0.3
    )
    null_free = {rv.name for rv in null.free_RVs}
    full_free = {rv.name for rv in full.free_RVs}
    assert full_free - null_free == {"b_wr"}
    assert null_free - full_free == set()
    assert {rv.name for rv in full.observed_RVs} == {"y_exit"}
    assert "nw_pre" not in full_free


def _passing_tables():
    diagnostics = pd.DataFrame({"gate_pass": [True] * 8})
    comparisons = pd.DataFrame(
        {
            "comparison_valid": [True] * 4,
            "elpd_difference_full_minus_null": [5.0, 4.5, 5.5, 4.2],
        }
    )
    risk = pd.DataFrame(
        {
            "population": [
                "all_words",
                "all_words",
                "words_le_25",
                "words_le_25",
            ],
            "slope_prior_sd": [0.3, 1.0, 0.3, 1.0],
            "reference_high_words": [5.0] * 4,
            "risk_difference_median": [0.20, 0.23, 0.18, 0.21],
            "risk_difference_prob_positive": [0.99, 0.98, 0.97, 0.99],
        }
    )
    return diagnostics, comparisons, risk


def test_promotion_rule_is_conjunctive_and_treats_sub_four_elpd_as_inconclusive():
    diagnostics, comparisons, risk = _passing_tables()
    passed = PROBE.promotion_decision(diagnostics, comparisons, risk)
    assert passed["status"] == "promote"

    comparisons.loc[0, "elpd_difference_full_minus_null"] = 3.99
    failed = PROBE.promotion_decision(diagnostics, comparisons, risk)
    assert failed["status"] == "do_not_promote"
    assert (
        failed["checks"][
            "all_four_full_models_discriminating_by_at_least_4_elpd"
        ]
        is False
    )
