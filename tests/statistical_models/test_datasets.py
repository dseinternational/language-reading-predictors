# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the multi-dataset layer (#165): DatasetSpec + LongitudinalPanel loader.

Uses a synthetic Byrne-shaped CSV in ``tmp_path`` - the real participant data
(CC-BY-4.0) is never read in unit tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from language_reading_predictors.statistical_models.datasets import (
    RLM_DATASET,
    RLM_MEASURES,
    DatasetSpec,
    StudyMeasure,
    resolve_dataset,
)
from language_reading_predictors.statistical_models.preprocessing import (
    LongitudinalPanel,
    load_longitudinal_panel,
)

_GROUP_LABELS = {1: "Down syndrome", 2: "Average readers", 3: "Reading-matched"}


def _write_synthetic(tmp_path, *, drop_one_wave=False, ceiling_violation=False):
    """A Byrne-shaped long CSV: 9 subjects (3 per group) x 3 waves of ``basread``."""
    rng = np.random.default_rng(3)
    rows = []
    for grp in (1, 2, 3):
        for k in range(3):
            sid = f"S{grp}{k}"
            for t in (1, 2, 3):
                val = int(rng.integers(0, 80)) + 3 * (t - 1)
                rows.append(
                    {"subject_id": sid, "time": t, "readgrp": grp, "basread": val}
                )
    df = pd.DataFrame(rows)
    if drop_one_wave:
        # Subject S10 loses its wave-2 basread -> not complete-case.
        df.loc[(df.subject_id == "S10") & (df.time == 2), "basread"] = np.nan
    if ceiling_violation:
        df.loc[df.index[0], "basread"] = 999
    path = tmp_path / "rlm_synth_long.csv"
    df.to_csv(path, index=False)
    return path


def _dataset(path: Path) -> DatasetSpec:
    return DatasetSpec(
        study_id="rlm_test",
        label="synthetic",
        path=path,
        group_labels=_GROUP_LABELS,
    )


def test_load_longitudinal_panel_shapes(tmp_path):
    path = _write_synthetic(tmp_path)
    panel = load_longitudinal_panel(
        _dataset(path), [RLM_MEASURES["basread"]], waves=(1, 2, 3)
    )
    assert isinstance(panel, LongitudinalPanel)
    assert panel.n_subjects == 9
    assert panel.n_waves == 3
    assert panel.n_obs == 27  # 9 subjects x 3 waves
    assert panel.n_phases == 2
    assert panel.dropped_subjects == 0
    assert panel.n_trials["basread"] == 87
    assert panel.group_codes == [1, 2, 3]
    assert panel.group_labels == list(_GROUP_LABELS.values())
    assert panel.counts["basread"].shape == (9, 3)
    assert panel.obs_mask["basread"].all()  # complete-case -> all observed
    # WavePanel-compatible container accessors (drop-in for the pipeline header).
    assert panel.n_children == panel.n_subjects


def test_complete_case_drops_incomplete_subject(tmp_path):
    path = _write_synthetic(tmp_path, drop_one_wave=True)
    panel = load_longitudinal_panel(
        _dataset(path), [RLM_MEASURES["basread"]], waves=(1, 2, 3)
    )
    assert panel.dropped_subjects == 1
    assert panel.n_subjects == 8
    assert "S10" not in panel.subject_ids


def test_dropped_rows_is_observation_rows_not_subjects(tmp_path):
    # dropped_rows is an observation-row count (subjects x waves), not a subject
    # count, so config.json / the pipeline header stay in row units (#171 review).
    path = _write_synthetic(tmp_path, drop_one_wave=True)
    panel = load_longitudinal_panel(
        _dataset(path), [RLM_MEASURES["basread"]], waves=(1, 2, 3)
    )
    assert panel.dropped_subjects == 1
    assert panel.dropped_rows == panel.dropped_subjects * panel.n_waves == 3


def test_available_case_keeps_incomplete_subject(tmp_path):
    path = _write_synthetic(tmp_path, drop_one_wave=True)
    panel = load_longitudinal_panel(
        _dataset(path),
        [RLM_MEASURES["basread"]],
        waves=(1, 2, 3),
        complete_case=False,
    )
    assert panel.dropped_subjects == 0
    assert panel.n_subjects == 9
    # The dropped wave shows up as a masked cell rather than a removed subject.
    assert not panel.obs_mask["basread"].all()


def test_ceiling_guard_raises(tmp_path):
    path = _write_synthetic(tmp_path, ceiling_violation=True)
    with pytest.raises(ValueError, match="exceeds measure ceiling"):
        load_longitudinal_panel(
            _dataset(path), [RLM_MEASURES["basread"]], waves=(1, 2, 3)
        )


def test_missing_column_raises(tmp_path):
    path = _write_synthetic(tmp_path)
    bad = StudyMeasure("nope", "not_a_column", 10, "Missing")
    with pytest.raises(ValueError, match="missing required columns"):
        load_longitudinal_panel(_dataset(path), [bad], waves=(1, 2, 3))


def test_empty_measures_raises(tmp_path):
    # An empty measures list would otherwise fail obscurely in complete-case
    # selection (keep stays None); reject it up front (#171 review).
    path = _write_synthetic(tmp_path)
    with pytest.raises(ValueError, match="at least one measure"):
        load_longitudinal_panel(_dataset(path), [], waves=(1, 2, 3))


def test_unknown_group_code_raises(tmp_path):
    # A group code with no label must fail clearly at load, not with a downstream
    # KeyError when building group_labels (#171 review).
    path = _write_synthetic(tmp_path)
    dataset = DatasetSpec(
        study_id="rlm_test",
        label="synthetic",
        path=path,
        group_labels={1: "Down syndrome", 2: "Average readers"},  # missing code 3
    )
    with pytest.raises(ValueError, match="no label in"):
        load_longitudinal_panel(dataset, [RLM_MEASURES["basread"]], waves=(1, 2, 3))


def test_rlm_dataset_registered():
    dataset, measures = resolve_dataset("rlm")
    assert dataset is RLM_DATASET
    assert dataset.study_id == "rlm"
    assert measures["basread"].n_trials == 87
    assert measures["basread"].n_trials_confirmed
    assert dataset.group_labels[1] == "Down syndrome"


def test_resolve_unknown_study_raises():
    with pytest.raises(KeyError, match="Unknown study_id"):
        resolve_dataset("does_not_exist")
