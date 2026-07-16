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


def _write_synthetic(
    tmp_path, *, drop_one_wave=False, ceiling_violation=False, extension=False
):
    """A Byrne-shaped long CSV: 9 subjects (3 per group) x 3 waves of ``basread``.

    ``extension`` adds a ragged follow-up tail (#338): wave 4 for two of the
    three children in every group, and a wave 5 observed **only** in group 1 -
    the Byrne-shaped Down-syndrome-only extension.
    """
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
            if extension:
                if k < 2:  # one child per group attrits before wave 4
                    rows.append(
                        {
                            "subject_id": sid,
                            "time": 4,
                            "readgrp": grp,
                            "basread": int(rng.integers(0, 80)) + 9,
                        }
                    )
                if grp == 1:  # wave 5 exists only in group 1
                    rows.append(
                        {
                            "subject_id": sid,
                            "time": 5,
                            "readgrp": grp,
                            "basread": int(rng.integers(0, 80)) + 12,
                        }
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
    assert panel.n_trials["basread"] == 90
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


def test_extension_waves_append_kept_subjects_only(tmp_path):
    # #338: extension waves sit outside the complete-case rule - kept subjects
    # contribute wherever observed, dropped subjects contribute nothing.
    path = _write_synthetic(tmp_path, drop_one_wave=True, extension=True)
    panel = load_longitudinal_panel(
        _dataset(path),
        [RLM_MEASURES["basread"]],
        waves=(1, 2, 3),
        extension_waves=(4, 5),
    )
    # S10 is not complete-case on the core, so its wave-4 row must be dropped.
    assert panel.dropped_subjects == 1
    assert "S10" not in panel.subject_ids
    assert panel.waves == (1, 2, 3)
    assert panel.extension_waves == (4, 5)
    assert panel.all_waves == (1, 2, 3, 4, 5)
    ext_rows = panel.long[panel.long["time"].isin([4, 5])]
    assert "S10" not in set(ext_rows["subject_id"])
    # Core: 8 subjects x 3 waves; extension: wave-4 kept observers (5 of 6 -
    # S10 dropped) + the kept group-1 wave-5 rows (2 of 3 - S10 dropped).
    assert panel.n_obs == 8 * 3 + 5 + 2
    # Supported cells: full core grid + wave-4 cells + group-1 wave 5 only.
    cells = panel.cells("basread")
    assert (1, 5) in cells
    assert (2, 5) not in cells and (3, 5) not in cells
    assert {(g, w) for g in (1, 2, 3) for w in (1, 2, 3, 4)}.issubset(set(cells))


def test_duplicate_extension_row_raises(tmp_path):
    # #358 review: the extension tail has no complete-case row-count check, so a
    # duplicated (subject, wave) row would silently reweight that cell - it must
    # be rejected at load.
    path = _write_synthetic(tmp_path, extension=True)
    df = pd.read_csv(path)
    dup = df[(df.subject_id == "S11") & (df.time == 4)]
    pd.concat([df, dup], ignore_index=True).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate extension-wave rows"):
        load_longitudinal_panel(
            _dataset(path),
            [RLM_MEASURES["basread"]],
            waves=(1, 2, 3),
            extension_waves=(4, 5),
        )


def test_extension_wave_overlapping_core_raises(tmp_path):
    path = _write_synthetic(tmp_path, extension=True)
    with pytest.raises(ValueError, match="overlap the complete-case core"):
        load_longitudinal_panel(
            _dataset(path),
            [RLM_MEASURES["basread"]],
            waves=(1, 2, 3),
            extension_waves=(3, 4),
        )


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
    # BAS Word Reading has 90 words (#338 ceilings sign-off, 2026-07-16); the
    # previous 87 was the observed extract maximum mislabelled as confirmed.
    assert measures["basread"].n_trials == 90
    assert measures["basread"].n_trials_confirmed
    assert dataset.group_labels[1] == "Down syndrome"


# Ceilings per the #338 research + data-owner sign-off (2026-07-16). Confirmed
# measures carry the instrument's published maximum; the three still-provisional
# measures keep the observed extract maximum with n_trials_confirmed=False until
# their manuals (1992 BAS Spelling Scale, 1983 Basic Number Skills, 1993 WORD)
# are checked. basmat (wave-3+ only) joined for the #338 window extension
# (lrp-rlm-hg-009, fitted on its own later-wave window).
_RLM_CONFIRMED = {
    "basread": 90,
    "bpvs": 32,
    "trog": 20,
    "basdig": 34,
    "bassim": 21,
    "basmat": 28,
}
_RLM_PROVISIONAL = {
    "basspel": 18,
    "woco": 31,
    "basnum": 60,
}


@pytest.mark.parametrize("symbol, ceiling", sorted(_RLM_CONFIRMED.items()))
def test_rlm_confirmed_ceilings(symbol, ceiling):
    _dataset_spec, measures = resolve_dataset("rlm")
    assert symbol in measures, f"{symbol} not registered in RLM_MEASURES"
    m = measures[symbol]
    assert m.column == symbol
    assert m.n_trials == ceiling
    assert m.n_trials_confirmed is True


@pytest.mark.parametrize("symbol, ceiling", sorted(_RLM_PROVISIONAL.items()))
def test_rlm_provisional_ceilings(symbol, ceiling):
    _dataset_spec, measures = resolve_dataset("rlm")
    assert symbol in measures, f"{symbol} not registered in RLM_MEASURES"
    m = measures[symbol]
    assert m.column == symbol
    assert m.n_trials == ceiling
    # Provisional: the ceiling is the observed maximum, not a confirmed instrument
    # maximum, so this flag must stay False until a data-owner confirms it.
    assert m.n_trials_confirmed is False


def test_resolve_unknown_study_raises():
    with pytest.raises(KeyError, match="Unknown study_id"):
        resolve_dataset("does_not_exist")
