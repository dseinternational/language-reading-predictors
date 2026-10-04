# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Saved results must meet the corrected statistical contracts."""

from copy import deepcopy
import json

import pandas as pd
import pytest

from language_reading_predictors.statistical_models.release import evaluate_publication
from language_reading_predictors.statistical_models.release.family_checks import (
    _mediation_link_sensitivity_release_failures,
    _pooled_levels_release_failures,
)
from .test_release_decision import _fit_dir


def _pooled_config():
    return {
        "kind": "pooled_levels",
        "n_obs": 210,
        "resolved_run_plan": {"bounded_predictor_transform": "haldane_logit", "mechanism_is_covariate": False},
        "extra": {"n_child_wave_rows": 210, "exposure_transform": "haldane_logit", "exposure_scale": [0.0, 1.0]},
        "reuse_contract": {"n_obs": 210, "fitted_subject_identity": {"n_rows": 210}},
    }


def test_corrected_pooled_record_passes_and_clipped_or_misaligned_records_fail():
    current = _pooled_config()
    assert not _pooled_levels_release_failures(current)
    legacy = deepcopy(current)
    del legacy["resolved_run_plan"]["bounded_predictor_transform"]
    assert _pooled_levels_release_failures(legacy)
    wrong_identity = deepcopy(current)
    wrong_identity["reuse_contract"]["fitted_subject_identity"]["n_rows"] = 214
    assert _pooled_levels_release_failures(wrong_identity)


def test_publication_withholds_a_pooled_fit_with_the_old_transform(tmp_path):
    directory = _fit_dir(tmp_path, kind="pooled_levels")
    config = json.loads((directory / "config.json").read_text())
    config.update(_pooled_config())
    del config["resolved_run_plan"]["bounded_predictor_transform"]
    (directory / "config.json").write_text(json.dumps(config))
    decision = evaluate_publication(directory)
    assert decision.stage == "computation"
    assert any("predictor-transform declaration is stale" in item for item in decision.failing_checks)


@pytest.mark.parametrize("multiplier,passes", [(1.0, True), (1.5, False)])
def test_floor_sensitivity_must_match_the_primary_zero_bias_effect(tmp_path, multiplier, passes):
    config = {
        "kind": "mediation",
        "resolved_run_plan": {"score_mean_link": "three_choice_guessing_floor", "estimand": "natural"},
    }
    pd.DataFrame(
        [{"quantity": "NIE", "prob_median": 0.026, "prob_lo": -0.011, "prob_hi": 0.065, "prob_pos": 0.879}]
    ).to_csv(tmp_path / "mediation_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "delta": 0,
                "nie_median": multiplier * 0.026,
                "nie_lo": multiplier * (-0.011),
                "nie_hi": multiplier * 0.065,
                "nie_prob_pos": 0.879,
            }
        ]
    ).to_csv(tmp_path / "mediation_sensitivity.csv", index=False)
    failures, missing = _mediation_link_sensitivity_release_failures(tmp_path, config)
    assert not missing
    assert (not failures) is passes


def test_missing_floor_sensitivity_cannot_pass(tmp_path):
    config = {"kind": "mediation", "resolved_run_plan": {"score_mean_link": "three_choice_guessing_floor"}}
    assert _mediation_link_sensitivity_release_failures(tmp_path, config)[1]
