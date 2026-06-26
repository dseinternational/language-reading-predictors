# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Registration and leakage-guard tests for the taught-vocabulary ML models.

LRP23 (gain) and LRP24 (achievement level) target the directly-taught expressive
vocabulary score (``b1extau``). Because the Block 1 expressive *total* ``b1exto``
equals taught + not-taught (``b1extau + b1exnt``), it contains the target and must
never appear in the predictor set — these tests guard that invariant.
"""

from __future__ import annotations

from language_reading_predictors.models.registry import MODELS


def test_taught_vocab_models_registered():
    assert {"lrp23", "lrp24"} <= set(MODELS)


def test_lrp23_gain_target_and_baseline():
    cfg = MODELS["lrp23"]
    assert cfg.target_var == "b1extau_gain"
    # GainModel auto-includes the baseline level as a predictor of the gain.
    assert "b1extau" in cfg.predictor_vars
    assert cfg.target_var not in cfg.predictor_vars


def test_lrp24_level_target_excluded_from_predictors():
    cfg = MODELS["lrp24"]
    assert cfg.target_var == "b1extau"
    assert "b1extau" not in cfg.predictor_vars


def test_taught_vocab_models_exclude_tautological_total():
    # b1exto = b1extau + b1exnt contains the target construct → leakage.
    for mid in ("lrp23", "lrp24"):
        assert "b1exto" not in MODELS[mid].predictor_vars, mid
    # The standardised vocabulary tests are legitimate (correlated, not supersets)
    # predictors and should remain available in the baseline.
    assert "eowpvt" in MODELS["lrp24"].predictor_vars
