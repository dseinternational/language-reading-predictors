# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Registration and leakage-guard tests for the taught-vocabulary ML models.

LRP-RLI-GBG-002 (gain) and LRP-RLI-GBL-002 (achievement level) target the
directly-taught expressive vocabulary score (``b1extau``). Because the Block 1
expressive *total* ``b1exto`` equals taught + not-taught (``b1extau + b1exnt``), it
contains the target and must never appear in the predictor set — these tests guard
that invariant.

The GB registry ``MODELS`` is keyed on the canonical CLI id since #168 Phase 2
(``lrp-rli-gbg-002``); legacy ids (``lrpgbg02``) are no longer registry keys.
"""

from __future__ import annotations

import pandas as pd
import pytest

from language_reading_predictors.models.registry import MODELS


def test_taught_vocab_models_registered():
    assert {"lrp-rli-gbg-002", "lrp-rli-gbl-002"} <= set(MODELS)


def test_lrpgbg02_gain_target_and_baseline():
    cfg = MODELS["lrp-rli-gbg-002"]
    assert cfg.target_var == "b1extau_gain"
    # GainModel auto-includes the baseline level as a predictor of the gain.
    assert "b1extau" in cfg.predictor_vars
    assert cfg.target_var not in cfg.predictor_vars


def test_lrpgbl02_level_target_excluded_from_predictors():
    cfg = MODELS["lrp-rli-gbl-002"]
    assert cfg.target_var == "b1extau"
    assert "b1extau" not in cfg.predictor_vars


def test_taught_vocab_models_exclude_tautological_total():
    # b1exto = b1extau + b1exnt contains the target construct → leakage.
    for mid in ("lrp-rli-gbg-002", "lrp-rli-gbl-002"):
        assert "b1exto" not in MODELS[mid].predictor_vars, mid
    # The standardised vocabulary tests are legitimate (correlated, not supersets)
    # predictors and should remain available in the baseline.
    assert "eowpvt" in MODELS["lrp-rli-gbl-002"].predictor_vars


# --- #116 Phase B: the four new block-1 vocab / phonetic-spelling outcomes ---

# canonical number (3-digit) -> (gain target, level target, the tautological total
# to exclude or None). MODELS keys are canonical CLI ids since #168 Phase 2.
_PHASE_B = {
    "001": ("b1retau_gain", "b1retau", "b1reto"),   # taught receptive vocab
    "003": ("b1rent_gain", "b1rent", "b1reto"),     # not-taught receptive vocab
    "004": ("b1exnt_gain", "b1exnt", "b1exto"),     # not-taught expressive vocab
    "011": ("spphon_gain", "spphon", None),         # phonetic spelling (no total)
}


def test_phase_b_outcomes_registered_with_targets():
    for nnn, (gain_t, level_t, _total) in _PHASE_B.items():
        g, lvl = MODELS[f"lrp-rli-gbg-{nnn}"], MODELS[f"lrp-rli-gbl-{nnn}"]
        assert g.target_var == gain_t and lvl.target_var == level_t
        # gain auto-includes its baseline; level excludes its own target.
        assert level_t in g.predictor_vars and gain_t not in g.predictor_vars
        assert level_t not in lvl.predictor_vars


def test_phase_b_models_exclude_tautological_totals():
    for nnn, (_g, _l, total) in _PHASE_B.items():
        for mid in (f"lrp-rli-gbg-{nnn}", f"lrp-rli-gbl-{nnn}"):
            if total is not None:
                assert total not in MODELS[mid].predictor_vars, mid


# --- log1p pipeline domain guard (defensive; no registered model uses it) ---


def _log_pipeline_with_y(y_values):
    """Build an LGBMLogPipeline with a minimal context whose target is ``y_values``."""
    from language_reading_predictors.models.common import ModelConfig, RunConfig
    from language_reading_predictors.models.lgbm_log_pipeline import LGBMLogPipeline

    cfg = ModelConfig(
        model_id="tmp_log",
        description="tmp",
        target_var="y",
        predictor_vars=["x"],
        model_params={"n_estimators": 5},
    )
    pipe = LGBMLogPipeline(cfg, RunConfig.from_name("dev"))
    pipe.context.y = pd.Series(y_values, dtype="float64")
    return pipe


def test_log_pipeline_rejects_target_at_or_below_minus_one():
    pipe = _log_pipeline_with_y([0.0, 5.0, -1.0])  # -1 is out of log1p's domain
    with pytest.raises(ValueError, match="LGBMLogPipeline"):
        pipe.configure_model()


def test_log_pipeline_accepts_valid_target():
    # y > -1 everywhere: configure_model must succeed and build the pipeline.
    pipe = _log_pipeline_with_y([0.0, 2.5, -0.5, 10.0])
    pipe.configure_model()
    assert pipe.context.pipeline is not None
