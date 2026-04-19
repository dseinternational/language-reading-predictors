# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP15: Predictors of phoneme-blending gains.

``LRP15`` is the baseline exploratory model for phoneme-blending
gains (``blending_gain``). ``blending`` is a phoneme-blending
(phonological awareness) score on a 0–10 scale: the child selects
which of three pictures depicts the word formed by segmented
phonemes spoken by the examiner.

The target is **mildly right-skewed** (``blending_gain`` min −5,
max 7, median 0, mean 0.48, std 2.15, skewness 0.51, with ~35%
negative and ~20% zero observations, n ≈ 161). Similar in shape
to LRP07 (``rowpvt_gain``, skew 0.04) but with heavier zero
pile-up.

No feature selection has been run for LRP15 yet — the MAE-tuned
params below are the starting point for later feature-selection
variants.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP15 has not yet been through iterative feature selection.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the full 34-predictor set (DEFAULT_GAIN, which already
# includes blending as a level predictor), no outlier exclusion (Optuna
# 150 trials, 10-split GroupKFold, seed 47, scoring=mae,
# lgbm_objective=mae). Tuner-inner CV MAE 1.5156 ± 0.3385. n=161.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 183,
    "learning_rate": 0.030870701532861023,
    "num_leaves": 54,
    "max_depth": 11,
    "min_child_samples": 18,
    "subsample": 0.8788051456884195,
    "subsample_freq": 1,
    "colsample_bytree": 0.9820473187671805,
    "reg_alpha": 0.023891411068064673,
    "reg_lambda": 0.001894563310128035,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP15(GainModel):
    """Phoneme-blending gain predictors — baseline (all data, MAE-tuned).

    Uses the full :attr:`Predictors.DEFAULT_GAIN` predictor set
    (``blending`` is already a member, so the GainModel auto-include
    is a no-op) with MAE-tuned hyperparameters and no outlier
    exclusion. Serves as the starting point for feature-selection
    work on the blending gain-prediction task.
    """

    model_id = "lrp15"
    target_var = V.BLENDING_GAIN
    description = (
        "LightGBM — phoneme-blending gain predictors "
        "(34 predictors, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    selection_steps = _SELECTION_STEPS
    shap_scatter_specs = [
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    ]
    notes = (
        "Baseline exploratory model for phoneme-blending gains "
        "(blending_gain). Uses the full default gain predictor set "
        "(blending is already a level predictor in that set, so the "
        "GainModel auto-include is a no-op) without outlier "
        "exclusion, and MAE-tuned params from an Optuna 150-trial "
        "study — no feature selection has been applied yet. Target "
        "is mildly right-skewed (skew 0.51) with ~20% zero and ~35% "
        "negative gains — comparable in shape to LRP07 but with "
        "heavier zero pile-up."
    )
