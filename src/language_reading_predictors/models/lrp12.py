# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP12: Predictors of receptive-grammar (TROG-2) level.

``LRP12`` is the baseline exploratory model for receptive-grammar
level (``trog``). The ``trog`` score is the items-correct total
from the Test for Reception of Grammar 2 (TROG-2; Bishop 2003),
covering eight grammatical constructs in blocks of four items
(32 items total; observed max 27 in this sample).

The target is near-Gaussian (``trog`` min 3, max 27, median 14,
mean 14.31, std 4.83, skewness 0.29, n ≈ 215) — cleaner
distribution than most LRP level targets. No floor or ceiling
pathology visible at this sample range.

No feature selection has been run for LRP12 yet — the MAE-tuned
params below (Optuna 150-trial study) are the starting point for
later feature-selection variants.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP12 has not yet been through iterative feature selection.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the full 32-predictor set (DEFAULT_LEVEL minus trog),
# no outlier exclusion (Optuna 150 trials, 10-split GroupKFold,
# seed 47, scoring=mae, lgbm_objective=mae). Tuner-inner CV MAE
# 2.8480 ± 0.4607. n=215.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 81,
    "learning_rate": 0.09962855350007978,
    "num_leaves": 58,
    "max_depth": 9,
    "min_child_samples": 19,
    "subsample": 0.9884209442730713,
    "subsample_freq": 1,
    "colsample_bytree": 0.7988255993346676,
    "reg_alpha": 0.7967408063153419,
    "reg_lambda": 0.31818246166130987,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP12(LevelModel):
    """TROG-2 receptive-grammar level predictors — baseline (all data, MAE-tuned).

    Uses the full :attr:`Predictors.DEFAULT_LEVEL` predictor set
    (minus the target ``trog``) with MAE-tuned hyperparameters and
    no outlier exclusion. Serves as the starting point for
    feature-selection work on the TROG level-prediction task.
    """

    model_id = "lrp12"
    target_var = V.TROG
    description = (
        "LightGBM — TROG-2 (receptive grammar) level predictors "
        "(32 predictors, MAE-tuned, no outlier exclusion)"
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
        "Baseline exploratory model for TROG-2 receptive-grammar "
        "level (trog). Uses the full default level predictor set "
        "(minus the target) without outlier exclusion, and MAE-tuned "
        "params from an Optuna 150-trial study — no feature selection "
        "has been applied yet. Target is near-Gaussian (skew 0.29)."
    )
