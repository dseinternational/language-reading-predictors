# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP06: Predictors of letter-sound knowledge level.

``LRP06`` is the exploratory model for letter-sound knowledge level
(``yarclet``). It is MAE-tuned on the full 32-predictor
:attr:`Predictors.DEFAULT_LEVEL` set (minus the target), with no
outlier exclusion, designed to identify the most important
influences on letter-sound knowledge level.

The target is **left-skewed with a ceiling at 32** (``yarclet`` min
0, max 32, median 21, skewness −0.60, n ≈ 214). The ceiling effect
(95th percentile = 31, 99th = 32) means many children score at or
near the instrument maximum — a different pathology from the
right-skewed / floor-at-0 targets of LRP02 and LRP04. Log / log1p
transforms are inappropriate here because the skew is in the wrong
direction; a reflection-log or quantile objective might be
considered later.

The predictor set will be reduced by iterative importance-based
feature selection under the MAE-tuned params (see
``notes/202604171421-lrp06-feature-selection.md``). This is the
initial tuned baseline; no feature-selection steps yet.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP06 has not yet been through iterative feature selection. When
# selection variants are introduced, record their rationale here as
# ``SelectionStep`` entries and chain from ``LRP06`` the same way
# ``lrp02.py`` / ``lrp04.py`` does.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the full 32-predictor DEFAULT_LEVEL set, no outlier
# exclusion (Optuna 150 trials, 10-split GroupKFold, seed 47,
# scoring=mae, lgbm_objective=mae). Tuner-inner CV MAE 4.2827 ±
# 0.9503. n=214.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 55,
    "learning_rate": 0.056216642069165164,
    "num_leaves": 20,
    "max_depth": 10,
    "min_child_samples": 25,
    "subsample": 0.7471713662504078,
    "subsample_freq": 1,
    "colsample_bytree": 0.9126203830943295,
    "reg_alpha": 0.003945304557177304,
    "reg_lambda": 0.12518330653891066,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRP06(LevelModel):
    """Letter-sound knowledge level predictors — exploratory (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_LEVEL` predictor set
    (minus the target ``yarclet``) with MAE-tuned hyperparameters
    and no outlier exclusion. The starting point for feature
    selection on the letter-sound knowledge level-prediction task.
    """

    model_id = "lrp06"
    target_var = V.YARCLET
    description = (
        "LightGBM — letter-sound knowledge level predictors "
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
        "Exploratory model for identifying important predictors of "
        "letter-sound knowledge level (yarclet). MAE-tuned on the "
        "full 32-predictor DEFAULT_LEVEL set without outlier exclusion "
        "so importance rankings reflect the full range of outcomes. "
        "Note the target has a ceiling effect at 32 — many "
        "observations cluster at the instrument maximum, producing a "
        "left-skewed distribution. Feature-selection variants to "
        "follow. See notes/202604171421-lrp06-feature-selection.md."
    )
