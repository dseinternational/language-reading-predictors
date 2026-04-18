# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP10: Predictors of receptive-grammar level (CELF).

``LRP10`` is the exploratory model for receptive-grammar level
(``celf``). It is MAE-tuned on the full 32-predictor
:attr:`Predictors.DEFAULT_LEVEL` set (minus the target), with no
outlier exclusion, designed to identify the most important
influences on receptive-grammar level.

The target is **mildly left-skewed** (``celf`` min 0, max 18,
median 11, mean 10.88, std 4.24, skewness −0.37, n ≈ 214). The
max of 18 is the instrument maximum but the 95th percentile is
below it, so there is no strong ceiling pathology (unlike LRP06's
``yarclet`` which piles at 32). Transforms are unlikely to be
required.

The predictor set will be reduced by iterative importance-based
feature selection under the MAE-tuned params (see
``notes/202604181400-lrp10-feature-selection.md``). This is the
initial tuned baseline; no feature-selection steps yet.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP10 has not yet been through iterative feature selection. When
# selection variants are introduced, record their rationale here as
# ``SelectionStep`` entries and chain from ``LRP10`` the same way
# ``lrp02.py`` / ``lrp04.py`` / ``lrp06.py`` / ``lrp08.py`` does.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the full 32-predictor set (DEFAULT_LEVEL minus celf),
# no outlier exclusion (Optuna 150 trials, 10-split GroupKFold,
# seed 47, scoring=mae, lgbm_objective=mae). Tuner-inner CV MAE
# 2.4105 ± 0.4283. n=214.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 40,
    "learning_rate": 0.1982861337355289,
    "num_leaves": 10,
    "max_depth": 4,
    "min_child_samples": 26,
    "subsample": 0.8360938053375331,
    "subsample_freq": 1,
    "colsample_bytree": 0.9188006872642313,
    "reg_alpha": 4.722131822233457,
    "reg_lambda": 0.0038920553782081447,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, untuned) ───────────────────────────────────


class LRP10(LevelModel):
    """CELF receptive-grammar level predictors — exploratory (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_LEVEL` predictor set
    (minus the target ``celf``) with MAE-tuned hyperparameters and
    no outlier exclusion. The starting point for feature selection
    on the CELF level-prediction task.
    """

    model_id = "lrp10"
    target_var = V.CELF
    description = (
        "LightGBM — CELF (receptive grammar) level predictors "
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
        "CELF receptive-grammar level (celf). MAE-tuned on the full "
        "32-predictor DEFAULT_LEVEL set without outlier exclusion so "
        "importance rankings reflect the full range of outcomes. "
        "Target is mildly left-skewed (skew −0.37); the max of 18 "
        "is the instrument maximum but there is no strong ceiling "
        "effect (unlike LRP06's yarclet which piles at 32). "
        "Feature-selection variants to follow. See "
        "notes/202604181400-lrp10-feature-selection.md."
    )
