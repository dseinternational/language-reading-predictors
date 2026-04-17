# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP04: Predictors of expressive-vocabulary level.

``LRP04`` is the exploratory model for expressive-vocabulary level
(``eowpvt``). It is MAE-tuned on the full 32-predictor
:attr:`Predictors.DEFAULT_LEVEL` set (minus the target), with no
outlier exclusion, designed to identify the most important
influences on expressive-vocabulary level.

The target is mildly right-skewed (``eowpvt`` min 8, max 77,
median 33, skewness 0.63, n ≈ 215). No hard floor at 0 (unlike
``ewrswr`` in LRP02), so the motivation for a ``log1p`` transform
is less compelling — but a question for future investigation.

The predictor set will be reduced by iterative importance-based
feature selection under the MAE-tuned params (see
``notes/202604171240-lrp04-feature-selection.md``). This is the
initial tuned baseline; no feature-selection steps yet.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP04 has not yet been through iterative feature selection. When
# selection variants are introduced, record their rationale here as
# ``SelectionStep`` entries and chain from ``LRP04`` the same way
# ``lrp02.py`` does.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the full 32-predictor DEFAULT_LEVEL set, no outlier
# exclusion (Optuna 150 trials, 10-split GroupKFold, seed 47,
# scoring=mae, lgbm_objective=mae). Tuner-inner CV MAE 6.1434 ± 1.0213.
# n=215.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 298,
    "learning_rate": 0.01949555639719653,
    "num_leaves": 17,
    "max_depth": 3,
    "min_child_samples": 4,
    "subsample": 0.6877801522354924,
    "subsample_freq": 1,
    "colsample_bytree": 0.6417173784614608,
    "reg_alpha": 0.0022197152846654793,
    "reg_lambda": 0.0027559915350836594,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRP04(LevelModel):
    """Expressive-vocabulary level predictors — exploratory (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_LEVEL` predictor set
    (minus the target ``eowpvt``) with MAE-tuned hyperparameters and
    no outlier exclusion. The starting point for feature selection
    on the expressive-vocabulary level-prediction task.
    """

    model_id = "lrp04"
    target_var = V.EOWPVT
    description = (
        "LightGBM — expressive-vocabulary level predictors "
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
        "expressive-vocabulary level (eowpvt). MAE-tuned on the full "
        "32-predictor DEFAULT_LEVEL set without outlier exclusion so "
        "importance rankings reflect the full range of outcomes. "
        "Feature-selection variants to follow. See "
        "notes/202604171240-lrp04-feature-selection.md."
    )
