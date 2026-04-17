# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP08: Predictors of receptive vocabulary level.

``LRP08`` is the exploratory model for receptive vocabulary level
(``rowpvt``). It is MAE-tuned on the full 32-predictor
:attr:`Predictors.DEFAULT_LEVEL` set (minus the target), with no
outlier exclusion, designed to identify the most important
influences on receptive vocabulary level.

The target is **essentially symmetric and near-Gaussian** (``rowpvt``
min 11, max 82, median 42, mean 41.1, std 14.1, skewness 0.04,
n ≈ 215). No floor, no ceiling, no heavy tail — the cleanest
target distribution of any LRP model to date. Transforms are
unnecessary; standard MAE and RMSE objectives should behave well.

The predictor set will be reduced by iterative importance-based
feature selection under the MAE-tuned params (see
``notes/202604171715-lrp08-feature-selection.md``). This is the
initial tuned baseline; no feature-selection steps yet.
"""

from language_reading_predictors.data_variables import Variables as V  # noqa: F401
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP08 has not yet been through iterative feature selection. When
# selection variants are introduced, record their rationale here as
# ``SelectionStep`` entries and chain from ``LRP08`` the same way
# ``lrp02.py`` / ``lrp04.py`` / ``lrp06.py`` does.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the full 32-predictor set (DEFAULT_LEVEL minus rowpvt),
# no outlier exclusion (Optuna 150 trials, 10-split GroupKFold,
# seed 47, scoring=mae, lgbm_objective=mae). Tuner-inner CV MAE
# 7.0639 ± 1.6708. n=215.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 118,
    "learning_rate": 0.05326460861031385,
    "num_leaves": 20,
    "max_depth": 9,
    "min_child_samples": 18,
    "subsample": 0.7266073045486356,
    "subsample_freq": 1,
    "colsample_bytree": 0.9618870696184477,
    "reg_alpha": 0.010880373065807128,
    "reg_lambda": 0.00393504737305517,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRP08(LevelModel):
    """Receptive vocabulary level predictors — exploratory (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_LEVEL` predictor set
    (minus the target ``rowpvt``) with MAE-tuned hyperparameters
    and no outlier exclusion. The starting point for feature
    selection on the receptive vocabulary level-prediction task.
    """

    model_id = "lrp08"
    target_var = V.ROWPVT
    description = (
        "LightGBM — receptive vocabulary level predictors "
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
        "receptive vocabulary level (rowpvt). MAE-tuned on the "
        "full 32-predictor DEFAULT_LEVEL set without outlier exclusion "
        "so importance rankings reflect the full range of outcomes. "
        "Target is essentially symmetric / near-Gaussian (skew 0.04, "
        "no floor or ceiling) — cleanest target distribution of any "
        "LRP model to date. Feature-selection variants to follow. "
        "See notes/202604171715-lrp08-feature-selection.md."
    )
