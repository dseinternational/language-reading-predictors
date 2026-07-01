# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG12: Predictors of word-reading gains.

``LRPGBG12`` is the exploratory model for word-reading gains
(``ewrswr_gain``) — MAE-tuned with no outlier exclusion, designed to
identify the most important influences on reading gains across the full
range of outcomes. ``ewrswr_gain`` is moderately right-skewed (−4 to 21,
median 2, skewness 1.33).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# MAE-tuned (Optuna 150-trial, seed 47) on the earlier pruned selected set;
# retained as the full-set baseline (retune-pending).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 52,
    "learning_rate": 0.11822278042172524,
    "num_leaves": 51,
    "max_depth": 7,
    "min_child_samples": 16,
    "subsample": 0.6491876371548958,
    "subsample_freq": 1,
    "colsample_bytree": 0.7993121576506287,
    "reg_alpha": 0.02832146282334302,
    "reg_lambda": 6.243279073188195,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBG12(GainModel):
    """Word-reading gain predictors — exploratory model (MAE-tuned, all data).

    Full ``Predictors.DEFAULT_GAIN`` set, MAE-tuned (params retune-pending).
    """

    model_id = "lrpgbg12"
    target_var = V.EWRSWR_GAIN
    description = (
        "LightGBM — word-reading gain predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    include = [V.EWRSWR]
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 53
    shap_scatter_specs = [
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
        ShapScatterSpec(
            color_by=V.EWRSWR,
            description="All predictors, coloured by baseline word-reading (ewrswr)",
        ),
    ]
    notes = (
        "Exploratory model for word-reading gains (ewrswr_gain). Fits the "
        "full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature "
        "selection in favour of full-set ranking); hyperparameters are "
        "retained from the earlier pruned-set Optuna tune (retune-pending). "
        "Gain models are near-noise (baseline-driven, regression to the "
        "mean) — treat the ranking as exploratory."
    )
