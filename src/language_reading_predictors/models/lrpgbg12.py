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


# MAE-tuned (Optuna 150-trial, seed 47, GroupKFold cv=53) on the full
# ``Predictors.DEFAULT_GAIN`` set; best mean cross-validated MAE 2.94 (#116 reporting
# refresh, superseding the earlier pruned-subset tune).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 580,
    "learning_rate": 0.014771985445699121,
    "num_leaves": 58,
    "max_depth": 5,
    "min_child_samples": 35,
    "subsample": 0.9555461290933629,
    "subsample_freq": 1,
    "colsample_bytree": 0.899629213906584,
    "reg_alpha": 0.0026334274053896006,
    "reg_lambda": 0.002036072137956731,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBG12(GainModel):
    """Word-reading gain predictors — exploratory model (MAE-tuned, all data).

    Full ``Predictors.DEFAULT_GAIN`` set, MAE-tuned on the full set (#116).
    """

    model_id = "lrpgbg12"
    target_var = V.EWRSWR_GAIN
    description = (
        "LightGBM — word-reading gain predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    include = (V.EWRSWR,)
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 53
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
        ShapScatterSpec(
            color_by=V.EWRSWR,
            description="All predictors, coloured by baseline word-reading (ewrswr)",
        ),
    )
    notes = (
        "Exploratory model for word-reading gains (ewrswr_gain). Fits the "
        "full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature "
        "selection in favour of full-set ranking); hyperparameters were "
        "re-tuned by Optuna on the full set (150 trials, seed 47; #116 "
        "reporting refresh). Gain models are near-noise (baseline-driven, "
        "regression to the mean) — treat the ranking as exploratory."
    )
