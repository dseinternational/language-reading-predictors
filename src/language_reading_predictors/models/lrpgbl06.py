# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL06: Predictors of expressive-vocabulary level.

``LRPGBL06`` is the exploratory model for expressive-vocabulary level
(``eowpvt``). The target is mildly right-skewed (``eowpvt`` min 8, max
77, median 33, skewness 0.63, n ≈ 215) with no hard floor at 0.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# MAE-tuned (Optuna 150-trial, seed 47) on the earlier pruned selected set;
# retained as the full-set baseline (retune-pending).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 68,
    "learning_rate": 0.08572327409762025,
    "num_leaves": 18,
    "max_depth": 3,
    "min_child_samples": 4,
    "subsample": 0.871675111170649,
    "subsample_freq": 1,
    "colsample_bytree": 0.9388695366733577,
    "reg_alpha": 0.001037151277752068,
    "reg_lambda": 0.019926160950819845,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL06(LevelModel):
    """Expressive-vocabulary level predictors — exploratory (MAE-tuned, all data).

    Full ``Predictors.DEFAULT_LEVEL`` set, MAE-tuned (params retune-pending).
    """

    model_id = "lrpgbl06"
    target_var = V.EOWPVT
    description = (
        "LightGBM — expressive-vocabulary level predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for expressive-vocabulary level (eowpvt). Fits the "
        "full DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature "
        "selection in favour of full-set ranking); hyperparameters are "
        "retained from the earlier pruned-set Optuna tune (retune-pending). "
        "Treat the ranking as exploratory."
    )
