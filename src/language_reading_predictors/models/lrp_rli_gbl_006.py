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


# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 6.83.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 17.946872999999997,
    "n_estimators": 165,
    "learning_rate": 0.023608756184565618,
    "num_leaves": 34,
    "max_depth": 3,
    "min_child_samples": 5,
    "subsample": 0.64923004859638,
    "subsample_freq": 1,
    "colsample_bytree": 0.6143607715916712,
    "reg_alpha": 0.003145954084038142,
    "reg_lambda": 0.01625220124143413,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL06(LevelModel):
    """Expressive-vocabulary level predictors — exploratory (Huber-tuned, all data).

    Full ``Predictors.DEFAULT_LEVEL`` set, Huber-tuned on the full set (#169).
    """

    model_id = "lrp-rli-gbl-006"
    target_var = V.EOWPVT
    description = (
        "LightGBM — expressive-vocabulary level predictors "
        "(full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for expressive-vocabulary level (eowpvt). Fits the "
        "full DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature "
        "selection in favour of full-set ranking); hyperparameters are "
        "re-tuned by Optuna on the full set (150 trials, seed 47; Huber retune of 2026-09-22, superseding #169). "
        "Treat the ranking as exploratory."
    )
