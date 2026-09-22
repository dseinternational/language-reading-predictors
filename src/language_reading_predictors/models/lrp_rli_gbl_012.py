# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL12: Predictors of word-reading level.

``LRPGBL12`` is the exploratory model for word-reading level (``ewrswr``) —
Huber-tuned with no outlier exclusion. The target is heavily right-skewed
(min 0, median 6.5, max 64) with a hard floor at 0.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 6.59.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 12.961630499999998,
    "n_estimators": 142,
    "learning_rate": 0.01650130408077383,
    "num_leaves": 57,
    "max_depth": 7,
    "min_child_samples": 7,
    "subsample": 0.6841960666432189,
    "subsample_freq": 1,
    "colsample_bytree": 0.8134755501758186,
    "reg_alpha": 1.1938060916954094,
    "reg_lambda": 0.001078373857024471,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL12(LevelModel):
    """Word-reading level predictors — exploratory model (Huber-tuned, all data).

    Full ``Predictors.DEFAULT_LEVEL`` set, Huber-tuned on the full set (#116).
    """

    model_id = "lrp-rli-gbl-012"
    target_var = V.EWRSWR
    description = (
        "LightGBM — word-reading level predictors "
        "(full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for word-reading level (ewrswr). Fits the full "
        "DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature "
        "selection in favour of full-set ranking); hyperparameters were "
        "re-tuned by Optuna on the full set (150 trials, seed 47; #116 "
        "reporting refresh). Treat the ranking as exploratory."
    )
