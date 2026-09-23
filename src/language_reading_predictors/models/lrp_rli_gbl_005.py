# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL05: Predictors of receptive vocabulary level.

``LRPGBL05`` is the exploratory model for receptive vocabulary level
(``rowpvt``). The target is **essentially symmetric and near-Gaussian**
(``rowpvt`` min 11, max 82, median 42, mean 41.1, std 14.1, skewness
0.04, n ≈ 215) — no floor, no ceiling, no heavy tail; the cleanest
target distribution of any LRP model.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 8.14.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 19.940969999999997,
    "n_estimators": 236,
    "learning_rate": 0.012378008540944419,
    "num_leaves": 38,
    "max_depth": 10,
    "min_child_samples": 4,
    "subsample": 0.6208058053831443,
    "subsample_freq": 1,
    "colsample_bytree": 0.8772283364228272,
    "reg_alpha": 0.0035737136884697257,
    "reg_lambda": 0.001150586346893409,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL05(LevelModel):
    """Receptive vocabulary level predictors — exploratory (Huber-tuned, all data).

    Full ``Predictors.DEFAULT_LEVEL`` set, Huber-tuned on the full set (#169).
    """

    model_id = "lrp-rli-gbl-005"
    target_var = V.ROWPVT
    description = (
        "LightGBM — receptive vocabulary level predictors "
        "(full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for receptive vocabulary level (rowpvt). Fits the "
        "full DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature "
        "selection in favour of full-set ranking); hyperparameters are "
        "re-tuned by Optuna on the full set (150 trials, seed 47; Huber retune of 2026-09-22, superseding #169). "
        "Target is near-Gaussian (skew 0.04). Treat the ranking as "
        "exploratory."
    )
