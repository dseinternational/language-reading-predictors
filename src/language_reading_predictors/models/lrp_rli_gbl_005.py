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


# MAE-tuned (Optuna 150-trial, seed 47) on the earlier pruned selected set;
# retained as the full-set baseline (retune-pending).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 235,
    "learning_rate": 0.012246385977646894,
    "num_leaves": 58,
    "max_depth": 3,
    "min_child_samples": 6,
    "subsample": 0.6471382352609089,
    "subsample_freq": 1,
    "colsample_bytree": 0.6827041658949718,
    "reg_alpha": 0.021220537844795304,
    "reg_lambda": 0.21769887246316527,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL05(LevelModel):
    """Receptive vocabulary level predictors — exploratory (MAE-tuned, all data).

    Full ``Predictors.DEFAULT_LEVEL`` set, MAE-tuned (params retune-pending).
    """

    model_id = "lrp-rli-gbl-005"
    target_var = V.ROWPVT
    description = (
        "LightGBM — receptive vocabulary level predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for receptive vocabulary level (rowpvt). Fits the "
        "full DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature "
        "selection in favour of full-set ranking); hyperparameters are "
        "retained from the earlier pruned-set Optuna tune (retune-pending). "
        "Target is near-Gaussian (skew 0.04). Treat the ranking as "
        "exploratory."
    )
