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


# MAE-tuned by Optuna on the full predictor set (150 trials, seed 47;
# #169 retune, superseding the earlier pruned-set tune).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 103,
    "learning_rate": 0.09552591224037643,
    "num_leaves": 20,
    "max_depth": 9,
    "min_child_samples": 10,
    "subsample": 0.9654535806625875,
    "subsample_freq": 1,
    "colsample_bytree": 0.6456936545386559,
    "reg_alpha": 0.0032086780420558176,
    "reg_lambda": 0.0024703568024118366,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL06(LevelModel):
    """Expressive-vocabulary level predictors — exploratory (MAE-tuned, all data).

    Full ``Predictors.DEFAULT_LEVEL`` set, MAE-tuned on the full set (#169).
    """

    model_id = "lrp-rli-gbl-006"
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
        "re-tuned by Optuna on the full set (150 trials, seed 47; #169). "
        "Treat the ranking as exploratory."
    )
