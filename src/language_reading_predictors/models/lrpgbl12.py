# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL12: Predictors of word-reading level.

``LRPGBL12`` is the exploratory model for word-reading level (``ewrswr``) —
MAE-tuned with no outlier exclusion. The target is heavily right-skewed
(min 0, median 6.5, max 64) with a hard floor at 0.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# MAE-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51) on the full
# ``Predictors.DEFAULT_LEVEL`` set; best out-of-fold MAE 5.89 (#116 reporting
# refresh, superseding the earlier pruned-subset tune).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 12,
    "learning_rate": 0.1742660450386943,
    "num_leaves": 21,
    "max_depth": 11,
    "min_child_samples": 4,
    "subsample": 0.7247819669156791,
    "subsample_freq": 1,
    "colsample_bytree": 0.6022781859914981,
    "reg_alpha": 0.3169609291119701,
    "reg_lambda": 0.0010032986789946063,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL12(LevelModel):
    """Word-reading level predictors — exploratory model (MAE-tuned, all data).

    Full ``Predictors.DEFAULT_LEVEL`` set, MAE-tuned on the full set (#116).
    """

    model_id = "lrpgbl12"
    target_var = V.EWRSWR
    description = (
        "LightGBM — word-reading level predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for word-reading level (ewrswr). Fits the full "
        "DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature "
        "selection in favour of full-set ranking); hyperparameters were "
        "re-tuned by Optuna on the full set (150 trials, seed 47; #116 "
        "reporting refresh). Treat the ranking as exploratory."
    )
