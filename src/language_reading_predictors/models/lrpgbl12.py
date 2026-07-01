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


# MAE-tuned (Optuna 150-trial, seed 47) on the earlier pruned selected set;
# retained as the full-set baseline (retune-pending).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 61,
    "learning_rate": 0.05087448729350299,
    "num_leaves": 56,
    "max_depth": 10,
    "min_child_samples": 30,
    "subsample": 0.9231435244978656,
    "subsample_freq": 1,
    "colsample_bytree": 0.6478837495498934,
    "reg_alpha": 0.003266918043241777,
    "reg_lambda": 2.7414726117714094,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL12(LevelModel):
    """Word-reading level predictors — exploratory model (MAE-tuned, all data).

    Full ``Predictors.DEFAULT_LEVEL`` set, MAE-tuned (params retune-pending).
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
        "selection in favour of full-set ranking); hyperparameters are "
        "retained from the earlier pruned-set Optuna tune (retune-pending). "
        "Treat the ranking as exploratory."
    )
