# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL15: Predictors of receptive-grammar (TROG-2) level.

``LRPGBL15`` is the exploratory model for receptive-grammar level (``trog``).
The ``trog`` score is the items-correct total from the Test for Reception
of Grammar 2 (TROG-2; Bishop 2003), covering eight grammatical constructs.
The target is near-Gaussian (min 3, max 27, median 14, mean 14.31, std
4.83, skewness 0.29, n ≈ 215) — cleaner than most LRP level targets.

Fits the full ``Predictors.DEFAULT_LEVEL`` set; hyperparameters are
retained from the earlier pruned-set tune (retune-pending, #116 Phase D).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# MAE-tuned (Optuna 150-trial, seed 47) on the earlier pruned selected set;
# retained as the full-set baseline (retune-pending).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 124,
    "learning_rate": 0.0362414989436608,
    "num_leaves": 39,
    "max_depth": 5,
    "min_child_samples": 26,
    "subsample": 0.698337301930694,
    "subsample_freq": 1,
    "colsample_bytree": 0.8852598662130469,
    "reg_alpha": 0.0019791895982314697,
    "reg_lambda": 3.1759013241066048,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL15(LevelModel):
    """TROG-2 receptive-grammar level predictors — exploratory (MAE-tuned, all data).

    Full ``Predictors.DEFAULT_LEVEL`` set, MAE-tuned (params retune-pending).
    """

    model_id = "lrp-rli-gbl-015"
    target_var = V.TROG
    description = (
        "LightGBM — TROG-2 (receptive grammar) level predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for receptive-grammar level (trog). Fits the full "
        "DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature "
        "selection in favour of full-set ranking); hyperparameters are "
        "retained from the earlier pruned-set Optuna tune (retune-pending). "
        "Treat the ranking as exploratory."
    )
