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
re-tuned by Optuna on the full set (150 trials, seed 47; Huber retune of 2026-09-22, superseding #169).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 3.30.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 5.982290999999999,
    "n_estimators": 44,
    "learning_rate": 0.1412237081472766,
    "num_leaves": 50,
    "max_depth": 5,
    "min_child_samples": 35,
    "subsample": 0.7509821242941188,
    "subsample_freq": 1,
    "colsample_bytree": 0.7018814041663605,
    "reg_alpha": 7.343167733411776,
    "reg_lambda": 1.2882970068298008,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL15(LevelModel):
    """TROG-2 receptive-grammar level predictors — exploratory (Huber-tuned, all data).

    Full ``Predictors.DEFAULT_LEVEL`` set, Huber-tuned on the full set (#169).
    """

    model_id = "lrp-rli-gbl-015"
    target_var = V.TROG
    description = (
        "LightGBM — TROG-2 (receptive grammar) level predictors "
        "(full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for receptive-grammar level (trog). Fits the full "
        "DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature "
        "selection in favour of full-set ranking); hyperparameters are "
        "re-tuned by Optuna on the full set (150 trials, seed 47; Huber retune of 2026-09-22, superseding #169). "
        "Treat the ranking as exploratory."
    )
