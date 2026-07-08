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
re-tuned by Optuna on the full set (150 trials, seed 47; #169).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# MAE-tuned by Optuna on the full predictor set (150 trials, seed 47;
# #169 retune, superseding the earlier pruned-set tune).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 63,
    "learning_rate": 0.18075869091750218,
    "num_leaves": 39,
    "max_depth": 4,
    "min_child_samples": 33,
    "subsample": 0.8156060719055651,
    "subsample_freq": 1,
    "colsample_bytree": 0.7150116968481384,
    "reg_alpha": 0.4155429932879105,
    "reg_lambda": 2.4687897146238416,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL15(LevelModel):
    """TROG-2 receptive-grammar level predictors — exploratory (MAE-tuned, all data).

    Full ``Predictors.DEFAULT_LEVEL`` set, MAE-tuned on the full set (#169).
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
        "re-tuned by Optuna on the full set (150 trials, seed 47; #169). "
        "Treat the ranking as exploratory."
    )
