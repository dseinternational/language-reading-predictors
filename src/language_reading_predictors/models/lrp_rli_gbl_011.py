# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL11: Predictors of phonetic-spelling level.

``LRPGBL11`` is the exploratory model for the phonetic-spelling level (``spphon``),
the level companion to :mod:`lrp_rli_gbg_011`. Added in #116 Phase B.

Predictor set: :attr:`Predictors.DEFAULT_LEVEL` minus the target. No leakage
sibling to drop. As a level model a high naive R² is partly concurrent
same-construct correlation; combined with the heavy floor on ``spphon`` (~78% at
zero; #119/#144), read the ranking as exploratory.

Status: Huber-tuned by Optuna on the full predictor set (150 trials, seed 47;
#169), superseding the earlier parameters borrowed from the phonics-adjacent
letter-sounds analogue LRPGBL09.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 10.78.
# Huber threshold alpha = 1.345 x mean |y - median| (MAD is zero)
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 25.404158878504674,
    "n_estimators": 200,
    "learning_rate": 0.02397205933190059,
    "num_leaves": 19,
    "max_depth": 11,
    "min_child_samples": 6,
    "subsample": 0.6165439389592248,
    "subsample_freq": 1,
    "colsample_bytree": 0.9748327146731446,
    "reg_alpha": 0.0735502354247838,
    "reg_lambda": 0.015620257966383323,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL11(LevelModel):
    """Phonetic-spelling level predictors — exploratory (Huber, all data)."""

    model_id = "lrp-rli-gbl-011"
    target_var = V.SPPHON
    description = (
        "LightGBM — phonetic-spelling level predictors "
        "(DEFAULT_LEVEL, Huber, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for predictors of phonetic-spelling level (spphon). "
        "spphon is heavily floored (~78% at zero), so read the ranking as "
        "exploratory. Hyperparameters Huber-tuned by Optuna on the full set "
        "(150 trials, seed 47; Huber retune of 2026-09-22, superseding #169)."
    )
