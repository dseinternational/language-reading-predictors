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

Status: MAE-tuned by Optuna on the full predictor set (150 trials, seed 47;
#169), superseding the earlier parameters borrowed from the phonics-adjacent
letter-sounds analogue LRPGBL09.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# MAE-tuned by Optuna on the full predictor set (150 trials, seed 47; #169).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 115,
    "learning_rate": 0.052587676389702374,
    "num_leaves": 42,
    "max_depth": 6,
    "min_child_samples": 8,
    "subsample": 0.9326615778078384,
    "subsample_freq": 1,
    "colsample_bytree": 0.8026282562980487,
    "reg_alpha": 0.001425287049531991,
    "reg_lambda": 1.8818793395665099,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL11(LevelModel):
    """Phonetic-spelling level predictors — exploratory (MAE, all data)."""

    model_id = "lrp-rli-gbl-011"
    target_var = V.SPPHON
    description = (
        "LightGBM — phonetic-spelling level predictors "
        "(DEFAULT_LEVEL, MAE, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for predictors of phonetic-spelling level (spphon). "
        "spphon is heavily floored (~78% at zero), so read the ranking as "
        "exploratory. Hyperparameters MAE-tuned by Optuna on the full set "
        "(150 trials, seed 47; #169)."
    )
