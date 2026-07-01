# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL11: Predictors of phonetic-spelling level.

``LRPGBL11`` is the exploratory model for the phonetic-spelling level (``spphon``),
the level companion to :mod:`lrpgbg11`. Added in #116 Phase B.

Predictor set: :attr:`Predictors.DEFAULT_LEVEL` minus the target. No leakage
sibling to drop. As a level model a high naive R² is partly concurrent
same-construct correlation; combined with the heavy floor on ``spphon`` (~78% at
zero; #119/#144), read the ranking as exploratory.

Status: initial exploratory baseline; hyperparameters borrowed from the
phonics-adjacent letter-sounds analogue LRPGBL09 pending a target-specific tune
(``scripts/tune_model.py lrpgbl11``).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# Borrowed from LRPGBL09 (letter-sounds level) pending a target tune.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 53,
    "learning_rate": 0.037371968457437586,
    "num_leaves": 29,
    "max_depth": 10,
    "min_child_samples": 8,
    "subsample": 0.7474456109857174,
    "subsample_freq": 1,
    "colsample_bytree": 0.9746583954334843,
    "reg_alpha": 1.6618715685357452,
    "reg_lambda": 1.2133803578125177,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL11(LevelModel):
    """Phonetic-spelling level predictors — exploratory (MAE, all data)."""

    model_id = "lrpgbl11"
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
        "exploratory. Hyperparameters borrowed from lrpgbl09 (letter sounds) "
        "pending a target-specific tune."
    )
