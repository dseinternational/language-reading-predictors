# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL04: Predictors of not-taught-expressive-vocabulary level.

``LRPGBL04`` is the exploratory model for the *not-taught* expressive-vocabulary
level (``b1exnt`` — Block 1 not-directly-taught expressive vocabulary score), the
level companion to :mod:`lrpgbg04`. Added in #116 Phase B.

Predictor set: :attr:`Predictors.DEFAULT_LEVEL` minus the target, **minus**
``b1exto`` (the Block 1 expressive total = taught + not-taught). As a level model
a high naive R² is partly concurrent same-construct correlation; read the ranking
as exploratory. The not-taught denominator (12 items) is unconfirmed (#144).

Status: initial exploratory baseline; hyperparameters borrowed from LRPGBL02
pending a target-specific tune (``scripts/tune_model.py lrpgbl04``).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# Borrowed from LRPGBL02 (block-1 taught-vocabulary level) pending a target tune.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 45,
    "learning_rate": 0.07573022964806482,
    "num_leaves": 30,
    "max_depth": 6,
    "min_child_samples": 10,
    "subsample": 0.8737230089192473,
    "subsample_freq": 1,
    "colsample_bytree": 0.7169131631393786,
    "reg_alpha": 0.0022764472298362187,
    "reg_lambda": 0.003357533830874894,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL04(LevelModel):
    """Not-taught expressive-vocabulary level predictors — exploratory (MAE, all data)."""

    model_id = "lrpgbl04"
    target_var = V.B1EXNT
    description = (
        "LightGBM — not-taught expressive-vocabulary level predictors "
        "(DEFAULT_LEVEL minus b1exto, MAE, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    exclude = [V.B1EXTO]
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for predictors of not-taught expressive-vocabulary level "
        "(b1exnt). b1exto (Block 1 expressive total = taught + not-taught) excluded "
        "to avoid target leakage. Hyperparameters borrowed from lrpgbl02 pending a "
        "tune; 12-item denominator unconfirmed (#144)."
    )
