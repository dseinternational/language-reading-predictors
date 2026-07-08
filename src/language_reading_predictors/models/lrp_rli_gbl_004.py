# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL04: Predictors of not-taught-expressive-vocabulary level.

``LRPGBL04`` is the exploratory model for the *not-taught* expressive-vocabulary
level (``b1exnt`` — Block 1 not-directly-taught expressive vocabulary score), the
level companion to :mod:`lrp_rli_gbg_004`. Added in #116 Phase B.

Predictor set: :attr:`Predictors.DEFAULT_LEVEL` minus the target, **minus**
``b1exto`` (the Block 1 expressive total = taught + not-taught). As a level model
a high naive R² is partly concurrent same-construct correlation; read the ranking
as exploratory. The not-taught denominator (12 items) is unconfirmed (#144).

Status: MAE-tuned by Optuna on the full predictor set (150 trials, seed 47;
#169), superseding the earlier parameters borrowed from LRPGBL02.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# MAE-tuned by Optuna on the full predictor set (150 trials, seed 47; #169).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 167,
    "learning_rate": 0.07211542310486023,
    "num_leaves": 38,
    "max_depth": 3,
    "min_child_samples": 5,
    "subsample": 0.9377647274254018,
    "subsample_freq": 1,
    "colsample_bytree": 0.6370586979003762,
    "reg_alpha": 0.0011378215315324038,
    "reg_lambda": 0.015748809647514038,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL04(LevelModel):
    """Not-taught expressive-vocabulary level predictors — exploratory (MAE, all data)."""

    model_id = "lrp-rli-gbl-004"
    target_var = V.B1EXNT
    description = (
        "LightGBM — not-taught expressive-vocabulary level predictors "
        "(DEFAULT_LEVEL minus b1exto, MAE, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    exclude = (V.B1EXTO,)
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for predictors of not-taught expressive-vocabulary level "
        "(b1exnt). b1exto (Block 1 expressive total = taught + not-taught) excluded "
        "to avoid target leakage. Hyperparameters MAE-tuned by Optuna on the full "
        "set (150 trials, seed 47; #169); 12-item denominator unconfirmed (#144)."
    )
