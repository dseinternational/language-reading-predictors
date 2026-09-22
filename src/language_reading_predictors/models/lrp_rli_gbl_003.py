# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL03: Predictors of not-taught-receptive-vocabulary level.

``LRPGBL03`` is the exploratory model for the *not-taught* receptive-vocabulary
level (``b1rent`` — Block 1 not-directly-taught receptive vocabulary score), the
level companion to :mod:`lrp_rli_gbg_003`. Added in #116 Phase B.

Predictor set: :attr:`Predictors.DEFAULT_LEVEL` minus the target, **minus**
``b1reto`` (the Block 1 receptive total = taught + not-taught). As a level model
a high naive R² is partly concurrent same-construct correlation; read the ranking
as exploratory. The not-taught denominator (12 items) is unconfirmed (#144).

Status: Huber-tuned by Optuna on the full predictor set (150 trials, seed 47;
#169), superseding the earlier parameters borrowed from LRPGBL02.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 1.48.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 1.9940969999999998,
    "n_estimators": 142,
    "learning_rate": 0.01597241271468781,
    "num_leaves": 40,
    "max_depth": 4,
    "min_child_samples": 4,
    "subsample": 0.6133708919586373,
    "subsample_freq": 1,
    "colsample_bytree": 0.7114648276137164,
    "reg_alpha": 0.2597608897184164,
    "reg_lambda": 0.0031643953352305967,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL03(LevelModel):
    """Not-taught receptive-vocabulary level predictors — exploratory (Huber, all data)."""

    model_id = "lrp-rli-gbl-003"
    target_var = V.B1RENT
    description = (
        "LightGBM — not-taught receptive-vocabulary level predictors "
        "(DEFAULT_LEVEL minus b1reto, Huber, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    exclude = (V.B1RETO,)
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for predictors of not-taught receptive-vocabulary level "
        "(b1rent). b1reto (Block 1 receptive total = taught + not-taught) excluded "
        "to avoid target leakage. Hyperparameters Huber-tuned by Optuna on the full "
        "set (150 trials, seed 47; Huber retune of 2026-09-22, superseding #169); 12-item denominator unconfirmed (#144)."
    )
