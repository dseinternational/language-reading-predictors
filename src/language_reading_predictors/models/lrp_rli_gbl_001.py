# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL01: Predictors of taught-receptive-vocabulary level.

``LRPGBL01`` is the exploratory model for the *taught* receptive-vocabulary level
(``b1retau`` — Block 1 directly-taught receptive vocabulary score), the level
companion to :mod:`lrp_rli_gbg_001` and the receptive analogue of :mod:`lrp_rli_gbl_002`
(taught expressive vocabulary level). Added in #116 Phase B.

Predictor set: :attr:`Predictors.DEFAULT_LEVEL` minus the target, **minus**
``b1reto`` (the Block 1 receptive total = taught + not-taught, which contains the
target construct directly — mirrors the ``b1exto`` exclusion in LRPGBL02). As a
level model, a high naive R² is partly concurrent same-construct correlation;
read the ranking as exploratory.

Status: Huber-tuned by Optuna on the full predictor set (150 trials, seed 47;
#169), superseding the earlier parameters borrowed from LRPGBL02.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 2.24.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 5.982290999999999,
    "n_estimators": 42,
    "learning_rate": 0.13104600945507308,
    "num_leaves": 24,
    "max_depth": 3,
    "min_child_samples": 5,
    "subsample": 0.7131076695369919,
    "subsample_freq": 1,
    "colsample_bytree": 0.7850787375162626,
    "reg_alpha": 0.1878914830359851,
    "reg_lambda": 0.31090454994152084,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL01(LevelModel):
    """Taught receptive-vocabulary level predictors — exploratory (Huber, all data)."""

    model_id = "lrp-rli-gbl-001"
    target_var = V.B1RETAU
    description = (
        "LightGBM — taught receptive-vocabulary level predictors "
        "(DEFAULT_LEVEL minus b1reto, Huber, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    exclude = (V.B1RETO,)
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for predictors of taught receptive-vocabulary level "
        "(b1retau), receptive analogue of lrpgbl02. b1reto (Block 1 receptive "
        "total = taught + not-taught) excluded to avoid target leakage. "
        "Hyperparameters Huber-tuned by Optuna on the full set (150 trials, seed 47; "
        "#169)."
    )
