# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG03: Predictors of not-taught-receptive-vocabulary gains.

``LRPGBG03`` is the exploratory model for *not-taught* receptive-vocabulary gains
(``b1rent_gain`` — change in the Block 1 not-directly-taught receptive vocabulary
score), the generalisation/transfer counterpart to the taught set. Added in #116
Phase B; the not-taught block-1 sets index transfer beyond the trained words.

Predictor set: :attr:`Predictors.DEFAULT_GAIN` plus the auto-included baseline
``b1rent`` (via :class:`GainModel`), **minus** ``b1reto`` (the Block 1 receptive
total = taught + not-taught, which contains the target directly).

Status: Huber-tuned by Optuna on the full predictor set (150 trials, seed 47;
#169). Note the not-taught denominator (12 items) is unconfirmed in the data
dictionary (#144); the GB ranking does not depend on it, but read magnitudes
with that caveat.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 1.50.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 1.9940969999999998,
    "n_estimators": 59,
    "learning_rate": 0.16204239038650206,
    "num_leaves": 62,
    "max_depth": 8,
    "min_child_samples": 39,
    "subsample": 0.9101527657742315,
    "subsample_freq": 1,
    "colsample_bytree": 0.6021561931883592,
    "reg_alpha": 2.4094869112238944,
    "reg_lambda": 0.003159645834176642,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBG03(GainModel):
    """Not-taught receptive-vocabulary gain predictors — exploratory (Huber, all data)."""

    model_id = "lrp-rli-gbg-003"
    target_var = V.B1RENT_GAIN
    description = (
        "LightGBM — not-taught receptive-vocabulary gain predictors "
        "(DEFAULT_GAIN minus b1reto, Huber, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    exclude = (V.B1RETO,)
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for predictors of not-taught receptive-vocabulary gains "
        "(b1rent_gain), the transfer counterpart to the taught set. b1reto (Block 1 "
        "receptive total = taught + not-taught) excluded to avoid target leakage. "
        "Hyperparameters Huber-tuned by Optuna on the full set (150 trials, seed 47; "
        "#169); 12-item denominator unconfirmed (#144)."
    )
