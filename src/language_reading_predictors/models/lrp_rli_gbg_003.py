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

Status: MAE-tuned by Optuna on the full predictor set (150 trials, seed 47;
#169). Note the not-taught denominator (12 items) is unconfirmed in the data
dictionary (#144); the GB ranking does not depend on it, but read magnitudes
with that caveat.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# MAE-tuned by Optuna on the full predictor set (150 trials, seed 47; #169).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 366,
    "learning_rate": 0.0452552675311885,
    "num_leaves": 62,
    "max_depth": 7,
    "min_child_samples": 39,
    "subsample": 0.9990722272349716,
    "subsample_freq": 1,
    "colsample_bytree": 0.9840389492131241,
    "reg_alpha": 0.2547748447643193,
    "reg_lambda": 0.10812141530625737,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBG03(GainModel):
    """Not-taught receptive-vocabulary gain predictors — exploratory (MAE, all data)."""

    model_id = "lrp-rli-gbg-003"
    target_var = V.B1RENT_GAIN
    description = (
        "LightGBM — not-taught receptive-vocabulary gain predictors "
        "(DEFAULT_GAIN minus b1reto, MAE, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    exclude = (V.B1RETO,)
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for predictors of not-taught receptive-vocabulary gains "
        "(b1rent_gain), the transfer counterpart to the taught set. b1reto (Block 1 "
        "receptive total = taught + not-taught) excluded to avoid target leakage. "
        "Hyperparameters MAE-tuned by Optuna on the full set (150 trials, seed 47; "
        "#169); 12-item denominator unconfirmed (#144)."
    )
