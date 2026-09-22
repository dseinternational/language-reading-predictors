# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG04: Predictors of not-taught-expressive-vocabulary gains.

``LRPGBG04`` is the exploratory model for *not-taught* expressive-vocabulary gains
(``b1exnt_gain`` — change in the Block 1 not-directly-taught expressive vocabulary
score), the expressive counterpart to :mod:`lrp_rli_gbg_003` and the generalisation
counterpart to the taught set :mod:`lrp_rli_gbg_002`. Added in #116 Phase B.

Predictor set: :attr:`Predictors.DEFAULT_GAIN` plus the auto-included baseline
``b1exnt`` (via :class:`GainModel`), **minus** ``b1exto`` (the Block 1 expressive
total = taught + not-taught, which contains the target directly — same exclusion
as LRPGBG02).

Status: Huber-tuned by Optuna on the full predictor set (150 trials, seed 47;
#169). The not-taught denominator (12 items) is unconfirmed in the data
dictionary (#144).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 1.25.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 1.9940969999999998,
    "n_estimators": 53,
    "learning_rate": 0.19393288859786456,
    "num_leaves": 51,
    "max_depth": 11,
    "min_child_samples": 38,
    "subsample": 0.8139288421776413,
    "subsample_freq": 1,
    "colsample_bytree": 0.7547032639349376,
    "reg_alpha": 0.0015884594029627022,
    "reg_lambda": 0.014928546216563083,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBG04(GainModel):
    """Not-taught expressive-vocabulary gain predictors — exploratory (Huber, all data)."""

    model_id = "lrp-rli-gbg-004"
    target_var = V.B1EXNT_GAIN
    description = (
        "LightGBM — not-taught expressive-vocabulary gain predictors "
        "(DEFAULT_GAIN minus b1exto, Huber, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    exclude = (V.B1EXTO,)
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for predictors of not-taught expressive-vocabulary gains "
        "(b1exnt_gain), the transfer counterpart to lrpgbg02. b1exto (Block 1 "
        "expressive total = taught + not-taught) excluded to avoid target leakage. "
        "Hyperparameters Huber-tuned by Optuna on the full set (150 trials, seed 47; "
        "#169); 12-item denominator unconfirmed (#144)."
    )
