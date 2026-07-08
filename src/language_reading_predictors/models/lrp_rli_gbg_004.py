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

Status: MAE-tuned by Optuna on the full predictor set (150 trials, seed 47;
#169). The not-taught denominator (12 items) is unconfirmed in the data
dictionary (#144).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# MAE-tuned by Optuna on the full predictor set (150 trials, seed 47; #169).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 597,
    "learning_rate": 0.010518099280120749,
    "num_leaves": 53,
    "max_depth": 7,
    "min_child_samples": 25,
    "subsample": 0.987550260331751,
    "subsample_freq": 1,
    "colsample_bytree": 0.9906399995368773,
    "reg_alpha": 0.094575911092152,
    "reg_lambda": 0.46887935276860904,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBG04(GainModel):
    """Not-taught expressive-vocabulary gain predictors — exploratory (MAE, all data)."""

    model_id = "lrp-rli-gbg-004"
    target_var = V.B1EXNT_GAIN
    description = (
        "LightGBM — not-taught expressive-vocabulary gain predictors "
        "(DEFAULT_GAIN minus b1exto, MAE, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    exclude = (V.B1EXTO,)
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for predictors of not-taught expressive-vocabulary gains "
        "(b1exnt_gain), the transfer counterpart to lrpgbg02. b1exto (Block 1 "
        "expressive total = taught + not-taught) excluded to avoid target leakage. "
        "Hyperparameters MAE-tuned by Optuna on the full set (150 trials, seed 47; "
        "#169); 12-item denominator unconfirmed (#144)."
    )
