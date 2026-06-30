# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG04: Predictors of not-taught-expressive-vocabulary gains.

``LRPGBG04`` is the exploratory model for *not-taught* expressive-vocabulary gains
(``b1exnt_gain`` — change in the Block 1 not-directly-taught expressive vocabulary
score), the expressive counterpart to :mod:`lrpgbg03` and the generalisation
counterpart to the taught set :mod:`lrpgbg02`. Added in #116 Phase B.

Predictor set: :attr:`Predictors.DEFAULT_GAIN` plus the auto-included baseline
``b1exnt`` (via :class:`GainModel`), **minus** ``b1exto`` (the Block 1 expressive
total = taught + not-taught, which contains the target directly — same exclusion
as LRPGBG02).

Status: initial exploratory baseline; hyperparameters borrowed from the block-1
vocabulary analogue LRPGBG02 pending a target-specific tune
(``scripts/tune_model.py lrpgbg04``). The not-taught denominator (12 items) is
unconfirmed in the data dictionary (#144).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# Borrowed from LRPGBG02 (block-1 taught-vocabulary gain) pending a target tune.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 25,
    "learning_rate": 0.06478548258507148,
    "num_leaves": 48,
    "max_depth": 11,
    "min_child_samples": 10,
    "subsample": 0.9835210793717761,
    "subsample_freq": 1,
    "colsample_bytree": 0.9203322386497722,
    "reg_alpha": 0.036771486040166265,
    "reg_lambda": 0.00745726685285877,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBG04(GainModel):
    """Not-taught expressive-vocabulary gain predictors — exploratory (MAE, all data)."""

    model_id = "lrpgbg04"
    target_var = V.B1EXNT_GAIN
    description = (
        "LightGBM — not-taught expressive-vocabulary gain predictors "
        "(DEFAULT_GAIN minus b1exto, MAE, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    exclude = [V.B1EXTO]
    selection_steps = []
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for predictors of not-taught expressive-vocabulary gains "
        "(b1exnt_gain), the transfer counterpart to lrpgbg02. b1exto (Block 1 "
        "expressive total = taught + not-taught) excluded to avoid target leakage. "
        "Hyperparameters borrowed from lrpgbg02 pending a tune; 12-item denominator "
        "unconfirmed (#144)."
    )
