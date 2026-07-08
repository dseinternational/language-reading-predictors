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
    "n_estimators": 59,
    "learning_rate": 0.03190399484024271,
    "num_leaves": 8,
    "max_depth": 8,
    "min_child_samples": 40,
    "subsample": 0.9763813572652899,
    "subsample_freq": 1,
    "colsample_bytree": 0.613483529665649,
    "reg_alpha": 0.5397054763728967,
    "reg_lambda": 0.0032934986093588266,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL03(LevelModel):
    """Not-taught receptive-vocabulary level predictors — exploratory (MAE, all data)."""

    model_id = "lrp-rli-gbl-003"
    target_var = V.B1RENT
    description = (
        "LightGBM — not-taught receptive-vocabulary level predictors "
        "(DEFAULT_LEVEL minus b1reto, MAE, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    exclude = (V.B1RETO,)
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for predictors of not-taught receptive-vocabulary level "
        "(b1rent). b1reto (Block 1 receptive total = taught + not-taught) excluded "
        "to avoid target leakage. Hyperparameters MAE-tuned by Optuna on the full "
        "set (150 trials, seed 47; #169); 12-item denominator unconfirmed (#144)."
    )
