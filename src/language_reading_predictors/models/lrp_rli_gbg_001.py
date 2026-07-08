# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG01: Predictors of taught-receptive-vocabulary gains.

``LRPGBG01`` is the exploratory model for *taught* receptive-vocabulary gains
(``b1retau_gain`` — change in the Block 1 directly-taught receptive vocabulary
score). It is the receptive analogue of :mod:`lrp_rli_gbg_002` (taught *expressive*
vocabulary gains) and one of the four block-1 vocabulary outcomes added in
#116 Phase B so the predictor ranking covers taught/not-taught receptive and
expressive vocabulary, not only the standardised tests.

Predictor set: :attr:`Predictors.DEFAULT_GAIN` plus the auto-included baseline
``b1retau`` (via :class:`GainModel`), **minus** ``b1reto``. The Block 1 receptive
*total* ``b1reto`` equals taught + not-taught (``b1retau + b1rent``), so it
contains the target/baseline construct directly; keeping it would make the model
a between-tests calibration of the same instrument rather than an identification
of substantive predictors (mirrors the ``b1exto`` exclusion in LRPGBG02).

Status: MAE-tuned by Optuna on the full predictor set (150 trials, seed 47;
#169), superseding the earlier parameters borrowed from the block-1
taught-vocabulary analogue LRPGBG02. Importance rankings — the purpose of this
model — are robust to reasonable parameters.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# MAE-tuned by Optuna on the full predictor set (150 trials, seed 47; #169).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 131,
    "learning_rate": 0.0347582835056579,
    "num_leaves": 12,
    "max_depth": 4,
    "min_child_samples": 10,
    "subsample": 0.7324520324408716,
    "subsample_freq": 1,
    "colsample_bytree": 0.95992562989133,
    "reg_alpha": 0.002356197673272227,
    "reg_lambda": 0.16513568663783953,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBG01(GainModel):
    """Taught receptive-vocabulary gain predictors — exploratory (MAE, all data).

    Uses :attr:`Predictors.DEFAULT_GAIN` plus the auto-included baseline
    ``b1retau`` and minus the tautological total ``b1reto`` (see module
    docstring), with no outlier exclusion.
    """

    model_id = "lrp-rli-gbg-001"
    target_var = V.B1RETAU_GAIN
    description = (
        "LightGBM — taught receptive-vocabulary gain predictors "
        "(DEFAULT_GAIN minus b1reto, MAE, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    exclude = (V.B1RETO,)
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for predictors of taught receptive-vocabulary gains "
        "(b1retau_gain), the receptive analogue of lrpgbg02. b1reto (Block 1 "
        "receptive total = taught + not-taught) is excluded to avoid target "
        "leakage. Hyperparameters MAE-tuned by Optuna on the full set (150 trials, "
        "seed 47; #169)."
    )
