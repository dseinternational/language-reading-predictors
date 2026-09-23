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

Status: Huber-tuned by Optuna on the full predictor set (150 trials, seed 47;
#169), superseding the earlier parameters borrowed from the block-1
taught-vocabulary analogue LRPGBG02. Importance rankings — the purpose of this
model — are robust to reasonable parameters.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 2.45.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 3.9881939999999996,
    "n_estimators": 310,
    "learning_rate": 0.011165720021047947,
    "num_leaves": 37,
    "max_depth": 11,
    "min_child_samples": 12,
    "subsample": 0.6617485391272644,
    "subsample_freq": 1,
    "colsample_bytree": 0.9792691444088106,
    "reg_alpha": 5.539232607944625,
    "reg_lambda": 0.02223896890412845,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBG01(GainModel):
    """Taught receptive-vocabulary gain predictors — exploratory (Huber, all data).

    Uses :attr:`Predictors.DEFAULT_GAIN` plus the auto-included baseline
    ``b1retau`` and minus the tautological total ``b1reto`` (see module
    docstring), with no outlier exclusion.
    """

    model_id = "lrp-rli-gbg-001"
    target_var = V.B1RETAU_GAIN
    description = (
        "LightGBM — taught receptive-vocabulary gain predictors "
        "(DEFAULT_GAIN minus b1reto, Huber, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    exclude = (V.B1RETO,)
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for predictors of taught receptive-vocabulary gains "
        "(b1retau_gain), the receptive analogue of lrpgbg02. b1reto (Block 1 "
        "receptive total = taught + not-taught) is excluded to avoid target "
        "leakage. Hyperparameters Huber-tuned by Optuna on the full set (150 trials, "
        "seed 47; #169)."
    )
