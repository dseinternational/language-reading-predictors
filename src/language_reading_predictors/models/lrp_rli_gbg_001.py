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

Status: initial exploratory baseline. Hyperparameters are borrowed from the
block-1 taught-vocabulary analogue LRPGBG02 as a reasonable starting point; a
target-specific Optuna tune (``scripts/tune_model.py lrpgbg01``) is a follow-up.
Importance rankings — the purpose of this model — are robust to reasonable
parameters.
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
        "leakage. Hyperparameters borrowed from lrpgbg02 pending a target-specific "
        "tune."
    )
