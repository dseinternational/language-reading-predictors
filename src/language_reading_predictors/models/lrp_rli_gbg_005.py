# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG05: Predictors of receptive vocabulary gains.

``LRPGBG05`` is the exploratory model for receptive vocabulary gains
(``rowpvt_gain``). It is MAE-tuned on the full
:attr:`Predictors.DEFAULT_GAIN` set (with the ``rowpvt`` baseline
auto-included) and no outlier exclusion, designed to identify the most
important influences on receptive vocabulary gains.

The target is **essentially symmetric** (``rowpvt_gain`` min ≈ −20,
max ≈ 34, median 5, mean 3.84, skewness 0.04, with ~29% negative
and ~3% zero observations, n ≈ 161). Cleaner distribution than any
previous gain target — no skew and no pile-up at zero.

Fits the full ``Predictors.DEFAULT_GAIN`` set; hyperparameters are
re-tuned by Optuna on the full set (150 trials, seed 47; #169).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned by Optuna on the full predictor set (150 trials, seed 47;
# #169 retune, superseding the earlier pruned-set tune).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 52,
    "learning_rate": 0.033583242538720685,
    "num_leaves": 45,
    "max_depth": 12,
    "min_child_samples": 7,
    "subsample": 0.66908083247909,
    "subsample_freq": 1,
    "colsample_bytree": 0.8607281557180144,
    "reg_alpha": 0.003497407529698995,
    "reg_lambda": 0.0014738297333053436,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRPGBG05(GainModel):
    """Receptive vocabulary gain predictors — exploratory (MAE-tuned, all data).

    Full ``Predictors.DEFAULT_GAIN`` set, MAE-tuned on the full
    set (#169). Uses the full predictor set plus the base variable
    ``rowpvt`` (auto-included via :class:`GainModel`) with no outlier
    exclusion.
    """

    model_id = "lrp-rli-gbg-005"
    target_var = V.ROWPVT_GAIN
    description = (
        "LightGBM — receptive vocabulary gain predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for rowpvt_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; #169). Gain models are near-noise (baseline-driven regression to the mean) - treat the ranking as exploratory."
    )
