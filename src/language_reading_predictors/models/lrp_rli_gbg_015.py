# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG15: Predictors of receptive-grammar (TROG-2) gains.

``LRPGBG15`` is the baseline exploratory model for receptive-grammar
gains (``trog_gain``). The ``trog`` score is the items-correct
total from the Test for Reception of Grammar 2 (TROG-2; Bishop
2003), covering eight grammatical constructs in blocks of four
items (32 items total; observed max 27 in this sample).

The target is mildly left-skewed (``trog_gain`` min ≈ −10,
max ≈ 12, median 2, mean 1.22, std 4.19, skewness −0.17, with
~34% negative and ~8% zero observations, n ≈ 161). Closer in
shape to LRPGBG05 (``rowpvt_gain``, skew 0.04) and LRPGBG09
(``yarclet_gain``, skew 0.45) than to the heavier-skewed gain
targets.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter sets ─────────────────────────────────────────────────

# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 3.43.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 5.982290999999999,
    "n_estimators": 12,
    "learning_rate": 0.1818310698055082,
    "num_leaves": 59,
    "max_depth": 12,
    "min_child_samples": 24,
    "subsample": 0.8198252404298846,
    "subsample_freq": 1,
    "colsample_bytree": 0.6602371000217966,
    "reg_alpha": 0.16386330109480204,
    "reg_lambda": 0.007966332332012117,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, Huber-tuned) ─────────────────────────────────


class LRPGBG15(GainModel):
    """TROG-2 receptive-grammar gain predictors — baseline (all data, Huber-tuned).

    Full ``Predictors.DEFAULT_GAIN`` set, Huber-tuned on the full set (#169).
    """

    model_id = "lrp-rli-gbg-015"
    target_var = V.TROG_GAIN
    description = (
        "LightGBM — TROG-2 (receptive grammar) gain predictors "
        "(full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for trog_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; Huber retune of 2026-09-22, superseding #169). Gain models are near-noise (baseline-driven regression to the mean) - treat the ranking as exploratory."
    )
