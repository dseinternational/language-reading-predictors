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

# MAE-tuned (Optuna 150-trial, seed 47) on the earlier pruned selected set;
# retained as the full-set baseline (retune-pending).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 49,
    "learning_rate": 0.04211208724730419,
    "num_leaves": 62,
    "max_depth": 6,
    "min_child_samples": 11,
    "subsample": 0.6190215422270804,
    "subsample_freq": 1,
    "colsample_bytree": 0.7610373169800141,
    "reg_alpha": 0.032621101944747564,
    "reg_lambda": 0.01492191446502865,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRPGBG15(GainModel):
    """TROG-2 receptive-grammar gain predictors — baseline (all data, MAE-tuned).

    Full ``Predictors.DEFAULT_GAIN`` set, MAE-tuned (params retune-pending).
    """

    model_id = "lrp-rli-gbg-015"
    target_var = V.TROG_GAIN
    description = (
        "LightGBM — TROG-2 (receptive grammar) gain predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for trog_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters are retained from the earlier pruned-set Optuna tune (retune-pending). Gain models are near-noise (baseline-driven regression to the mean) - treat the ranking as exploratory."
    )
