# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG07: Predictors of expressive-information (APT) gains.

``LRPGBG07`` is the baseline exploratory model for expressive-
information gains (``aptinfo_gain``). ``aptinfo`` is the
information raw score from the Action Picture Test (Renfrew,
1997): the child is shown pictures and asked to describe them,
with scoring of the information content of the response (as
distinct from its grammatical structure, which is scored
separately as ``aptgram`` — LRPGBG08/18).

The target is mildly right-skewed (``aptinfo_gain`` min −7,
max 16, median 2.5, mean 2.61, std 4.44, skewness 0.25, with
~29% negative and ~4% zero observations, n ≈ 160). The low
zero-mass is unusual — most children show measurable change
from timepoint to timepoint (cf LRPGBG15 `trog_gain` ~8% zero,
LRPGBG08 `aptgram_gain` ~11% zero, LRPGBG13 `nonword_gain` ~48%
zero).

Fits the full ``Predictors.DEFAULT_GAIN`` set; hyperparameters are
re-tuned by Optuna on the full set (150 trials, seed 47; Huber retune of 2026-09-22, superseding #169).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter sets ─────────────────────────────────────────────────

# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 3.87.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 6.979339499999999,
    "n_estimators": 251,
    "learning_rate": 0.03147277019144279,
    "num_leaves": 10,
    "max_depth": 6,
    "min_child_samples": 33,
    "subsample": 0.8264195586859768,
    "subsample_freq": 1,
    "colsample_bytree": 0.9965282286339251,
    "reg_alpha": 1.1546392206152165,
    "reg_lambda": 0.01006119066142894,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, Huber-tuned) ─────────────────────────────────


class LRPGBG07(GainModel):
    """APT expressive-information gain predictors — baseline (all data, Huber-tuned).

    Full ``Predictors.DEFAULT_GAIN`` set, Huber-tuned on the full
    set (#169). ``aptinfo`` is already a member, so the GainModel
    auto-include is a no-op; no outlier exclusion.
    """

    model_id = "lrp-rli-gbg-007"
    target_var = V.APTINFO_GAIN
    description = (
        "LightGBM — APT expressive-information gain predictors "
        "(full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for aptinfo_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; Huber retune of 2026-09-22, superseding #169). Gain models are near-noise (baseline-driven regression to the mean) - treat the ranking as exploratory."
    )
