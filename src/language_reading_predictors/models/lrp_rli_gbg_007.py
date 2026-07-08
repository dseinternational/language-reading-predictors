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
retained from the earlier pruned-set tune (retune-pending, #116 Phase D).
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
    "n_estimators": 275,
    "learning_rate": 0.03817935546050526,
    "num_leaves": 29,
    "max_depth": 12,
    "min_child_samples": 18,
    "subsample": 0.6220122627331144,
    "subsample_freq": 1,
    "colsample_bytree": 0.6383469112272381,
    "reg_alpha": 0.3358114008334534,
    "reg_lambda": 0.06761501795300534,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRPGBG07(GainModel):
    """APT expressive-information gain predictors — baseline (all data, MAE-tuned).

    Full ``Predictors.DEFAULT_GAIN`` set, MAE-tuned (params
    retune-pending). ``aptinfo`` is already a member, so the GainModel
    auto-include is a no-op; no outlier exclusion.
    """

    model_id = "lrp-rli-gbg-007"
    target_var = V.APTINFO_GAIN
    description = (
        "LightGBM — APT expressive-information gain predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for aptinfo_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters are retained from the earlier pruned-set Optuna tune (retune-pending). Gain models are near-noise (baseline-driven regression to the mean) - treat the ranking as exploratory."
    )
