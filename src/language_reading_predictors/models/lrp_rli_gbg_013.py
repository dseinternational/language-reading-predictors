# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG13: Predictors of non-word reading gains.

``LRPGBG13`` is the baseline exploratory model for non-word reading
gains (``nonword_gain``). ``nonword`` is an items-correct score
from a non-word decoding task (observed range 0–6).

The target is **heavily zero-loaded** (``nonword_gain`` min −6,
max 6, median 0, mean 0.41, std 1.66, skewness 0.40, with ~19%
negative and **~48% zero** observations, n ≈ 153). Half the
children show no change between timepoints — consistent with the
floor-heavy nonword level distribution (57% zero at any given
timepoint). Tree models will struggle to predict non-zero gains
reliably.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# MAE-tuned by Optuna on the full predictor set (150 trials, seed 47;
# #169 retune, superseding the earlier pruned-set tune).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 109,
    "learning_rate": 0.13809890473906328,
    "num_leaves": 38,
    "max_depth": 5,
    "min_child_samples": 26,
    "subsample": 0.9672677202248349,
    "subsample_freq": 1,
    "colsample_bytree": 0.7420738914522781,
    "reg_alpha": 0.0015094066894091216,
    "reg_lambda": 0.0012184431150826054,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBG13(GainModel):
    """Non-word reading gain predictors — baseline (all data, MAE-tuned).

    Full ``Predictors.DEFAULT_GAIN`` set, MAE-tuned on the full set (#169).
    """

    model_id = "lrp-rli-gbg-013"
    target_var = V.NONWORD_GAIN
    description = (
        "LightGBM — non-word reading gain predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for nonword_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; #169). Gain models are near-noise (baseline-driven regression to the mean) - treat the ranking as exploratory."
    )
