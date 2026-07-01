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


# MAE-tuned (Optuna 150-trial, seed 47) on the earlier pruned selected set;
# retained as the full-set baseline (retune-pending).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 562,
    "learning_rate": 0.04923477608940171,
    "num_leaves": 57,
    "max_depth": 12,
    "min_child_samples": 33,
    "subsample": 0.8923475866868639,
    "subsample_freq": 1,
    "colsample_bytree": 0.8409770178651289,
    "reg_alpha": 2.5883086592596545,
    "reg_lambda": 0.026085831362143107,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBG13(GainModel):
    """Non-word reading gain predictors — baseline (all data, MAE-tuned).

    Full ``Predictors.DEFAULT_GAIN`` set, MAE-tuned (params retune-pending).
    """

    model_id = "lrpgbg13"
    target_var = V.NONWORD_GAIN
    description = (
        "LightGBM — non-word reading gain predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for nonword_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters are retained from the earlier pruned-set Optuna tune (retune-pending). Gain models are near-noise (baseline-driven regression to the mean) - treat the ranking as exploratory."
    )
