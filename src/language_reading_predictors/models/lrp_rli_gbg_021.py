# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG21: Predictors of DEAP vowel articulation gains (``deappvo_gain``).

``deappvo`` is vowel accuracy from the Diagnostic Evaluation of
Articulation and Phonology (Dodd et al., 2006) — the proportion of
sounds correctly produced in a picture-naming task.

The gain target spans min -23.5, max 17.8, median 0.00, mean 0.19,
std 5.74, skew -0.62, with ~28% negative and ~31% zero
observations (n = 152). Regression from the mean dominates gain
targets across the suite.

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable vowel
articulation is and from what, to inform whether the shared DAG
needs a speech-sound accuracy node. It is not a causal or
intention-to-treat estimate.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline



# ── hyperparameters ──────────────────────────────────────────────────────
# MAE-tuned by Optuna on the full predictor set (150 trials, seed 47;
# #169 retune, superseding the earlier pruned-set tune).

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.041272150647050684,
    "num_leaves": 38,
    "max_depth": 7,
    "min_child_samples": 28,
    "subsample": 0.9386885187525864,
    "colsample_bytree": 0.632624327638653,
    "reg_alpha": 0.07355977005015595,
    "reg_lambda": 0.0077027070740037975,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 382,
}


class LRPGBG21(GainModel):
    """DEAP vowel articulation gains predictors — baseline (MAE-tuned)."""

    model_id = "lrp-rli-gbg-021"
    target_var = V.DEAPPVO_GAIN
    description = (
        "LightGBM — DEAP vowel articulation gains predictors (full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    )
    notes = (
        "Exploratory model for deappvo_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; #169). Gain models are near-noise (baseline-driven regression to the mean) — treat the ranking as exploratory."
    )
