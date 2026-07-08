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
# MAE-tuned (Optuna 150-trial, seed 47) on the earlier pruned selected set;
# retained as the full-set baseline (retune-pending).

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.047737397768147234,
    "num_leaves": 53,
    "max_depth": 3,
    "min_child_samples": 20,
    "subsample": 0.8313091023641015,
    "colsample_bytree": 0.7904910654511403,
    "reg_alpha": 1.9756763955805776,
    "reg_lambda": 1.8953881224366262,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 166,
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
        "Exploratory model for deappvo_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters are retained from the earlier pruned-set Optuna tune (retune-pending). Gain models are near-noise (baseline-driven regression to the mean) — treat the ranking as exploratory."
    )
