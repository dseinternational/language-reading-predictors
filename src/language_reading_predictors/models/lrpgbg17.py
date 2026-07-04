# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG17: Predictors of Early Repetition Battery nonword repetition gains (``erbnw_gain``).

``erbnw`` is the number of nonwords correctly repeated from the
Early Repetition Battery (a repetition task indexing verbal /
phonological short-term memory).

The gain target spans min -6.0, max 10.0, median 1.00, mean 1.26,
std 3.15, skew 0.04, with ~22% negative and ~20% zero observations
(n = 147). Regression from the mean dominates gain targets across
the suite.

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable nonword
repetition is and from what, to inform whether the shared DAG
needs a verbal / phonological short-term memory node. It is not a
causal or intention-to-treat estimate.
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
    "learning_rate": 0.11750810775578611,
    "num_leaves": 45,
    "max_depth": 4,
    "min_child_samples": 17,
    "subsample": 0.6097606249206079,
    "colsample_bytree": 0.758360640735803,
    "reg_alpha": 0.01182776312555698,
    "reg_lambda": 0.18152030603308358,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 82,
}


class LRPGBG17(GainModel):
    """Early Repetition Battery nonword repetition gains predictors — baseline (MAE-tuned)."""

    model_id = "lrpgbg17"
    target_var = V.ERBNW_GAIN
    description = (
        "LightGBM — Early Repetition Battery nonword repetition gains predictors (full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    )
    notes = (
        "Exploratory model for erbnw_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters are retained from the earlier pruned-set Optuna tune (retune-pending). Gain models are near-noise (baseline-driven regression to the mean) — treat the ranking as exploratory."
    )
