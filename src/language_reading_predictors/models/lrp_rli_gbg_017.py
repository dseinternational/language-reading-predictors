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
# MAE-tuned by Optuna on the full predictor set (150 trials, seed 47;
# #169 retune, superseding the earlier pruned-set tune).

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.153985052602863,
    "num_leaves": 12,
    "max_depth": 9,
    "min_child_samples": 9,
    "subsample": 0.9733873170757179,
    "colsample_bytree": 0.9663624745836672,
    "reg_alpha": 0.0013989511279540352,
    "reg_lambda": 0.03188526883833447,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 3,
}


class LRPGBG17(GainModel):
    """Early Repetition Battery nonword repetition gains predictors — baseline (MAE-tuned)."""

    model_id = "lrp-rli-gbg-017"
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
        "Exploratory model for erbnw_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; #169). Gain models are near-noise (baseline-driven regression to the mean) — treat the ranking as exploratory."
    )
