# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL17: Predictors of Early Repetition Battery nonword repetition level (``erbnw``).

``erbnw`` is the number of nonwords correctly repeated from the
Early Repetition Battery (a repetition task indexing verbal /
phonological short-term memory).

The target spans min 0.0, max 18.0, median 9.00, mean 8.98, std
4.74, skew 0.01 (n = 202).

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable nonword
repetition is and from what, to inform whether the shared DAG
needs a verbal / phonological short-term memory node. It is not a
causal or available-case modified ITT estimate.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline



# ── hyperparameters (Huber-tuned) ──────────────────────────────────────────

# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 2.31.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 7.976387999999999,
    "learning_rate": 0.014233227095559997,
    "num_leaves": 8,
    "max_depth": 10,
    "min_child_samples": 35,
    "subsample": 0.9628075945838528,
    "colsample_bytree": 0.912625532617394,
    "reg_alpha": 1.645261397149483,
    "reg_lambda": 0.0028424663500279572,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 381,
}


class LRPGBL17(LevelModel):
    """Early Repetition Battery nonword repetition level predictors — baseline (Huber-tuned)."""

    model_id = "lrp-rli-gbl-017"
    target_var = V.ERBNW
    description = (
        "LightGBM — Early Repetition Battery nonword repetition level predictors (full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    )
    notes = (
        "Exploratory model for erbnw (level). Fits the full DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; Huber retune of 2026-09-22, superseding #169). Treat the ranking as exploratory."
    )
