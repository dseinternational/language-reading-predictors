# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL18: Predictors of Early Repetition Battery word repetition level (``erbword``).

``erbword`` is the number of real words correctly repeated from
the Early Repetition Battery (a repetition task indexing verbal /
phonological short-term memory).

The target spans min 0.0, max 28.0, median 12.00, mean 11.35, std
5.22, skew -0.25 (n = 203).

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable word repetition
is and from what, to inform whether the shared DAG needs a verbal
/ phonological short-term memory node. It is not a causal or
available-case modified ITT estimate.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline



# ── hyperparameters (Huber-tuned) ──────────────────────────────────────────

# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 2.28.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 7.976387999999999,
    "learning_rate": 0.034994329338801464,
    "num_leaves": 19,
    "max_depth": 9,
    "min_child_samples": 10,
    "subsample": 0.6039084201218891,
    "colsample_bytree": 0.8376810322589427,
    "reg_alpha": 0.0025844301009303452,
    "reg_lambda": 0.16040121214683786,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 165,
}


class LRPGBL18(LevelModel):
    """Early Repetition Battery word repetition level predictors — baseline (Huber-tuned)."""

    model_id = "lrp-rli-gbl-018"
    target_var = V.ERBWORD
    description = (
        "LightGBM — Early Repetition Battery word repetition level predictors (full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    )
    notes = (
        "Exploratory model for erbword (level). Fits the full DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; Huber retune of 2026-09-22, superseding #169). Treat the ranking as exploratory."
    )
