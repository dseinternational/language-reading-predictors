# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Documentation edited by a LLM-based AI tool (Codex/GPT-6).

"""
lrp-rli-gbg-018: Predictors of Early Repetition Battery word repetition gains (``erbword_gain``).

``erbword`` is the number of real words correctly repeated from
the Early Repetition Battery (a repetition task indexing verbal /
phonological short-term memory).

Rankings describe predictive associations, not causal effects.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline



# ── hyperparameters ──────────────────────────────────────────────────────
# Huber-tuned by Optuna on the full predictor set (150 trials, seed 47;
# #169 retune, superseding the earlier pruned-set tune).

# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 2.07.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 1.9940969999999998,
    "learning_rate": 0.0360353868010488,
    "num_leaves": 63,
    "max_depth": 7,
    "min_child_samples": 14,
    "subsample": 0.603248902559168,
    "colsample_bytree": 0.8494519806133378,
    "reg_alpha": 0.012676036468714916,
    "reg_lambda": 0.006247155361786454,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 95,
}


class LRPGBG18(GainModel):
    """Early Repetition Battery word repetition gains predictors — baseline (Huber-tuned)."""

    model_id = "lrp-rli-gbg-018"
    target_var = V.ERBWORD_GAIN
    description = (
        "LightGBM — Early Repetition Battery word repetition gains predictors (full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    )
    notes = (
        "Exploratory gain model for erbword_gain using DEFAULT_GAIN and its own baseline. "
        "Huber parameters come from the 22 September 2026 retune. Read predictive "
        "performance and ranking stability from the fitted outputs; baseline "
        "associations can reflect score limits or regression to the mean."
    )
