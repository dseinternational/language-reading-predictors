# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Documentation edited by a LLM-based AI tool (Codex/GPT-6).

"""
lrp-rli-gbg-019: Predictors of Early Repetition Battery total repetition gains (``erbto_gain``).

``erbto`` is the ERB total score from the Early Repetition Battery
(a repetition task indexing verbal / phonological short-term
memory). It is a composite of ``erbword``, ``erbnw`` (in the
candidate pool), so read the ranking alongside the level model's same-instrument check.

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
# on the full default predictor set; best mean cross-validated RMSE 4.06.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 5.982290999999999,
    "learning_rate": 0.11071574309046359,
    "num_leaves": 10,
    "max_depth": 5,
    "min_child_samples": 32,
    "subsample": 0.6051330371164608,
    "colsample_bytree": 0.678405152921178,
    "reg_alpha": 0.03692670745714472,
    "reg_lambda": 0.01097366227602514,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 76,
}


class LRPGBG19(GainModel):
    """Early Repetition Battery total repetition gains predictors — baseline (Huber-tuned)."""

    model_id = "lrp-rli-gbg-019"
    target_var = V.ERBTO_GAIN
    description = (
        "LightGBM — Early Repetition Battery total repetition gains predictors (full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    )
    notes = (
        "Exploratory gain model for erbto_gain using DEFAULT_GAIN and its own baseline. "
        "Huber parameters come from the 22 September 2026 retune. Read predictive "
        "performance and ranking stability from the fitted outputs; baseline "
        "associations can reflect score limits or regression to the mean."
    )
