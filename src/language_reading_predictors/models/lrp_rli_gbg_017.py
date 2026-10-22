# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Documentation edited by a LLM-based AI tool (Codex/GPT-6).

"""
lrp-rli-gbg-017: Predictors of Early Repetition Battery nonword repetition gains (``erbnw_gain``).

``erbnw`` is the number of nonwords correctly repeated from the
Early Repetition Battery (a repetition task indexing verbal /
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
# on the full default predictor set; best mean cross-validated RMSE 2.57.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 3.9881939999999996,
    "learning_rate": 0.1391607836594236,
    "num_leaves": 52,
    "max_depth": 3,
    "min_child_samples": 14,
    "subsample": 0.9086121322708418,
    "colsample_bytree": 0.9571232006383583,
    "reg_alpha": 2.101998945258291,
    "reg_lambda": 0.0032000768731771036,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 22,
}


class LRPGBG17(GainModel):
    """Early Repetition Battery nonword repetition gains predictors — baseline (Huber-tuned)."""

    model_id = "lrp-rli-gbg-017"
    target_var = V.ERBNW_GAIN
    description = (
        "LightGBM — Early Repetition Battery nonword repetition gains predictors (full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    )
    notes = (
        "Exploratory gain model for erbnw_gain using DEFAULT_GAIN and its own baseline. "
        "Huber parameters come from the 22 September 2026 retune. Read predictive "
        "performance and ranking stability from the fitted outputs; baseline "
        "associations can reflect score limits or regression to the mean."
    )
