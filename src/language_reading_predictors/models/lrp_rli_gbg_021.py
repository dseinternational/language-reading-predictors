# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Documentation edited by a LLM-based AI tool (Codex/GPT-6).

"""
lrp-rli-gbg-021: Predictors of DEAP vowel articulation gains (``deappvo_gain``).

``deappvo`` is vowel accuracy from the Diagnostic Evaluation of
Articulation and Phonology (Dodd et al., 2006) — the percentage of
sounds correctly produced in a picture-naming task.

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
# on the full default predictor set; best mean cross-validated RMSE 3.66.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 4.107839820000004,
    "learning_rate": 0.12251125718120537,
    "num_leaves": 55,
    "max_depth": 3,
    "min_child_samples": 13,
    "subsample": 0.7693450189299881,
    "colsample_bytree": 0.9258184360556404,
    "reg_alpha": 0.34310199295854016,
    "reg_lambda": 0.014315316291876488,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 41,
}


class LRPGBG21(GainModel):
    """DEAP vowel articulation gains predictors — baseline (Huber-tuned)."""

    model_id = "lrp-rli-gbg-021"
    target_var = V.DEAPPVO_GAIN
    description = (
        "LightGBM — DEAP vowel articulation gains predictors (full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    )
    notes = (
        "Exploratory gain model for deappvo_gain using DEFAULT_GAIN and its own baseline. "
        "Huber parameters come from the 22 September 2026 retune. Read predictive "
        "performance and ranking stability from the fitted outputs; baseline "
        "associations can reflect score limits or regression to the mean."
    )
