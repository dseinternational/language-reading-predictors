# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Documentation edited by a LLM-based AI tool (Codex/GPT-6).

"""
lrp-rli-gbg-022: Predictors of DEAP average articulation gains (``deappav_gain``).

``deappav`` is the DEAP picture-naming average from the Diagnostic
Evaluation of Articulation and Phonology (Dodd et al., 2006) — the
proportion of sounds correctly produced in a picture-naming task.
It is a composite of ``deappin``, ``deappvo``, ``deappfi`` (in the
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
# on the full default predictor set; best mean cross-validated RMSE 3.97.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 5.772910814999984,
    "learning_rate": 0.021613815087377218,
    "num_leaves": 61,
    "max_depth": 7,
    "min_child_samples": 14,
    "subsample": 0.994282857777598,
    "colsample_bytree": 0.700484440534385,
    "reg_alpha": 3.3588630855272434,
    "reg_lambda": 0.00624331514748244,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 38,
}


class LRPGBG22(GainModel):
    """DEAP average articulation gains predictors — baseline (Huber-tuned)."""

    model_id = "lrp-rli-gbg-022"
    target_var = V.DEAPPAV_GAIN
    description = (
        "LightGBM — DEAP average articulation gains predictors (full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    )
    notes = (
        "Exploratory gain model for deappav_gain using DEFAULT_GAIN and its own baseline. "
        "Huber parameters come from the 22 September 2026 retune. Read predictive "
        "performance and ranking stability from the fitted outputs; baseline "
        "associations can reflect score limits or regression to the mean."
    )
