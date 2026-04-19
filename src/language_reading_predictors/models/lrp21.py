# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP21: Predictors of DEAP fine-articulation gains.

``LRP21`` is the baseline exploratory model for DEAP fine-
articulation gains (``deappfi_gain``). ``deappfi`` is a
percentage-scale articulation measure from the Diagnostic
Evaluation of Articulation and Phonology (Dodd et al., 2006) —
the proportion of sounds correctly produced when the child is
asked to name pictures. ``deappfi`` specifically scores the
*final* consonant of each word (distinct from ``deappin``
initial and ``deappvo`` voicing).

The target is nearly symmetric but with a heavy two-sided
spread (``deappfi_gain`` min −56.9, max 56.0, median 0.01,
mean 0.84, std 13.28, skewness −0.32, with **~48% negative**
and ~2% zero observations, n ≈ 152). **Heavy regression from
the ceiling is the dominant story** — children at the top of
the scale tend to drop back between timepoints while those at
the floor improve.

DEAP measures have been used as predictors across every other
model in the suite but never as targets until LRP21/22.

No feature selection has been run for LRP21 yet — the MAE-tuned
params below are the starting point for later feature-selection
variants.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP21 has not yet been through iterative feature selection.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the full 34-predictor set (DEFAULT_GAIN, which already
# includes deappfi), no outlier exclusion (Optuna 150 trials, 10-split
# GroupKFold, seed 47, scoring=mae, lgbm_objective=mae). Tuner-inner
# CV MAE 8.0843 ± 2.7100. n=152.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 19,
    "learning_rate": 0.16968181492926776,
    "num_leaves": 22,
    "max_depth": 10,
    "min_child_samples": 7,
    "subsample": 0.6373225764155953,
    "subsample_freq": 1,
    "colsample_bytree": 0.8514678053914846,
    "reg_alpha": 0.0035935468143608543,
    "reg_lambda": 0.008743117776035333,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP21(GainModel):
    """DEAP fine-articulation gain predictors — baseline (all data, MAE-tuned).

    Uses the full :attr:`Predictors.DEFAULT_GAIN` predictor set
    (``deappfi`` is already a member, so the GainModel auto-include
    is a no-op) with MAE-tuned hyperparameters and no outlier
    exclusion. Serves as the starting point for feature-selection
    work on the deappfi gain-prediction task.
    """

    model_id = "lrp21"
    target_var = V.DEAPPFI_GAIN
    description = (
        "LightGBM — DEAP fine-articulation gain predictors "
        "(34 predictors, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    selection_steps = _SELECTION_STEPS
    shap_scatter_specs = [
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    ]
    notes = (
        "Baseline exploratory model for DEAP fine-articulation "
        "gains (deappfi_gain). Uses the full default gain predictor "
        "set (deappfi is already a level predictor in that set, so "
        "the GainModel auto-include is a no-op) without outlier "
        "exclusion, and MAE-tuned params from an Optuna 150-trial "
        "study — no feature selection has been applied yet. Target "
        "is nearly symmetric (skew −0.32) with heavy two-sided "
        "spread (std 13.3 on 0-100 percentage scale); ceiling-driven "
        "regression is the dominant dynamic. First articulation "
        "target in the suite (DEAP used only as predictor in LRP01-LRP20)."
    )
