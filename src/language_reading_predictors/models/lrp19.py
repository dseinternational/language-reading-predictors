# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP19: Predictors of expressive-information (APT) gains.

``LRP19`` is the baseline exploratory model for expressive-
information gains (``aptinfo_gain``). ``aptinfo`` is the
information raw score from the Action Picture Test (Renfrew,
1997): the child is shown pictures and asked to describe them,
with scoring of the information content of the response (as
distinct from its grammatical structure, which is scored
separately as ``aptgram`` — LRP17/18).

The target is mildly right-skewed (``aptinfo_gain`` min −7,
max 16, median 2.5, mean 2.61, std 4.44, skewness 0.25, with
~29% negative and ~4% zero observations, n ≈ 160). The low
zero-mass is unusual — most children show measurable change
from timepoint to timepoint (cf LRP11 `trog_gain` ~8% zero,
LRP17 `aptgram_gain` ~11% zero, LRP13 `nonword_gain` ~48%
zero).

No feature selection has been run for LRP19 yet — the MAE-tuned
params below are the starting point for later feature-selection
variants.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP19 has not yet been through iterative feature selection.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the full 34-predictor set (DEFAULT_GAIN, which already
# includes aptinfo), no outlier exclusion (Optuna 150 trials, 10-split
# GroupKFold, seed 47, scoring=mae, lgbm_objective=mae). Tuner-inner
# CV MAE 3.2172 ± 0.5688. n=160.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 165,
    "learning_rate": 0.08050120726641911,
    "num_leaves": 23,
    "max_depth": 11,
    "min_child_samples": 27,
    "subsample": 0.7831373259488615,
    "subsample_freq": 1,
    "colsample_bytree": 0.956934350260005,
    "reg_alpha": 0.02005919343903236,
    "reg_lambda": 0.009655485090866587,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP19(GainModel):
    """APT expressive-information gain predictors — baseline (all data, MAE-tuned).

    Uses the full :attr:`Predictors.DEFAULT_GAIN` predictor set
    (``aptinfo`` is already a member, so the GainModel auto-include
    is a no-op) with MAE-tuned hyperparameters and no outlier
    exclusion. Serves as the starting point for feature-selection
    work on the aptinfo gain-prediction task.
    """

    model_id = "lrp19"
    target_var = V.APTINFO_GAIN
    description = (
        "LightGBM — APT expressive-information gain predictors "
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
        "Baseline exploratory model for APT expressive-information "
        "gains (aptinfo_gain). Uses the full default gain predictor "
        "set (aptinfo is already a level predictor in that set, so "
        "the GainModel auto-include is a no-op) without outlier "
        "exclusion, and MAE-tuned params from an Optuna 150-trial "
        "study — no feature selection has been applied yet. Target "
        "is mildly right-skewed (skew 0.25), unusually low zero "
        "mass (~4%). Pair partner to LRP17 (aptgram_gain)."
    )
