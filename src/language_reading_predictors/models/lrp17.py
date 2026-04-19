# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP17: Predictors of expressive-grammar (APT) gains.

``LRP17`` is the baseline exploratory model for expressive-grammar
gains (``aptgram_gain``). ``aptgram`` is the grammar raw score
from the Action Picture Test (Renfrew, 1997) — the child is shown
pictures and asked to describe them, with scoring of the
grammatical structure of the response.

The target is mildly right-skewed (``aptgram_gain`` min −11,
max 16, median 1, mean 1.49, std 4.34, skewness 0.31, with ~32%
negative and ~11% zero observations, n ≈ 158). Similar gain-shape
to LRP09 (``celf_gain``, skew 0.14).

``aptgram`` is the expressive-grammar parallel to ``trog``
(LRP11/12 receptive grammar) — the pair addresses the
expressive vs receptive grammar asymmetry that is a live
question in DS language research.

No feature selection has been run for LRP17 yet — the MAE-tuned
params below are the starting point for later feature-selection
variants.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP17 has not yet been through iterative feature selection.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the full 34-predictor set (DEFAULT_GAIN, which already
# includes aptgram), no outlier exclusion (Optuna 150 trials, 10-split
# GroupKFold, seed 47, scoring=mae, lgbm_objective=mae). Tuner-inner
# CV MAE 3.1562 ± 0.4001. n=158.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 52,
    "learning_rate": 0.14753791101613759,
    "num_leaves": 18,
    "max_depth": 9,
    "min_child_samples": 21,
    "subsample": 0.8413663342544724,
    "subsample_freq": 1,
    "colsample_bytree": 0.8025716954980706,
    "reg_alpha": 0.07212716898249655,
    "reg_lambda": 4.453890230795632,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP17(GainModel):
    """APT expressive-grammar gain predictors — baseline (all data, MAE-tuned).

    Uses the full :attr:`Predictors.DEFAULT_GAIN` predictor set
    (``aptgram`` is already a member, so the GainModel auto-include
    is a no-op) with MAE-tuned hyperparameters and no outlier
    exclusion. Serves as the starting point for feature-selection
    work on the aptgram gain-prediction task.
    """

    model_id = "lrp17"
    target_var = V.APTGRAM_GAIN
    description = (
        "LightGBM — APT expressive-grammar gain predictors "
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
        "Baseline exploratory model for APT expressive-grammar "
        "gains (aptgram_gain). Uses the full default gain predictor "
        "set (aptgram is already a level predictor in that set, so "
        "the GainModel auto-include is a no-op) without outlier "
        "exclusion, and MAE-tuned params from an Optuna 150-trial "
        "study — no feature selection has been applied yet. Target "
        "is mildly right-skewed (skew 0.31). Pair partner to LRP11 "
        "(trog_gain, receptive grammar)."
    )
