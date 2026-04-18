# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP11: Predictors of receptive-grammar (TROG-2) gains.

``LRP11`` is the baseline exploratory model for receptive-grammar
gains (``trog_gain``). The ``trog`` score is the items-correct
total from the Test for Reception of Grammar 2 (TROG-2; Bishop
2003), covering eight grammatical constructs in blocks of four
items (32 items total; observed max 27 in this sample).

The target is mildly left-skewed (``trog_gain`` min ≈ −10,
max ≈ 12, median 2, mean 1.22, std 4.19, skewness −0.17, with
~34% negative and ~8% zero observations, n ≈ 161). Closer in
shape to LRP07 (``rowpvt_gain``, skew 0.04) and LRP05
(``yarclet_gain``, skew 0.45) than to the heavier-skewed gain
targets.

No tuning has been run for LRP11 yet — it runs on a reasonable
``_LGBM_BASELINE_PARAMS`` dict so later feature-selection variants
have a documented starting point.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP11 has not yet been through iterative feature selection.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the full 34-predictor set (DEFAULT_GAIN, which already
# includes trog), no outlier exclusion (Optuna 150 trials, 10-split
# GroupKFold, seed 47, scoring=mae, lgbm_objective=mae). Tuner-inner
# CV MAE 3.2512 ± 0.5931. n=161.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 88,
    "learning_rate": 0.03804924080103806,
    "num_leaves": 26,
    "max_depth": 7,
    "min_child_samples": 11,
    "subsample": 0.6044183792162923,
    "subsample_freq": 1,
    "colsample_bytree": 0.7621630316696983,
    "reg_alpha": 0.006180867550340822,
    "reg_lambda": 0.47920183828458845,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, untuned) ───────────────────────────────────


class LRP11(GainModel):
    """TROG-2 receptive-grammar gain predictors — baseline (all data, untuned).

    Uses the full :attr:`Predictors.DEFAULT_GAIN` predictor set
    (``trog`` is already a member, so the GainModel auto-include
    is a no-op) with a reasonable ``_LGBM_BASELINE_PARAMS`` set.
    Serves as the starting point for feature-selection and tuning
    work on the TROG gain-prediction task.
    """

    model_id = "lrp11"
    target_var = V.TROG_GAIN
    description = (
        "LightGBM — TROG-2 (receptive grammar) gain predictors "
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
        "Baseline exploratory model for TROG-2 receptive-grammar "
        "gains (trog_gain). Uses the full default gain predictor "
        "set (trog is already included as a level predictor so the "
        "GainModel auto-include is a no-op) without outlier "
        "exclusion, and a reasonable _LGBM_BASELINE_PARAMS starting "
        "point — no feature selection or hyperparameter tuning has "
        "been applied yet. Target is mildly left-skewed (skew −0.17)."
    )
