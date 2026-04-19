# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP13: Predictors of non-word reading gains.

``LRP13`` is the baseline exploratory model for non-word reading
gains (``nonword_gain``). ``nonword`` is an items-correct score
from a non-word decoding task (observed range 0–6).

The target is **heavily zero-loaded** (``nonword_gain`` min −6,
max 6, median 0, mean 0.41, std 1.66, skewness 0.40, with ~19%
negative and **~48% zero** observations, n ≈ 153). Half the
children show no change between timepoints — consistent with the
floor-heavy nonword level distribution (57% zero at any given
timepoint). Tree models will struggle to predict non-zero gains
reliably.

No tuning has been run for LRP13 yet — it runs on a reasonable
``_LGBM_BASELINE_PARAMS`` dict so later feature-selection variants
have a documented starting point.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


_SELECTION_STEPS: list[SelectionStep] = []


# MAE-tuned on the full 34-predictor set (DEFAULT_GAIN, which already
# includes nonword), no outlier exclusion (Optuna 150 trials, 10-split
# GroupKFold, seed 47, scoring=mae, lgbm_objective=mae). Tuner-inner
# CV MAE 0.9801 ± 0.3358. n=153.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 108,
    "learning_rate": 0.0674865249242372,
    "num_leaves": 42,
    "max_depth": 5,
    "min_child_samples": 18,
    "subsample": 0.9714636655718227,
    "subsample_freq": 1,
    "colsample_bytree": 0.6334050957187943,
    "reg_alpha": 0.019528666718259695,
    "reg_lambda": 0.0897529592003205,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRP13(GainModel):
    """Non-word reading gain predictors — baseline (all data, untuned).

    Uses the full :attr:`Predictors.DEFAULT_GAIN` predictor set
    (``nonword`` is already a member) with a reasonable
    ``_LGBM_BASELINE_PARAMS`` set. Serves as the starting point
    for feature-selection and tuning work on the non-word-reading
    gain-prediction task.
    """

    model_id = "lrp13"
    target_var = V.NONWORD_GAIN
    description = (
        "LightGBM — non-word reading gain predictors "
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
        "Baseline exploratory model for non-word reading gains "
        "(nonword_gain). Uses the full default gain predictor set "
        "(nonword is already a level predictor in that set, so the "
        "GainModel auto-include is a no-op) without outlier "
        "exclusion, and a reasonable _LGBM_BASELINE_PARAMS starting "
        "point — no feature selection or hyperparameter tuning has "
        "been applied yet. Target is heavily zero-loaded (~48% zero, "
        "19% negative) — a different pathology from the other gain "
        "targets."
    )
