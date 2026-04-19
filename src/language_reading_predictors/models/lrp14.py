# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP14: Predictors of non-word reading level.

``LRP14`` is the baseline exploratory model for non-word reading
level (``nonword``). ``nonword`` is an items-correct score from
a non-word decoding task (observed range 0–6 in this sample).

The target is **heavily floor-loaded** (``nonword`` min 0, max 6,
median 0, mean 1.24, std 1.81, skewness 1.38, with **57% at
zero**, n ≈ 215). Non-word reading / phonological decoding is a
late-emerging skill — most children in this sample have not yet
started to decode non-words reliably, so the data are genuinely
half zeros. This differs from the LRP02 / LRP06 right-skewed
reading targets where zeros are a minority.

Log or quantile transforms may be more appropriate than a plain
regression here; plan for a ``lrp14_log`` variant in follow-up PRs.

No tuning has been run for LRP14 yet — it runs on a reasonable
``_LGBM_BASELINE_PARAMS`` dict so later feature-selection variants
have a documented starting point.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


_SELECTION_STEPS: list[SelectionStep] = []


# MAE-tuned on the full 32-predictor set (DEFAULT_LEVEL minus
# nonword), no outlier exclusion (Optuna 150 trials, 10-split
# GroupKFold, seed 47, scoring=mae, lgbm_objective=mae). Tuner-inner
# CV MAE 0.8696 ± 0.3104. n=215.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 156,
    "learning_rate": 0.04173182097406151,
    "num_leaves": 19,
    "max_depth": 8,
    "min_child_samples": 29,
    "subsample": 0.8875496084128299,
    "subsample_freq": 1,
    "colsample_bytree": 0.6482600725432665,
    "reg_alpha": 0.0028665242167077703,
    "reg_lambda": 0.09374915139778504,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRP14(LevelModel):
    """Non-word reading level predictors — baseline (all data, untuned).

    Uses the full :attr:`Predictors.DEFAULT_LEVEL` predictor set
    (minus the target ``nonword``) and a reasonable
    ``_LGBM_BASELINE_PARAMS`` set. Serves as the starting point for
    feature-selection and tuning work on the non-word-reading
    level-prediction task.
    """

    model_id = "lrp14"
    target_var = V.NONWORD
    description = (
        "LightGBM — non-word reading level predictors "
        "(32 predictors, MAE-tuned, no outlier exclusion)"
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
        "Baseline exploratory model for non-word reading level "
        "(nonword). Uses the full default level predictor set "
        "(minus the target) without outlier exclusion, and a "
        "reasonable _LGBM_BASELINE_PARAMS starting point — no "
        "feature selection or hyperparameter tuning has been applied "
        "yet. Target is heavily floor-loaded (57% at zero, skew 1.38) "
        "— most children have not yet started decoding non-words. "
        "Log or quantile transforms may be more appropriate in a "
        "follow-up variant."
    )
