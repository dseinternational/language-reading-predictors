# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP16: Predictors of phoneme-blending level.

``LRP16`` is the baseline exploratory model for phoneme-blending
level (``blending``). A phoneme-blending (phonological awareness)
score on a 0–10 scale: the child selects which of three pictures
depicts the word formed by segmented phonemes spoken by the
examiner.

The target is **essentially symmetric** (``blending`` min 0, max
10, median 6, mean 5.76, std 2.55, skewness 0.01, n ≈ 215) — one
of the cleanest distributions in the suite. The coarse 0–10
scale may cap achievable R² (similar to CELF's 0–18 scale in
LRP10).

No feature selection has been run for LRP16 yet — the MAE-tuned
params below are the starting point for later feature-selection
variants.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP16 has not yet been through iterative feature selection.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the full 32-predictor set (DEFAULT_LEVEL minus blending),
# no outlier exclusion (Optuna 150 trials, 10-split GroupKFold,
# seed 47, scoring=mae, lgbm_objective=mae). Tuner-inner CV MAE
# 1.7000 ± 0.4065. n=215.
#
# Tuned CV MAE is slightly *worse* than the 500-tree baseline
# (1.7000 vs 1.6570) — unusual for this suite, likely because the
# coarse 0–10 target admits little headroom for a shallower tune
# to find. See notes/202604182100-lrp15-lrp16-blending-baseline.md.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 104,
    "learning_rate": 0.023145759415568873,
    "num_leaves": 34,
    "max_depth": 4,
    "min_child_samples": 33,
    "subsample": 0.610302836120071,
    "subsample_freq": 1,
    "colsample_bytree": 0.9183475839745446,
    "reg_alpha": 0.007417570178866872,
    "reg_lambda": 0.5356836718169591,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP16(LevelModel):
    """Phoneme-blending level predictors — baseline (all data, MAE-tuned).

    Uses the full :attr:`Predictors.DEFAULT_LEVEL` predictor set
    (minus the target ``blending``) with MAE-tuned hyperparameters
    and no outlier exclusion. Serves as the starting point for
    feature-selection work on the blending level-prediction task.
    """

    model_id = "lrp16"
    target_var = V.BLENDING
    description = (
        "LightGBM — phoneme-blending level predictors "
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
        "Baseline exploratory model for phoneme-blending level "
        "(blending). Uses the full default level predictor set "
        "(minus the target) without outlier exclusion, and MAE-tuned "
        "params from an Optuna 150-trial study — no feature "
        "selection has been applied yet. Target is essentially "
        "symmetric (skew 0.01) on a coarse 0-10 scale; coarse scale "
        "may cap achievable R² (similar to CELF's 0-18 scale in "
        "LRP10). MAE-tuned CV is marginally worse than the 500-tree "
        "baseline here — unusual for this suite."
    )
