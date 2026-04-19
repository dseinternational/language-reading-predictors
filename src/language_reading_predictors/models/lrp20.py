# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP20: Predictors of expressive-information (APT) level.

``LRP20`` is the baseline exploratory model for expressive-
information level (``aptinfo``). ``aptinfo`` is the information
raw score from the Action Picture Test (Renfrew, 1997): the child
is shown pictures and asked to describe them, with scoring of the
information content of the response (as distinct from its
grammatical structure, which is scored separately as ``aptgram``
— LRP17/18).

The target is **essentially symmetric** (``aptinfo`` min 0,
max 37.5, median 16.5, mean 16.97, std 7.93, skewness 0.24,
with ~1% at zero, n ≈ 214) — one of the cleanest distributions
in the suite, comparable to LRP12 (`trog`, skew 0.29) and
LRP16 (`blending`, skew 0.01) and much cleaner than the paired
LRP18 (`aptgram`, skew 1.23).

No feature selection has been run for LRP20 yet — the MAE-tuned
params below are the starting point for later feature-selection
variants.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP20 has not yet been through iterative feature selection.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the full 32-predictor set (DEFAULT_LEVEL minus aptinfo),
# no outlier exclusion (Optuna 150 trials, 10-split GroupKFold,
# seed 47, scoring=mae, lgbm_objective=mae). Tuner-inner CV MAE
# 2.8282 ± 0.7825. n=214.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 445,
    "learning_rate": 0.02047434426043014,
    "num_leaves": 40,
    "max_depth": 11,
    "min_child_samples": 14,
    "subsample": 0.9066656920775948,
    "subsample_freq": 1,
    "colsample_bytree": 0.8232912822965548,
    "reg_alpha": 0.0030342120360389493,
    "reg_lambda": 0.04233508438838369,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP20(LevelModel):
    """APT expressive-information level predictors — baseline (all data, MAE-tuned).

    Uses the full :attr:`Predictors.DEFAULT_LEVEL` predictor set
    (minus the target ``aptinfo``) with MAE-tuned hyperparameters
    and no outlier exclusion. Serves as the starting point for
    feature-selection work on the aptinfo level-prediction task.
    """

    model_id = "lrp20"
    target_var = V.APTINFO
    description = (
        "LightGBM — APT expressive-information level predictors "
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
        "Baseline exploratory model for APT expressive-information "
        "level (aptinfo). Uses the full default level predictor set "
        "(minus the target) without outlier exclusion, and MAE-tuned "
        "params from an Optuna 150-trial study — no feature "
        "selection has been applied yet. Target is essentially "
        "symmetric (skew 0.24) on a 0-37.5 scale — one of the "
        "cleanest distributions in the suite. Pair partner to LRP18 "
        "(aptgram, expressive grammar)."
    )
