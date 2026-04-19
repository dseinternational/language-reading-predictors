# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP22: Predictors of DEAP fine-articulation level.

``LRP22`` is the baseline exploratory model for DEAP fine-
articulation level (``deappfi``). ``deappfi`` is a
percentage-scale articulation measure from the Diagnostic
Evaluation of Articulation and Phonology (Dodd et al., 2006) —
the proportion of sounds correctly produced when the child is
asked to name pictures. ``deappfi`` specifically scores the
*final* consonant of each word (distinct from ``deappin``
initial and ``deappvo`` voicing).

The target is **left-skewed with ceiling** (``deappfi`` min
5.4, max 95.2, median 66.6, mean 60.3, std 20.9, skewness
**−0.87**, n ≈ 207). Ceiling effects are possible at the top
end — several children score in the 90s. No zeros in the
sample (unlike the reading targets where floor effects
dominate).

DEAP measures have been used as predictors across every other
model in the suite but never as targets until LRP21/22. First
articulation-domain target.

No feature selection has been run for LRP22 yet — the MAE-tuned
params below are the starting point for later feature-selection
variants.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP22 has not yet been through iterative feature selection.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the full 32-predictor set (DEFAULT_LEVEL minus deappfi),
# no outlier exclusion (Optuna 150 trials, 10-split GroupKFold,
# seed 47, scoring=mae, lgbm_objective=mae). Tuner-inner CV MAE
# 9.6719 ± 1.3824. n=207.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 385,
    "learning_rate": 0.011170674378747147,
    "num_leaves": 7,
    "max_depth": 7,
    "min_child_samples": 16,
    "subsample": 0.9825474567123886,
    "subsample_freq": 1,
    "colsample_bytree": 0.9030132868657452,
    "reg_alpha": 0.014779086055498235,
    "reg_lambda": 8.093500291393644,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP22(LevelModel):
    """DEAP fine-articulation level predictors — baseline (all data, MAE-tuned).

    Uses the full :attr:`Predictors.DEFAULT_LEVEL` predictor set
    (minus the target ``deappfi``) with MAE-tuned hyperparameters
    and no outlier exclusion. Serves as the starting point for
    feature-selection work on the deappfi level-prediction task.
    """

    model_id = "lrp22"
    target_var = V.DEAPPFI
    description = (
        "LightGBM — DEAP fine-articulation level predictors "
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
        "Baseline exploratory model for DEAP fine-articulation "
        "level (deappfi). Uses the full default level predictor set "
        "(minus the target) without outlier exclusion, and MAE-tuned "
        "params from an Optuna 150-trial study — no feature "
        "selection has been applied yet. Target is left-skewed with "
        "ceiling effects (skew −0.87) on a 0-100 percentage scale. "
        "First articulation-domain target in the suite (DEAP used "
        "only as predictor in LRP01-LRP20)."
    )
