# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP18: Predictors of expressive-grammar (APT) level.

``LRP18`` is the baseline exploratory model for expressive-grammar
level (``aptgram``). ``aptgram`` is the grammar raw score from the
Action Picture Test (Renfrew, 1997) — the child is shown pictures
and asked to describe them, with scoring of the grammatical
structure of the response.

The target is **right-skewed** (``aptgram`` min 0, max 28,
median 6, mean 7.63, std 6.34, skewness 1.23, with ~9% at zero,
n ≈ 211) — comparable in skew magnitude to LRP02's ``ewrswr``
baseline, and a heavier floor than the receptive-grammar target
``trog`` (LRP12, skew 0.29).

``aptgram`` is the expressive-grammar parallel to ``trog``
(LRP12 receptive grammar) — the pair addresses the expressive vs
receptive grammar asymmetry that is a live question in DS
language research. The right-skew motivates a later log-transform
variant (mirroring LRP02's ``lrp02_log``).

No feature selection has been run for LRP18 yet — the MAE-tuned
params below are the starting point for later feature-selection
variants.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP18 has not yet been through iterative feature selection.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the full 32-predictor set (DEFAULT_LEVEL minus aptgram),
# no outlier exclusion (Optuna 150 trials, 10-split GroupKFold,
# seed 47, scoring=mae, lgbm_objective=mae). Tuner-inner CV MAE
# 2.5698 ± 0.7605. n=211.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 161,
    "learning_rate": 0.02782903247800992,
    "num_leaves": 55,
    "max_depth": 3,
    "min_child_samples": 6,
    "subsample": 0.6236226731956364,
    "subsample_freq": 1,
    "colsample_bytree": 0.622990613640347,
    "reg_alpha": 0.004433904771755985,
    "reg_lambda": 0.0010353549751789706,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP18(LevelModel):
    """APT expressive-grammar level predictors — baseline (all data, MAE-tuned).

    Uses the full :attr:`Predictors.DEFAULT_LEVEL` predictor set
    (minus the target ``aptgram``) with MAE-tuned hyperparameters
    and no outlier exclusion. Serves as the starting point for
    feature-selection work on the aptgram level-prediction task.
    """

    model_id = "lrp18"
    target_var = V.APTGRAM
    description = (
        "LightGBM — APT expressive-grammar level predictors "
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
        "Baseline exploratory model for APT expressive-grammar "
        "level (aptgram). Uses the full default level predictor set "
        "(minus the target) without outlier exclusion, and MAE-tuned "
        "params from an Optuna 150-trial study — no feature "
        "selection has been applied yet. Target is right-skewed "
        "(skew 1.23) on a 0-28 scale; a log-transform variant is a "
        "natural follow-up (mirroring LRP02's lrp02_log). Pair "
        "partner to LRP12 (trog, receptive grammar)."
    )
