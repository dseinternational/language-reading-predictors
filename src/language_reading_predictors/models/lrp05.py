# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP05: Predictors of letter-sound knowledge gains.

``LRP05`` is the exploratory model for letter-sound knowledge gains
(``yarclet_gain``). It is MAE-tuned on the full 34-predictor
:attr:`Predictors.DEFAULT_GAIN` set (plus the auto-included base
variable ``yarclet``), with no outlier exclusion, designed to
identify the most important influences on letter-sound knowledge
gains.

The target is signed with a mild right tail (``yarclet_gain`` min ≈
−17, max ≈ 24, median 2, skewness 0.45, with ~22% negative and ~12%
zero observations, n ≈ 160). Similar shape to ``eowpvt_gain``
(LRP03) and milder than ``ewrswr_gain`` (LRP01).

The predictor set will be reduced by iterative importance-based
feature selection under the MAE-tuned params (see
``notes/202604171421-lrp05-feature-selection.md``). This is the
initial tuned baseline; no feature-selection steps yet.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP05 has not yet been through iterative feature selection. When
# selection variants are introduced, record their rationale here as
# ``SelectionStep`` entries and chain from ``LRP05`` the same way
# ``lrp01.py`` / ``lrp03.py`` does.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the full 34-predictor set (DEFAULT_GAIN + yarclet), no
# outlier exclusion (Optuna 150 trials, 10-split GroupKFold, seed 47,
# scoring=mae, lgbm_objective=mae). Tuner-inner CV MAE 3.3963 ±
# 0.6532. n=160.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 178,
    "learning_rate": 0.016929338277499147,
    "num_leaves": 43,
    "max_depth": 12,
    "min_child_samples": 13,
    "subsample": 0.723464871420595,
    "subsample_freq": 1,
    "colsample_bytree": 0.8693009164664598,
    "reg_alpha": 0.003920651895404464,
    "reg_lambda": 0.035232577401592906,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRP05(GainModel):
    """Letter-sound knowledge gain predictors — exploratory (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_GAIN` predictor set plus
    the base variable ``yarclet`` (auto-included via :class:`GainModel`)
    with MAE-tuned hyperparameters and no outlier exclusion. The
    starting point for feature selection on the letter-sound
    knowledge gain-prediction task.
    """

    model_id = "lrp05"
    target_var = V.YARCLET_GAIN
    description = (
        "LightGBM — letter-sound knowledge gain predictors "
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
        "Exploratory model for identifying important predictors of "
        "letter-sound knowledge gains (yarclet_gain). MAE-tuned on "
        "the full 34-predictor set (DEFAULT_GAIN + yarclet) without "
        "outlier exclusion so importance rankings reflect the full "
        "range of outcomes. Feature-selection variants to follow. "
        "See notes/202604171421-lrp05-feature-selection.md."
    )
