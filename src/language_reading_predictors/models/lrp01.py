# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP01: Predictors of word reading gains.

``LRP01`` is the primary (exploratory) model — MAE-tuned with no outlier
exclusion — designed to identify the most important influences on reading
gains across the full range of outcomes. ``LRP01Prediction`` is the
prediction-focused variant with outlier exclusion and RMSE-tuned params.

Both share the same 6-predictor set selected via iterative importance-based
feature selection under an MAE objective
(see ``notes/202604161432-lrp01-feature-selection-mae.md``).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Documents the 34 → 6 feature-selection history under MAE-tuned params
# with no outlier exclusion (n=157).

_SELECTION_STEPS = [
    SelectionStep(
        removed=[
            V.GROUP, V.APTGRAM, V.EARINF, V.SPPHON, V.ERBWORD, V.ROWPVT,
            V.ERBNW, V.AREA, V.B1RETO, V.NONWORD, V.NUMCHIL, V.BEHAV,
            V.DEAPPIN, V.YARCSI, V.VISION,
            V.DEAPPVO, V.APTINFO, V.MUMEDUPOST16,
        ],
        notes=(
            "Remove 18 features with zero or negative permutation importance "
            "in the MAE-tuned baseline (dev config, 5-fold CV, n=157)."
        ),
        date="2026-04-16",
        metrics_before={"cv_mae_mean": 2.996},
        metrics_after={"cv_mae_mean": 2.967},
    ),
    SelectionStep(
        removed=[V.DEAPPFI, V.AGESPEAK, V.TIME, V.EWRSWR, V.AGEBOOKS],
        notes=(
            "Remove 5 features with importance <= 0.0003 or negative: "
            "deappfi, agespeak, time, ewrswr, agebooks."
        ),
        date="2026-04-16",
        metrics_before={"cv_mae_mean": 2.967},
        metrics_after={"cv_mae_mean": 2.954},
    ),
    SelectionStep(
        removed=[V.DADEDUPOST16, V.GENDER],
        notes="Remove dadedupost16 (0.002) and gender (0.001).",
        date="2026-04-16",
        metrics_before={"cv_mae_mean": 2.954},
        metrics_after={"cv_mae_mean": 2.939},
    ),
    SelectionStep(
        removed=[V.TROG],
        notes="Remove trog (0.004, rank 9/9).",
        date="2026-04-16",
        metrics_before={"cv_mae_mean": 2.939},
        metrics_after={"cv_mae_mean": 2.920},
    ),
    SelectionStep(
        removed=[V.EOWPVT],
        notes="Remove eowpvt (0.004). b1exto and celf cover language.",
        date="2026-04-16",
        metrics_before={"cv_mae_mean": 2.920},
        metrics_after={"cv_mae_mean": 2.899},
    ),
    SelectionStep(
        removed=[V.HEARING],
        notes="Remove hearing (0.006, rank 7/7).",
        date="2026-04-16",
        metrics_before={"cv_mae_mean": 2.899},
        metrics_after={"cv_mae_mean": 2.893},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on 6 predictors, no outlier exclusion (Optuna 50 trials,
# 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Best trial #42, CV MAE 2.8394 ± 0.8535. n=157.
_LGBM_MAE_PARAMS: dict[str, float | int] = {
    "objective": "mae",
    "n_estimators": 118,
    "learning_rate": 0.020503143245054957,
    "num_leaves": 13,
    "max_depth": 11,
    "min_child_samples": 30,
    "subsample": 0.9479940166756426,
    "subsample_freq": 1,
    "colsample_bytree": 0.6024743488626004,
    "reg_alpha": 0.0375788355410621,
    "reg_lambda": 0.0037690200347546363,
    "n_jobs": 16,
    "verbosity": -1,
}

# RMSE-tuned on 6 predictors, no outlier exclusion (Optuna 50 trials,
# 10-split GroupKFold, seed 47, scoring=rmse, lgbm_objective=regression).
# Best trial #36, CV RMSE 3.8126 ± 0.9152. n=157.
_LGBM_RMSE_PARAMS: dict[str, float | int] = {
    "objective": "regression",
    "n_estimators": 50,
    "learning_rate": 0.09226000258126973,
    "num_leaves": 30,
    "max_depth": 9,
    "min_child_samples": 29,
    "subsample": 0.6347769811026076,
    "subsample_freq": 1,
    "colsample_bytree": 0.7232328649750049,
    "reg_alpha": 0.01764013107887275,
    "reg_lambda": 0.07888648720556594,
    "n_jobs": 16,
    "verbosity": -1,
}


# ── primary model (exploratory) ─────────────────────────────────────────


class LRP01(GainModel):
    """Word-reading gain predictors — exploratory model (MAE-tuned, all data).

    This is the primary model for identifying which predictors influence
    word-reading gains. It uses MAE-tuned hyperparameters with no outlier
    exclusion so that the full range of outcomes — including the children
    who made the largest gains — informs the importance rankings.

    For prediction accuracy on typical cases, use ``LRP01Prediction``.
    """

    model_id = "lrp01"
    target_var = V.EWRSWR_GAIN
    description = (
        "LightGBM — word-reading gain predictors "
        "(6 predictors, MAE-tuned, no outlier exclusion)"
    )
    include = [V.EWRSWR]
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 53
    outlier_threshold = None
    selection_steps = _SELECTION_STEPS
    notes = (
        "Exploratory model for identifying important predictors of reading "
        "gains. Uses MAE objective and no outlier exclusion so that importance "
        "rankings reflect the full range of outcomes. See "
        "notes/202604161432-lrp01-feature-selection-mae.md."
    )


# ── prediction variant ──────────────────────────────────────────────────


class LRP01Prediction(GainModel):
    """Word-reading gain predictors — prediction-focused (RMSE-tuned, all data).

    Optimised for prediction accuracy. Uses RMSE-tuned hyperparameters with
    the same 6-predictor set and no outlier exclusion. Same data as the
    exploratory model.
    """

    model_id = "lrp01_prediction"
    variant_of = "lrp01"
    target_var = V.EWRSWR_GAIN
    description = (
        "LightGBM — word-reading gain predictors "
        "(6 predictors, RMSE-tuned)"
    )
    include = [V.EWRSWR]
    pipeline_cls = LGBMPipeline
    params = _LGBM_RMSE_PARAMS
    cv_splits = 53
    outlier_threshold = None
    selection_steps = _SELECTION_STEPS
    notes = (
        "Prediction-focused variant. Same 6 predictors and data as the "
        "exploratory model, but RMSE-tuned for best prediction accuracy. "
        "CV RMSE 3.8126 (inner tuning)."
    )
