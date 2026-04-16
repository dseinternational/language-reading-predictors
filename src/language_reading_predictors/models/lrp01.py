# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP01: Predictors of word reading gains.

``LRP01`` is the primary (exploratory) model — MAE-tuned with no outlier
exclusion — designed to identify the most important influences on reading
gains across the full range of outcomes. ``LRP01Prediction`` is the
prediction-focused variant with outlier exclusion and RMSE-tuned params.

Both share the same 15-predictor set selected via iterative importance-based
feature selection (see ``notes/202604161333-lrp01-feature-selection.md``).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# These document the 34 → 15 feature-selection history. The steps are
# defined once and referenced by both the exploratory and prediction models.

_SELECTION_STEPS = [
    # Select01: drop 6 zero-importance features (34 → 28)
    SelectionStep(
        removed=[V.GROUP, V.APTGRAM, V.SPPHON, V.AREA, V.BEHAV, V.VISION],
        notes=(
            "Remove 6 features with exactly zero permutation importance "
            "in the tuned baseline (dev config, 5-fold CV): group, aptgram, "
            "spphon, area, behav, vision."
        ),
        date="2026-04-16",
        metrics_before={"cv_rmse_mean": 3.302, "cv_r2_mean": 0.122},
        metrics_after={"cv_rmse_mean": 3.294, "cv_r2_mean": 0.129},
    ),
    # Select02: drop 6 low-importance features (28 → 22)
    SelectionStep(
        removed=[V.YARCSI, V.ROWPVT, V.EARINF, V.ERBNW, V.NONWORD, V.MUMEDUPOST16],
        notes=(
            "Remove 6 features with importance < 0.002 in select01 "
            "(dev config, 5-fold CV): yarcsi (0.000), rowpvt (0.001), "
            "earinf (0.001), erbnw (0.001), nonword (0.001), "
            "mumedupost16 (0.001)."
        ),
        date="2026-04-16",
        metrics_before={"cv_rmse_mean": 3.294, "cv_r2_mean": 0.129},
        metrics_after={"cv_rmse_mean": 3.272, "cv_r2_mean": 0.139},
    ),
    # Select03: drop 6 low-importance / cluster-redundant (22 → 16)
    SelectionStep(
        removed=[V.HEARING, V.DADEDUPOST16, V.B1RETO, V.EWRSWR, V.ERBWORD, V.NUMCHIL],
        notes=(
            "Remove 6 features: hearing (0.012, singleton), "
            "dadedupost16 (0.011, singleton), b1reto (0.018, redundant "
            "with eowpvt in cluster 3), ewrswr (0.017, redundant with "
            "yarclet in cluster 1), erbword (0.026, redundant with "
            "deappfi/deappin in cluster 9), numchil (0.020, singleton). "
            "Importance values from test config 10-fold CV."
        ),
        date="2026-04-16",
        metrics_before={"cv_rmse_mean": 3.272, "cv_r2_mean": 0.139},
        metrics_after={"cv_rmse_mean": 3.241, "cv_r2_mean": 0.157},
    ),
    # Select04: drop agespeak (16 → 15)
    SelectionStep(
        removed=[V.AGESPEAK],
        notes="Remove agespeak (importance 0.003, rank 16/16 in select03).",
        date="2026-04-16",
        metrics_before={"cv_rmse_mean": 3.241, "cv_r2_mean": 0.157},
        metrics_after={"cv_rmse_mean": 3.211, "cv_r2_mean": 0.172},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# RMSE-tuned on 34 predictors, outliers excluded (Optuna 50 trials,
# 10-split GroupKFold, seed 47). Best trial #14, CV RMSE 3.3145 ± 0.5423.
# Used during feature selection and retained for the prediction variant.
_LGBM_RMSE_TUNED_PARAMS: dict[str, float | int] = {
    "n_estimators": 83,
    "learning_rate": 0.061852240742933245,
    "num_leaves": 34,
    "max_depth": 12,
    "min_child_samples": 31,
    "subsample": 0.8360372539990215,
    "subsample_freq": 1,
    "colsample_bytree": 0.8786907349593146,
    "reg_alpha": 1.4894437395338633,
    "reg_lambda": 5.25756227415291,
    "n_jobs": 16,
    "verbosity": -1,
}

# MAE-tuned on 15 predictors, no outlier exclusion (Optuna 50 trials,
# 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Best trial #11, CV MAE 2.8476 ± 0.7610. n=157. Only 29 estimators —
# a conservative ensemble that does not chase outlier residuals.
_LGBM_MAE_TUNED_PARAMS: dict[str, float | int] = {
    "objective": "mae",
    "n_estimators": 29,
    "learning_rate": 0.19020993655265472,
    "num_leaves": 46,
    "max_depth": 12,
    "min_child_samples": 40,
    "subsample": 0.7337465509959111,
    "subsample_freq": 1,
    "colsample_bytree": 0.9969182249863432,
    "reg_alpha": 0.01030462958842614,
    "reg_lambda": 1.028398001881727,
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
        "(15 predictors, MAE-tuned, no outlier exclusion)"
    )
    include = [V.EWRSWR]
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_TUNED_PARAMS
    cv_splits = 53
    outlier_threshold = None
    selection_steps = _SELECTION_STEPS
    notes = (
        "Exploratory model for identifying important predictors of reading "
        "gains. Uses MAE objective and no outlier exclusion so that importance "
        "rankings reflect the full range of outcomes. See "
        "notes/202604161333-lrp01-feature-selection.md for the full history."
    )


# ── prediction variant ──────────────────────────────────────────────────


class LRP01Prediction(GainModel):
    """Word-reading gain predictors — prediction-focused (RMSE-tuned, outliers excluded).

    Optimised for prediction accuracy on typical cases. Excludes outlier
    gains (>= 15) and uses RMSE-tuned hyperparameters. Same 15-predictor
    set as the exploratory model.
    """

    model_id = "lrp01_prediction"
    variant_of = "lrp01"
    target_var = V.EWRSWR_GAIN
    description = (
        "LightGBM — word-reading gain predictors "
        "(15 predictors, RMSE-tuned, outliers excluded)"
    )
    include = [V.EWRSWR]
    pipeline_cls = LGBMPipeline
    params = _LGBM_RMSE_TUNED_PARAMS
    cv_splits = 53
    outlier_threshold = 15.0
    selection_steps = _SELECTION_STEPS
    notes = (
        "Prediction-focused variant. Excludes outlier gains >= 15 and uses "
        "RMSE-tuned hyperparameters for best prediction accuracy on typical "
        "cases. CV RMSE 3.211 (dev config)."
    )
