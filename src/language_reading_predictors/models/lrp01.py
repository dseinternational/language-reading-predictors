# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP01: Predictors of word reading gains.

``LRP01`` is the base model using all default gain predictors.
``LRP01Select01`` is the first selection variant with Optuna-tuned
hyperparameters. Additional variants subclass the appropriate ancestor
to chain feature-selection decisions.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline
from language_reading_predictors.models.registry import DEFAULT_LGBM_PARAMS


class LRP01(GainModel):
    """Word-reading gain predictors — all default gain predictors."""

    model_id = "lrp01"
    target_var = V.EWRSWR_GAIN
    description = "LightGBM — word-reading gain predictors (all predictors)"
    include = [V.EWRSWR]
    pipeline_cls = LGBMPipeline
    params = DEFAULT_LGBM_PARAMS
    cv_splits = 53
    outlier_threshold = 15.0


# Tuned LightGBM hyperparameters from Optuna (30 trials, 10-split GroupKFold).
# Early stopping uses an inner GroupShuffleSplit slice of each training fold —
# the outer val fold is never shown to `early_stopping`, so the reported CV
# RMSE and `best_iteration_` are independent. Tuning runs on raw NaN (no
# mean-impute) to match the fit pipeline. Best trial #14, inner CV RMSE
# 3.3145 ± 0.5423. Mean best iteration 83 replaces the default 1200 — the
# untuned LGBM was massively over-training at n=152.
#
# RandomForest variant retired 2026-04-12 after timing showed tuned LGBM was
# ~30× faster end-to-end at equivalent CV RMSE. See
# notes/202604121234-lightgbm-rf-comparison-tuning.md and
# notes/202604121451-lightgbm-model-selection.md for the full history.
_LGBM_TUNED_PARAMS: dict[str, float | int] = {
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

_PDP_FEATURES = [
    V.AGE,
    V.ATTEND,
    V.YARCLET,
    V.CELF,
    V.TROG,
    V.EOWPVT,
    V.B1EXTO,
    V.BLENDING,
    V.TIME,
    V.NONWORD,
    V.B1RETO,
    V.B2RETO,
    V.APTINFO,
    V.DEAPPIN,
    V.APTGRAM,
    V.ROWPVT,
    V.BEHAV,
    V.B2EXTO,
    V.VISION,
]


class LRP01Select01(LRP01):
    """Tuned variant with Optuna hyperparameters — same predictor set."""

    model_id = "lrp01_select01"
    variant_of = "lrp01"
    description = "LightGBM — word-reading gain predictors (outliers excluded)"
    params = _LGBM_TUNED_PARAMS
    pdp_features = _PDP_FEATURES
    selection_steps = [
        SelectionStep(
            notes=(
                "Final accepted model. LightGBM with Optuna-tuned hyperparameters "
                "(tuning round 1, inner CV RMSE 3.3145 ± 0.5423). See notes/ for "
                "the RF→LGBM consolidation history."
            ),
            date="2026-04-12",
            metrics_before={"cv_rmse_mean": 3.45, "cv_rmse_std": 0.52},
            metrics_after={"cv_rmse_mean": 3.31, "cv_rmse_std": 0.54},
        ),
    ]
