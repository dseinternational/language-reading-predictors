# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP01 — word-reading gain predictors.

Holds the final ``lrp01`` model, its LightGBM sibling ``lrp01_lgbm``, and
any historical selection variants (``lrp01_select01``, ``lrp01_select02``,
...). Variants carry ``variant_of="lrp01"`` so that ``fit_model.py all``
skips them unless ``--include-variants`` is passed.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline
from language_reading_predictors.models.registry import (
    DEFAULT_LGBM_PARAMS,
    _gain_model,
)

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


# ── final model ─────────────────────────────────────────────────────────

_gain_model(
    "lrp01",
    V.EWRSWR_GAIN,
    description="Random Forest — word-reading gain predictors (outliers excluded)",
    include=[V.EWRSWR],
    cv_splits=53,
    outlier_threshold=15.0,
    pdp_features=_PDP_FEATURES,
    notes="Final accepted model. See 'Feature selection history' for variants.",
)

# ── LightGBM sibling ────────────────────────────────────────────────────

_gain_model(
    "lrp01_lgbm",
    V.EWRSWR_GAIN,
    description="LightGBM — word-reading gain predictors (outliers excluded)",
    include=[V.EWRSWR],
    cv_splits=53,
    outlier_threshold=15.0,
    pdp_features=_PDP_FEATURES,
    pipeline_cls=LGBMPipeline,
    params=DEFAULT_LGBM_PARAMS,
)

# ── selection variants (historical) ─────────────────────────────────────

# Tuned RandomForest hyperparameters from Optuna (30 trials, 10-split
# GroupKFold, seed 47, raw NaN to match the fit pipeline). Best trial #22,
# inner CV RMSE 3.3678 ± 0.5601. See output/tuning/lrp01/ for the full study.
_RF_TUNED_SELECT01: dict[str, float | int | bool | str] = {
    "n_estimators": 1800,
    "max_depth": 11,
    "min_samples_leaf": 16,
    "min_samples_split": 4,
    "max_features": 0.37549488805461834,
    "bootstrap": False,
    "criterion": "squared_error",
    "n_jobs": 16,
}

_gain_model(
    "lrp01_select01",
    V.EWRSWR_GAIN,
    description="RandomForest — lrp01 tuning round 1 (Optuna TPE, 30 trials)",
    include=[V.EWRSWR],
    cv_splits=53,
    outlier_threshold=15.0,
    pdp_features=_PDP_FEATURES,
    params=_RF_TUNED_SELECT01,
    variant_of="lrp01",
    notes=(
        "Optuna TPE hyperparameter tuning against 10-split GroupKFold on "
        "raw NaN (sklearn RF handles missingness natively since 1.4). Best "
        "trial #22, inner CV RMSE 3.3678 ± 0.5601. The search prefers a "
        "deeper-but-leaf-constrained forest (max_depth=11, "
        "min_samples_leaf=16, max_features≈0.38, bootstrap=False, "
        "n_estimators=1800) over the scikit-learn defaults."
    ),
)

# Tuned LightGBM hyperparameters from Optuna (30 trials, 10-split GroupKFold).
# Early stopping uses an inner GroupShuffleSplit slice of each training fold —
# the outer val fold is never shown to `early_stopping`, so the reported CV
# RMSE and `best_iteration_` are independent. Tuning runs on raw NaN (no
# mean-impute) to match the fit pipeline. See output/tuning/lrp01_lgbm/.
_LGBM_TUNED_SELECT01: dict[str, float | int] = {
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

_gain_model(
    "lrp01_lgbm_select01",
    V.EWRSWR_GAIN,
    description="LightGBM — lrp01 tuning round 1 (Optuna TPE, 30 trials)",
    include=[V.EWRSWR],
    cv_splits=53,
    outlier_threshold=15.0,
    pdp_features=_PDP_FEATURES,
    pipeline_cls=LGBMPipeline,
    params=_LGBM_TUNED_SELECT01,
    variant_of="lrp01_lgbm",
    notes=(
        "Optuna TPE hyperparameter tuning against 10-split GroupKFold. Early "
        "stopping uses an inner GroupShuffleSplit slice of each training fold "
        "(20% of groups, 50 rounds patience, ceiling 2000) so the outer val "
        "fold never leaks into `best_iteration_`. Tuning runs on raw NaN "
        "(no mean-impute) to match the fit pipeline. Best trial #14, inner "
        "CV RMSE 3.3145 ± 0.5423. Mean best iteration 83 replaces the "
        "default 1200 — the untuned LGBM was massively over-training at n=152."
    ),
)
