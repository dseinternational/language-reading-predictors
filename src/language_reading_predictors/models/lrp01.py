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

_gain_model(
    "lrp01_select01",
    V.EWRSWR_GAIN,
    description="lrp01 selection round 1 — baseline Predictors.DEFAULT_GAIN",
    include=[V.EWRSWR],
    cv_splits=53,
    outlier_threshold=15.0,
    pdp_features=_PDP_FEATURES,
    variant_of="lrp01",
    notes=(
        "Baseline selection round. No drops — establishes the champion "
        "permutation-importance ranking used to guide later rounds."
    ),
)

# Tuned LightGBM hyperparameters from Optuna (30 trials, 10-split GroupKFold).
# Early stopping uses an inner GroupShuffleSplit slice of each training fold —
# the outer val fold is never shown to `early_stopping`, so the reported CV
# RMSE and `best_iteration_` are independent. See output/tuning/lrp01_lgbm/.
_LGBM_TUNED_SELECT01: dict[str, float | int] = {
    "n_estimators": 21,
    "learning_rate": 0.19927560307078432,
    "num_leaves": 46,
    "max_depth": 12,
    "min_child_samples": 33,
    "subsample": 0.9056186653894883,
    "subsample_freq": 1,
    "colsample_bytree": 0.6737858250693828,
    "reg_alpha": 0.038234494308696715,
    "reg_lambda": 0.009546754601000738,
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
        "fold never leaks into `best_iteration_`. Best trial #12, inner CV "
        "RMSE 3.2797 ± 0.5703. Mean best iteration 21 replaces the default "
        "1200 — the untuned LGBM was massively over-training at n=152."
    ),
)
