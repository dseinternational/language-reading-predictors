# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL09: Predictors of letter-sound knowledge level.

``LRPGBL09`` is the exploratory model for letter-sound knowledge level
(``yarclet``). It is Huber-tuned on the full 32-predictor
:attr:`Predictors.DEFAULT_LEVEL` set (minus the target), with no
outlier exclusion, designed to identify the most important
influences on letter-sound knowledge level.

The target is **left-skewed with a ceiling at 32** (``yarclet`` min
0, max 32, median 21, skewness −0.60, n ≈ 214). The ceiling effect
(95th percentile = 31, 99th = 32) means many children score at or
near the instrument maximum — a different pathology from the
right-skewed / floor-at-0 targets of LRPGBL12 and LRPGBL06. Log / log1p
transforms are inappropriate here because the skew is in the wrong
direction; a reflection-log or quantile objective might be
considered later.

Fits the full ``Predictors.DEFAULT_LEVEL`` set; hyperparameters were re-tuned
by Optuna on the full set (150 trials, seed 47; Huber retune of 2026-09-22, superseding #169).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter sets ─────────────────────────────────────────────────

# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 5.27.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 11.964581999999998,
    "n_estimators": 66,
    "learning_rate": 0.03288220867605954,
    "num_leaves": 60,
    "max_depth": 12,
    "min_child_samples": 39,
    "subsample": 0.9815485823826032,
    "subsample_freq": 1,
    "colsample_bytree": 0.6622798324217657,
    "reg_alpha": 0.009742388984108551,
    "reg_lambda": 0.7856175922673118,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, Huber-tuned) ──────────────────────────────


class LRPGBL09(LevelModel):
    """Letter-sound knowledge level predictors — exploratory (Huber-tuned, all data).

    Full ``Predictors.DEFAULT_LEVEL`` set, Huber-tuned on the full set (#169).
    """

    model_id = "lrp-rli-gbl-009"
    target_var = V.YARCLET
    description = (
        "LightGBM — letter-sound knowledge level predictors "
        "(full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for yarclet (level). Fits the full DEFAULT_LEVEL "
        "predictor set (#116 Phase D retired hard feature selection in favour "
        "of full-set ranking); hyperparameters are re-tuned by Optuna on the full set "
        "(150 trials, seed 47; Huber retune of 2026-09-22, superseding #169). Treat the ranking as "
        "exploratory."
    )
