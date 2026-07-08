# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL09: Predictors of letter-sound knowledge level.

``LRPGBL09`` is the exploratory model for letter-sound knowledge level
(``yarclet``). It is MAE-tuned on the full 32-predictor
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

Fits the full ``Predictors.DEFAULT_LEVEL`` set; hyperparameters are retained
from the earlier pruned-set tune (retune-pending, #116 Phase D).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned (Optuna 150-trial, seed 47) on the earlier pruned selected set;
# retained as the full-set baseline (retune-pending).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 53,
    "learning_rate": 0.037371968457437586,
    "num_leaves": 29,
    "max_depth": 10,
    "min_child_samples": 8,
    "subsample": 0.7474456109857174,
    "subsample_freq": 1,
    "colsample_bytree": 0.9746583954334843,
    "reg_alpha": 1.6618715685357452,
    "reg_lambda": 1.2133803578125177,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRPGBL09(LevelModel):
    """Letter-sound knowledge level predictors — exploratory (MAE-tuned, all data).

    Full ``Predictors.DEFAULT_LEVEL`` set, MAE-tuned (params retune-pending).
    """

    model_id = "lrp-rli-gbl-009"
    target_var = V.YARCLET
    description = (
        "LightGBM — letter-sound knowledge level predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for yarclet (level). Fits the full DEFAULT_LEVEL "
        "predictor set (#116 Phase D retired hard feature selection in favour "
        "of full-set ranking); hyperparameters are retained from the earlier "
        "pruned-set Optuna tune (retune-pending). Treat the ranking as "
        "exploratory."
    )
