# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL14: Predictors of basic concept knowledge level (CELF).

``LRPGBL14`` is the exploratory model for basic concept knowledge level
(``celf``). The ``celf`` score is drawn from the Clinical Evaluation of
Language Fundamentals Preschool 2nd Ed (Wiig, Secord & Semel 2006); in
this study only the basic-concept-knowledge subtest (18 linguistic
concepts) was administered — so ``celf`` is a lexical/semantic concept
measure, NOT a grammar measure (grammar is covered by ``trog`` for
receptive and ``aptgram`` for expressive grammar).

The target is **mildly left-skewed** (``celf`` min 0, max 18, median 11,
mean 10.88, std 4.24, skewness −0.37, n ≈ 214). The max of 18 is the
instrument maximum but the 95th percentile is below it, so there is no
strong ceiling pathology.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 2.81.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 5.982290999999999,
    "n_estimators": 72,
    "learning_rate": 0.026970478314762396,
    "num_leaves": 14,
    "max_depth": 9,
    "min_child_samples": 22,
    "subsample": 0.6631215463960534,
    "subsample_freq": 1,
    "colsample_bytree": 0.6259460182587492,
    "reg_alpha": 0.09700354052715723,
    "reg_lambda": 0.015087909782796149,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL14(LevelModel):
    """CELF basic concept knowledge level predictors — exploratory (Huber-tuned, all data).

    Full ``Predictors.DEFAULT_LEVEL`` set, Huber-tuned on the full set (#169).
    """

    model_id = "lrp-rli-gbl-014"
    target_var = V.CELF
    description = (
        "LightGBM — CELF (basic concept knowledge) level predictors "
        "(full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for basic concept knowledge level (celf). Fits the "
        "full DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature "
        "selection in favour of full-set ranking); hyperparameters are "
        "re-tuned by Optuna on the full set (150 trials, seed 47; Huber retune of 2026-09-22, superseding #169). "
        "CELF here is a lexical/semantic concept measure, not grammar. Treat "
        "the ranking as exploratory."
    )
