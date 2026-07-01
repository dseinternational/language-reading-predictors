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


# MAE-tuned (Optuna 150-trial, seed 47) on the earlier pruned selected set;
# retained as the full-set baseline (retune-pending).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 65,
    "learning_rate": 0.043738091375742166,
    "num_leaves": 44,
    "max_depth": 4,
    "min_child_samples": 37,
    "subsample": 0.9853192989415035,
    "subsample_freq": 1,
    "colsample_bytree": 0.9140199440486373,
    "reg_alpha": 0.06457861635292404,
    "reg_lambda": 0.013949746526542768,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL14(LevelModel):
    """CELF basic concept knowledge level predictors — exploratory (MAE-tuned, all data).

    Full ``Predictors.DEFAULT_LEVEL`` set, MAE-tuned (params retune-pending).
    """

    model_id = "lrpgbl14"
    target_var = V.CELF
    description = (
        "LightGBM — CELF (basic concept knowledge) level predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for basic concept knowledge level (celf). Fits the "
        "full DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature "
        "selection in favour of full-set ranking); hyperparameters are "
        "retained from the earlier pruned-set Optuna tune (retune-pending). "
        "CELF here is a lexical/semantic concept measure, not grammar. Treat "
        "the ranking as exploratory."
    )
