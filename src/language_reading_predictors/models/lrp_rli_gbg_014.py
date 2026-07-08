# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG14: Predictors of basic concept knowledge gains (CELF).

``LRPGBG14`` is the exploratory model for basic concept knowledge
gains (``celf_gain``). The ``celf`` score is drawn from the
Clinical Evaluation of Language Fundamentals Preschool 2nd Ed
(Wiig, Secord & Semel 2006) and in this study only the basic-
concept-knowledge subtest (18 linguistic concepts) was
administered — so ``celf`` is a lexical/semantic concept measure,
NOT a grammar measure (the grammar measures in this study are
``trog`` for receptive grammar and ``aptgram`` for expressive
grammar).

LRPGBG14 is MAE-tuned on the full 34-predictor
:attr:`Predictors.DEFAULT_GAIN` set (which already includes
``celf`` as a level predictor — the GainModel's auto-include is a
no-op here), with no outlier exclusion, designed to identify the
most important influences on basic concept knowledge gains.

The target is **mildly right-skewed** (``celf_gain`` min ≈ −8,
max ≈ 10, median 1, mean 1.14, std 3.20, skewness 0.14, with ~26%
negative and ~17% zero observations, n ≈ 160). The zero pile-up
is heavier than in LRPGBG05 / LRPGBG09 gains (17% vs 3%/12%) —
consistent with the coarser 0-18 CELF raw-score scale.

Fits the full ``Predictors.DEFAULT_GAIN`` set; hyperparameters are
re-tuned by Optuna on the full set (150 trials, seed 47; #169).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned by Optuna on the full predictor set (150 trials, seed 47;
# #169 retune, superseding the earlier pruned-set tune).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 27,
    "learning_rate": 0.1990927990559259,
    "num_leaves": 33,
    "max_depth": 12,
    "min_child_samples": 10,
    "subsample": 0.7250568376375872,
    "subsample_freq": 1,
    "colsample_bytree": 0.851649578232224,
    "reg_alpha": 0.007458595069008448,
    "reg_lambda": 1.1204774780118514,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRPGBG14(GainModel):
    """CELF basic concept knowledge gain predictors — exploratory (MAE-tuned, all data).

    Full ``Predictors.DEFAULT_GAIN`` set, MAE-tuned on the full set (#169).
    """

    model_id = "lrp-rli-gbg-014"
    target_var = V.CELF_GAIN
    description = (
        "LightGBM — CELF (basic concept knowledge) gain predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for celf_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; #169). Gain models are near-noise (baseline-driven regression to the mean) - treat the ranking as exploratory."
    )
