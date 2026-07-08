# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG06: Predictors of expressive-vocabulary gains.

``LRPGBG06`` is the exploratory model for expressive-vocabulary gains
(``eowpvt_gain``). It is MAE-tuned on the full 34-predictor
:attr:`Predictors.DEFAULT_GAIN` set (plus the auto-included base
variable ``eowpvt``), with no outlier exclusion, designed to
identify the most important influences on expressive-vocabulary
gains.

The target is signed (``eowpvt_gain`` min ≈ −13, max ≈ 28, median 3,
skewness 0.32, with ~25% negative observations and n ≈ 161). That's
much milder skew than LRPGBG12's ``ewrswr_gain`` and nearly symmetric —
a log / signed-log transform may or may not help and is a question
for future investigation.

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
    "n_estimators": 192,
    "learning_rate": 0.010015643165526266,
    "num_leaves": 52,
    "max_depth": 6,
    "min_child_samples": 6,
    "subsample": 0.8255230681355167,
    "subsample_freq": 1,
    "colsample_bytree": 0.8439752145892472,
    "reg_alpha": 0.008190665978651132,
    "reg_lambda": 0.24072005148863818,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRPGBG06(GainModel):
    """Expressive-vocabulary gain predictors — exploratory (MAE-tuned, all data).

    Full ``Predictors.DEFAULT_GAIN`` set, MAE-tuned on the full
    set (#169). Uses the full predictor set plus the base variable
    ``eowpvt`` (auto-included via :class:`GainModel`) with no outlier
    exclusion.
    """

    model_id = "lrp-rli-gbg-006"
    target_var = V.EOWPVT_GAIN
    description = (
        "LightGBM — expressive-vocabulary gain predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for eowpvt_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; #169). Gain models are near-noise (baseline-driven regression to the mean) - treat the ranking as exploratory."
    )
