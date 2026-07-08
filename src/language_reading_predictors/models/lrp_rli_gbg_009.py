# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG09: Predictors of letter-sound knowledge gains.

``LRPGBG09`` is the exploratory model for letter-sound knowledge gains
(``yarclet_gain``). It is MAE-tuned on the full 34-predictor
:attr:`Predictors.DEFAULT_GAIN` set (plus the auto-included base
variable ``yarclet``), with no outlier exclusion, designed to
identify the most important influences on letter-sound knowledge
gains.

The target is signed with a mild right tail (``yarclet_gain`` min ≈
−17, max ≈ 24, median 2, skewness 0.45, with ~22% negative and ~12%
zero observations, n ≈ 160). Similar shape to ``eowpvt_gain``
(LRPGBG06) and milder than ``ewrswr_gain`` (LRPGBG12).

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
    "n_estimators": 184,
    "learning_rate": 0.01699899204903742,
    "num_leaves": 60,
    "max_depth": 11,
    "min_child_samples": 12,
    "subsample": 0.8528409409308849,
    "subsample_freq": 1,
    "colsample_bytree": 0.9583730472706107,
    "reg_alpha": 0.0037508752320932538,
    "reg_lambda": 0.020801203155819466,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRPGBG09(GainModel):
    """Letter-sound knowledge gain predictors — exploratory (MAE-tuned, all data).

    Full ``Predictors.DEFAULT_GAIN`` set, MAE-tuned on the full
    set (#169). Uses the full predictor set plus the base variable
    ``yarclet`` (auto-included via :class:`GainModel`) with no outlier
    exclusion.
    """

    model_id = "lrp-rli-gbg-009"
    target_var = V.YARCLET_GAIN
    description = (
        "LightGBM — letter-sound knowledge gain predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for yarclet_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; #169). Gain models are near-noise (baseline-driven regression to the mean) - treat the ranking as exploratory."
    )
