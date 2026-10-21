# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Documentation edited by a LLM-based AI tool (Codex/GPT-6).

"""
lrp-rli-gbg-009: Predictors of letter-sound knowledge gains.

``LRPGBG09`` is the exploratory model for letter-sound knowledge gains
(``yarclet_gain``). It is Huber-tuned on the full
:attr:`Predictors.DEFAULT_GAIN` set (plus the auto-included base
variable ``yarclet``), with no outlier exclusion, used to rank predictors of
letter-sound knowledge gains.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter sets ─────────────────────────────────────────────────

# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 3.88.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 5.982290999999999,
    "n_estimators": 181,
    "learning_rate": 0.0147576273215656,
    "num_leaves": 51,
    "max_depth": 10,
    "min_child_samples": 13,
    "subsample": 0.8388537605803461,
    "subsample_freq": 1,
    "colsample_bytree": 0.9710276594976839,
    "reg_alpha": 0.1573664318194167,
    "reg_lambda": 0.07280758683932663,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, Huber-tuned) ──────────────────────────────


class LRPGBG09(GainModel):
    """Letter-sound knowledge gain predictors — exploratory (Huber-tuned, all data).

    Full ``Predictors.DEFAULT_GAIN`` set, Huber-tuned on the full
    set. Uses the full predictor set plus the base variable
    ``yarclet`` (auto-included via :class:`GainModel`) with no outlier
    exclusion.
    """

    model_id = "lrp-rli-gbg-009"
    target_var = V.YARCLET_GAIN
    description = (
        "LightGBM — letter-sound knowledge gain predictors "
        "(full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory gain model for yarclet_gain using DEFAULT_GAIN and its own baseline. "
        "Huber parameters come from the 22 September 2026 retune. Read predictive "
        "performance and ranking stability from the fitted outputs; baseline "
        "associations can reflect score limits or regression to the mean."
    )
