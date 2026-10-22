# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Documentation edited by a LLM-based AI tool (Codex/GPT-6).

"""
lrp-rli-gbg-005: Predictors of receptive vocabulary gains.

``LRPGBG05`` is the exploratory model for receptive vocabulary gains
(``rowpvt_gain``). It is Huber-tuned on the full
:attr:`Predictors.DEFAULT_GAIN` set (with the ``rowpvt`` baseline
auto-included) and no outlier exclusion, designed to rank predictors of receptive vocabulary gains.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter sets ─────────────────────────────────────────────────

# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 8.01.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 9.970484999999998,
    "n_estimators": 66,
    "learning_rate": 0.15484591361429514,
    "num_leaves": 48,
    "max_depth": 6,
    "min_child_samples": 39,
    "subsample": 0.6625127808402785,
    "subsample_freq": 1,
    "colsample_bytree": 0.6310391246947331,
    "reg_alpha": 0.35731706070703834,
    "reg_lambda": 4.907301983701988,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, Huber-tuned) ──────────────────────────────


class LRPGBG05(GainModel):
    """Receptive vocabulary gain predictors — exploratory (Huber-tuned, all data).

    Full ``Predictors.DEFAULT_GAIN`` set, Huber-tuned on the full
    set. Uses the full predictor set plus the base variable
    ``rowpvt`` (auto-included via :class:`GainModel`) with no outlier
    exclusion.
    """

    model_id = "lrp-rli-gbg-005"
    target_var = V.ROWPVT_GAIN
    description = (
        "LightGBM — receptive vocabulary gain predictors "
        "(full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory gain model for rowpvt_gain using DEFAULT_GAIN and its own baseline. "
        "Huber parameters come from the 22 September 2026 retune. Read predictive "
        "performance and ranking stability from the fitted outputs; baseline "
        "associations can reflect score limits or regression to the mean."
    )
