# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Documentation edited by a LLM-based AI tool (Codex/GPT-6).

"""
lrp-rli-gbg-013: Predictors of non-word reading gains.

``LRPGBG13`` is the baseline exploratory model for non-word reading
gains (``nonword_gain``). ``nonword`` is an items-correct score
from a non-word decoding task (observed range 0–6).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 1.12.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 1.9940969999999998,
    "n_estimators": 1191,
    "learning_rate": 0.011225495442360225,
    "num_leaves": 27,
    "max_depth": 9,
    "min_child_samples": 39,
    "subsample": 0.8389636298118259,
    "subsample_freq": 1,
    "colsample_bytree": 0.9212996072348311,
    "reg_alpha": 0.1305950709055434,
    "reg_lambda": 6.274735959418554,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBG13(GainModel):
    """Non-word reading gain predictors — baseline (all data, Huber-tuned).

    Full ``Predictors.DEFAULT_GAIN`` set, Huber-tuned on the full set (#169).
    """

    model_id = "lrp-rli-gbg-013"
    target_var = V.NONWORD_GAIN
    description = (
        "LightGBM — non-word reading gain predictors "
        "(full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory gain model for nonword_gain using DEFAULT_GAIN and its own baseline. "
        "Huber parameters come from the 22 September 2026 retune. Read predictive "
        "performance and ranking stability from the fitted outputs; baseline "
        "associations can reflect score limits or regression to the mean."
    )
