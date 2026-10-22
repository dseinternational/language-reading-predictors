# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Documentation edited by a LLM-based AI tool (Codex/GPT-6).

"""
lrp-rli-gbg-007: Predictors of expressive-information (APT) gains.

``LRPGBG07`` is the baseline exploratory model for expressive-
information gains (``aptinfo_gain``). ``aptinfo`` is the
information raw score from the Action Picture Test (Renfrew,
1997): the child is shown pictures and asked to describe them,
with scoring of the information content of the response (as
distinct from its grammatical structure, which is scored
separately as ``aptgram`` — ``lrp-rli-gbg-008``).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter sets ─────────────────────────────────────────────────

# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 3.87.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 6.979339499999999,
    "n_estimators": 251,
    "learning_rate": 0.03147277019144279,
    "num_leaves": 10,
    "max_depth": 6,
    "min_child_samples": 33,
    "subsample": 0.8264195586859768,
    "subsample_freq": 1,
    "colsample_bytree": 0.9965282286339251,
    "reg_alpha": 1.1546392206152165,
    "reg_lambda": 0.01006119066142894,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, Huber-tuned) ─────────────────────────────────


class LRPGBG07(GainModel):
    """APT expressive-information gain predictors — baseline (all data, Huber-tuned).

    Full ``Predictors.DEFAULT_GAIN`` set, Huber-tuned on the full
    set (#169). ``aptinfo`` is already a member, so the GainModel
    auto-include is a no-op; no outlier exclusion.
    """

    model_id = "lrp-rli-gbg-007"
    target_var = V.APTINFO_GAIN
    description = (
        "LightGBM — APT expressive-information gain predictors "
        "(full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory gain model for aptinfo_gain using DEFAULT_GAIN and its own baseline. "
        "Huber parameters come from the 22 September 2026 retune. Read predictive "
        "performance and ranking stability from the fitted outputs; baseline "
        "associations can reflect score limits or regression to the mean."
    )
