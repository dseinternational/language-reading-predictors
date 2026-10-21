# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Documentation edited by a LLM-based AI tool (Codex/GPT-6).

"""
lrp-rli-gbg-010: Predictors of phoneme-blending gains.

``LRPGBG10`` is the baseline exploratory model for phoneme-blending
gains (``blending_gain``). ``blending`` is a phoneme-blending
(phonological awareness) score on a 0–10 scale: the child selects
which of three pictures depicts the word formed by segmented
phonemes spoken by the examiner.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter sets ─────────────────────────────────────────────────

# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 1.80.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 1.9940969999999998,
    "n_estimators": 170,
    "learning_rate": 0.028554158042751736,
    "num_leaves": 49,
    "max_depth": 5,
    "min_child_samples": 16,
    "subsample": 0.6210520684732574,
    "subsample_freq": 1,
    "colsample_bytree": 0.7655389769688665,
    "reg_alpha": 0.05740131921620294,
    "reg_lambda": 0.11858012641679934,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, Huber-tuned) ─────────────────────────────────


class LRPGBG10(GainModel):
    """Phoneme-blending gain predictors — baseline (all data, Huber-tuned).

    Full ``Predictors.DEFAULT_GAIN`` set, Huber-tuned on the full
    set (#169). ``blending`` is already a member, so the GainModel
    auto-include is a no-op; no outlier exclusion.
    """

    model_id = "lrp-rli-gbg-010"
    target_var = V.BLENDING_GAIN
    description = (
        "LightGBM — phoneme-blending gain predictors "
        "(full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory gain model for blending_gain using DEFAULT_GAIN and its own baseline. "
        "Huber parameters come from the 22 September 2026 retune. Read predictive "
        "performance and ranking stability from the fitted outputs; baseline "
        "associations can reflect score limits or regression to the mean."
    )
