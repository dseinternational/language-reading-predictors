# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG08: Predictors of expressive-grammar (APT) gains.

``LRPGBG08`` is the baseline exploratory model for expressive-grammar
gains (``aptgram_gain``). ``aptgram`` is the grammar raw score
from the Action Picture Test (Renfrew, 1997) — the child is shown
pictures and asked to describe them, with scoring of the
grammatical structure of the response.

The target is mildly right-skewed (``aptgram_gain`` min −11,
max 16, median 1, mean 1.49, std 4.34, skewness 0.31, with ~32%
negative and ~11% zero observations, n ≈ 158). Similar gain-shape
to LRPGBG14 (``celf_gain``, skew 0.14).

``aptgram`` is the expressive-grammar parallel to ``trog``
(LRPGBG15/12 receptive grammar) — the pair addresses the
expressive vs receptive grammar asymmetry that is a live
question in DS language research.

Fits the full ``Predictors.DEFAULT_GAIN`` set; hyperparameters are
retained from the earlier pruned-set tune (retune-pending, #116 Phase D).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned (Optuna 150-trial, seed 47) on the earlier pruned selected set;
# retained as the full-set baseline (retune-pending).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 46,
    "learning_rate": 0.06161429194155265,
    "num_leaves": 36,
    "max_depth": 4,
    "min_child_samples": 6,
    "subsample": 0.7361638914240527,
    "subsample_freq": 1,
    "colsample_bytree": 0.9284978358011612,
    "reg_alpha": 0.006282476618386439,
    "reg_lambda": 0.04070671472487475,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRPGBG08(GainModel):
    """APT expressive-grammar gain predictors — baseline (all data, MAE-tuned).

    Full ``Predictors.DEFAULT_GAIN`` set, MAE-tuned (params
    retune-pending). ``aptgram`` is already a member, so the GainModel
    auto-include is a no-op; no outlier exclusion.
    """

    model_id = "lrpgbg08"
    target_var = V.APTGRAM_GAIN
    description = (
        "LightGBM — APT expressive-grammar gain predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for aptgram_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters are retained from the earlier pruned-set Optuna tune (retune-pending). Gain models are near-noise (baseline-driven regression to the mean) - treat the ranking as exploratory."
    )
