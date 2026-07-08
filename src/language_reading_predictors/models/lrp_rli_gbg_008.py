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
    "n_estimators": 86,
    "learning_rate": 0.03181238295963342,
    "num_leaves": 44,
    "max_depth": 3,
    "min_child_samples": 5,
    "subsample": 0.9999807450737386,
    "subsample_freq": 1,
    "colsample_bytree": 0.9701145699411898,
    "reg_alpha": 0.009346004721992875,
    "reg_lambda": 0.00592519603996152,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRPGBG08(GainModel):
    """APT expressive-grammar gain predictors — baseline (all data, MAE-tuned).

    Full ``Predictors.DEFAULT_GAIN`` set, MAE-tuned on the full
    set (#169). ``aptgram`` is already a member, so the GainModel
    auto-include is a no-op; no outlier exclusion.
    """

    model_id = "lrp-rli-gbg-008"
    target_var = V.APTGRAM_GAIN
    description = (
        "LightGBM — APT expressive-grammar gain predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for aptgram_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; #169). Gain models are near-noise (baseline-driven regression to the mean) - treat the ranking as exploratory."
    )
