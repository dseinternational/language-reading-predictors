# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL08: Predictors of expressive-grammar (APT) level.

``LRPGBL08`` is the baseline exploratory model for expressive-grammar
level (``aptgram``). ``aptgram`` is the grammar raw score from the
Action Picture Test (Renfrew, 1997) — the child is shown pictures
and asked to describe them, with scoring of the grammatical
structure of the response.

The target is **right-skewed** (``aptgram`` min 0, max 28,
median 6, mean 7.63, std 6.34, skewness 1.23, with ~9% at zero,
n ≈ 211) — comparable in skew magnitude to LRPGBL12's ``ewrswr``
baseline, and a heavier floor than the receptive-grammar target
``trog`` (LRPGBL15, skew 0.29).

``aptgram`` is the expressive-grammar parallel to ``trog``
(LRPGBL15 receptive grammar) — the pair addresses the expressive vs
receptive grammar asymmetry that is a live question in DS
language research. The right-skew motivates a later log-transform
variant (mirroring LRPGBL12's ``lrpgbl12_log``).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned by Optuna on the full predictor set (150 trials, seed 47;
# #169 retune, superseding the earlier pruned-set tune).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 102,
    "learning_rate": 0.03816906301061165,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 5,
    "subsample": 0.6939079492149587,
    "subsample_freq": 1,
    "colsample_bytree": 0.7369501404010328,
    "reg_alpha": 0.06357374004205678,
    "reg_lambda": 0.0021109116292999516,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRPGBL08(LevelModel):
    """APT expressive-grammar level predictors — baseline (all data, MAE-tuned).

    Full ``Predictors.DEFAULT_LEVEL`` set, MAE-tuned on the full set (#169).
    """

    model_id = "lrp-rli-gbl-008"
    target_var = V.APTGRAM
    description = (
        "LightGBM — APT expressive-grammar level predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for aptgram (level). Fits the full DEFAULT_LEVEL "
        "predictor set (#116 Phase D retired hard feature selection in favour "
        "of full-set ranking); hyperparameters are re-tuned by Optuna on the full set "
        "(150 trials, seed 47; #169). Treat the ranking as "
        "exploratory."
    )
