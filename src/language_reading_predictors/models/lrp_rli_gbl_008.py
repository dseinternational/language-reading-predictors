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

# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 3.18.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 7.976387999999999,
    "n_estimators": 122,
    "learning_rate": 0.02177205848025236,
    "num_leaves": 29,
    "max_depth": 12,
    "min_child_samples": 12,
    "subsample": 0.6906018073631475,
    "subsample_freq": 1,
    "colsample_bytree": 0.8534058520061745,
    "reg_alpha": 0.002629556363761465,
    "reg_lambda": 0.05789277957989755,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, Huber-tuned) ─────────────────────────────────


class LRPGBL08(LevelModel):
    """APT expressive-grammar level predictors — baseline (all data, Huber-tuned).

    Full ``Predictors.DEFAULT_LEVEL`` set, Huber-tuned on the full set (#169).
    """

    model_id = "lrp-rli-gbl-008"
    target_var = V.APTGRAM
    description = (
        "LightGBM — APT expressive-grammar level predictors "
        "(full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for aptgram (level). Fits the full DEFAULT_LEVEL "
        "predictor set (#116 Phase D retired hard feature selection in favour "
        "of full-set ranking); hyperparameters are re-tuned by Optuna on the full set "
        "(150 trials, seed 47; Huber retune of 2026-09-22, superseding #169). Treat the ranking as "
        "exploratory."
    )
