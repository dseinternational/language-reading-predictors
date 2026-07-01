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

# MAE-tuned (Optuna 150-trial, seed 47) on the earlier pruned selected set;
# retained as the full-set baseline (retune-pending).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 290,
    "learning_rate": 0.041412982334710226,
    "num_leaves": 48,
    "max_depth": 4,
    "min_child_samples": 4,
    "subsample": 0.7867626106611368,
    "subsample_freq": 1,
    "colsample_bytree": 0.6621049966005043,
    "reg_alpha": 8.554800291256814,
    "reg_lambda": 0.02745428375418847,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRPGBL08(LevelModel):
    """APT expressive-grammar level predictors — baseline (all data, MAE-tuned).

    Full ``Predictors.DEFAULT_LEVEL`` set, MAE-tuned (params retune-pending).
    """

    model_id = "lrpgbl08"
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
        "of full-set ranking); hyperparameters are retained from the earlier "
        "pruned-set Optuna tune (retune-pending). Treat the ranking as "
        "exploratory."
    )
