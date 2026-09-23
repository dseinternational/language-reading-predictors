# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG16: Predictors of DEAP fine-articulation gains.

``LRPGBG16`` is the baseline exploratory model for DEAP fine-
articulation gains (``deappfi_gain``). ``deappfi`` is a
percentage-scale articulation measure from the Diagnostic
Evaluation of Articulation and Phonology (Dodd et al., 2006) —
the proportion of sounds correctly produced when the child is
asked to name pictures. ``deappfi`` specifically scores the
*final* consonant of each word (distinct from ``deappin``
initial and ``deappvo`` voicing).

The target is nearly symmetric but with a heavy two-sided
spread (``deappfi_gain`` min −56.9, max 56.0, median 0.01,
mean 0.84, std 13.28, skewness −0.32, with **~48% negative**
and ~2% zero observations, n ≈ 152). **Heavy regression from
the ceiling is the dominant story** — children at the top of
the scale tend to drop back between timepoints while those at
the floor improve.

DEAP measures have been used as predictors across every other
model in the suite but never as targets until LRPGBG16/22.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter sets ─────────────────────────────────────────────────

# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 9.49.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 12.692427405000009,
    "n_estimators": 57,
    "learning_rate": 0.12832922120578538,
    "num_leaves": 56,
    "max_depth": 7,
    "min_child_samples": 37,
    "subsample": 0.713578378387809,
    "subsample_freq": 1,
    "colsample_bytree": 0.7689941185782706,
    "reg_alpha": 3.609909835138867,
    "reg_lambda": 5.7411849028018,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, Huber-tuned) ─────────────────────────────────


class LRPGBG16(GainModel):
    """DEAP fine-articulation gain predictors — baseline (all data, Huber-tuned).

    Full ``Predictors.DEFAULT_GAIN`` set, Huber-tuned on the full set (#169).
    """

    model_id = "lrp-rli-gbg-016"
    target_var = V.DEAPPFI_GAIN
    description = (
        "LightGBM — DEAP fine-articulation gain predictors "
        "(full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for deappfi_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; Huber retune of 2026-09-22, superseding #169). Gain models are near-noise (baseline-driven regression to the mean) - treat the ranking as exploratory."
    )
