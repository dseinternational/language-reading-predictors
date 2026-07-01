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

# MAE-tuned (Optuna 150-trial, seed 47) on the earlier pruned selected set;
# retained as the full-set baseline (retune-pending).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 11,
    "learning_rate": 0.18894563671851897,
    "num_leaves": 35,
    "max_depth": 5,
    "min_child_samples": 26,
    "subsample": 0.8495160600596453,
    "subsample_freq": 1,
    "colsample_bytree": 0.8734189036490825,
    "reg_alpha": 0.0015837285659681297,
    "reg_lambda": 0.2283963702135058,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRPGBG16(GainModel):
    """DEAP fine-articulation gain predictors — baseline (all data, MAE-tuned).

    Full ``Predictors.DEFAULT_GAIN`` set, MAE-tuned (params retune-pending).
    """

    model_id = "lrpgbg16"
    target_var = V.DEAPPFI_GAIN
    description = (
        "LightGBM — DEAP fine-articulation gain predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for deappfi_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters are retained from the earlier pruned-set Optuna tune (retune-pending). Gain models are near-noise (baseline-driven regression to the mean) - treat the ranking as exploratory."
    )
