# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG20: Predictors of DEAP initial-consonant articulation gains (``deappin_gain``).

``deappin`` is initial-consonant accuracy from the Diagnostic
Evaluation of Articulation and Phonology (Dodd et al., 2006) — the
proportion of sounds correctly produced in a picture-naming task.

The gain target spans min -12.1, max 23.8, median 1.18, mean 1.02,
std 6.48, skew 0.30, with ~42% negative and ~5% zero observations
(n = 152). Regression from the mean dominates gain targets across
the suite.

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable initial-
consonant articulation is and from what, to inform whether the
shared DAG needs a speech-sound accuracy node. It is not a causal
or intention-to-treat estimate.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline



# ── hyperparameters ──────────────────────────────────────────────────────
# MAE-tuned (Optuna 150-trial, seed 47) on the earlier pruned selected set;
# retained as the full-set baseline (retune-pending).

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.12870587965203345,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 32,
    "subsample": 0.9903120177414865,
    "colsample_bytree": 0.7839600465161084,
    "reg_alpha": 0.09725138724589176,
    "reg_lambda": 0.11980050171655834,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 83,
}


class LRPGBG20(GainModel):
    """DEAP initial-consonant articulation gains predictors — baseline (MAE-tuned)."""

    model_id = "lrp-rli-gbg-020"
    target_var = V.DEAPPIN_GAIN
    description = (
        "LightGBM — DEAP initial-consonant articulation gains predictors (full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    )
    notes = (
        "Exploratory model for deappin_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters are retained from the earlier pruned-set Optuna tune (retune-pending). Gain models are near-noise (baseline-driven regression to the mean) — treat the ranking as exploratory."
    )
