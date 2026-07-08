# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG22: Predictors of DEAP average articulation gains (``deappav_gain``).

``deappav`` is the DEAP picture-naming average from the Diagnostic
Evaluation of Articulation and Phonology (Dodd et al., 2006) — the
proportion of sounds correctly produced in a picture-naming task.
It is a composite of ``deappin``, ``deappvo``, ``deappfi`` (in the
candidate pool), but gain targets are near-noise (regression-to-
the-mean dominates), so the level model carries the same-
instrument check.

The gain target spans min -13.9, max 12.9, median 0.20, mean 0.42,
std 4.67, skew -0.09, with ~45% negative and ~2% zero observations
(n = 152). Regression from the mean dominates gain targets across
the suite.

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable average
articulation is and from what, to inform whether the shared DAG
needs a speech-sound accuracy node. It is not a causal or
intention-to-treat estimate.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline



# ── hyperparameters ──────────────────────────────────────────────────────
# MAE-tuned by Optuna on the full predictor set (150 trials, seed 47;
# #169 retune, superseding the earlier pruned-set tune).

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.13156119317157547,
    "num_leaves": 46,
    "max_depth": 9,
    "min_child_samples": 8,
    "subsample": 0.7108872311249903,
    "colsample_bytree": 0.6427188747891276,
    "reg_alpha": 0.3864822977101127,
    "reg_lambda": 0.002007833299733497,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 9,
}


class LRPGBG22(GainModel):
    """DEAP average articulation gains predictors — baseline (MAE-tuned)."""

    model_id = "lrp-rli-gbg-022"
    target_var = V.DEAPPAV_GAIN
    description = (
        "LightGBM — DEAP average articulation gains predictors (full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    )
    notes = (
        "Exploratory model for deappav_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; #169). Gain models are near-noise (baseline-driven regression to the mean) — treat the ranking as exploratory."
    )
