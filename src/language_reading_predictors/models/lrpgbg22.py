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
# MAE-tuned (Optuna 150-trial, seed 47) on the earlier pruned selected set;
# retained as the full-set baseline (retune-pending).

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.18291066454567442,
    "num_leaves": 20,
    "max_depth": 12,
    "min_child_samples": 5,
    "subsample": 0.8201087570197031,
    "colsample_bytree": 0.8675365863766314,
    "reg_alpha": 0.0070609566714952225,
    "reg_lambda": 0.1711341570967381,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 5,
}


class LRPGBG22(GainModel):
    """DEAP average articulation gains predictors — baseline (MAE-tuned)."""

    model_id = "lrpgbg22"
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
        "Exploratory model for deappav_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters are retained from the earlier pruned-set Optuna tune (retune-pending). Gain models are near-noise (baseline-driven regression to the mean) — treat the ranking as exploratory."
    )
