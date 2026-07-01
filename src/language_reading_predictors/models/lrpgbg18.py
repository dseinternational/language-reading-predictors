# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG18: Predictors of Early Repetition Battery word repetition gains (``erbword_gain``).

``erbword`` is the number of real words correctly repeated from
the Early Repetition Battery (a repetition task indexing verbal /
phonological short-term memory).

The gain target spans min -9.0, max 12.0, median 1.00, mean 1.03,
std 3.00, skew 0.35, with ~22% negative and ~22% zero observations
(n = 148). Regression from the mean dominates gain targets across
the suite.

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable word repetition
is and from what, to inform whether the shared DAG needs a verbal
/ phonological short-term memory node. It is not a causal or
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
    "learning_rate": 0.04590079068925787,
    "num_leaves": 49,
    "max_depth": 4,
    "min_child_samples": 14,
    "subsample": 0.615810127489842,
    "colsample_bytree": 0.9408669501013529,
    "reg_alpha": 0.0063643336825072545,
    "reg_lambda": 3.0856651422754178,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 136,
}


class LRPGBG18(GainModel):
    """Early Repetition Battery word repetition gains predictors — baseline (MAE-tuned)."""

    model_id = "lrpgbg18"
    target_var = V.ERBWORD_GAIN
    description = (
        "LightGBM — Early Repetition Battery word repetition gains predictors (full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = [
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    ]
    notes = (
        "Exploratory model for erbword_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters are retained from the earlier pruned-set Optuna tune (retune-pending). Gain models are near-noise (baseline-driven regression to the mean) — treat the ranking as exploratory."
    )
