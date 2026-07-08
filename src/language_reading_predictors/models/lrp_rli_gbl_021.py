# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL21: Predictors of DEAP vowel articulation level (``deappvo``).

``deappvo`` is vowel accuracy from the Diagnostic Evaluation of
Articulation and Phonology (Dodd et al., 2006) — the proportion of
sounds correctly produced in a picture-naming task.

The target spans min 68.8, max 100.0, median 98.04, mean 95.86,
std 5.72, skew -2.13 (n = 207).

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable vowel
articulation is and from what, to inform whether the shared DAG
needs a speech-sound accuracy node. It is not a causal or
intention-to-treat estimate.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline



# ── hyperparameters (MAE-tuned) ──────────────────────────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.02887441463824519,
    "num_leaves": 56,
    "max_depth": 5,
    "min_child_samples": 22,
    "subsample": 0.6173113687985987,
    "colsample_bytree": 0.8539703371725147,
    "reg_alpha": 0.003946302374821871,
    "reg_lambda": 0.007877191398796134,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 140,
}


class LRPGBL21(LevelModel):
    """DEAP vowel articulation level predictors — baseline (MAE-tuned)."""

    model_id = "lrp-rli-gbl-021"
    target_var = V.DEAPPVO
    description = (
        "LightGBM — DEAP vowel articulation level predictors (full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    )
    notes = (
        "Exploratory model for deappvo (level). Fits the full DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; #169). Treat the ranking as exploratory."
    )

