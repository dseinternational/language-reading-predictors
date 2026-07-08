# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL20: Predictors of DEAP initial-consonant articulation level (``deappin``).

``deappin`` is initial-consonant accuracy from the Diagnostic
Evaluation of Articulation and Phonology (Dodd et al., 2006) — the
proportion of sounds correctly produced in a picture-naming task.

The target spans min 25.9, max 92.4, median 71.67, mean 69.19, std
12.69, skew -0.65 (n = 207).

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable initial-
consonant articulation is and from what, to inform whether the
shared DAG needs a speech-sound accuracy node. It is not a causal
or intention-to-treat estimate.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline



# ── hyperparameters (MAE-tuned) ──────────────────────────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.020192620279150416,
    "num_leaves": 59,
    "max_depth": 6,
    "min_child_samples": 21,
    "subsample": 0.8145188411849295,
    "colsample_bytree": 0.7524136367524303,
    "reg_alpha": 0.047432982905265264,
    "reg_lambda": 0.16631104170004232,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 149,
}


class LRPGBL20(LevelModel):
    """DEAP initial-consonant articulation level predictors — baseline (MAE-tuned)."""

    model_id = "lrp-rli-gbl-020"
    target_var = V.DEAPPIN
    description = (
        "LightGBM — DEAP initial-consonant articulation level predictors (full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    )
    notes = (
        "Exploratory model for deappin (level). Fits the full DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; #169). Treat the ranking as exploratory."
    )

