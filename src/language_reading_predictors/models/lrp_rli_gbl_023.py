# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL23: Predictors of DEAP composite articulation level (``deapp_c``).

``deapp_c`` is the DEAP picture-naming composite from the
Diagnostic Evaluation of Articulation and Phonology (Dodd et al.,
2006) — the proportion of sounds correctly produced in a picture-
naming task. It is a composite — ``deappin``, ``deappvo``,
``deappfi`` are its components and remain in the candidate pool,
so a high naive R² is mechanical (see the same-skill-excluded
ranking view, ``ranking_excluding_same_skill.csv``).

The target spans min 141.2, max 284.6, median 234.56, mean 225.33,
std 34.54, skew -0.67 (n = 207).

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable composite
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
    "learning_rate": 0.015140353513327712,
    "num_leaves": 30,
    "max_depth": 3,
    "min_child_samples": 5,
    "subsample": 0.6993055967918627,
    "colsample_bytree": 0.9278904239375344,
    "reg_alpha": 0.6026526085851373,
    "reg_lambda": 0.2686118758705613,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 990,
}


class LRPGBL23(LevelModel):
    """DEAP composite articulation level predictors — baseline (MAE-tuned)."""

    model_id = "lrp-rli-gbl-023"
    target_var = V.DEAPP_C
    description = (
        "LightGBM — DEAP composite articulation level predictors (full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    )
    notes = (
        "Exploratory model for deapp_c (level). Fits the full DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; #169). Treat the ranking as exploratory."
    )

