# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL19: Predictors of Early Repetition Battery total repetition level (``erbto``).

``erbto`` is the ERB total score from the Early Repetition Battery
(a repetition task indexing verbal / phonological short-term
memory). It is a composite — ``erbword``, ``erbnw`` are its
components and remain in the candidate pool, so a high naive R² is
mechanical (see the same-skill-excluded ranking view, ``ranking_excluding_same_skill.csv``).

The target spans min 1.0, max 36.0, median 21.00, mean 20.17, std
9.47, skew -0.21 (n = 202).

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable total repetition
is and from what, to inform whether the shared DAG needs a verbal
/ phonological short-term memory node. It is not a causal or
intention-to-treat estimate.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline



# ── hyperparameters (MAE-tuned) ──────────────────────────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.011278679136334829,
    "num_leaves": 58,
    "max_depth": 3,
    "min_child_samples": 4,
    "subsample": 0.964337374219075,
    "colsample_bytree": 0.9546390335084767,
    "reg_alpha": 2.6119920182981073,
    "reg_lambda": 0.9965848470912236,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 1097,
}


class LRPGBL19(LevelModel):
    """Early Repetition Battery total repetition level predictors — baseline (MAE-tuned)."""

    model_id = "lrp-rli-gbl-019"
    target_var = V.ERBTO
    description = (
        "LightGBM — Early Repetition Battery total repetition level predictors (full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    )
    notes = (
        "Exploratory model for erbto (level). Fits the full DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; #169). Treat the ranking as exploratory."
    )

