# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL17: Predictors of Early Repetition Battery nonword repetition level (``erbnw``).

``erbnw`` is the number of nonwords correctly repeated from the
Early Repetition Battery (a repetition task indexing verbal /
phonological short-term memory).

The target spans min 0.0, max 18.0, median 9.00, mean 8.98, std
4.74, skew 0.01 (n = 202).

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable nonword
repetition is and from what, to inform whether the shared DAG
needs a verbal / phonological short-term memory node. It is not a
causal or intention-to-treat estimate.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline



# ── hyperparameters (MAE-tuned) ──────────────────────────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.016448486269199713,
    "num_leaves": 10,
    "max_depth": 8,
    "min_child_samples": 23,
    "subsample": 0.8128550741391448,
    "colsample_bytree": 0.9858283410142396,
    "reg_alpha": 0.016258934164446813,
    "reg_lambda": 0.014253178652321749,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 384,
}


class LRPGBL17(LevelModel):
    """Early Repetition Battery nonword repetition level predictors — baseline (MAE-tuned)."""

    model_id = "lrpgbl17"
    target_var = V.ERBNW
    description = (
        "LightGBM — Early Repetition Battery nonword repetition level predictors (full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = [
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    ]
    notes = (
        "Exploratory model for erbnw (level). Fits the full DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters are retained from the earlier pruned-set Optuna tune (retune-pending). Treat the ranking as exploratory."
    )

