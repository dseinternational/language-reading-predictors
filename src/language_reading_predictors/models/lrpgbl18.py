# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL18: Predictors of Early Repetition Battery word repetition level (``erbword``).

``erbword`` is the number of real words correctly repeated from
the Early Repetition Battery (a repetition task indexing verbal /
phonological short-term memory).

The target spans min 0.0, max 28.0, median 12.00, mean 11.35, std
5.22, skew -0.25 (n = 203).

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable word repetition
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
    "learning_rate": 0.12648578524796544,
    "num_leaves": 10,
    "max_depth": 4,
    "min_child_samples": 6,
    "subsample": 0.6442114045867877,
    "colsample_bytree": 0.6424565278401814,
    "reg_alpha": 0.1708379704840523,
    "reg_lambda": 0.018815743974597524,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 133,
}


class LRPGBL18(LevelModel):
    """Early Repetition Battery word repetition level predictors — baseline (MAE-tuned)."""

    model_id = "lrpgbl18"
    target_var = V.ERBWORD
    description = (
        "LightGBM — Early Repetition Battery word repetition level predictors (full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = [
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    ]
    notes = (
        "Exploratory model for erbword (level). Fits the full DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters are retained from the earlier pruned-set Optuna tune (retune-pending). Treat the ranking as exploratory."
    )

