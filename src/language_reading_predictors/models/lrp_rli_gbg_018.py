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
# MAE-tuned by Optuna on the full predictor set (150 trials, seed 47;
# #169 retune, superseding the earlier pruned-set tune).

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.0772530439017306,
    "num_leaves": 48,
    "max_depth": 10,
    "min_child_samples": 15,
    "subsample": 0.6590294316797071,
    "colsample_bytree": 0.7522340207839537,
    "reg_alpha": 3.024954788315427,
    "reg_lambda": 2.375517100604047,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 51,
}


class LRPGBG18(GainModel):
    """Early Repetition Battery word repetition gains predictors — baseline (MAE-tuned)."""

    model_id = "lrp-rli-gbg-018"
    target_var = V.ERBWORD_GAIN
    description = (
        "LightGBM — Early Repetition Battery word repetition gains predictors (full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    )
    notes = (
        "Exploratory model for erbword_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; #169). Gain models are near-noise (baseline-driven regression to the mean) — treat the ranking as exploratory."
    )
