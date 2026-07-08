# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG19: Predictors of Early Repetition Battery total repetition gains (``erbto_gain``).

``erbto`` is the ERB total score from the Early Repetition Battery
(a repetition task indexing verbal / phonological short-term
memory). It is a composite of ``erbword``, ``erbnw`` (in the
candidate pool), but gain targets are near-noise (regression-to-
the-mean dominates), so the level model carries the same-
instrument check.

The gain target spans min -16.0, max 15.0, median 2.00, mean 2.05,
std 5.04, skew -0.25, with ~27% negative and ~7% zero observations
(n = 147). Regression from the mean dominates gain targets across
the suite.

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable total repetition
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
    "learning_rate": 0.1835524846851142,
    "num_leaves": 50,
    "max_depth": 6,
    "min_child_samples": 29,
    "subsample": 0.8978385954772342,
    "colsample_bytree": 0.9080795000334347,
    "reg_alpha": 0.006017571585924246,
    "reg_lambda": 0.2230422591511532,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 10,
}


class LRPGBG19(GainModel):
    """Early Repetition Battery total repetition gains predictors — baseline (MAE-tuned)."""

    model_id = "lrp-rli-gbg-019"
    target_var = V.ERBTO_GAIN
    description = (
        "LightGBM — Early Repetition Battery total repetition gains predictors (full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    )
    notes = (
        "Exploratory model for erbto_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; #169). Gain models are near-noise (baseline-driven regression to the mean) — treat the ranking as exploratory."
    )
