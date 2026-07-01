# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL24: Predictors of language sample mean length of utterance level (``lsammlu``).

``lsammlu`` is the mean length of utterance from a coded sample of
the child's spontaneous connected speech.

The target spans min 1.0, max 3.8, median 2.14, mean 2.25, std
0.73, skew 0.31 (n = 106).

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable mean length of
utterance is and from what, to inform whether the shared DAG needs
a spontaneous connected speech node. It is not a causal or
intention-to-treat estimate. The language-sample measures are
recorded at t1–t2 only, so this level model is doubly exploratory
(≈106 rows, two waves) and no gain model is fitted. The other
language-sample measures are absent from the default predictor
pool (recorded at t1–t2 only), so this model cannot be carried by
same-instrument siblings.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline



# ── hyperparameters (MAE-tuned) ──────────────────────────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.06395203888918281,
    "num_leaves": 42,
    "max_depth": 6,
    "min_child_samples": 9,
    "subsample": 0.9489941264970362,
    "colsample_bytree": 0.8413850745885425,
    "reg_alpha": 4.54924369762865,
    "reg_lambda": 0.35495819366481224,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 48,
}


class LRPGBL24(LevelModel):
    """language sample mean length of utterance level predictors — baseline (MAE-tuned)."""

    model_id = "lrpgbl24"
    target_var = V.LSAMMLU
    description = (
        "LightGBM — language sample mean length of utterance level predictors (full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = [
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    ]
    notes = (
        "Exploratory model for lsammlu (level). Fits the full DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters are retained from the earlier pruned-set Optuna tune (retune-pending). Treat the ranking as exploratory."
    )
