# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL27: Predictors of language sample unique words level (``lsamun``).

``lsamun`` is the number of unique words (lexical diversity) from
a coded sample of the child's spontaneous connected speech.

The target spans min 1.0, max 86.0, median 36.00, mean 36.13, std
20.73, skew 0.32 (n = 106).

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable unique words is
and from what, to inform whether the shared DAG needs a
spontaneous connected speech node. It is not a causal or
available-case modified ITT estimate. The language-sample measures are
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



# ── hyperparameters (Huber-tuned) ──────────────────────────────────────────

# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=51, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 13.22.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 29.911454999999997,
    "learning_rate": 0.04641005068519118,
    "num_leaves": 10,
    "max_depth": 8,
    "min_child_samples": 25,
    "subsample": 0.8848970731510472,
    "colsample_bytree": 0.7215828658590007,
    "reg_alpha": 7.727947336883153,
    "reg_lambda": 0.003059154522478216,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 117,
}


class LRPGBL27(LevelModel):
    """language sample unique words level predictors — baseline (Huber-tuned)."""

    model_id = "lrp-rli-gbl-027"
    target_var = V.LSAMUN
    description = (
        "LightGBM — language sample unique words level predictors (full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    )
    notes = (
        "Exploratory model for lsamun (level). Fits the full DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; Huber retune of 2026-09-22, superseding #169). Treat the ranking as exploratory."
    )
