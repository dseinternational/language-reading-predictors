# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL25: Predictors of language sample maximum utterance length level (``lsammax``).

``lsammax`` is the maximum utterance length from a coded sample of
the child's spontaneous connected speech.

The target spans min 1.0, max 13.0, median 5.00, mean 5.22, std
2.18, skew 0.67 (n = 106).

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable maximum
utterance length is and from what, to inform whether the shared
DAG needs a spontaneous connected speech node. It is not a causal
or available-case modified ITT estimate. The language-sample measures are
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
# on the full default predictor set; best mean cross-validated RMSE 1.63.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 1.9940969999999998,
    "learning_rate": 0.010354595521033166,
    "num_leaves": 14,
    "max_depth": 10,
    "min_child_samples": 9,
    "subsample": 0.6045229209918934,
    "colsample_bytree": 0.6247821130253598,
    "reg_alpha": 7.887932192593137,
    "reg_lambda": 0.0013568180004135968,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 787,
}


class LRPGBL25(LevelModel):
    """language sample maximum utterance length level predictors — baseline (Huber-tuned)."""

    model_id = "lrp-rli-gbl-025"
    target_var = V.LSAMMAX
    description = (
        "LightGBM — language sample maximum utterance length level predictors (full predictor set, Huber-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    )
    notes = (
        "Exploratory model for lsammax (level). Fits the full DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; Huber retune of 2026-09-22, superseding #169). Treat the ranking as exploratory."
    )
