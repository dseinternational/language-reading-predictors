# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL28: Predictors of language sample total words level (``lsamto``).

``lsamto`` is the total number of words produced from a coded
sample of the child's spontaneous connected speech.

The target spans min 3.0, max 221.0, median 68.50, mean 76.59, std
51.97, skew 0.65 (n = 106).

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable total words is
and from what, to inform whether the shared DAG needs a
spontaneous connected speech node. It is not a causal or
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
    "learning_rate": 0.11285393630253474,
    "num_leaves": 55,
    "max_depth": 3,
    "min_child_samples": 8,
    "subsample": 0.6652315813492752,
    "colsample_bytree": 0.9379836846271299,
    "reg_alpha": 1.8470975741347961,
    "reg_lambda": 0.03382949123797818,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 57,
}


class LRPGBL28(LevelModel):
    """language sample total words level predictors — baseline (MAE-tuned)."""

    model_id = "lrp-rli-gbl-028"
    target_var = V.LSAMTO
    description = (
        "LightGBM — language sample total words level predictors (full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    )
    notes = (
        "Exploratory model for lsamto (level). Fits the full DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; #169). Treat the ranking as exploratory."
    )
