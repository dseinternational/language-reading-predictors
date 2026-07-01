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
or intention-to-treat estimate. The language-sample measures are
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
    "learning_rate": 0.1329073999399487,
    "num_leaves": 53,
    "max_depth": 6,
    "min_child_samples": 12,
    "subsample": 0.7884118562815289,
    "colsample_bytree": 0.8063199668371345,
    "reg_alpha": 0.002165193941670985,
    "reg_lambda": 0.6271374829795932,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 31,
}


class LRPGBL25(LevelModel):
    """language sample maximum utterance length level predictors — baseline (MAE-tuned)."""

    model_id = "lrpgbl25"
    target_var = V.LSAMMAX
    description = (
        "LightGBM — language sample maximum utterance length level predictors (full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    shap_scatter_specs = [
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    ]
    notes = (
        "Exploratory model for lsammax (level). Fits the full DEFAULT_LEVEL predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters are retained from the earlier pruned-set Optuna tune (retune-pending). Treat the ranking as exploratory."
    )
