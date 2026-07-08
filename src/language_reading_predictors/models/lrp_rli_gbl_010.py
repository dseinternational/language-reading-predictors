# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL10: Predictors of phoneme-blending level.

``LRPGBL10`` is the baseline exploratory model for phoneme-blending
level (``blending``). A phoneme-blending (phonological awareness)
score on a 0–10 scale: the child selects which of three pictures
depicts the word formed by segmented phonemes spoken by the
examiner.

The target is **essentially symmetric** (``blending`` min 0, max
10, median 6, mean 5.76, std 2.55, skewness 0.01, n ≈ 215) — one
of the cleanest distributions in the suite. The coarse 0–10
scale may cap achievable R² (similar to CELF's 0–18 scale in
LRPGBL14).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned by Optuna on the full predictor set (150 trials, seed 47;
# #169 retune, superseding the earlier pruned-set tune).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 82,
    "learning_rate": 0.037177313586346,
    "num_leaves": 51,
    "max_depth": 4,
    "min_child_samples": 35,
    "subsample": 0.6372320482795071,
    "subsample_freq": 1,
    "colsample_bytree": 0.8957463676991052,
    "reg_alpha": 0.1907196568296802,
    "reg_lambda": 0.014054850452382656,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRPGBL10(LevelModel):
    """Phoneme-blending level predictors — baseline (all data, MAE-tuned).

    Full ``Predictors.DEFAULT_LEVEL`` set, MAE-tuned on the full set (#169).
    """

    model_id = "lrp-rli-gbl-010"
    target_var = V.BLENDING
    description = (
        "LightGBM — phoneme-blending level predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for blending (level). Fits the full DEFAULT_LEVEL "
        "predictor set (#116 Phase D retired hard feature selection in favour "
        "of full-set ranking); hyperparameters are re-tuned by Optuna on the full set "
        "(150 trials, seed 47; #169). Treat the ranking as "
        "exploratory."
    )
