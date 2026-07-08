# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG10: Predictors of phoneme-blending gains.

``LRPGBG10`` is the baseline exploratory model for phoneme-blending
gains (``blending_gain``). ``blending`` is a phoneme-blending
(phonological awareness) score on a 0–10 scale: the child selects
which of three pictures depicts the word formed by segmented
phonemes spoken by the examiner.

The target is **mildly right-skewed** (``blending_gain`` min −5,
max 7, median 0, mean 0.48, std 2.15, skewness 0.51, with ~35%
negative and ~20% zero observations, n ≈ 161). Similar in shape
to LRPGBG05 (``rowpvt_gain``, skew 0.04) but with heavier zero
pile-up.

Fits the full ``Predictors.DEFAULT_GAIN`` set; hyperparameters are
re-tuned by Optuna on the full set (150 trials, seed 47; #169).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned by Optuna on the full predictor set (150 trials, seed 47;
# #169 retune, superseding the earlier pruned-set tune).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 310,
    "learning_rate": 0.010714935008511606,
    "num_leaves": 38,
    "max_depth": 3,
    "min_child_samples": 28,
    "subsample": 0.9694286818989883,
    "subsample_freq": 1,
    "colsample_bytree": 0.9789309744826044,
    "reg_alpha": 0.05302028851722132,
    "reg_lambda": 0.005516762267184367,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRPGBG10(GainModel):
    """Phoneme-blending gain predictors — baseline (all data, MAE-tuned).

    Full ``Predictors.DEFAULT_GAIN`` set, MAE-tuned on the full
    set (#169). ``blending`` is already a member, so the GainModel
    auto-include is a no-op; no outlier exclusion.
    """

    model_id = "lrp-rli-gbg-010"
    target_var = V.BLENDING_GAIN
    description = (
        "LightGBM — phoneme-blending gain predictors "
        "(full predictor set, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for blending_gain (gain). Fits the full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature selection in favour of full-set ranking); hyperparameters were re-tuned by Optuna on the full set (150 trials, seed 47; #169). Gain models are near-noise (baseline-driven regression to the mean) - treat the ranking as exploratory."
    )
