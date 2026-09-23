# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG12: Predictors of word-reading gains.

``LRPGBG12`` is the exploratory model for word-reading gains
(``ewrswr_gain``) — Huber-tuned with no outlier exclusion, designed to
identify the most important influences on reading gains across the full
range of outcomes. ``ewrswr_gain`` is moderately right-skewed (−4 to 21,
median 2, skewness 1.33).
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# Huber-tuned (Optuna 150-trial, seed 47, GroupKFold cv=53, RMSE scoring)
# on the full default predictor set; best mean cross-validated RMSE 3.40.
# Huber threshold alpha = 1.345 x 1.4826 x MAD
# of the tuned target
# (2026-09-22 Huber retune, superseding the #169 MAE tune).
_LGBM_HUBER_PARAMS: dict[str, float | int | str] = {
    "objective": "huber",
    "alpha": 3.9881939999999996,
    "n_estimators": 81,
    "learning_rate": 0.14156173630831012,
    "num_leaves": 46,
    "max_depth": 9,
    "min_child_samples": 38,
    "subsample": 0.7403116960534286,
    "subsample_freq": 1,
    "colsample_bytree": 0.6093814119277299,
    "reg_alpha": 2.022454244751553,
    "reg_lambda": 0.004710335482424086,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBG12(GainModel):
    """Word-reading gain predictors — exploratory model (Huber-tuned, all data).

    Full ``Predictors.DEFAULT_GAIN`` set, Huber-tuned on the full set (#116).
    """

    model_id = "lrp-rli-gbg-012"
    target_var = V.EWRSWR_GAIN
    description = (
        "LightGBM — word-reading gain predictors "
        "(full predictor set, Huber-tuned, no outlier exclusion)"
    )
    include = (V.EWRSWR,)
    pipeline_cls = LGBMPipeline
    params = _LGBM_HUBER_PARAMS
    cv_splits = 53
    shap_scatter_specs = (
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
        ShapScatterSpec(
            color_by=V.EWRSWR,
            description="All predictors, coloured by baseline word-reading (ewrswr)",
        ),
        # Classical dependence pairs shown under "Selected dependence pairs"
        # in the report template. They were dropped when uniform feature
        # selection cut the model to three predictors (#102) and not restored
        # when #116 Phase D brought the full predictor set back.
        ShapScatterSpec(
            predictors=[V.AGE],
            color_by=V.YARCLET,
            description="age vs yarclet (letter-sound knowledge)",
        ),
        ShapScatterSpec(
            predictors=[V.AGE],
            color_by=V.CELF,
            description="age vs celf (receptive language)",
        ),
        ShapScatterSpec(
            predictors=[V.YARCLET],
            color_by=V.BLENDING,
            description="yarclet vs blending (phonological prerequisites)",
        ),
        ShapScatterSpec(
            predictors=[V.CELF],
            color_by=V.B1EXTO,
            description="celf vs b1exto (receptive vs expressive language)",
        ),
    )
    notes = (
        "Exploratory model for word-reading gains (ewrswr_gain). Fits the "
        "full DEFAULT_GAIN predictor set (#116 Phase D retired hard feature "
        "selection in favour of full-set ranking); hyperparameters were "
        "re-tuned by Optuna on the full set (150 trials, seed 47; #116 "
        "reporting refresh). Gain models are near-noise (baseline-driven, "
        "regression to the mean) — treat the ranking as exploratory."
    )
