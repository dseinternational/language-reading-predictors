# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP01: Predictors of word-reading gains.

``LRP01`` is the exploratory model for word-reading gains
(``ewrswr_gain``) — MAE-tuned with no outlier exclusion, designed to
identify the most important influences on reading gains across the full
range of outcomes. ``ewrswr_gain`` is moderately right-skewed (−4 to 21,
median 2, skewness 1.33).

Uniform feature selection (2026-06-21): reduced from the full
34-predictor :attr:`Predictors.DEFAULT_GAIN` set to 3 predictors via a
distance-correlation redundancy filter (dcor >= 0.70) plus an importance
noise-floor cut, with the baseline measure (``ewrswr``) force-kept as the
regression-to-the-mean anchor, then re-tuned. See the SelectionStep and
notes/202606211200-uniform-gb-fs.md.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.CELF, V.BLENDING, V.ERBWORD, V.DEAPPVO, V.APTINFO, V.GROUP, V.NUMCHIL,
            V.GENDER, V.ROWPVT, V.SPPHON, V.HEARING, V.EARINF, V.YARCSI, V.BEHAV,
            V.B1RETO, V.AREA, V.VISION, V.AGEBOOKS, V.MUMEDUPOST16, V.DADEDUPOST16,
            V.AGESPEAK, V.DEAPPFI, V.APTGRAM, V.ERBNW, V.DEAPPIN, V.TIME, V.NONWORD,
            V.EOWPVT, V.TROG, V.B1EXTO, V.YARCLET
        ],
        notes=(
            "Uniform feature selection (2026-06-21): from the full 34-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). The baseline measure was force-kept (regression-to-the-mean anchor). Reduces to 3 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Applied uniformly across all GB models; see notes/202606211200-uniform-gb-fs.md."
        ),
        date="2026-06-21",
        metrics_before={"cv_mae_mean": 2.9136},
        metrics_after={"cv_mae_mean": 2.9014},
    ),
]


# MAE-tuned on the 3-predictor uniform-selected set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 2.9014.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 52,
    "learning_rate": 0.11822278042172524,
    "num_leaves": 51,
    "max_depth": 7,
    "min_child_samples": 16,
    "subsample": 0.6491876371548958,
    "subsample_freq": 1,
    "colsample_bytree": 0.7993121576506287,
    "reg_alpha": 0.02832146282334302,
    "reg_lambda": 6.243279073188195,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRP01(GainModel):
    """Word-reading gain predictors — exploratory model (MAE-tuned, all data).

    Uniform-selected subset of :attr:`Predictors.DEFAULT_GAIN` with the
    baseline ``ewrswr`` force-kept, MAE-tuned, no outlier exclusion. See
    the SelectionStep and the module docstring.
    """

    model_id = "lrp01"
    target_var = V.EWRSWR_GAIN
    description = (
        "LightGBM — word-reading gain predictors "
        "(3 predictors, MAE-tuned, no outlier exclusion)"
    )
    include = [V.EWRSWR]
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 53
    selection_steps = _SELECTION_STEPS
    shap_scatter_specs = [
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
        ShapScatterSpec(
            color_by=V.EWRSWR,
            description="All predictors, coloured by baseline word-reading (ewrswr)",
        ),
    ]
    notes = (
        "Exploratory model for word-reading gains (ewrswr_gain). Uniform "
        "feature selection (2026-06-21) from the full 34-predictor "
        "DEFAULT_GAIN set to 3 predictors (distance-correlation redundancy "
        "filter + importance noise-floor cut; baseline ewrswr force-kept; no "
        "dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner "
        "CV MAE 2.914 -> 2.901). Gain models are near-noise (baseline-driven, "
        "regression to the mean) — treat the reduced ranking as exploratory. "
        "See notes/202606211200-uniform-gb-fs.md."
    )
