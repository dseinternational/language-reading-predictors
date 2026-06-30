# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL05: Predictors of receptive vocabulary level.

``LRPGBL05`` is the exploratory model for receptive vocabulary level
(``rowpvt``). The target is **essentially symmetric and near-Gaussian**
(``rowpvt`` min 11, max 82, median 42, mean 41.1, std 14.1, skewness
0.04, n ≈ 215) — no floor, no ceiling, no heavy tail; the cleanest
target distribution of any LRP model.

Uniform feature selection (2026-06-21): reduced from the full
32-predictor :attr:`Predictors.DEFAULT_LEVEL` set (minus the target) to
7 predictors via a distance-correlation redundancy filter (dcor >= 0.70)
plus an importance noise-floor cut, then re-tuned — superseding the
earlier 17-predictor hand-selected set and clearing its residual
redundancy. No same-skill variant: the uniform filter already drops the
bespoke receptive-vocabulary sibling (``b1reto``). See the SelectionStep
and.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS, SelectionStep
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.B1RETO, V.EOWPVT, V.B1EXTO, V.DEAPPFI, V.GENDER, V.GROUP, V.AREA,
            V.EARINF, V.MUMEDUPOST16, V.DADEDUPOST16, V.AGEBOOKS, V.YARCLET,
            V.AGESPEAK, V.ERBNW, V.SPPHON, V.VISION, V.NUMCHIL, V.HEARING,
            V.DEAPPVO, V.YARCSI, V.BEHAV, V.APTGRAM, V.AGE, V.EWRSWR, V.BLENDING
        ],
        notes=(
            "Uniform feature selection (2026-06-21): from the full 32-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). Reduces to 7 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Applied uniformly across all GB models."
        ),
        date="2026-06-21",
        metrics_before={"cv_mae_mean": 7.0096},
        metrics_after={"cv_mae_mean": 7.2648},
    ),
]


# MAE-tuned on the 7-predictor uniform-selected set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 7.2648.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 235,
    "learning_rate": 0.012246385977646894,
    "num_leaves": 58,
    "max_depth": 3,
    "min_child_samples": 6,
    "subsample": 0.6471382352609089,
    "subsample_freq": 1,
    "colsample_bytree": 0.6827041658949718,
    "reg_alpha": 0.021220537844795304,
    "reg_lambda": 0.21769887246316527,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL05(LevelModel):
    """Receptive vocabulary level predictors — exploratory (MAE-tuned, all data).

    Uniform-selected subset of :attr:`Predictors.DEFAULT_LEVEL` (minus the
    target ``rowpvt``) with MAE-tuned hyperparameters and no outlier
    exclusion. See the SelectionStep and the module docstring.
    """

    model_id = "lrpgbl05"
    target_var = V.ROWPVT
    description = (
        "LightGBM — receptive vocabulary level predictors "
        "(7 predictors, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    selection_steps = _SELECTION_STEPS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for receptive vocabulary level (rowpvt). Uniform "
        "feature selection (2026-06-21) from the full 32-predictor "
        "DEFAULT_LEVEL set to 7 predictors (distance-correlation redundancy "
        "filter + importance noise-floor cut; no dcor >= 0.70 pairs remain), "
        "re-tuned on the reduced set (tuner-inner CV MAE 7.010 -> 7.265). "
        "Target is near-Gaussian (skew 0.04). Treat the reduced ranking as "
        "exploratory."
    )
