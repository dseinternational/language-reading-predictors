# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG13: Predictors of non-word reading gains.

``LRPGBG13`` is the baseline exploratory model for non-word reading
gains (``nonword_gain``). ``nonword`` is an items-correct score
from a non-word decoding task (observed range 0–6).

The target is **heavily zero-loaded** (``nonword_gain`` min −6,
max 6, median 0, mean 0.41, std 1.66, skewness 0.40, with ~19%
negative and **~48% zero** observations, n ≈ 153). Half the
children show no change between timepoints — consistent with the
floor-heavy nonword level distribution (57% zero at any given
timepoint). Tree models will struggle to predict non-zero gains
reliably.

Uniform feature selection (2026-06-21): reduced from the full 34-predictor set to 2 predictors via a distance-correlation redundancy filter plus an importance noise-floor cut, then re-tuned. See the SelectionStep below.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS, SelectionStep
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.EWRSWR, V.APTINFO, V.BLENDING, V.SPPHON, V.AGESPEAK, V.NUMCHIL,
            V.AGEBOOKS, V.AREA, V.VISION, V.GROUP, V.GENDER, V.MUMEDUPOST16,
            V.EARINF, V.EOWPVT, V.HEARING, V.DADEDUPOST16, V.DEAPPFI, V.APTGRAM,
            V.B1EXTO, V.BEHAV, V.YARCSI, V.B1RETO, V.TIME, V.ROWPVT, V.ERBWORD,
            V.DEAPPIN, V.CELF, V.AGE, V.ATTEND, V.DEAPPVO, V.TROG, V.YARCLET
        ],
        notes=(
            "Uniform feature selection (2026-06-21): from the full 34-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). The baseline measure was force-kept (regression-to-the-mean anchor). Reduces to 2 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Applied uniformly across all GB models."
        ),
        date="2026-06-21",
        metrics_before={"cv_mae_mean": 0.9825},
        metrics_after={"cv_mae_mean": 0.9329},
    ),
]


# MAE-tuned on the 2-predictor uniform-selected set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 0.9329.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 562,
    "learning_rate": 0.04923477608940171,
    "num_leaves": 57,
    "max_depth": 12,
    "min_child_samples": 33,
    "subsample": 0.8923475866868639,
    "subsample_freq": 1,
    "colsample_bytree": 0.8409770178651289,
    "reg_alpha": 2.5883086592596545,
    "reg_lambda": 0.026085831362143107,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBG13(GainModel):
    """Non-word reading gain predictors — baseline (all data, MAE-tuned).

    Uses a feature-selected subset of :attr:`Predictors.DEFAULT_GAIN`
    (``nonword`` is already a member) with MAE-tuned hyperparameters
    and no outlier exclusion. Feature selection was applied (2026-06-21 uniform); see the SelectionStep and the module docstring.
    """

    model_id = "lrpgbg13"
    target_var = V.NONWORD_GAIN
    description = (
        "LightGBM — non-word reading gain predictors "
        "(2 predictors, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    selection_steps = _SELECTION_STEPS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for nonword_gain (gain). Uniform feature selection (2026-06-21) from the full 34-predictor DEFAULT_GAIN set to 2 predictors (distance-correlation redundancy filter + importance noise-floor cut; baseline force-kept; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 0.983 -> 0.933). Gain models are near-noise (baseline-driven regression to the mean) - treat the reduced ranking as exploratory."
    )
