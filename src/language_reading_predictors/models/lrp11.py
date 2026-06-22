# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP11: Predictors of receptive-grammar (TROG-2) gains.

``LRP11`` is the baseline exploratory model for receptive-grammar
gains (``trog_gain``). The ``trog`` score is the items-correct
total from the Test for Reception of Grammar 2 (TROG-2; Bishop
2003), covering eight grammatical constructs in blocks of four
items (32 items total; observed max 27 in this sample).

The target is mildly left-skewed (``trog_gain`` min ≈ −10,
max ≈ 12, median 2, mean 1.22, std 4.19, skewness −0.17, with
~34% negative and ~8% zero observations, n ≈ 161). Closer in
shape to LRP07 (``rowpvt_gain``, skew 0.04) and LRP05
(``yarclet_gain``, skew 0.45) than to the heavier-skewed gain
targets.

Uniform feature selection (2026-06-21): reduced from the full 34-predictor set to 5 predictors via a distance-correlation redundancy filter plus an importance noise-floor cut, then re-tuned. See the SelectionStep below and notes/202606211200-uniform-gb-fs.md.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Feature selection (2026-06-21 uniform): distance-correlation
# redundancy filter + importance noise-floor cut; see the SelectionStep.

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.MUMEDUPOST16, V.NONWORD, V.DADEDUPOST16, V.AREA, V.NUMCHIL, V.EARINF,
            V.YARCLET, V.HEARING, V.APTINFO, V.AGEBOOKS, V.GROUP, V.B1EXTO,
            V.GENDER, V.VISION, V.ERBWORD, V.SPPHON, V.AGESPEAK, V.ROWPVT, V.BEHAV,
            V.TIME, V.AGE, V.DEAPPIN, V.YARCSI, V.EWRSWR, V.BLENDING, V.B1RETO,
            V.ERBNW, V.ATTEND, V.APTGRAM
        ],
        notes=(
            "Uniform feature selection (2026-06-21): from the full 34-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). The baseline measure was force-kept (regression-to-the-mean anchor). Reduces to 5 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Applied uniformly across all GB models; see notes/202606211200-uniform-gb-fs.md."
        ),
        date="2026-06-21",
        metrics_before={"cv_mae_mean": 3.0897},
        metrics_after={"cv_mae_mean": 3.0346},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 5-predictor uniform-selected set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 3.0346.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 49,
    "learning_rate": 0.04211208724730419,
    "num_leaves": 62,
    "max_depth": 6,
    "min_child_samples": 11,
    "subsample": 0.6190215422270804,
    "subsample_freq": 1,
    "colsample_bytree": 0.7610373169800141,
    "reg_alpha": 0.032621101944747564,
    "reg_lambda": 0.01492191446502865,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP11(GainModel):
    """TROG-2 receptive-grammar gain predictors — baseline (all data, MAE-tuned).

    Uses a feature-selected subset of :attr:`Predictors.DEFAULT_GAIN`
    (``trog`` is already a member, so the GainModel auto-include
    is a no-op) with MAE-tuned hyperparameters and no outlier
    exclusion. Feature selection was applied (2026-06-21 uniform); see the SelectionStep and the module docstring.
    """

    model_id = "lrp11"
    target_var = V.TROG_GAIN
    description = (
        "LightGBM — TROG-2 (receptive grammar) gain predictors "
        "(5 predictors, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    selection_steps = _SELECTION_STEPS
    shap_scatter_specs = [
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    ]
    notes = (
        "Exploratory model for trog_gain (gain). Uniform feature selection (2026-06-21) from the full 34-predictor DEFAULT_GAIN set to 5 predictors (distance-correlation redundancy filter + importance noise-floor cut; baseline force-kept; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 3.090 -> 3.035). Gain models are near-noise (baseline-driven regression to the mean) - treat the reduced ranking as exploratory. See notes/202606211200-uniform-gb-fs.md."
    )
