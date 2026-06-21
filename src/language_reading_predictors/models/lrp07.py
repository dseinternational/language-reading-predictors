# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP07: Predictors of receptive vocabulary gains.

``LRP07`` is the exploratory model for receptive vocabulary gains
(``rowpvt_gain``). It is MAE-tuned on the 12-predictor Select01 set
(down from the original 34-predictor
:attr:`Predictors.DEFAULT_GAIN` + ``rowpvt`` base), with no outlier
exclusion, designed to identify the most important influences on
receptive vocabulary gains.

The target is **essentially symmetric** (``rowpvt_gain`` min ≈ −20,
max ≈ 34, median 5, mean 3.84, skewness 0.04, with ~29% negative
and ~3% zero observations, n ≈ 161). Cleaner distribution than any
previous gain target — no skew and no pile-up at zero.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Documents the 34 → 12 feature-selection history under MAE-tuned
# params with no outlier exclusion (n=161).
# See notes/202604171715-lrp07-feature-selection.md for the full rationale.

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.MUMEDUPOST16, V.AGE, V.NUMCHIL, V.GROUP, V.GENDER, V.NONWORD,
            V.EARINF, V.AREA, V.VISION, V.EWRSWR, V.HEARING, V.SPPHON, V.AGEBOOKS,
            V.ERBNW, V.BLENDING, V.DEAPPVO, V.DADEDUPOST16, V.BEHAV, V.YARCSI,
            V.EOWPVT, V.AGESPEAK, V.APTINFO, V.DEAPPIN, V.APTGRAM, V.YARCLET,
            V.DEAPPFI, V.B1EXTO, V.ERBWORD, V.B1RETO, V.TIME, V.CELF, V.ATTEND
        ],
        notes=(
            "Uniform feature selection (2026-06-21): from the full 34-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). The baseline measure was force-kept (regression-to-the-mean anchor). Reduces to 2 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Applied uniformly across all GB models; see notes/202606211200-uniform-gb-fs.md."
        ),
        date="2026-06-21",
        metrics_before={"cv_mae_mean": 7.2241},
        metrics_after={"cv_mae_mean": 7.1434},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 2-predictor uniform-selected set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 7.1434.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 26,
    "learning_rate": 0.17328403199352835,
    "num_leaves": 23,
    "max_depth": 10,
    "min_child_samples": 37,
    "subsample": 0.6393895136146711,
    "subsample_freq": 1,
    "colsample_bytree": 0.7127905387176482,
    "reg_alpha": 0.5995768124191261,
    "reg_lambda": 2.3376116482336817,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRP07(GainModel):
    """Receptive vocabulary gain predictors — exploratory (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_GAIN` predictor set plus
    the base variable ``rowpvt`` (auto-included via :class:`GainModel`)
    with MAE-tuned hyperparameters and no outlier exclusion. The
    starting point for feature selection on the receptive vocabulary
    gain-prediction task.
    """

    model_id = "lrp07"
    target_var = V.ROWPVT_GAIN
    description = (
        "LightGBM — receptive vocabulary gain predictors "
        "(2 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for identifying important predictors of "
        "receptive vocabulary gains (rowpvt_gain). MAE-tuned on the "
        "12-predictor Select01 set (down from DEFAULT_GAIN + rowpvt's "
        "original 34) without outlier exclusion so importance "
        "rankings reflect the full range of outcomes. Target is "
        "essentially symmetric (skew 0.04) — cleanest gain target so "
        "far. See notes/202604171715-lrp07-feature-selection.md."
    )
