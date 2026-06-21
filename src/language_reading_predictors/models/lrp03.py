# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP03: Predictors of expressive-vocabulary gains.

``LRP03`` is the exploratory model for expressive-vocabulary gains
(``eowpvt_gain``). It is MAE-tuned on the full 34-predictor
:attr:`Predictors.DEFAULT_GAIN` set (plus the auto-included base
variable ``eowpvt``), with no outlier exclusion, designed to
identify the most important influences on expressive-vocabulary
gains.

The target is signed (``eowpvt_gain`` min ≈ −13, max ≈ 28, median 3,
skewness 0.32, with ~25% negative observations and n ≈ 161). That's
much milder skew than LRP01's ``ewrswr_gain`` and nearly symmetric —
a log / signed-log transform may or may not help and is a question
for future investigation.

The predictor set will be reduced by iterative importance-based
feature selection under the MAE-tuned params (see
``notes/202604171127-lpr03-feature-selection.md``). This is the
initial tuned baseline; no feature-selection steps yet.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Documents the 34 → 11 feature-selection history under MAE-tuned params
# with no outlier exclusion (n=161).
# See notes/202604171127-lpr03-feature-selection.md for the full rationale.

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.APTINFO, V.DEAPPIN, V.ERBWORD, V.ERBNW, V.NONWORD, V.AGESPEAK,
            V.VISION, V.EARINF, V.GROUP, V.CELF, V.GENDER, V.AREA, V.DADEDUPOST16,
            V.HEARING, V.NUMCHIL, V.AGEBOOKS, V.MUMEDUPOST16, V.BEHAV, V.DEAPPFI,
            V.YARCSI, V.B1RETO, V.EWRSWR, V.TIME, V.APTGRAM, V.BLENDING, V.B1EXTO,
            V.YARCLET, V.SPPHON, V.ROWPVT, V.ATTEND, V.AGE
        ],
        notes=(
            "Uniform feature selection (2026-06-21): from the full 34-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). The baseline measure was force-kept (regression-to-the-mean anchor). Reduces to 3 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Applied uniformly across all GB models; see notes/202606211200-uniform-gb-fs.md."
        ),
        date="2026-06-21",
        metrics_before={"cv_mae_mean": 5.1631},
        metrics_after={"cv_mae_mean": 5.2216},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 3-predictor uniform-selected set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 5.2216.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 175,
    "learning_rate": 0.012391770398684195,
    "num_leaves": 43,
    "max_depth": 3,
    "min_child_samples": 16,
    "subsample": 0.7649062066306234,
    "subsample_freq": 1,
    "colsample_bytree": 0.8825720972911073,
    "reg_alpha": 1.9954223064442116,
    "reg_lambda": 0.18963037439304298,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRP03(GainModel):
    """Expressive-vocabulary gain predictors — exploratory (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_GAIN` predictor set plus
    the base variable ``eowpvt`` (auto-included via :class:`GainModel`)
    with MAE-tuned hyperparameters and no outlier exclusion. The
    starting point for feature selection on the expressive-vocabulary
    gain-prediction task.
    """

    model_id = "lrp03"
    target_var = V.EOWPVT_GAIN
    description = (
        "LightGBM — expressive-vocabulary gain predictors "
        "(3 predictors, MAE-tuned, no outlier exclusion)"
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
        "expressive-vocabulary gains (eowpvt_gain). MAE-tuned on the "
        "full 34-predictor set (DEFAULT_GAIN + eowpvt) without outlier "
        "exclusion so importance rankings reflect the full range of "
        "outcomes. Feature-selection variants to follow. See "
        "notes/202604171127-lpr03-feature-selection.md."
    )
