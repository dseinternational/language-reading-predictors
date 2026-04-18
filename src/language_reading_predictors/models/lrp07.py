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

_SELECTION_STEPS = [
    SelectionStep(
        removed=[
            # Tier A — ≤ 0.005 importance in the 34-predictor MAE tune
            # (11 at exactly 0.000, 3 slightly negative)
            V.GENDER, V.EWRSWR, V.APTINFO, V.SPPHON, V.YARCSI,
            V.EOWPVT, V.BEHAV, V.YARCLET, V.TIME, V.VISION,
            V.EARINF, V.NONWORD, V.AREA, V.GROUP, V.HEARING,
            V.AGESPEAK, V.AGEBOOKS, V.APTGRAM, V.ERBNW, V.ERBWORD,
            V.NUMCHIL, V.AGE,
        ],
        notes=(
            "Aggressive one-shot cut from 34 → 12 predictors. Drops "
            "22 Tier-A features with importance ≤ 0.005 under the "
            "42-tree MAE-tuned model (11 at exactly 0.000 including "
            "eowpvt, yarclet, time, nonword, spphon, yarcsi, ewrswr, "
            "aptinfo — striking given the last three are top "
            "predictors on other tasks). eowpvt in particular is "
            "redundant with rowpvt under the few-fast-shallow tune; "
            "aptinfo/ewrswr/yarclet/time route through rowpvt rather "
            "than providing independent gain signal. The gain-vs-level "
            "stories diverge sharply for receptive vocabulary: LRP08 "
            "language-cluster predictors (aptinfo, eowpvt, celf, trog) "
            "route differently for gain prediction."
        ),
        date="2026-04-17",
        metrics_before={"cv_mae_mean": 7.003},
        metrics_after={"cv_mae_mean": 6.803},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 12-predictor Select01 set, no outlier exclusion
# (Optuna 150 trials, 10-split GroupKFold, seed 47, scoring=mae,
# lgbm_objective=mae). Tuner-inner CV MAE 6.7536 ± 1.2224. n=161.
# Supersedes the 34-predictor tune (tuner-inner 6.9827).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 31,
    "learning_rate": 0.1426966134045708,
    "num_leaves": 22,
    "max_depth": 6,
    "min_child_samples": 9,
    "subsample": 0.6664929179252206,
    "subsample_freq": 1,
    "colsample_bytree": 0.8199943081859693,
    "reg_alpha": 0.5510234406492407,
    "reg_lambda": 0.41628274953116934,
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
        "(12 predictors, MAE-tuned, no outlier exclusion)"
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
