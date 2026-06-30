# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP25: Predictors of Early Repetition Battery nonword repetition gains (``erbnw_gain``).

``erbnw`` is the number of nonwords correctly repeated from the
Early Repetition Battery (a repetition task indexing verbal /
phonological short-term memory).

The gain target spans min -6.0, max 10.0, median 1.00, mean 1.26,
std 3.15, skew 0.04, with ~22% negative and ~20% zero observations
(n = 147). Regression from the mean dominates gain targets across
the suite.

This is an exploratory gradient-boosting discovery model on the
same footing as LRP01–22: it asks how predictable nonword
repetition is and from what, to inform whether the shared DAG
needs a verbal / phonological short-term memory node. It is not a
causal or intention-to-treat estimate.

Uniform feature selection (2026-06-23): reduced from the full
34-predictor DEFAULT_GAIN set to 5 predictors via a distance-
correlation redundancy filter (dcor >= 0.70) plus an importance
noise-floor cut, then re-tuned. See the SelectionStep below and
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps ────────────────────────────────────────────

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.TIME, V.GROUP, V.AREA, V.GENDER, V.AGE, V.APTGRAM,
            V.APTINFO, V.B1EXTO, V.B1RETO, V.CELF, V.EOWPVT, V.ERBWORD,
            V.NONWORD, V.BLENDING, V.ROWPVT, V.SPPHON, V.TROG, V.YARCSI,
            V.DEAPPIN, V.DEAPPVO, V.EWRSWR, V.BEHAV, V.ATTEND, V.VISION,
            V.HEARING, V.EARINF, V.NUMCHIL, V.AGEBOOKS, V.DADEDUPOST16
        ],
        notes=(
            "Uniform feature selection (2026-06-23): from the full 34-predictor DEFAULT_GAIN set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). The baseline measure was force-kept (regression-to-the-mean anchor). Reduces to 5 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Same method as the LRP01–22 suite; scripts/rank_predictors.py (the full-set ranking that supersedes the retired hard-selection pass)."
        ),
        date="2026-06-23",
        metrics_before={"cv_mae_mean": 2.2711},
        metrics_after={"cv_mae_mean": 2.2464},
    ),
]


# ── hyperparameters (MAE-tuned on the reduced set) ───────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.11750810775578611,
    "num_leaves": 45,
    "max_depth": 4,
    "min_child_samples": 17,
    "subsample": 0.6097606249206079,
    "colsample_bytree": 0.758360640735803,
    "reg_alpha": 0.01182776312555698,
    "reg_lambda": 0.18152030603308358,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 82,
}


class LRP25(GainModel):
    """Early Repetition Battery nonword repetition gains predictors — baseline (MAE-tuned)."""

    model_id = "lrp25"
    target_var = V.ERBNW_GAIN
    description = (
        "LightGBM — Early Repetition Battery nonword repetition gains predictors (5 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for erbnw_gain (gain). Uniform feature selection (2026-06-23) from the full 34-predictor DEFAULT_GAIN set to 5 predictors (distance-correlation redundancy filter + importance noise-floor cut; baseline force-kept; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 2.250). Gain models are near-noise (baseline-driven regression to the mean) — treat the reduced ranking as exploratory."
    )
