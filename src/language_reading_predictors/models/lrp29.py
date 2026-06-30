# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP29: Predictors of Early Repetition Battery total repetition gains (``erbto_gain``).

``erbto`` is the ERB total score from the Early Repetition Battery
(a repetition task indexing verbal / phonological short-term
memory). It is a composite of ``erbword``, ``erbnw`` (in the
candidate pool), but gain targets are near-noise (regression-to-
the-mean dominates), so the level model carries the same-
instrument check.

The gain target spans min -16.0, max 15.0, median 2.00, mean 2.05,
std 5.04, skew -0.25, with ~27% negative and ~7% zero observations
(n = 147). Regression from the mean dominates gain targets across
the suite.

This is an exploratory gradient-boosting discovery model on the
same footing as LRP01–22: it asks how predictable total repetition
is and from what, to inform whether the shared DAG needs a verbal
/ phonological short-term memory node. It is not a causal or
intention-to-treat estimate.

Uniform feature selection (2026-06-23): reduced from the full
35-predictor DEFAULT_GAIN set to 10 predictors via a distance-
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
            V.GROUP, V.AREA, V.GENDER, V.APTGRAM, V.APTINFO, V.B1EXTO,
            V.B1RETO, V.CELF, V.EOWPVT, V.ERBWORD, V.NONWORD,
            V.BLENDING, V.SPPHON, V.TROG, V.YARCLET, V.DEAPPIN,
            V.DEAPPVO, V.DEAPPFI, V.EWRSWR, V.BEHAV, V.ATTEND, V.VISION,
            V.HEARING, V.NUMCHIL, V.AGEBOOKS
        ],
        notes=(
            "Uniform feature selection (2026-06-23): from the full 35-predictor DEFAULT_GAIN set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). The baseline measure was force-kept (regression-to-the-mean anchor). The standardised instrument was preferred over its bespoke taught sibling (``rowpvt`` <- ``b1reto``). Reduces to 10 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Same method as the LRP01–22 suite; scripts/rank_predictors.py (the full-set ranking that supersedes the retired hard-selection pass)."
        ),
        date="2026-06-23",
        metrics_before={"cv_mae_mean": 3.5817},
        metrics_after={"cv_mae_mean": 3.5552},
    ),
]


# ── hyperparameters (MAE-tuned on the reduced set) ───────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.05735427437840209,
    "num_leaves": 28,
    "max_depth": 11,
    "min_child_samples": 4,
    "subsample": 0.6232029530855322,
    "colsample_bytree": 0.9412725584011916,
    "reg_alpha": 0.0012049430230700346,
    "reg_lambda": 0.0037881121608362768,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 22,
}


class LRP29(GainModel):
    """Early Repetition Battery total repetition gains predictors — baseline (MAE-tuned)."""

    model_id = "lrp29"
    target_var = V.ERBTO_GAIN
    description = (
        "LightGBM — Early Repetition Battery total repetition gains predictors (10 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for erbto_gain (gain). Uniform feature selection (2026-06-23) from the full 35-predictor DEFAULT_GAIN set to 10 predictors (distance-correlation redundancy filter + importance noise-floor cut; baseline force-kept; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 3.548). Gain models are near-noise (baseline-driven regression to the mean) — treat the reduced ranking as exploratory."
    )
