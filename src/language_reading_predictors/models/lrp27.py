# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP27: Predictors of Early Repetition Battery word repetition gains (``erbword_gain``).

``erbword`` is the number of real words correctly repeated from
the Early Repetition Battery (a repetition task indexing verbal /
phonological short-term memory).

The gain target spans min -9.0, max 12.0, median 1.00, mean 1.03,
std 3.00, skew 0.35, with ~22% negative and ~22% zero observations
(n = 148). Regression from the mean dominates gain targets across
the suite.

This is an exploratory gradient-boosting discovery model on the
same footing as LRP01–22: it asks how predictable word repetition
is and from what, to inform whether the shared DAG needs a verbal
/ phonological short-term memory node. It is not a causal or
intention-to-treat estimate.

Uniform feature selection (2026-06-23): reduced from the full
34-predictor DEFAULT_GAIN set to 11 predictors via a distance-
correlation redundancy filter (dcor >= 0.70) plus an importance
noise-floor cut, then re-tuned. See the SelectionStep below and
notes/202606230900-predictability-speech-memory-language.md.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps ────────────────────────────────────────────

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.TIME, V.GROUP, V.AREA, V.GENDER, V.AGE, V.APTINFO,
            V.B1EXTO, V.ERBNW, V.NONWORD, V.BLENDING, V.ROWPVT,
            V.SPPHON, V.TROG, V.YARCLET, V.YARCSI, V.DEAPPIN, V.DEAPPVO,
            V.BEHAV, V.VISION, V.HEARING, V.EARINF, V.NUMCHIL,
            V.DADEDUPOST16
        ],
        notes=(
            "Uniform feature selection (2026-06-23): from the full 34-predictor DEFAULT_GAIN set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). The baseline measure was force-kept (regression-to-the-mean anchor). Reduces to 11 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Same method as the LRP01–22 suite; see scripts/uniform_feature_selection.py and notes/202606230900-predictability-speech-memory-language.md."
        ),
        date="2026-06-23",
        metrics_before={"cv_mae_mean": 1.9006},
        metrics_after={"cv_mae_mean": 1.7915},
    ),
]


# ── hyperparameters (MAE-tuned on the reduced set) ───────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.04590079068925787,
    "num_leaves": 49,
    "max_depth": 4,
    "min_child_samples": 14,
    "subsample": 0.615810127489842,
    "colsample_bytree": 0.9408669501013529,
    "reg_alpha": 0.0063643336825072545,
    "reg_lambda": 3.0856651422754178,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 136,
}


class LRP27(GainModel):
    """Early Repetition Battery word repetition gains predictors — baseline (MAE-tuned)."""

    model_id = "lrp27"
    target_var = V.ERBWORD_GAIN
    description = (
        "LightGBM — Early Repetition Battery word repetition gains predictors (11 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for erbword_gain (gain). Uniform feature selection (2026-06-23) from the full 34-predictor DEFAULT_GAIN set to 11 predictors (distance-correlation redundancy filter + importance noise-floor cut; baseline force-kept; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 1.788). Gain models are near-noise (baseline-driven regression to the mean) — treat the reduced ranking as exploratory. See notes/202606230900-predictability-speech-memory-language.md."
    )
