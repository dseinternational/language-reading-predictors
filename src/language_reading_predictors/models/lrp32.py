# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP32: Predictors of DEAP initial-consonant articulation level (``deappin``).

``deappin`` is initial-consonant accuracy from the Diagnostic
Evaluation of Articulation and Phonology (Dodd et al., 2006) — the
proportion of sounds correctly produced in a picture-naming task.

The target spans min 25.9, max 92.4, median 71.67, mean 69.19, std
12.69, skew -0.65 (n = 207).

This is an exploratory gradient-boosting discovery model on the
same footing as LRP01–22: it asks how predictable initial-
consonant articulation is and from what, to inform whether the
shared DAG needs a speech-sound accuracy node. It is not a causal
or intention-to-treat estimate.

Uniform feature selection (2026-06-23): reduced from the full
32-predictor DEFAULT_LEVEL set to 8 predictors via a distance-
correlation redundancy filter (dcor >= 0.70) plus an importance
noise-floor cut, then re-tuned. See the SelectionStep below and
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps ────────────────────────────────────────────

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.GROUP, V.AREA, V.GENDER, V.APTINFO, V.B1EXTO, V.B1RETO,
            V.EOWPVT, V.ERBWORD, V.NONWORD, V.BLENDING, V.ROWPVT,
            V.SPPHON, V.TROG, V.YARCLET, V.YARCSI, V.EWRSWR, V.BEHAV,
            V.VISION, V.HEARING, V.EARINF, V.NUMCHIL, V.AGEBOOKS,
            V.MUMEDUPOST16, V.DADEDUPOST16
        ],
        notes=(
            "Uniform feature selection (2026-06-23): from the full 32-predictor DEFAULT_LEVEL set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). Reduces to 8 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Same method as the LRP01–22 suite; scripts/rank_predictors.py (the full-set ranking that supersedes the retired hard-selection pass)."
        ),
        date="2026-06-23",
        metrics_before={"cv_mae_mean": 6.1678},
        metrics_after={"cv_mae_mean": 5.9409},
    ),
]


# ── hyperparameters (MAE-tuned on the reduced set) ───────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.08444251504264087,
    "num_leaves": 24,
    "max_depth": 11,
    "min_child_samples": 16,
    "subsample": 0.7942532101054347,
    "colsample_bytree": 0.704358096256459,
    "reg_alpha": 0.10378084540080104,
    "reg_lambda": 0.48707063622517416,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 67,
}


class LRP32(LevelModel):
    """DEAP initial-consonant articulation level predictors — baseline (MAE-tuned)."""

    model_id = "lrp32"
    target_var = V.DEAPPIN
    description = (
        "LightGBM — DEAP initial-consonant articulation level predictors (8 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for deappin (level). Uniform feature selection (2026-06-23) from the full 32-predictor DEFAULT_LEVEL set to 8 predictors (distance-correlation redundancy filter + importance noise-floor cut; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 5.557). Treat the reduced ranking as exploratory."
    )

