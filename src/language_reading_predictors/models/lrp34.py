# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP34: Predictors of DEAP vowel articulation level (``deappvo``).

``deappvo`` is vowel accuracy from the Diagnostic Evaluation of
Articulation and Phonology (Dodd et al., 2006) — the proportion of
sounds correctly produced in a picture-naming task.

The target spans min 68.8, max 100.0, median 98.04, mean 95.86,
std 5.72, skew -2.13 (n = 207).

This is an exploratory gradient-boosting discovery model on the
same footing as LRP01–22: it asks how predictable vowel
articulation is and from what, to inform whether the shared DAG
needs a speech-sound accuracy node. It is not a causal or
intention-to-treat estimate.

Uniform feature selection (2026-06-23): reduced from the full
32-predictor DEFAULT_LEVEL set to 9 predictors via a distance-
correlation redundancy filter (dcor >= 0.70) plus an importance
noise-floor cut, then re-tuned. See the SelectionStep below and
notes/202606230900-predictability-speech-memory-language.md.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps ────────────────────────────────────────────

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.GROUP, V.AREA, V.GENDER, V.AGE, V.APTGRAM, V.B1EXTO,
            V.B1RETO, V.CELF, V.EOWPVT, V.ERBNW, V.ERBWORD, V.NONWORD,
            V.BLENDING, V.SPPHON, V.TROG, V.YARCSI, V.DEAPPFI, V.EWRSWR,
            V.BEHAV, V.VISION, V.EARINF, V.MUMEDUPOST16, V.DADEDUPOST16
        ],
        notes=(
            "Uniform feature selection (2026-06-23): from the full 32-predictor DEFAULT_LEVEL set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). Reduces to 9 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Same method as the LRP01–22 suite; see scripts/rank_predictors.py (the full-set ranking that supersedes the retired hard-selection pass) and notes/202606230900-predictability-speech-memory-language.md."
        ),
        date="2026-06-23",
        metrics_before={"cv_mae_mean": 3.3875},
        metrics_after={"cv_mae_mean": 3.1617},
    ),
]


# ── hyperparameters (MAE-tuned on the reduced set) ───────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.05191805093255059,
    "num_leaves": 38,
    "max_depth": 10,
    "min_child_samples": 5,
    "subsample": 0.7988963079903608,
    "colsample_bytree": 0.953670356355704,
    "reg_alpha": 0.008098655906001905,
    "reg_lambda": 9.201037681033354,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 156,
}


class LRP34(LevelModel):
    """DEAP vowel articulation level predictors — baseline (MAE-tuned)."""

    model_id = "lrp34"
    target_var = V.DEAPPVO
    description = (
        "LightGBM — DEAP vowel articulation level predictors (9 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for deappvo (level). Uniform feature selection (2026-06-23) from the full 32-predictor DEFAULT_LEVEL set to 9 predictors (distance-correlation redundancy filter + importance noise-floor cut; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 3.026). Treat the reduced ranking as exploratory. See notes/202606230900-predictability-speech-memory-language.md."
    )

