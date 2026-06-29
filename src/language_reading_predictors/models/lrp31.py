# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP31: Predictors of DEAP initial-consonant articulation gains (``deappin_gain``).

``deappin`` is initial-consonant accuracy from the Diagnostic
Evaluation of Articulation and Phonology (Dodd et al., 2006) — the
proportion of sounds correctly produced in a picture-naming task.

The gain target spans min -12.1, max 23.8, median 1.18, mean 1.02,
std 6.48, skew 0.30, with ~42% negative and ~5% zero observations
(n = 152). Regression from the mean dominates gain targets across
the suite.

This is an exploratory gradient-boosting discovery model on the
same footing as LRP01–22: it asks how predictable initial-
consonant articulation is and from what, to inform whether the
shared DAG needs a speech-sound accuracy node. It is not a causal
or intention-to-treat estimate.

Uniform feature selection (2026-06-23): reduced from the full
34-predictor DEFAULT_GAIN set to 9 predictors via a distance-
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
            V.GROUP, V.GENDER, V.AGE, V.APTGRAM, V.APTINFO, V.B1EXTO,
            V.CELF, V.ERBNW, V.ERBWORD, V.BLENDING, V.ROWPVT, V.TROG,
            V.YARCSI, V.DEAPPVO, V.DEAPPFI, V.EWRSWR, V.BEHAV, V.ATTEND,
            V.AGESPEAK, V.VISION, V.HEARING, V.EARINF, V.NUMCHIL,
            V.AGEBOOKS, V.MUMEDUPOST16
        ],
        notes=(
            "Uniform feature selection (2026-06-23): from the full 34-predictor DEFAULT_GAIN set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). The baseline measure was force-kept (regression-to-the-mean anchor). Reduces to 9 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Same method as the LRP01–22 suite; see scripts/rank_predictors.py (the full-set ranking that supersedes the retired hard-selection pass) and notes/202606230900-predictability-speech-memory-language.md."
        ),
        date="2026-06-23",
        metrics_before={"cv_mae_mean": 4.9299},
        metrics_after={"cv_mae_mean": 4.7070},
    ),
]


# ── hyperparameters (MAE-tuned on the reduced set) ───────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.12870587965203345,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 32,
    "subsample": 0.9903120177414865,
    "colsample_bytree": 0.7839600465161084,
    "reg_alpha": 0.09725138724589176,
    "reg_lambda": 0.11980050171655834,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 83,
}


class LRP31(GainModel):
    """DEAP initial-consonant articulation gains predictors — baseline (MAE-tuned)."""

    model_id = "lrp31"
    target_var = V.DEAPPIN_GAIN
    description = (
        "LightGBM — DEAP initial-consonant articulation gains predictors (9 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for deappin_gain (gain). Uniform feature selection (2026-06-23) from the full 34-predictor DEFAULT_GAIN set to 9 predictors (distance-correlation redundancy filter + importance noise-floor cut; baseline force-kept; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 4.738). Gain models are near-noise (baseline-driven regression to the mean) — treat the reduced ranking as exploratory. See notes/202606230900-predictability-speech-memory-language.md."
    )
