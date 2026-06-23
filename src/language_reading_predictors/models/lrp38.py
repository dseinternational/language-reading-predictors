# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP38: Predictors of language sample mean length of utterance level (``lsammlu``).

``lsammlu`` is the mean length of utterance from a coded sample of
the child's spontaneous connected speech.

The target spans min 1.0, max 3.8, median 2.14, mean 2.25, std
0.73, skew 0.31 (n = 106).

This is an exploratory gradient-boosting discovery model on the
same footing as LRP01–22: it asks how predictable mean length of
utterance is and from what, to inform whether the shared DAG needs
a spontaneous connected speech node. It is not a causal or
intention-to-treat estimate. The language-sample measures are
recorded at t1–t2 only, so this level model is doubly exploratory
(≈106 rows, two waves) and no gain model is fitted. The other
language-sample measures are absent from the default predictor
pool (recorded at t1–t2 only), so this model cannot be carried by
same-instrument siblings.

Uniform feature selection (2026-06-23): reduced from the full
33-predictor DEFAULT_LEVEL set to 4 predictors via a distance-
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
            V.TIME, V.GROUP, V.AREA, V.GENDER, V.AGE, V.APTGRAM,
            V.B1EXTO, V.B1RETO, V.CELF, V.EOWPVT, V.ERBWORD, V.NONWORD,
            V.BLENDING, V.ROWPVT, V.SPPHON, V.TROG, V.YARCLET,
            V.DEAPPVO, V.DEAPPFI, V.EWRSWR, V.BEHAV, V.AGESPEAK,
            V.VISION, V.HEARING, V.EARINF, V.NUMCHIL, V.AGEBOOKS,
            V.MUMEDUPOST16, V.DADEDUPOST16
        ],
        notes=(
            "Uniform feature selection (2026-06-23): from the full 33-predictor DEFAULT_LEVEL set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). Reduces to 4 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Same method as the LRP01–22 suite; see scripts/uniform_feature_selection.py and notes/202606230900-predictability-speech-memory-language.md."
        ),
        date="2026-06-23",
        metrics_before={"cv_mae_mean": 0.4222},
        metrics_after={"cv_mae_mean": 0.3899},
    ),
]


# ── hyperparameters (MAE-tuned on the reduced set) ───────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.06395203888918281,
    "num_leaves": 42,
    "max_depth": 6,
    "min_child_samples": 9,
    "subsample": 0.9489941264970362,
    "colsample_bytree": 0.8413850745885425,
    "reg_alpha": 4.54924369762865,
    "reg_lambda": 0.35495819366481224,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 48,
}


class LRP38(LevelModel):
    """language sample mean length of utterance level predictors — baseline (MAE-tuned)."""

    model_id = "lrp38"
    target_var = V.LSAMMLU
    description = (
        "LightGBM — language sample mean length of utterance level predictors (4 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for lsammlu (level). Uniform feature selection (2026-06-23) from the full 33-predictor DEFAULT_LEVEL set to 4 predictors (distance-correlation redundancy filter + importance noise-floor cut; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 0.393). Treat the reduced ranking as exploratory. See notes/202606230900-predictability-speech-memory-language.md."
    )
