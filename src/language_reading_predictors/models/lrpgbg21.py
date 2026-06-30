# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG21: Predictors of DEAP vowel articulation gains (``deappvo_gain``).

``deappvo`` is vowel accuracy from the Diagnostic Evaluation of
Articulation and Phonology (Dodd et al., 2006) — the proportion of
sounds correctly produced in a picture-naming task.

The gain target spans min -23.5, max 17.8, median 0.00, mean 0.19,
std 5.74, skew -0.62, with ~28% negative and ~31% zero
observations (n = 152). Regression from the mean dominates gain
targets across the suite.

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable vowel
articulation is and from what, to inform whether the shared DAG
needs a speech-sound accuracy node. It is not a causal or
intention-to-treat estimate.

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
            V.AREA, V.GENDER, V.AGE, V.APTGRAM, V.APTINFO, V.B1EXTO,
            V.B1RETO, V.EOWPVT, V.ERBNW, V.ERBWORD, V.NONWORD, V.ROWPVT,
            V.YARCSI, V.DEAPPIN, V.EWRSWR, V.BEHAV, V.ATTEND,
            V.AGESPEAK, V.VISION, V.HEARING, V.EARINF, V.NUMCHIL,
            V.AGEBOOKS, V.MUMEDUPOST16, V.DADEDUPOST16
        ],
        notes=(
            "Uniform feature selection (2026-06-23): from the full 34-predictor DEFAULT_GAIN set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). The baseline measure was force-kept (regression-to-the-mean anchor). Reduces to 9 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Same method as the LRPGBG12–22 suite; see scripts/rank_predictors.py (the full-set ranking that supersedes the retired hard-selection pass) and notes/202606230900-predictability-speech-memory-language.md."
        ),
        date="2026-06-23",
        metrics_before={"cv_mae_mean": 3.2935},
        metrics_after={"cv_mae_mean": 3.0024},
    ),
]


# ── hyperparameters (MAE-tuned on the reduced set) ───────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.047737397768147234,
    "num_leaves": 53,
    "max_depth": 3,
    "min_child_samples": 20,
    "subsample": 0.8313091023641015,
    "colsample_bytree": 0.7904910654511403,
    "reg_alpha": 1.9756763955805776,
    "reg_lambda": 1.8953881224366262,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 166,
}


class LRPGBG21(GainModel):
    """DEAP vowel articulation gains predictors — baseline (MAE-tuned)."""

    model_id = "lrpgbg21"
    target_var = V.DEAPPVO_GAIN
    description = (
        "LightGBM — DEAP vowel articulation gains predictors (9 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for deappvo_gain (gain). Uniform feature selection (2026-06-23) from the full 34-predictor DEFAULT_GAIN set to 9 predictors (distance-correlation redundancy filter + importance noise-floor cut; baseline force-kept; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 3.027). Gain models are near-noise (baseline-driven regression to the mean) — treat the reduced ranking as exploratory. See notes/202606230900-predictability-speech-memory-language.md."
    )
