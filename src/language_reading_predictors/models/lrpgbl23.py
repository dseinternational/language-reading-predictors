# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL23: Predictors of DEAP composite articulation level (``deapp_c``).

``deapp_c`` is the DEAP picture-naming composite from the
Diagnostic Evaluation of Articulation and Phonology (Dodd et al.,
2006) — the proportion of sounds correctly produced in a picture-
naming task. It is a composite — ``deappin``, ``deappvo``,
``deappfi`` are its components and remain in the candidate pool,
so a high naive R² is mechanical (see the same-skill-excluded
ranking view, ``ranking_excluding_same_skill.csv``).

The target spans min 141.2, max 284.6, median 234.56, mean 225.33,
std 34.54, skew -0.67 (n = 207).

This is an exploratory gradient-boosting discovery model on the
same footing as LRPGBG12–22: it asks how predictable composite
articulation is and from what, to inform whether the shared DAG
needs a speech-sound accuracy node. It is not a causal or
intention-to-treat estimate.

Uniform feature selection (2026-06-23): reduced from the full
33-predictor DEFAULT_LEVEL set to 9 predictors via a distance-
correlation redundancy filter (dcor >= 0.70) plus an importance
noise-floor cut, then re-tuned. See the SelectionStep below.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps ────────────────────────────────────────────

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.TIME, V.AREA, V.GENDER, V.AGE, V.APTGRAM, V.APTINFO,
            V.B1EXTO, V.B1RETO, V.CELF, V.ERBWORD, V.NONWORD, V.ROWPVT,
            V.TROG, V.YARCLET, V.YARCSI, V.DEAPPIN, V.EWRSWR, V.BEHAV,
            V.AGESPEAK, V.HEARING, V.EARINF, V.AGEBOOKS, V.MUMEDUPOST16,
            V.DADEDUPOST16
        ],
        notes=(
            "Uniform feature selection (2026-06-23): from the full 33-predictor DEFAULT_LEVEL set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). Reduces to 9 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Same method as the LRPGBG12–22 suite; see scripts/rank_predictors.py (the full-set ranking that supersedes the retired hard-selection pass)."
        ),
        date="2026-06-23",
        metrics_before={"cv_mae_mean": 4.0242},
        metrics_after={"cv_mae_mean": 7.0796},
    ),
]


# ── hyperparameters (MAE-tuned on the reduced set) ───────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.06802196540541967,
    "num_leaves": 22,
    "max_depth": 6,
    "min_child_samples": 7,
    "subsample": 0.6253421582972084,
    "colsample_bytree": 0.9910781779426832,
    "reg_alpha": 0.46377779535901403,
    "reg_lambda": 0.052157411074455234,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 164,
}


class LRPGBL23(LevelModel):
    """DEAP composite articulation level predictors — baseline (MAE-tuned)."""

    model_id = "lrpgbl23"
    target_var = V.DEAPP_C
    description = (
        "LightGBM — DEAP composite articulation level predictors (9 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for deapp_c (level). Uniform feature selection (2026-06-23) from the full 33-predictor DEFAULT_LEVEL set to 9 predictors (distance-correlation redundancy filter + importance noise-floor cut; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 6.630). Treat the reduced ranking as exploratory."
    )

