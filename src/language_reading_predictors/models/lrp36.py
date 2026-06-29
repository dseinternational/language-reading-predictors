# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP36: Predictors of DEAP average articulation level (``deappav``).

``deappav`` is the DEAP picture-naming average from the Diagnostic
Evaluation of Articulation and Phonology (Dodd et al., 2006) — the
proportion of sounds correctly produced in a picture-naming task.
It is a composite — ``deappin``, ``deappvo``, ``deappfi`` are its
components and remain in the candidate pool, so a high naive R² is
mechanical (see the same-skill-excluded ranking view, ``ranking_excluding_same_skill.csv``).

The target spans min 47.1, max 94.9, median 77.70, mean 75.27, std
11.30, skew -0.70 (n = 207).

This is an exploratory gradient-boosting discovery model on the
same footing as LRP01–22: it asks how predictable average
articulation is and from what, to inform whether the shared DAG
needs a speech-sound accuracy node. It is not a causal or
intention-to-treat estimate.

Uniform feature selection (2026-06-23): reduced from the full
33-predictor DEFAULT_LEVEL set to 5 predictors via a distance-
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
            V.TIME, V.GROUP, V.AREA, V.AGE, V.APTINFO, V.B1EXTO,
            V.B1RETO, V.CELF, V.EOWPVT, V.ERBWORD, V.NONWORD,
            V.BLENDING, V.ROWPVT, V.SPPHON, V.TROG, V.YARCLET, V.YARCSI,
            V.DEAPPIN, V.EWRSWR, V.BEHAV, V.AGESPEAK, V.VISION,
            V.HEARING, V.EARINF, V.NUMCHIL, V.AGEBOOKS, V.MUMEDUPOST16,
            V.DADEDUPOST16
        ],
        notes=(
            "Uniform feature selection (2026-06-23): from the full 33-predictor DEFAULT_LEVEL set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). Reduces to 5 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Same method as the LRP01–22 suite; see scripts/rank_predictors.py (the full-set ranking that supersedes the retired hard-selection pass) and notes/202606230900-predictability-speech-memory-language.md."
        ),
        date="2026-06-23",
        metrics_before={"cv_mae_mean": 1.6517},
        metrics_after={"cv_mae_mean": 2.6954},
    ),
]


# ── hyperparameters (MAE-tuned on the reduced set) ───────────────────────

_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "learning_rate": 0.039654383749487036,
    "num_leaves": 41,
    "max_depth": 12,
    "min_child_samples": 4,
    "subsample": 0.7337465509959112,
    "colsample_bytree": 0.9969182249863433,
    "reg_alpha": 0.0014782261853234005,
    "reg_lambda": 7.04330979395619,
    "subsample_freq": 1,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 47,
    "n_estimators": 197,
}


class LRP36(LevelModel):
    """DEAP average articulation level predictors — baseline (MAE-tuned)."""

    model_id = "lrp36"
    target_var = V.DEAPPAV
    description = (
        "LightGBM — DEAP average articulation level predictors (5 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for deappav (level). Uniform feature selection (2026-06-23) from the full 33-predictor DEFAULT_LEVEL set to 5 predictors (distance-correlation redundancy filter + importance noise-floor cut; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 2.392). Treat the reduced ranking as exploratory. See notes/202606230900-predictability-speech-memory-language.md."
    )

