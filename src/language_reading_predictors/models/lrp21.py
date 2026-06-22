# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP21: Predictors of DEAP fine-articulation gains.

``LRP21`` is the baseline exploratory model for DEAP fine-
articulation gains (``deappfi_gain``). ``deappfi`` is a
percentage-scale articulation measure from the Diagnostic
Evaluation of Articulation and Phonology (Dodd et al., 2006) —
the proportion of sounds correctly produced when the child is
asked to name pictures. ``deappfi`` specifically scores the
*final* consonant of each word (distinct from ``deappin``
initial and ``deappvo`` voicing).

The target is nearly symmetric but with a heavy two-sided
spread (``deappfi_gain`` min −56.9, max 56.0, median 0.01,
mean 0.84, std 13.28, skewness −0.32, with **~48% negative**
and ~2% zero observations, n ≈ 152). **Heavy regression from
the ceiling is the dominant story** — children at the top of
the scale tend to drop back between timepoints while those at
the floor improve.

DEAP measures have been used as predictors across every other
model in the suite but never as targets until LRP21/22.

Uniform feature selection (2026-06-21): reduced from the full 34-predictor set to 6 predictors via a distance-correlation redundancy filter plus an importance noise-floor cut, then re-tuned. See the SelectionStep below and notes/202606211200-uniform-gb-fs.md.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Feature selection (2026-06-21 uniform): distance-correlation
# redundancy filter + importance noise-floor cut; see the SelectionStep.

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.B1EXTO, V.DEAPPIN, V.BEHAV, V.GROUP, V.AREA, V.VISION, V.EARINF,
            V.HEARING, V.NUMCHIL, V.AGEBOOKS, V.DADEDUPOST16, V.AGESPEAK,
            V.MUMEDUPOST16, V.GENDER, V.YARCSI, V.APTGRAM, V.SPPHON, V.ROWPVT,
            V.BLENDING, V.CELF, V.TIME, V.B1RETO, V.TROG, V.APTINFO, V.ATTEND,
            V.ERBWORD, V.YARCLET, V.ERBNW
        ],
        notes=(
            "Uniform feature selection (2026-06-21): from the full 34-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). The standardised instrument was preferred over its bespoke taught sibling where it did not reintroduce redundancy. The baseline measure was force-kept (regression-to-the-mean anchor). Reduces to 6 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Applied uniformly across all GB models; see notes/202606211200-uniform-gb-fs.md."
        ),
        date="2026-06-21",
        metrics_before={"cv_mae_mean": 8.7223},
        metrics_after={"cv_mae_mean": 8.2945},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 6-predictor uniform-selected set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 8.2945.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 11,
    "learning_rate": 0.18894563671851897,
    "num_leaves": 35,
    "max_depth": 5,
    "min_child_samples": 26,
    "subsample": 0.8495160600596453,
    "subsample_freq": 1,
    "colsample_bytree": 0.8734189036490825,
    "reg_alpha": 0.0015837285659681297,
    "reg_lambda": 0.2283963702135058,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP21(GainModel):
    """DEAP fine-articulation gain predictors — baseline (all data, MAE-tuned).

    Uses a feature-selected subset of :attr:`Predictors.DEFAULT_GAIN`
    (``deappfi`` is already a member, so the GainModel auto-include
    is a no-op) with MAE-tuned hyperparameters and no outlier
    exclusion. Feature selection was applied (2026-06-21 uniform); see the SelectionStep and the module docstring.
    """

    model_id = "lrp21"
    target_var = V.DEAPPFI_GAIN
    description = (
        "LightGBM — DEAP fine-articulation gain predictors "
        "(6 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for deappfi_gain (gain). Uniform feature selection (2026-06-21) from the full 34-predictor DEFAULT_GAIN set to 6 predictors (distance-correlation redundancy filter + importance noise-floor cut; baseline force-kept; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 8.722 -> 8.294). Gain models are near-noise (baseline-driven regression to the mean) - treat the reduced ranking as exploratory. See notes/202606211200-uniform-gb-fs.md."
    )
