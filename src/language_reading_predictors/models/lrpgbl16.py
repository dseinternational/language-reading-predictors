# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL16: Predictors of DEAP fine-articulation level.

``LRPGBL16`` is the baseline exploratory model for DEAP fine-
articulation level (``deappfi``). ``deappfi`` is a
percentage-scale articulation measure from the Diagnostic
Evaluation of Articulation and Phonology (Dodd et al., 2006) —
the proportion of sounds correctly produced when the child is
asked to name pictures. ``deappfi`` specifically scores the
*final* consonant of each word (distinct from ``deappin``
initial and ``deappvo`` voicing).

The target is **left-skewed with ceiling** (``deappfi`` min
5.4, max 95.2, median 66.6, mean 60.3, std 20.9, skewness
**−0.87**, n ≈ 207). Ceiling effects are possible at the top
end — several children score in the 90s. No zeros in the
sample (unlike the reading targets where floor effects
dominate).

DEAP measures have been used as predictors across every other
model in the suite but never as targets until LRPGBL16/22. First
articulation-domain target.

Uniform feature selection (2026-06-21): reduced from the full 32-predictor set to 7 predictors via a distance-correlation redundancy filter plus an importance noise-floor cut, then re-tuned. See the SelectionStep below.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS, SelectionStep
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Feature selection (2026-06-21 uniform): distance-correlation
# redundancy filter + importance noise-floor cut; see the SelectionStep.

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.GENDER, V.APTINFO, V.VISION, V.HEARING, V.EARINF, V.YARCSI,
            V.AGEBOOKS, V.BEHAV, V.SPPHON, V.CELF, V.AREA, V.GROUP, V.NUMCHIL,
            V.DEAPPVO, V.B1EXTO, V.MUMEDUPOST16, V.AGE, V.NONWORD, V.EOWPVT,
            V.ROWPVT, V.B1RETO, V.BLENDING, V.APTGRAM, V.ERBNW, V.ERBWORD
        ],
        notes=(
            "Uniform feature selection (2026-06-21): from the full 32-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). Reduces to 7 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Applied uniformly across all GB models."
        ),
        date="2026-06-21",
        metrics_before={"cv_mae_mean": 9.9866},
        metrics_after={"cv_mae_mean": 9.9368},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 7-predictor uniform-selected set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 9.9368.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 133,
    "learning_rate": 0.028088238000348178,
    "num_leaves": 37,
    "max_depth": 9,
    "min_child_samples": 6,
    "subsample": 0.6005851369100905,
    "subsample_freq": 1,
    "colsample_bytree": 0.9665121302617031,
    "reg_alpha": 0.0026948627929153086,
    "reg_lambda": 0.05949393188300806,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRPGBL16(LevelModel):
    """DEAP fine-articulation level predictors — baseline (all data, MAE-tuned).

    Uses a feature-selected subset of :attr:`Predictors.DEFAULT_LEVEL`
    (minus the target ``deappfi``) with MAE-tuned hyperparameters
    and no outlier exclusion. Feature selection was applied (2026-06-21 uniform); see the SelectionStep and the module docstring.
    """

    model_id = "lrpgbl16"
    target_var = V.DEAPPFI
    description = (
        "LightGBM — DEAP fine-articulation level predictors "
        "(7 predictors, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    selection_steps = _SELECTION_STEPS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for deappfi (level). Uniform feature selection (2026-06-21) from the full 32-predictor DEFAULT_LEVEL set to 7 predictors (distance-correlation redundancy filter + importance noise-floor cut; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 9.987 -> 9.937). Treat the reduced ranking as exploratory."
    )


# Same-skill (null) variant: MAE-tuned on the 6-predictor set after dropping
# deappin — DEAP initial-consonant accuracy, scored from the same picture-
# naming sample as the target deappfi. Tuner-inner CV MAE rises 9.958 -> 16.126
# and the chosen model is ~3 trees: removing the DEAP sibling collapses the
# signal to ~null (the deappfi finding).
_LGBM_MAE_PARAMS_NOCONSTRUCT: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 3,
    "learning_rate": 0.10821526553081377,
    "num_leaves": 49,
    "max_depth": 8,
    "min_child_samples": 6,
    "subsample": 0.6917683552854258,
    "subsample_freq": 1,
    "colsample_bytree": 0.7580400397784406,
    "reg_alpha": 0.17579698004549832,
    "reg_lambda": 0.0016202157050561594,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRPGBL16NoConstruct(LRPGBL16):
    """deappfi — same-skill (null) variant: DEAP same-sample sibling deappin dropped."""

    model_id = "lrpgbl16_noconstruct"
    variant_of = "lrpgbl16"
    description = (
        "LightGBM — deappfi predictors "
        "(6 predictors, same-skill reduced: DEAP sibling dropped -> null)"
    )
    params = _LGBM_MAE_PARAMS_NOCONSTRUCT
    selection_steps = [
        SelectionStep(
            removed=[V.DEAPPIN],
            notes=(
                "Same-skill (null) variant of lrpgbl16: drops deappin — DEAP initial-consonant accuracy, scored from the same picture-naming sample as the target deappfi (final-consonant accuracy), i.e. a parallel scoring of the same articulation performance. This is the deappfi null result: with the same-instrument DEAP sibling removed, tuner-inner CV MAE rises from 9.96 to 16.13 and the chosen model is ~3 trees (near-constant), confirming there is no non-articulation predictor of final-consonant accuracy at this n. The within-DEAP primary (lrpgbl16) is kept as a convergent-validity reference; this variant documents the null."
            ),
            date="2026-06-21",
            metrics_before={"cv_mae_mean": 9.9583},
            metrics_after={"cv_mae_mean": 16.1262},
        ),
    ]
    notes = (
        "Same-skill (null) variant of lrpgbl16: drops deappin (DEAP initial-consonant accuracy, same picture-naming sample as the target deappfi). Removing the parallel DEAP scoring collapses CV (tuner-inner MAE 9.96 -> 16.13, ~3-tree near-constant model): no non-articulation predictor of final-consonant accuracy at this n. The within-DEAP primary is kept as a convergent-validity reference."
    )
