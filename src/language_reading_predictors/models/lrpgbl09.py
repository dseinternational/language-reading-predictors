# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBL09: Predictors of letter-sound knowledge level.

``LRPGBL09`` is the exploratory model for letter-sound knowledge level
(``yarclet``). It is MAE-tuned on the full 32-predictor
:attr:`Predictors.DEFAULT_LEVEL` set (minus the target), with no
outlier exclusion, designed to identify the most important
influences on letter-sound knowledge level.

The target is **left-skewed with a ceiling at 32** (``yarclet`` min
0, max 32, median 21, skewness −0.60, n ≈ 214). The ceiling effect
(95th percentile = 31, 99th = 32) means many children score at or
near the instrument maximum — a different pathology from the
right-skewed / floor-at-0 targets of LRPGBL12 and LRPGBL06. Log / log1p
transforms are inappropriate here because the skew is in the wrong
direction; a reflection-log or quantile objective might be
considered later.

Uniform feature selection (2026-06-21) reduced the predictor set to the SelectionStep below via a distance-correlation redundancy filter plus an importance noise-floor cut.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS, SelectionStep
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Uniform feature-selection history (see the SelectionStep below).
# for the full rationale.

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.B1EXTO, V.B1RETO, V.ERBNW, V.SPPHON, V.GROUP, V.CELF, V.APTGRAM,
            V.MUMEDUPOST16, V.AGESPEAK, V.DADEDUPOST16, V.AGEBOOKS, V.HEARING,
            V.NUMCHIL, V.AREA, V.GENDER, V.VISION, V.BEHAV, V.EARINF, V.ROWPVT,
            V.APTINFO, V.TROG, V.YARCSI, V.DEAPPFI, V.DEAPPIN, V.AGE, V.DEAPPVO
        ],
        notes=(
            "Uniform feature selection (2026-06-21): from the full 32-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). The standardised instrument was preferred over its bespoke taught sibling where it did not reintroduce redundancy. Reduces to 6 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Applied uniformly across all GB models."
        ),
        date="2026-06-21",
        metrics_before={"cv_mae_mean": 4.6203},
        metrics_after={"cv_mae_mean": 4.3643},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 6-predictor uniform-selected set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 4.3643.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 53,
    "learning_rate": 0.037371968457437586,
    "num_leaves": 29,
    "max_depth": 10,
    "min_child_samples": 8,
    "subsample": 0.7474456109857174,
    "subsample_freq": 1,
    "colsample_bytree": 0.9746583954334843,
    "reg_alpha": 1.6618715685357452,
    "reg_lambda": 1.2133803578125177,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRPGBL09(LevelModel):
    """Letter-sound knowledge level predictors — exploratory (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_LEVEL` predictor set
    (minus the target ``yarclet``) with MAE-tuned hyperparameters
    and no outlier exclusion. The starting point for feature
    selection on the letter-sound knowledge level-prediction task.
    """

    model_id = "lrpgbl09"
    target_var = V.YARCLET
    description = (
        "LightGBM — letter-sound knowledge level predictors "
        "(6 predictors, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    selection_steps = _SELECTION_STEPS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for yarclet (level). Uniform feature selection (2026-06-21) from the full 32-predictor DEFAULT_LEVEL set to 6 predictors (distance-correlation redundancy filter + importance noise-floor cut; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 4.620 -> 4.364). Treat the reduced ranking as exploratory."
    )
