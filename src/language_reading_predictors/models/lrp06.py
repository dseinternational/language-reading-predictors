# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP06: Predictors of letter-sound knowledge level.

``LRP06`` is the exploratory model for letter-sound knowledge level
(``yarclet``). It is MAE-tuned on the full 32-predictor
:attr:`Predictors.DEFAULT_LEVEL` set (minus the target), with no
outlier exclusion, designed to identify the most important
influences on letter-sound knowledge level.

The target is **left-skewed with a ceiling at 32** (``yarclet`` min
0, max 32, median 21, skewness −0.60, n ≈ 214). The ceiling effect
(95th percentile = 31, 99th = 32) means many children score at or
near the instrument maximum — a different pathology from the
right-skewed / floor-at-0 targets of LRP02 and LRP04. Log / log1p
transforms are inappropriate here because the skew is in the wrong
direction; a reflection-log or quantile objective might be
considered later.

The predictor set will be reduced by iterative importance-based
feature selection under the MAE-tuned params (see
``notes/202604171421-lrp06-feature-selection.md``). This is the
initial tuned baseline; no feature-selection steps yet.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Documents the 32 → 10 feature-selection history under MAE-tuned params
# with no outlier exclusion (n=214).
# See notes/202604171421-lrp06-feature-selection.md for the full rationale.

_SELECTION_STEPS = [
    SelectionStep(
        removed=[
            # Tier A — importance ≤ 0.005 in the 32-predictor MAE tune
            V.AREA, V.CELF, V.HEARING, V.GENDER, V.BEHAV,
            V.APTINFO, V.YARCSI, V.TROG, V.APTGRAM, V.EARINF,
            V.NUMCHIL, V.AGESPEAK, V.DEAPPVO, V.AGEBOOKS,
            V.MUMEDUPOST16, V.ERBNW, V.ROWPVT, V.DEAPPFI,
            # Tier B — 0.006-0.010, redundant with retained
            # higher-importance siblings
            V.SPPHON,   # dcorr 0.78 with retained ewrswr (0.500)
            V.B1RETO,   # dcorr 0.76 with retained b1exto (0.105)
            V.EOWPVT,   # dcorr 0.80 with retained b1exto
            V.BLENDING, # dcorr 0.55 with retained ewrswr
        ],
        notes=(
            "Aggressive one-shot cut from 32 → 10 predictors. Drops "
            "18 Tier-A features with importance ≤ 0.005 (9 at exactly "
            "0.000 under the 55-tree MAE-tuned model) plus 4 Tier-B "
            "features at 0.006-0.010 that are redundant with retained "
            "higher-importance siblings: spphon/blending (ewrswr "
            "cluster, dcorr 0.78/0.55), b1reto/eowpvt (language "
            "cluster, dcorr 0.76/0.80 with b1exto). Surprising: "
            "aptinfo sits at exactly 0 importance here despite being "
            "top-2 on LRP04 — different construct relationship for "
            "letter-sound knowledge vs expressive vocabulary."
        ),
        date="2026-04-17",
        metrics_before={"cv_mae_mean": 4.422},
        metrics_after={"cv_mae_mean": 4.238},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 10-predictor Select01 set, no outlier exclusion
# (Optuna 150 trials, 10-split GroupKFold, seed 47, scoring=mae,
# lgbm_objective=mae). Tuner-inner CV MAE 4.0996 ± 0.8713. n=214.
# Supersedes the 32-predictor tune (tuner-inner 4.2827).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 277,
    "learning_rate": 0.015673816656256778,
    "num_leaves": 15,
    "max_depth": 10,
    "min_child_samples": 21,
    "subsample": 0.6507936233336762,
    "subsample_freq": 1,
    "colsample_bytree": 0.6399635717298359,
    "reg_alpha": 0.2906864644733162,
    "reg_lambda": 0.036855080194016676,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRP06(LevelModel):
    """Letter-sound knowledge level predictors — exploratory (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_LEVEL` predictor set
    (minus the target ``yarclet``) with MAE-tuned hyperparameters
    and no outlier exclusion. The starting point for feature
    selection on the letter-sound knowledge level-prediction task.
    """

    model_id = "lrp06"
    target_var = V.YARCLET
    description = (
        "LightGBM — letter-sound knowledge level predictors "
        "(10 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for identifying important predictors of "
        "letter-sound knowledge level (yarclet). MAE-tuned on the "
        "full 32-predictor DEFAULT_LEVEL set without outlier exclusion "
        "so importance rankings reflect the full range of outcomes. "
        "Note the target has a ceiling effect at 32 — many "
        "observations cluster at the instrument maximum, producing a "
        "left-skewed distribution. Feature-selection variants to "
        "follow. See notes/202604171421-lrp06-feature-selection.md."
    )
