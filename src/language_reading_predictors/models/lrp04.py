# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP04: Predictors of expressive-vocabulary level.

``LRP04`` is the exploratory model for expressive-vocabulary level
(``eowpvt``). It is MAE-tuned on the full 32-predictor
:attr:`Predictors.DEFAULT_LEVEL` set (minus the target), with no
outlier exclusion, designed to identify the most important
influences on expressive-vocabulary level.

The target is mildly right-skewed (``eowpvt`` min 8, max 77,
median 33, skewness 0.63, n ≈ 215). No hard floor at 0 (unlike
``ewrswr`` in LRP02), so the motivation for a ``log1p`` transform
is less compelling — but a question for future investigation.

The predictor set will be reduced by iterative importance-based
feature selection under the MAE-tuned params (see
``notes/202604171240-lrp04-feature-selection.md``). This is the
initial tuned baseline; no feature-selection steps yet.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Documents the 32 → 9 feature-selection history under MAE-tuned params
# with no outlier exclusion (n=215).
# See notes/202604171240-lrp04-feature-selection.md for the full rationale.

_SELECTION_STEPS = [
    SelectionStep(
        removed=[
            # Tier A — importance ≤ 0.005 in the 32-predictor MAE tune
            V.HEARING, V.GROUP, V.AREA, V.TIME, V.BEHAV, V.EARINF,
            V.DEAPPVO, V.NONWORD, V.GENDER, V.MUMEDUPOST16,
            V.SPPHON, V.YARCSI, V.ERBNW, V.BLENDING, V.VISION,
            V.ERBWORD, V.AGEBOOKS,
            # Tier B with redundancy support
            V.APTGRAM,          # dcorr 0.76 with retained aptinfo (0.107)
            V.TROG,             # dcorr 0.60-0.64 with language cluster
            V.DEAPPFI,          # dcorr 0.77 with retained deappin (pair)
            # Additional demographic/family drops
            V.AGESPEAK,
            V.DADEDUPOST16,
            V.NUMCHIL,
        ],
        notes=(
            "Aggressive one-shot cut from 32 → 9 predictors. Drops 17 "
            "Tier-A features with importance ≤ 0.005 (several with "
            "redundancy support: yarcsi/spphon/nonword dcorr 0.66-0.78 "
            "with retained ewrswr; erbword+erbnw dcorr 0.84 pair). Adds "
            "three Tier-B drops that have both low-ish importance and "
            "strong redundancy with retained higher-importance partners: "
            "aptgram (dcorr 0.76 with aptinfo), trog (dcorr 0.60-0.64 "
            "with language cluster), deappfi (dcorr 0.77 with deappin — "
            "keep one of the articulation pair). Finally drops three "
            "demographic/family features on importance grounds: "
            "agespeak (0.010), dadedupost16 (0.009, redundant with "
            "already-Tier-A mumedupost16), numchil (0.006)."
        ),
        date="2026-04-17",
        metrics_before={"cv_mae_mean": 6.156},
        metrics_after={"cv_mae_mean": 5.564},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 9-predictor Select01 set, no outlier exclusion
# (Optuna 150 trials, 10-split GroupKFold, seed 47, scoring=mae,
# lgbm_objective=mae). Tuner-inner CV MAE 5.5527 ± 1.2155. n=215.
# Supersedes the original 32-predictor tune (tuner-inner 6.1434).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 69,
    "learning_rate": 0.07553599741644976,
    "num_leaves": 32,
    "max_depth": 5,
    "min_child_samples": 6,
    "subsample": 0.859517008393526,
    "subsample_freq": 1,
    "colsample_bytree": 0.6834749241109254,
    "reg_alpha": 0.007614012937381988,
    "reg_lambda": 0.14649804504467065,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRP04(LevelModel):
    """Expressive-vocabulary level predictors — exploratory (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_LEVEL` predictor set
    (minus the target ``eowpvt``) with MAE-tuned hyperparameters and
    no outlier exclusion. The starting point for feature selection
    on the expressive-vocabulary level-prediction task.
    """

    model_id = "lrp04"
    target_var = V.EOWPVT
    description = (
        "LightGBM — expressive-vocabulary level predictors "
        "(9 predictors, MAE-tuned, no outlier exclusion)"
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
        "expressive-vocabulary level (eowpvt). MAE-tuned on the full "
        "32-predictor DEFAULT_LEVEL set without outlier exclusion so "
        "importance rankings reflect the full range of outcomes. "
        "Feature-selection variants to follow. See "
        "notes/202604171240-lrp04-feature-selection.md."
    )
