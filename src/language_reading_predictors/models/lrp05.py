# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP05: Predictors of letter-sound knowledge gains.

``LRP05`` is the exploratory model for letter-sound knowledge gains
(``yarclet_gain``). It is MAE-tuned on the full 34-predictor
:attr:`Predictors.DEFAULT_GAIN` set (plus the auto-included base
variable ``yarclet``), with no outlier exclusion, designed to
identify the most important influences on letter-sound knowledge
gains.

The target is signed with a mild right tail (``yarclet_gain`` min ≈
−17, max ≈ 24, median 2, skewness 0.45, with ~22% negative and ~12%
zero observations, n ≈ 160). Similar shape to ``eowpvt_gain``
(LRP03) and milder than ``ewrswr_gain`` (LRP01).

The predictor set will be reduced by iterative importance-based
feature selection under the MAE-tuned params (see
``notes/202604171421-lrp05-feature-selection.md``). This is the
initial tuned baseline; no feature-selection steps yet.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Documents the 34 → 15 feature-selection history under MAE-tuned params
# with no outlier exclusion (n=160).
# See notes/202604171421-lrp05-feature-selection.md for the full rationale.

_SELECTION_STEPS = [
    SelectionStep(
        removed=[
            # Tier A — importance ≤ 0.005 in the 34-predictor MAE tune
            V.BEHAV, V.GENDER, V.AREA, V.NONWORD, V.VISION,
            V.EARINF, V.HEARING, V.NUMCHIL, V.YARCSI, V.EWRSWR,
            V.ERBNW, V.AGEBOOKS, V.B1EXTO, V.APTINFO,
            # Tier B — 0.005-0.010, redundant with retained
            # higher-importance siblings or with Tier-A drops
            V.SPPHON,       # reading-cluster redundancy; ewrswr already going
            V.MUMEDUPOST16, # dcorr 0.56 with dadedupost16 (retained)
            V.GROUP,        # weak singleton
            V.TROG,         # language-cluster redundancy
            V.APTGRAM,      # language-cluster redundancy
        ],
        notes=(
            "Aggressive one-shot cut from 34 → 15 predictors. Drops "
            "14 Tier-A features with importance ≤ 0.005 (essentially "
            "zero-signal under the 178-tree MAE-tuned model) plus 5 "
            "Tier-B features at 0.006-0.010 where redundancy or "
            "construct overlap with retained features justifies the "
            "drop. Reading measures drop entirely (ewrswr, nonword, "
            "yarcsi, spphon) — the base yarclet + time + age + attend "
            "+ speech articulation story dominates letter-sound gain "
            "prediction at n=160, not other reading scores."
        ),
        date="2026-04-17",
        metrics_before={"cv_mae_mean": 3.349},
        metrics_after={"cv_mae_mean": 3.249},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 15-predictor Select01 set, no outlier exclusion
# (Optuna 150 trials, 10-split GroupKFold, seed 47, scoring=mae,
# lgbm_objective=mae). Tuner-inner CV MAE 3.3082 ± 0.5435. n=160.
# Supersedes the 34-predictor tune (tuner-inner 3.3963).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 35,
    "learning_rate": 0.17619154429377148,
    "num_leaves": 54,
    "max_depth": 11,
    "min_child_samples": 7,
    "subsample": 0.6553474984702952,
    "subsample_freq": 1,
    "colsample_bytree": 0.7792653744980131,
    "reg_alpha": 0.10822326112673696,
    "reg_lambda": 0.17613569608007879,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRP05(GainModel):
    """Letter-sound knowledge gain predictors — exploratory (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_GAIN` predictor set plus
    the base variable ``yarclet`` (auto-included via :class:`GainModel`)
    with MAE-tuned hyperparameters and no outlier exclusion. The
    starting point for feature selection on the letter-sound
    knowledge gain-prediction task.
    """

    model_id = "lrp05"
    target_var = V.YARCLET_GAIN
    description = (
        "LightGBM — letter-sound knowledge gain predictors "
        "(15 predictors, MAE-tuned, no outlier exclusion)"
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
        "letter-sound knowledge gains (yarclet_gain). MAE-tuned on "
        "the full 34-predictor set (DEFAULT_GAIN + yarclet) without "
        "outlier exclusion so importance rankings reflect the full "
        "range of outcomes. Feature-selection variants to follow. "
        "See notes/202604171421-lrp05-feature-selection.md."
    )
