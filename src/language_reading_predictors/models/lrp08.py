# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP08: Predictors of receptive vocabulary level.

``LRP08`` is the exploratory model for receptive vocabulary level
(``rowpvt``). It is MAE-tuned on the 17-predictor Select01 set
(down from the original 32-predictor
:attr:`Predictors.DEFAULT_LEVEL` minus target), with no outlier
exclusion, designed to identify the most important influences on
receptive vocabulary level.

The target is **essentially symmetric and near-Gaussian** (``rowpvt``
min 11, max 82, median 42, mean 41.1, std 14.1, skewness 0.04,
n ≈ 215). No floor, no ceiling, no heavy tail — the cleanest
target distribution of any LRP model to date. Transforms are
unnecessary; standard MAE and RMSE objectives should behave well.
"""

from language_reading_predictors.data_variables import Variables as V  # noqa: F401
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Documents the 32 → 17 feature-selection history under MAE-tuned
# params with no outlier exclusion (n=215).
# See notes/202604171715-lrp08-feature-selection.md for the full rationale.

_SELECTION_STEPS = [
    SelectionStep(
        removed=[
            # Tier A — ≤ 0.005 importance in the 32-predictor MAE tune
            V.GENDER, V.VISION, V.GROUP, V.EARINF, V.BEHAV,
            V.TIME, V.YARCSI, V.AREA, V.HEARING, V.BLENDING,
            V.DEAPPVO, V.AGESPEAK,
            # Tier B — 0.006 importance, redundant with retained
            # higher-importance siblings or demographic noise
            V.APTGRAM,   # dcorr ≈ 0.76 with retained aptinfo (0.067)
            V.ERBWORD,   # speech pair with retained erbnw (0.009)
            V.AGEBOOKS,  # demographic noise-floor
        ],
        notes=(
            "Moderate one-shot cut from 32 → 17 predictors. Drops "
            "12 Tier-A features with importance ≤ 0.005 under the "
            "118-tree MAE-tuned model plus 3 Tier-B redundant "
            "features (aptgram shares signal with aptinfo, erbword "
            "pair-redundant with erbnw, agebooks is near-zero "
            "demographic noise). Conservative relative to LRP06's "
            "32→10 cut — LRP08 has a flatter importance distribution "
            "(top b1reto only 0.095 vs LRP06 ewrswr 0.500) so the "
            "retained mid-tier features all contribute meaningfully. "
            "Keeps the full language cluster (b1reto, aptinfo, "
            "eowpvt, celf, trog, b1exto) and both reading controls "
            "(ewrswr, yarclet) for construct coverage."
        ),
        date="2026-04-17",
        metrics_before={"cv_mae_mean": 7.075},
        metrics_after={"cv_mae_mean": 6.966},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 17-predictor Select01 set, no outlier exclusion
# (Optuna 150 trials, 10-split GroupKFold, seed 47, scoring=mae,
# lgbm_objective=mae). Tuner-inner CV MAE 6.9048 ± 1.4492. n=215.
# Supersedes the 32-predictor tune (tuner-inner 7.0639).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 216,
    "learning_rate": 0.02084571088606304,
    "num_leaves": 22,
    "max_depth": 8,
    "min_child_samples": 10,
    "subsample": 0.7189672525215607,
    "subsample_freq": 1,
    "colsample_bytree": 0.8425605377858724,
    "reg_alpha": 0.004764523961768294,
    "reg_lambda": 0.6582222408420664,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRP08(LevelModel):
    """Receptive vocabulary level predictors — exploratory (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_LEVEL` predictor set
    (minus the target ``rowpvt``) with MAE-tuned hyperparameters
    and no outlier exclusion. The starting point for feature
    selection on the receptive vocabulary level-prediction task.
    """

    model_id = "lrp08"
    target_var = V.ROWPVT
    description = (
        "LightGBM — receptive vocabulary level predictors "
        "(17 predictors, MAE-tuned, no outlier exclusion)"
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
        "receptive vocabulary level (rowpvt). MAE-tuned on the "
        "17-predictor Select01 set (down from the original 32) "
        "without outlier exclusion so importance rankings reflect "
        "the full range of outcomes. Target is essentially symmetric "
        "/ near-Gaussian (skew 0.04, no floor or ceiling) — cleanest "
        "target distribution of any LRP model to date. "
        "See notes/202604171715-lrp08-feature-selection.md."
    )
