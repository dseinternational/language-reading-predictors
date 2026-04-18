# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP10: Predictors of receptive-grammar level (CELF).

``LRP10`` is the exploratory model for receptive-grammar level
(``celf``). It is MAE-tuned on the 12-predictor Select01 set
(down from the original 32-predictor
:attr:`Predictors.DEFAULT_LEVEL` minus target), with no outlier
exclusion, designed to identify the most important influences on
receptive-grammar level.

The target is **mildly left-skewed** (``celf`` min 0, max 18,
median 11, mean 10.88, std 4.24, skewness −0.37, n ≈ 214). The
max of 18 is the instrument maximum but the 95th percentile is
below it, so there is no strong ceiling pathology (unlike LRP06's
``yarclet`` which piles at 32). Transforms are unlikely to be
required.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Documents the 32 → 12 feature-selection history under MAE-tuned
# params with no outlier exclusion (n=214).
# See notes/202604181400-lrp10-feature-selection.md for the full rationale.

_SELECTION_STEPS = [
    SelectionStep(
        removed=[
            # Tier A — ≤ 0.005 importance in the 32-predictor MAE tune
            # (3 L1-zeroed: yarcsi, time, blending)
            V.EARINF, V.NUMCHIL, V.AREA, V.NONWORD, V.BEHAV,
            V.GROUP, V.GENDER, V.DEAPPFI, V.HEARING, V.YARCSI,
            V.TIME, V.BLENDING,
            # Tier B — 0.006-0.014, redundant or near-noise under dcorr audit
            V.APTGRAM,    # 0.014; dcorr 0.769 with retained aptinfo
            V.ERBWORD,    # 0.010; dcorr 0.713 with retained deappin; 0.839 with erbnw
            V.AGEBOOKS,   # 0.009; demographic near-noise
            V.YARCLET,    # 0.008; dcorr 0.690 with retained ewrswr
            V.B1EXTO,     # 0.007; dcorr 0.807 with retained eowpvt
            V.SPPHON,     # 0.006; dcorr 0.786 with retained ewrswr
            V.DEAPPVO,    # 0.006; near-noise; no dcorr ≥ 0.60 with retained
            V.ERBNW,      # 0.006; dcorr 0.839 with dropped erbword (pair-redundant)
        ],
        notes=(
            "Correlation-informed one-shot cut from 32 → 12 predictors. "
            "A full 32×32 dcorr audit identified dense redundancy in "
            "the language/speech/reading clusters (40+ pairs at dcorr "
            "≥ 0.60) — much higher redundancy than LRP09's gain task. "
            "Drops 12 Tier-A features at ≤ 0.005 importance (3 already "
            "L1-zeroed by the extreme reg_alpha 4.72 tune) plus 8 "
            "Tier-B features where importance is low AND dcorr with "
            "retained sibling is ≥ 0.69 (aptgram/aptinfo, erbword/"
            "deappin, yarclet/ewrswr, b1exto/eowpvt, spphon/ewrswr, "
            "erbnw/erbword) or where both pair members are near-noise "
            "(deappvo). Retains one reading-cluster representative "
            "(ewrswr at 0.008) as construct control even though it is "
            "low-importance. Note the retained set still has internal "
            "redundancy (eowpvt/aptinfo dcorr 0.767, eowpvt/rowpvt "
            "0.718, b1reto/aptinfo 0.740) — reducing further requires "
            "construct-driven Select02 cuts."
        ),
        date="2026-04-18",
        metrics_before={"cv_mae_mean": 2.405},
        metrics_after={"cv_mae_mean": 2.415},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 12-predictor Select01 set, no outlier exclusion
# (Optuna 150 trials, 10-split GroupKFold, seed 47, scoring=mae,
# lgbm_objective=mae). Tuner-inner CV MAE 2.3282 ± 0.3972. n=214.
# Supersedes the 32-predictor tune (tuner-inner 2.4105).
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 87,
    "learning_rate": 0.08133302519220838,
    "num_leaves": 30,
    "max_depth": 11,
    "min_child_samples": 5,
    "subsample": 0.8452803637308646,
    "subsample_freq": 1,
    "colsample_bytree": 0.6453260102696494,
    "reg_alpha": 2.470284680568493,
    "reg_lambda": 0.0024313249182264067,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, untuned) ───────────────────────────────────


class LRP10(LevelModel):
    """CELF receptive-grammar level predictors — exploratory (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_LEVEL` predictor set
    (minus the target ``celf``) with MAE-tuned hyperparameters and
    no outlier exclusion. The starting point for feature selection
    on the CELF level-prediction task.
    """

    model_id = "lrp10"
    target_var = V.CELF
    description = (
        "LightGBM — CELF (receptive grammar) level predictors "
        "(12 predictors, MAE-tuned, no outlier exclusion)"
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
        "CELF receptive-grammar level (celf). MAE-tuned on the "
        "12-predictor Select01 set (down from the original 32 via "
        "an aggressive correlation-informed cut that dropped the "
        "full zero-importance tail plus 8 Tier-B features redundant "
        "with retained siblings). Without outlier exclusion so "
        "importance rankings reflect the full range of outcomes. "
        "Target is mildly left-skewed (skew −0.37); the max of 18 "
        "is the instrument maximum but there is no strong ceiling "
        "effect (unlike LRP06's yarclet which piles at 32). See "
        "notes/202604181400-lrp10-feature-selection.md."
    )
