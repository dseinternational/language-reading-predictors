# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP10: Predictors of basic concept knowledge level (CELF) —
construct-reduced to isolate non-vocabulary signal.

``LRP10`` is the exploratory model for basic concept knowledge
level (``celf``). The ``celf`` score is drawn from the Clinical
Evaluation of Language Fundamentals Preschool 2nd Ed (Wiig,
Secord & Semel 2006) and in this study only the basic-concept-
knowledge subtest (18 linguistic concepts) was administered — so
``celf`` is a lexical/semantic concept measure, NOT a grammar
measure (the grammar measures in this study are ``trog`` for
receptive grammar and ``aptgram`` for expressive grammar).

LRP10 is MAE-tuned on the 10-predictor Select02 set (down from
the original 32-predictor :attr:`Predictors.DEFAULT_LEVEL` minus
target via Select01's 32→12 correlation-informed cut, then
Select02's construct-driven drop of the top two vocabulary
predictors ``eowpvt`` and ``b1reto``). No outlier exclusion. The
Select02 cut deliberately trades prediction accuracy for
interpretability: the model now answers "what predicts basic
concept knowledge beyond vocabulary?" rather than a pure accuracy
optimum.

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

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.B1RETO, V.ERBNW, V.B1EXTO, V.BEHAV, V.NUMCHIL, V.HEARING, V.VISION,
            V.APTINFO, V.GENDER, V.EARINF, V.GROUP, V.AREA, V.BLENDING,
            V.DADEDUPOST16, V.ERBWORD, V.YARCSI, V.DEAPPFI, V.MUMEDUPOST16,
            V.AGEBOOKS, V.NONWORD, V.SPPHON, V.YARCLET, V.APTGRAM, V.TIME,
            V.AGESPEAK, V.EWRSWR, V.DEAPPVO, V.ROWPVT, V.TROG
        ],
        notes=(
            "Uniform feature selection (2026-06-21): from the full 32-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). Reduces to 3 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Applied uniformly across all GB models; see notes/202606211200-uniform-gb-fs.md."
        ),
        date="2026-06-21",
        metrics_before={"cv_mae_mean": 2.4964},
        metrics_after={"cv_mae_mean": 2.5670},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 3-predictor uniform-selected set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 2.5670.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 65,
    "learning_rate": 0.043738091375742166,
    "num_leaves": 44,
    "max_depth": 4,
    "min_child_samples": 37,
    "subsample": 0.9853192989415035,
    "subsample_freq": 1,
    "colsample_bytree": 0.9140199440486373,
    "reg_alpha": 0.06457861635292404,
    "reg_lambda": 0.013949746526542768,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRP10(LevelModel):
    """CELF basic concept knowledge level predictors — exploratory (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_LEVEL` predictor set
    (minus the target ``celf``) with MAE-tuned hyperparameters and
    no outlier exclusion. The starting point for feature selection
    on the CELF basic concept knowledge level-prediction task.
    """

    model_id = "lrp10"
    target_var = V.CELF
    description = (
        "LightGBM — CELF (basic concept knowledge) level predictors "
        "(3 predictors, MAE-tuned, construct-reduced "
        "to exclude the top two vocabulary predictors)"
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
        "CELF basic concept knowledge level (celf) BEYOND the two "
        "strongest vocabulary handles in the dataset. CELF in this "
        "study assesses 18 basic linguistic concepts (a lexical / "
        "semantic measure, NOT a grammar measure — grammar is "
        "covered by trog and aptgram). Construct-reduced to 10 "
        "predictors via Select01 (32→12 correlation-informed cut) "
        "then Select02 (drop eowpvt and b1reto — the top two "
        "Select01 predictors). Mirrors LRP04's construct-driven "
        "Select02 drop of b1exto. Without outlier exclusion so "
        "importance rankings reflect the full range of outcomes. "
        "Target is mildly left-skewed (skew −0.37); the max of 18 "
        "is the instrument maximum but there is no strong ceiling "
        "effect (unlike LRP06's yarclet which piles at 32). See "
        "notes/202604181400-lrp10-feature-selection.md."
    )
