# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP09: Predictors of basic concept knowledge gains (CELF).

``LRP09`` is the exploratory model for basic concept knowledge
gains (``celf_gain``). The ``celf`` score is drawn from the
Clinical Evaluation of Language Fundamentals Preschool 2nd Ed
(Wiig, Secord & Semel 2006) and in this study only the basic-
concept-knowledge subtest (18 linguistic concepts) was
administered — so ``celf`` is a lexical/semantic concept measure,
NOT a grammar measure (the grammar measures in this study are
``trog`` for receptive grammar and ``aptgram`` for expressive
grammar).

LRP09 is MAE-tuned on the full 34-predictor
:attr:`Predictors.DEFAULT_GAIN` set (which already includes
``celf`` as a level predictor — the GainModel's auto-include is a
no-op here), with no outlier exclusion, designed to identify the
most important influences on basic concept knowledge gains.

The target is **mildly right-skewed** (``celf_gain`` min ≈ −8,
max ≈ 10, median 1, mean 1.14, std 3.20, skewness 0.14, with ~26%
negative and ~17% zero observations, n ≈ 160). The zero pile-up
is heavier than in LRP07 / LRP05 gains (17% vs 3%/12%) —
consistent with the coarser 0-18 CELF raw-score scale.

The predictor set will be reduced by iterative importance-based
feature selection under the MAE-tuned params (see
``notes/202604181400-lrp09-feature-selection.md``). This is the
initial tuned baseline; no feature-selection steps yet.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Documents the 34 → 19 feature-selection history under MAE-tuned
# params with no outlier exclusion (n=160).
# See notes/202604181400-lrp09-feature-selection.md for the full rationale.

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.BLENDING, V.HEARING, V.AGEBOOKS, V.VISION, V.BEHAV, V.AGESPEAK,
            V.DADEDUPOST16, V.EARINF, V.GROUP, V.NUMCHIL, V.AREA, V.MUMEDUPOST16,
            V.GENDER, V.SPPHON, V.DEAPPFI, V.YARCSI, V.EOWPVT, V.ATTEND, V.ERBNW,
            V.DEAPPIN, V.ERBWORD, V.B1EXTO, V.B1RETO, V.APTINFO, V.APTGRAM,
            V.DEAPPVO, V.TROG, V.TIME, V.EWRSWR
        ],
        notes=(
            "Uniform feature selection (2026-06-21): from the full 34-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). The baseline measure was force-kept (regression-to-the-mean anchor). Reduces to 5 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Applied uniformly across all GB models; see notes/202606211200-uniform-gb-fs.md."
        ),
        date="2026-06-21",
        metrics_before={"cv_mae_mean": 2.1621},
        metrics_after={"cv_mae_mean": 2.1929},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 5-predictor uniform-selected set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 2.1929.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 274,
    "learning_rate": 0.014134903867028734,
    "num_leaves": 37,
    "max_depth": 4,
    "min_child_samples": 4,
    "subsample": 0.9062849094869884,
    "subsample_freq": 1,
    "colsample_bytree": 0.9681718321490902,
    "reg_alpha": 0.03608716091718503,
    "reg_lambda": 0.07485707056926946,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRP09(GainModel):
    """CELF basic concept knowledge gain predictors — exploratory (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_GAIN` predictor set
    (``celf`` is already a member, so the GainModel auto-include is
    a no-op) with MAE-tuned hyperparameters and no outlier exclusion.
    The starting point for feature selection on the CELF basic
    concept knowledge gain-prediction task.
    """

    model_id = "lrp09"
    target_var = V.CELF_GAIN
    description = (
        "LightGBM — CELF (basic concept knowledge) gain predictors "
        "(5 predictors, MAE-tuned, no outlier exclusion)"
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
        "CELF basic concept knowledge gains (celf_gain). CELF in this "
        "study assesses 18 basic linguistic concepts (a lexical / "
        "semantic measure, NOT a grammar measure — grammar is covered "
        "by trog and aptgram). MAE-tuned on the 17-predictor Select02 "
        "set (down from the original 34 via Select01's 34→19 cut and "
        "Select02's redundancy-driven 19→17 cut of eowpvt and "
        "deappin) without outlier exclusion so importance rankings "
        "reflect the full range of outcomes. Target is mildly right-"
        "skewed (skew 0.14) with heavier zero pile-up than other gain "
        "targets (17% zero, vs 3-12% in LRP05/LRP07). See "
        "notes/202604181400-lrp09-feature-selection.md."
    )
