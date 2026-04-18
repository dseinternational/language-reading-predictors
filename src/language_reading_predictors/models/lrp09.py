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

_SELECTION_STEPS = [
    SelectionStep(
        removed=[
            # Tier A — ≤ 0.005 importance in the 34-predictor MAE tune
            V.AREA, V.VISION, V.NUMCHIL, V.EARINF, V.BEHAV, V.HEARING,
            V.GROUP, V.GENDER,
            # Tier B — 0.006-0.014, near-noise or redundant with retained
            V.YARCSI,         # 0.007; redundant with retained yarclet/ewrswr
            V.AGESPEAK,       # 0.008; demographic near-noise
            V.AGEBOOKS,       # 0.009; demographic near-noise
            V.DADEDUPOST16,   # 0.012; dcorr ≈ 0.56 with retained mumedupost16
            V.ERBWORD,        # 0.013; pair-redundant with retained erbnw (0.017)
            V.BLENDING,       # 0.013; reading-cluster redundancy with retained ewrswr/spphon
            V.TIME,           # 0.014; weak time-invariant signal under this tune
        ],
        notes=(
            "Moderate one-shot cut from 34 → 19 predictors. Drops "
            "8 Tier-A features (all at ≤ 0.007 importance) plus 7 "
            "Tier-B features at 0.008-0.014 justified by "
            "redundancy (yarcsi/blending with ewrswr/yarclet reading "
            "cluster, erbword with erbnw, dadedupost16 with "
            "mumedupost16) or demographic noise-floor status "
            "(agespeak, agebooks, time). Less aggressive than "
            "LRP07's 34→12 cut because LRP09's deep-tree many-"
            "estimator tune produced a shallower noise floor (no "
            "features at exactly 0 importance); the retained mid-"
            "tier features contribute meaningfully. Keeps both "
            "grammar measures (trog, aptgram) and the full DEAP "
            "articulation trio (deappfi, deappin, deappvo) for "
            "language-domain breadth — CELF measures basic concept "
            "knowledge (semantic/lexical), so grammar measures are "
            "kept as language-cluster controls rather than as "
            "construct-matched predictors."
        ),
        date="2026-04-18",
        metrics_before={"cv_mae_mean": 2.232},
        metrics_after={"cv_mae_mean": 2.185},
    ),
    SelectionStep(
        removed=[
            V.EOWPVT,   # dcorr 0.807 with retained b1exto (0.038); imp 0.026
            V.DEAPPIN,  # dcorr 0.767 with retained deappfi (0.057); imp 0.035
        ],
        notes=(
            "Redundancy-driven Select02: 19 → 17 predictors. Post-"
            "Select01 dcorr audit (see notes/202604181400-lrp09-"
            "feature-selection.md) identified two pairs at dcorr "
            "≥ 0.76 that Select01 had left in place: "
            "b1exto/eowpvt at 0.807 and deappfi/deappin at 0.767. "
            "Both drops target the lower-importance sibling in each "
            "pair (importance-based tie-break). Retained sibling "
            "should absorb the dropped feature's signal via the "
            "shared-information structure. Construct coverage is "
            "preserved: expressive-vocab still represented by b1exto "
            "(plus receptive rowpvt), articulation still represented "
            "by deappfi + deappvo (DEAP-fi and DEAP-vo). Does not "
            "address the remaining moderate-redundancy pairs "
            "(ewrswr/yarclet 0.690, b1exto/b1reto 0.756) — those "
            "are either construct-separable (reading level vs "
            "letter-sound) or deliberately kept for construct "
            "coverage."
        ),
        date="2026-04-18",
        metrics_before={"cv_mae_mean": 2.185},
        metrics_after={"cv_mae_mean": 2.182},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the full 34-predictor set (DEFAULT_GAIN, which already
# includes celf), no outlier exclusion (Optuna 150 trials, 10-split
# GroupKFold, seed 47, scoring=mae, lgbm_objective=mae). Tuner-inner
# CV MAE 2.2244 ± 0.3661. n=160.
#
# Select01 / Select02 reductions (19, then 17 predictors) both kept
# these 34-predictor tuned params as carry-forward. Both retunes on
# reduced sets produced dramatically different params (fewer trees,
# fast learning) but refit CV MAE was consistently worse than the
# carry-forward by 0.06-0.10. The many-slow-deep-with-strong-L2
# regime generalises well to reduced predictor sets; retunes find
# narrow high-variance optima that over-fit inner CV.
# See notes/202604181400-lrp09-feature-selection.md for details.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 194,
    "learning_rate": 0.012955887984432111,
    "num_leaves": 44,
    "max_depth": 12,
    "min_child_samples": 5,
    "subsample": 0.6522788933655064,
    "subsample_freq": 1,
    "colsample_bytree": 0.968420503348316,
    "reg_alpha": 0.0013680996955281002,
    "reg_lambda": 2.240185955527313,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, untuned) ───────────────────────────────────


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
