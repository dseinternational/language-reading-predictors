# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP22: Predictors of DEAP fine-articulation level.

``LRP22`` is the baseline exploratory model for DEAP fine-
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
model in the suite but never as targets until LRP21/22. First
articulation-domain target.

Feature selection applied 2026-06-20 (replication): reduced from the full 32-predictor set to 7 predictors via a distance-correlation redundancy filter (dcor >= 0.70, keep the highest-importance representative) plus an importance noise-floor cut, then re-tuned on the reduced set. See the SelectionStep below and notes/202606201500-gb-replication-findings.md.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Feature selection (2026-06-20 replication): distance-correlation
# redundancy filter + importance noise-floor cut; see the SelectionStep.

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.GROUP, V.AREA, V.GENDER, V.AGE, V.APTGRAM, V.APTINFO, V.B1EXTO,
            V.B1RETO, V.CELF, V.EOWPVT, V.ERBNW, V.ERBWORD, V.NONWORD, V.BLENDING,
            V.ROWPVT, V.SPPHON, V.YARCSI, V.DEAPPVO, V.BEHAV, V.VISION, V.HEARING,
            V.EARINF, V.NUMCHIL, V.AGEBOOKS, V.MUMEDUPOST16
        ],
        notes=(
            "Feature selection (replication, 2026-06-20): from the full 32-predictor set, a distance-correlation filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative per cluster) plus removal of features at/below the 0.005 importance floor. Reduces to 7 predictors with no dcor >= 0.70 pairs remaining; pooled refit-CV held under matched hyperparameters, then the set was re-tuned. See notes/202606201500-gb-replication-findings.md."
        ),
        date="2026-06-20",
        metrics_before={"cv_mae_mean": 9.9456},
        metrics_after={"cv_mae_mean": 9.9583},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 7-predictor replication-selected set, no outlier
# exclusion (Optuna 150 trials, 10-split GroupKFold, seed 47, scoring=mae,
# lgbm_objective=mae). Tuner-inner CV MAE 9.9583. Supersedes the full-set tune.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 125,
    "learning_rate": 0.03930092267917853,
    "num_leaves": 59,
    "max_depth": 9,
    "min_child_samples": 11,
    "subsample": 0.7527306921042428,
    "subsample_freq": 1,
    "colsample_bytree": 0.9080847854288628,
    "reg_alpha": 0.030835094299706333,
    "reg_lambda": 0.06855294982918957,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP22(LevelModel):
    """DEAP fine-articulation level predictors — baseline (all data, MAE-tuned).

    Uses a feature-selected subset of :attr:`Predictors.DEFAULT_LEVEL`
    (minus the target ``deappfi``) with MAE-tuned hyperparameters
    and no outlier exclusion. Feature selection was applied (2026-06-20 replication); see the SelectionStep and the module docstring.
    """

    model_id = "lrp22"
    target_var = V.DEAPPFI
    description = (
        "LightGBM — DEAP fine-articulation level predictors "
        "(7 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for deappfi (level). Feature-selected (2026-06-20 replication) from the full 32-predictor default set to 7 predictors via a distance-correlation redundancy filter (no dcor >= 0.70 pairs remain) plus an importance noise-floor cut, then re-tuned on the reduced set (tuner-inner CV MAE 9.946 -> 9.958). Only the dominant predictor is robustly above the importance noise floor; treat the reduced ranking as exploratory. See the SelectionStep and notes/202606201500-gb-replication-findings.md."
    )
