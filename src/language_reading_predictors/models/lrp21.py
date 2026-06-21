# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP21: Predictors of DEAP fine-articulation gains.

``LRP21`` is the baseline exploratory model for DEAP fine-
articulation gains (``deappfi_gain``). ``deappfi`` is a
percentage-scale articulation measure from the Diagnostic
Evaluation of Articulation and Phonology (Dodd et al., 2006) —
the proportion of sounds correctly produced when the child is
asked to name pictures. ``deappfi`` specifically scores the
*final* consonant of each word (distinct from ``deappin``
initial and ``deappvo`` voicing).

The target is nearly symmetric but with a heavy two-sided
spread (``deappfi_gain`` min −56.9, max 56.0, median 0.01,
mean 0.84, std 13.28, skewness −0.32, with **~48% negative**
and ~2% zero observations, n ≈ 152). **Heavy regression from
the ceiling is the dominant story** — children at the top of
the scale tend to drop back between timepoints while those at
the floor improve.

DEAP measures have been used as predictors across every other
model in the suite but never as targets until LRP21/22.

Feature selection applied 2026-06-20 (replication): reduced from the full 34-predictor set to 6 predictors via a distance-correlation redundancy filter (dcor >= 0.70, keep the highest-importance representative) plus an importance noise-floor cut, then re-tuned on the reduced set. See the SelectionStep below and notes/202606201500-gb-replication-findings.md.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Feature selection (2026-06-20 replication): distance-correlation
# redundancy filter + importance noise-floor cut; see the SelectionStep.

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.TIME, V.GROUP, V.AREA, V.GENDER, V.APTGRAM, V.APTINFO, V.B1EXTO,
            V.B1RETO, V.CELF, V.ERBNW, V.ERBWORD, V.BLENDING, V.ROWPVT, V.SPPHON,
            V.TROG, V.YARCLET, V.YARCSI, V.DEAPPIN, V.BEHAV, V.ATTEND, V.AGESPEAK,
            V.VISION, V.HEARING, V.EARINF, V.NUMCHIL, V.AGEBOOKS, V.MUMEDUPOST16,
            V.DADEDUPOST16
        ],
        notes=(
            "Feature selection (replication, 2026-06-20): from the full 34-predictor set, a distance-correlation filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative per cluster) plus removal of features at/below the 0.005 importance floor. The standardised instrument was preferred over its intervention-taught sibling (eowpvt<-b1exto / rowpvt<-b1reto) where it did not cost CV. Reduces to 6 predictors with no dcor >= 0.70 pairs remaining; pooled refit-CV held under matched hyperparameters, then the set was re-tuned. See notes/202606201500-gb-replication-findings.md."
        ),
        date="2026-06-20",
        metrics_before={"cv_mae_mean": 8.2309},
        metrics_after={"cv_mae_mean": 8.2692},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 6-predictor replication-selected set, no outlier
# exclusion (Optuna 150 trials, 10-split GroupKFold, seed 47, scoring=mae,
# lgbm_objective=mae). Tuner-inner CV MAE 8.2692. Supersedes the full-set tune.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 147,
    "learning_rate": 0.012056446653615095,
    "num_leaves": 12,
    "max_depth": 5,
    "min_child_samples": 4,
    "subsample": 0.7871728250594918,
    "subsample_freq": 1,
    "colsample_bytree": 0.7486825897864793,
    "reg_alpha": 2.6696685908920714,
    "reg_lambda": 0.07257530687413892,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP21(GainModel):
    """DEAP fine-articulation gain predictors — baseline (all data, MAE-tuned).

    Uses a feature-selected subset of :attr:`Predictors.DEFAULT_GAIN`
    (``deappfi`` is already a member, so the GainModel auto-include
    is a no-op) with MAE-tuned hyperparameters and no outlier
    exclusion. Feature selection was applied (2026-06-20 replication); see the SelectionStep and the module docstring.
    """

    model_id = "lrp21"
    target_var = V.DEAPPFI_GAIN
    description = (
        "LightGBM — DEAP fine-articulation gain predictors "
        "(6 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for deappfi_gain (gain). Feature-selected (2026-06-20 replication) from the full 34-predictor default set to 6 predictors via a distance-correlation redundancy filter (no dcor >= 0.70 pairs remain) plus an importance noise-floor cut, then re-tuned on the reduced set (tuner-inner CV MAE 8.231 -> 8.269). Only the dominant predictor is robustly above the importance noise floor; treat the reduced ranking as exploratory. See the SelectionStep and notes/202606201500-gb-replication-findings.md."
    )
