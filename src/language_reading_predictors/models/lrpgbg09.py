# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG09: Predictors of letter-sound knowledge gains.

``LRPGBG09`` is the exploratory model for letter-sound knowledge gains
(``yarclet_gain``). It is MAE-tuned on the full 34-predictor
:attr:`Predictors.DEFAULT_GAIN` set (plus the auto-included base
variable ``yarclet``), with no outlier exclusion, designed to
identify the most important influences on letter-sound knowledge
gains.

The target is signed with a mild right tail (``yarclet_gain`` min ≈
−17, max ≈ 24, median 2, skewness 0.45, with ~22% negative and ~12%
zero observations, n ≈ 160). Similar shape to ``eowpvt_gain``
(LRPGBG06) and milder than ``ewrswr_gain`` (LRPGBG12).

Uniform feature selection (2026-06-21) reduced the predictor set to the SelectionStep below via a distance-correlation redundancy filter plus an importance noise-floor cut; see ``notes/202606211200-uniform-gb-fs.md``.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS, SelectionStep
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Uniform feature-selection history (see the SelectionStep below).
# See notes/202606211200-uniform-gb-fs.md for the full rationale.

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.ERBWORD, V.EWRSWR, V.AGEBOOKS, V.GROUP, V.AGESPEAK, V.MUMEDUPOST16,
            V.EARINF, V.GENDER, V.DEAPPIN, V.NUMCHIL, V.BEHAV, V.AREA, V.VISION,
            V.HEARING, V.EOWPVT, V.TROG, V.CELF, V.B1RETO, V.YARCSI, V.NONWORD,
            V.B1EXTO, V.ROWPVT, V.DEAPPVO, V.APTGRAM, V.APTINFO, V.BLENDING
        ],
        notes=(
            "Uniform feature selection (2026-06-21): from the full 34-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). The baseline measure was force-kept (regression-to-the-mean anchor). Reduces to 8 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Applied uniformly across all GB models; see notes/202606211200-uniform-gb-fs.md."
        ),
        date="2026-06-21",
        metrics_before={"cv_mae_mean": 3.3941},
        metrics_after={"cv_mae_mean": 3.3569},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 8-predictor uniform-selected set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 3.3569.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 41,
    "learning_rate": 0.0853489590463849,
    "num_leaves": 42,
    "max_depth": 12,
    "min_child_samples": 21,
    "subsample": 0.688114728810087,
    "subsample_freq": 1,
    "colsample_bytree": 0.9723492415919919,
    "reg_alpha": 0.001990819330098672,
    "reg_lambda": 3.298314330689827,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRPGBG09(GainModel):
    """Letter-sound knowledge gain predictors — exploratory (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_GAIN` predictor set plus
    the base variable ``yarclet`` (auto-included via :class:`GainModel`)
    with MAE-tuned hyperparameters and no outlier exclusion. The
    starting point for feature selection on the letter-sound
    knowledge gain-prediction task.
    """

    model_id = "lrpgbg09"
    target_var = V.YARCLET_GAIN
    description = (
        "LightGBM — letter-sound knowledge gain predictors "
        "(8 predictors, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    selection_steps = _SELECTION_STEPS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for yarclet_gain (gain). Uniform feature selection (2026-06-21) from the full 34-predictor DEFAULT_GAIN set to 8 predictors (distance-correlation redundancy filter + importance noise-floor cut; baseline force-kept; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 3.394 -> 3.357). Gain models are near-noise (baseline-driven regression to the mean) - treat the reduced ranking as exploratory. See notes/202606211200-uniform-gb-fs.md."
    )
