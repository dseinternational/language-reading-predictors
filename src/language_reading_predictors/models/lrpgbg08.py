# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG08: Predictors of expressive-grammar (APT) gains.

``LRPGBG08`` is the baseline exploratory model for expressive-grammar
gains (``aptgram_gain``). ``aptgram`` is the grammar raw score
from the Action Picture Test (Renfrew, 1997) — the child is shown
pictures and asked to describe them, with scoring of the
grammatical structure of the response.

The target is mildly right-skewed (``aptgram_gain`` min −11,
max 16, median 1, mean 1.49, std 4.34, skewness 0.31, with ~32%
negative and ~11% zero observations, n ≈ 158). Similar gain-shape
to LRPGBG14 (``celf_gain``, skew 0.14).

``aptgram`` is the expressive-grammar parallel to ``trog``
(LRPGBG15/12 receptive grammar) — the pair addresses the
expressive vs receptive grammar asymmetry that is a live
question in DS language research.

Uniform feature selection (2026-06-21): reduced from the full 34-predictor set to 6 predictors via a distance-correlation redundancy filter plus an importance noise-floor cut, then re-tuned. See the SelectionStep below.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS, SelectionStep
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Feature selection (2026-06-21 uniform): distance-correlation
# redundancy filter + importance noise-floor cut; see the SelectionStep.

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.B1RETO, V.BLENDING, V.BEHAV, V.AGEBOOKS, V.YARCSI, V.DADEDUPOST16,
            V.EARINF, V.NUMCHIL, V.HEARING, V.AREA, V.GENDER, V.VISION, V.GROUP,
            V.EWRSWR, V.AGESPEAK, V.MUMEDUPOST16, V.EOWPVT, V.CELF, V.ERBNW,
            V.YARCLET, V.DEAPPVO, V.B1EXTO, V.DEAPPIN, V.NONWORD, V.DEAPPFI, V.TIME,
            V.APTINFO, V.ATTEND
        ],
        notes=(
            "Uniform feature selection (2026-06-21): from the full 34-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). The standardised instrument was preferred over its bespoke taught sibling where it did not reintroduce redundancy. The baseline measure was force-kept (regression-to-the-mean anchor). Reduces to 6 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Applied uniformly across all GB models."
        ),
        date="2026-06-21",
        metrics_before={"cv_mae_mean": 3.2280},
        metrics_after={"cv_mae_mean": 2.8658},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 6-predictor uniform-selected set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 2.8658.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 46,
    "learning_rate": 0.06161429194155265,
    "num_leaves": 36,
    "max_depth": 4,
    "min_child_samples": 6,
    "subsample": 0.7361638914240527,
    "subsample_freq": 1,
    "colsample_bytree": 0.9284978358011612,
    "reg_alpha": 0.006282476618386439,
    "reg_lambda": 0.04070671472487475,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRPGBG08(GainModel):
    """APT expressive-grammar gain predictors — baseline (all data, MAE-tuned).

    Uses a feature-selected subset of :attr:`Predictors.DEFAULT_GAIN`
    (``aptgram`` is already a member, so the GainModel auto-include
    is a no-op) with MAE-tuned hyperparameters and no outlier
    exclusion. Feature selection was applied (2026-06-21 uniform); see the SelectionStep and the module docstring.
    """

    model_id = "lrpgbg08"
    target_var = V.APTGRAM_GAIN
    description = (
        "LightGBM — APT expressive-grammar gain predictors "
        "(6 predictors, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    selection_steps = _SELECTION_STEPS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for aptgram_gain (gain). Uniform feature selection (2026-06-21) from the full 34-predictor DEFAULT_GAIN set to 6 predictors (distance-correlation redundancy filter + importance noise-floor cut; baseline force-kept; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 3.228 -> 2.866). Gain models are near-noise (baseline-driven regression to the mean) - treat the reduced ranking as exploratory."
    )
