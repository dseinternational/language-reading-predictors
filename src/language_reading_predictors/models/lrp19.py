# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP19: Predictors of expressive-information (APT) gains.

``LRP19`` is the baseline exploratory model for expressive-
information gains (``aptinfo_gain``). ``aptinfo`` is the
information raw score from the Action Picture Test (Renfrew,
1997): the child is shown pictures and asked to describe them,
with scoring of the information content of the response (as
distinct from its grammatical structure, which is scored
separately as ``aptgram`` — LRP17/18).

The target is mildly right-skewed (``aptinfo_gain`` min −7,
max 16, median 2.5, mean 2.61, std 4.44, skewness 0.25, with
~29% negative and ~4% zero observations, n ≈ 160). The low
zero-mass is unusual — most children show measurable change
from timepoint to timepoint (cf LRP11 `trog_gain` ~8% zero,
LRP17 `aptgram_gain` ~11% zero, LRP13 `nonword_gain` ~48%
zero).

Feature selection applied 2026-06-20 (replication): reduced from the full 34-predictor set to 5 predictors via a distance-correlation redundancy filter (dcor >= 0.70, keep the highest-importance representative) plus an importance noise-floor cut, then re-tuned on the reduced set. See the SelectionStep below and notes/202606201500-gb-replication-findings.md.
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
            V.TIME, V.GROUP, V.AREA, V.GENDER, V.AGE, V.APTGRAM, V.B1EXTO, V.B1RETO,
            V.CELF, V.EOWPVT, V.ERBNW, V.NONWORD, V.BLENDING, V.ROWPVT, V.YARCLET,
            V.YARCSI, V.DEAPPIN, V.DEAPPFI, V.EWRSWR, V.BEHAV, V.ATTEND, V.AGESPEAK,
            V.VISION, V.HEARING, V.EARINF, V.NUMCHIL, V.AGEBOOKS, V.MUMEDUPOST16,
            V.DADEDUPOST16
        ],
        notes=(
            "Feature selection (replication, 2026-06-20): from the full 34-predictor set, a distance-correlation filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative per cluster) plus removal of features at/below the 0.005 importance floor. Reduces to 5 predictors with no dcor >= 0.70 pairs remaining; pooled refit-CV held under matched hyperparameters, then the set was re-tuned. See notes/202606201500-gb-replication-findings.md."
        ),
        date="2026-06-20",
        metrics_before={"cv_mae_mean": 3.2994},
        metrics_after={"cv_mae_mean": 3.0865},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 5-predictor replication-selected set, no outlier
# exclusion (Optuna 150 trials, 10-split GroupKFold, seed 47, scoring=mae,
# lgbm_objective=mae). Tuner-inner CV MAE 3.0865. Supersedes the full-set tune.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 175,
    "learning_rate": 0.05711307834022608,
    "num_leaves": 21,
    "max_depth": 5,
    "min_child_samples": 18,
    "subsample": 0.6005936605399571,
    "subsample_freq": 1,
    "colsample_bytree": 0.6122693866663788,
    "reg_alpha": 0.1279507418298321,
    "reg_lambda": 0.016218957713668814,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP19(GainModel):
    """APT expressive-information gain predictors — baseline (all data, MAE-tuned).

    Uses a feature-selected subset of :attr:`Predictors.DEFAULT_GAIN`
    (``aptinfo`` is already a member, so the GainModel auto-include
    is a no-op) with MAE-tuned hyperparameters and no outlier
    exclusion. Feature selection was applied (2026-06-20 replication); see the SelectionStep and the module docstring.
    """

    model_id = "lrp19"
    target_var = V.APTINFO_GAIN
    description = (
        "LightGBM — APT expressive-information gain predictors "
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
        "Exploratory model for aptinfo_gain (gain). Feature-selected (2026-06-20 replication) from the full 34-predictor default set to 5 predictors via a distance-correlation redundancy filter (no dcor >= 0.70 pairs remain) plus an importance noise-floor cut, then re-tuned on the reduced set (tuner-inner CV MAE 3.299 -> 3.086). Only the dominant predictor is robustly above the importance noise floor; treat the reduced ranking as exploratory. See the SelectionStep and notes/202606201500-gb-replication-findings.md."
    )
