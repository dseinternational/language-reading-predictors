# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP17: Predictors of expressive-grammar (APT) gains.

``LRP17`` is the baseline exploratory model for expressive-grammar
gains (``aptgram_gain``). ``aptgram`` is the grammar raw score
from the Action Picture Test (Renfrew, 1997) — the child is shown
pictures and asked to describe them, with scoring of the
grammatical structure of the response.

The target is mildly right-skewed (``aptgram_gain`` min −11,
max 16, median 1, mean 1.49, std 4.34, skewness 0.31, with ~32%
negative and ~11% zero observations, n ≈ 158). Similar gain-shape
to LRP09 (``celf_gain``, skew 0.14).

``aptgram`` is the expressive-grammar parallel to ``trog``
(LRP11/12 receptive grammar) — the pair addresses the
expressive vs receptive grammar asymmetry that is a live
question in DS language research.

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
            V.TIME, V.GROUP, V.AREA, V.GENDER, V.APTINFO, V.B1EXTO, V.B1RETO,
            V.CELF, V.EOWPVT, V.ERBNW, V.NONWORD, V.BLENDING, V.YARCLET, V.YARCSI,
            V.DEAPPIN, V.DEAPPVO, V.DEAPPFI, V.EWRSWR, V.BEHAV, V.ATTEND,
            V.AGESPEAK, V.VISION, V.HEARING, V.EARINF, V.NUMCHIL, V.AGEBOOKS,
            V.MUMEDUPOST16, V.DADEDUPOST16
        ],
        notes=(
            "Feature selection (replication, 2026-06-20): from the full 34-predictor set, a distance-correlation filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative per cluster) plus removal of features at/below the 0.005 importance floor. The standardised instrument was preferred over its intervention-taught sibling (eowpvt<-b1exto / rowpvt<-b1reto) where it did not cost CV. Reduces to 6 predictors with no dcor >= 0.70 pairs remaining; pooled refit-CV held under matched hyperparameters, then the set was re-tuned. See notes/202606201500-gb-replication-findings.md."
        ),
        date="2026-06-20",
        metrics_before={"cv_mae_mean": 3.1989},
        metrics_after={"cv_mae_mean": 2.8717},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 6-predictor replication-selected set, no outlier
# exclusion (Optuna 150 trials, 10-split GroupKFold, seed 47, scoring=mae,
# lgbm_objective=mae). Tuner-inner CV MAE 2.8717. Supersedes the full-set tune.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 40,
    "learning_rate": 0.05514443358934577,
    "num_leaves": 46,
    "max_depth": 8,
    "min_child_samples": 6,
    "subsample": 0.7001225594922503,
    "subsample_freq": 1,
    "colsample_bytree": 0.9628942702118503,
    "reg_alpha": 0.0069664119243308875,
    "reg_lambda": 0.0021645658003044267,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP17(GainModel):
    """APT expressive-grammar gain predictors — baseline (all data, MAE-tuned).

    Uses a feature-selected subset of :attr:`Predictors.DEFAULT_GAIN`
    (``aptgram`` is already a member, so the GainModel auto-include
    is a no-op) with MAE-tuned hyperparameters and no outlier
    exclusion. Feature selection was applied (2026-06-20 replication); see the SelectionStep and the module docstring.
    """

    model_id = "lrp17"
    target_var = V.APTGRAM_GAIN
    description = (
        "LightGBM — APT expressive-grammar gain predictors "
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
        "Exploratory model for aptgram_gain (gain). Feature-selected (2026-06-20 replication) from the full 34-predictor default set to 6 predictors via a distance-correlation redundancy filter (no dcor >= 0.70 pairs remain) plus an importance noise-floor cut, then re-tuned on the reduced set (tuner-inner CV MAE 3.199 -> 2.872). Only the dominant predictor is robustly above the importance noise floor; treat the reduced ranking as exploratory. See the SelectionStep and notes/202606201500-gb-replication-findings.md."
    )
