# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP20: Predictors of expressive-information (APT) level.

``LRP20`` is the baseline exploratory model for expressive-
information level (``aptinfo``). ``aptinfo`` is the information
raw score from the Action Picture Test (Renfrew, 1997): the child
is shown pictures and asked to describe them, with scoring of the
information content of the response (as distinct from its
grammatical structure, which is scored separately as ``aptgram``
— LRP17/18).

The target is **essentially symmetric** (``aptinfo`` min 0,
max 37.5, median 16.5, mean 16.97, std 7.93, skewness 0.24,
with ~1% at zero, n ≈ 214) — one of the cleanest distributions
in the suite, comparable to LRP12 (`trog`, skew 0.29) and
LRP16 (`blending`, skew 0.01) and much cleaner than the paired
LRP18 (`aptgram`, skew 1.23).

Feature selection applied 2026-06-20 (replication): reduced from the full 32-predictor set to 8 predictors via a distance-correlation redundancy filter (dcor >= 0.70, keep the highest-importance representative) plus an importance noise-floor cut, then re-tuned on the reduced set. See the SelectionStep below and notes/202606201500-gb-replication-findings.md.
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
            V.TIME, V.GROUP, V.AREA, V.GENDER, V.B1RETO, V.CELF, V.EOWPVT,
            V.ERBWORD, V.NONWORD, V.SPPHON, V.TROG, V.YARCLET, V.YARCSI, V.DEAPPIN,
            V.DEAPPFI, V.BEHAV, V.AGESPEAK, V.VISION, V.HEARING, V.EARINF,
            V.NUMCHIL, V.AGEBOOKS, V.MUMEDUPOST16, V.DADEDUPOST16
        ],
        notes=(
            "Feature selection (replication, 2026-06-20): from the full 32-predictor set, a distance-correlation filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative per cluster) plus removal of features at/below the 0.005 importance floor. Reduces to 8 predictors with no dcor >= 0.70 pairs remaining; pooled refit-CV held under matched hyperparameters, then the set was re-tuned. See notes/202606201500-gb-replication-findings.md."
        ),
        date="2026-06-20",
        metrics_before={"cv_mae_mean": 2.7868},
        metrics_after={"cv_mae_mean": 2.6872},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 8-predictor replication-selected set, no outlier
# exclusion (Optuna 150 trials, 10-split GroupKFold, seed 47, scoring=mae,
# lgbm_objective=mae). Tuner-inner CV MAE 2.6872. Supersedes the full-set tune.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 494,
    "learning_rate": 0.02731502756725624,
    "num_leaves": 49,
    "max_depth": 7,
    "min_child_samples": 4,
    "subsample": 0.9595524946260233,
    "subsample_freq": 1,
    "colsample_bytree": 0.7938963172103678,
    "reg_alpha": 0.004752944147643233,
    "reg_lambda": 5.286553325794671,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP20(LevelModel):
    """APT expressive-information level predictors — baseline (all data, MAE-tuned).

    Uses a feature-selected subset of :attr:`Predictors.DEFAULT_LEVEL`
    (minus the target ``aptinfo``) with MAE-tuned hyperparameters
    and no outlier exclusion. Feature selection was applied (2026-06-20 replication); see the SelectionStep and the module docstring.
    """

    model_id = "lrp20"
    target_var = V.APTINFO
    description = (
        "LightGBM — APT expressive-information level predictors "
        "(8 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for aptinfo (level). Feature-selected (2026-06-20 replication) from the full 32-predictor default set to 8 predictors via a distance-correlation redundancy filter (no dcor >= 0.70 pairs remain) plus an importance noise-floor cut, then re-tuned on the reduced set (tuner-inner CV MAE 2.787 -> 2.687). Only the dominant predictor is robustly above the importance noise floor; treat the reduced ranking as exploratory. See the SelectionStep and notes/202606201500-gb-replication-findings.md."
    )


# Construct-reduced variant: MAE-tuned on the 7-predictor set after
# additionally dropping same-construct (language_composite) predictors
# (aptgram). Tuner-inner CV MAE 3.0591.
_LGBM_MAE_PARAMS_NOCONSTRUCT: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 362,
    "learning_rate": 0.025217381297931024,
    "num_leaves": 27,
    "max_depth": 11,
    "min_child_samples": 5,
    "subsample": 0.8396622868674787,
    "subsample_freq": 1,
    "colsample_bytree": 0.6124355048380657,
    "reg_alpha": 0.011797647797568587,
    "reg_lambda": 0.0010321960298320736,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRP20NoConstruct(LRP20):
    """aptinfo — construct-reduced (language_composite dropped)."""

    model_id = "lrp20_noconstruct"
    variant_of = "lrp20"
    description = (
        "LightGBM — aptinfo predictors "
        "(7 predictors, construct-reduced)"
    )
    params = _LGBM_MAE_PARAMS_NOCONSTRUCT
    selection_steps = [
        SelectionStep(
            removed=[V.APTGRAM],
            notes=(
                "Construct-reduced variant of lrp20: drops the same-construct (language_composite) predictors (aptgram) from the primary set to ask what predicts aptinfo beyond its sibling measures. Pooled CV falls accordingly; re-tuned on the reduced set. See notes/202606201500-gb-replication-findings.md."
            ),
            date="2026-06-20",
            metrics_after={"cv_mae_mean": 3.0591},
        ),
    ]
    notes = (
        "Construct-reduced variant of lrp20: drops the same-construct (language_composite) predictors (aptgram) from the primary set to ask what predicts aptinfo beyond its sibling measures. Pooled CV falls accordingly; re-tuned on the reduced set. See notes/202606201500-gb-replication-findings.md."
    )
