# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP04: Predictors of expressive-vocabulary level.

``LRP04`` is the exploratory model for expressive-vocabulary level
(``eowpvt``). It is MAE-tuned on the full 32-predictor
:attr:`Predictors.DEFAULT_LEVEL` set (minus the target), with no
outlier exclusion, designed to identify the most important
influences on expressive-vocabulary level.

The target is mildly right-skewed (``eowpvt`` min 8, max 77,
median 33, skewness 0.63, n ≈ 215). No hard floor at 0 (unlike
``ewrswr`` in LRP02), so the motivation for a ``log1p`` transform
is less compelling — but a question for future investigation.

The predictor set will be reduced by iterative importance-based
feature selection under the MAE-tuned params (see
``notes/202604171240-lrp04-feature-selection.md``). This is the
initial tuned baseline; no feature-selection steps yet.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Documents the 32 → 7 feature-selection history under MAE-tuned params
# with no outlier exclusion (n=215).
# See notes/202604171240-lrp04-feature-selection.md for the full rationale.

_SELECTION_STEPS = [
    SelectionStep(
        removed=[
            # Tier A — importance ≤ 0.005 in the 32-predictor MAE tune
            V.HEARING, V.GROUP, V.AREA, V.TIME, V.BEHAV, V.EARINF,
            V.DEAPPVO, V.NONWORD, V.GENDER, V.MUMEDUPOST16,
            V.SPPHON, V.YARCSI, V.ERBNW, V.BLENDING, V.VISION,
            V.ERBWORD, V.AGEBOOKS,
            # Tier B with redundancy support
            V.APTGRAM,          # dcorr 0.76 with retained aptinfo (0.107)
            V.TROG,             # dcorr 0.60-0.64 with language cluster
            V.DEAPPFI,          # dcorr 0.77 with retained deappin (pair)
            # Additional demographic/family drops
            V.AGESPEAK,
            V.DADEDUPOST16,
            V.NUMCHIL,
        ],
        notes=(
            "Aggressive one-shot cut from 32 → 9 predictors. Drops 17 "
            "Tier-A features with importance ≤ 0.005 (several with "
            "redundancy support: yarcsi/spphon/nonword dcorr 0.66-0.78 "
            "with retained ewrswr; erbword+erbnw dcorr 0.84 pair). Adds "
            "three Tier-B drops that have both low-ish importance and "
            "strong redundancy with retained higher-importance partners: "
            "aptgram (dcorr 0.76 with aptinfo), trog (dcorr 0.60-0.64 "
            "with language cluster), deappfi (dcorr 0.77 with deappin — "
            "keep one of the articulation pair). Finally drops three "
            "demographic/family features on importance grounds: "
            "agespeak (0.010), dadedupost16 (0.009, redundant with "
            "already-Tier-A mumedupost16), numchil (0.006)."
        ),
        date="2026-04-17",
        metrics_before={"cv_mae_mean": 6.156},
        metrics_after={"cv_mae_mean": 5.564},
    ),
    SelectionStep(
        removed=[
            V.B1EXTO,   # rank 1 importance (0.164) but tautological:
                        # another expressive-vocabulary test, same
                        # construct as the target eowpvt. Keeping it
                        # turns the model into a between-tests
                        # calibration study rather than an
                        # identification of non-vocabulary predictors
                        # of expressive vocabulary.
            V.B1RETO,   # rank 9 importance (0.026). Dcorr 0.74-0.76
                        # with retained aptinfo / rowpvt / b1exto. With
                        # rowpvt retained as the standardised receptive
                        # vocabulary measure, b1reto adds no
                        # independent receptive-language signal.
        ],
        notes=(
            "Construct-driven drops: b1exto is another expressive-"
            "vocabulary instrument — same construct as the target "
            "eowpvt — so its high importance is largely tautological. "
            "Removing it reframes the model as 'non-vocabulary "
            "predictors of expressive vocabulary' rather than "
            "'between-test calibration'. b1reto is redundant with "
            "the retained rowpvt (standardised receptive vocabulary "
            "test), dcorr 0.74. CV metrics are expected to degrade "
            "because b1exto is the single strongest predictor (0.164) "
            "— the trade is worse metrics for a more interpretable "
            "model."
        ),
        date="2026-04-17",
        metrics_before={"cv_mae_mean": 5.572},
        metrics_after={"cv_mae_mean": 6.114},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 7-predictor Select02 set, no outlier exclusion
# (Optuna 150 trials, 10-split GroupKFold, seed 47, scoring=mae,
# lgbm_objective=mae). Tuner-inner CV MAE 6.1385 ± 1.1345. n=215.
# Supersedes earlier tunes:
#   32-predictor        (tuner-inner 6.1434)
#   9-predictor Select01 (tuner-inner 5.5527)
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 45,
    "learning_rate": 0.07573022964806482,
    "num_leaves": 30,
    "max_depth": 6,
    "min_child_samples": 10,
    "subsample": 0.8737230089192473,
    "subsample_freq": 1,
    "colsample_bytree": 0.7169131631393786,
    "reg_alpha": 0.0022764472298362187,
    "reg_lambda": 0.003357533830874894,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRP04(LevelModel):
    """Expressive-vocabulary level predictors — exploratory (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_LEVEL` predictor set
    (minus the target ``eowpvt``) with MAE-tuned hyperparameters and
    no outlier exclusion. The starting point for feature selection
    on the expressive-vocabulary level-prediction task.
    """

    model_id = "lrp04"
    target_var = V.EOWPVT
    description = (
        "LightGBM — expressive-vocabulary level predictors "
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
        "Exploratory model for identifying important predictors of "
        "expressive-vocabulary level (eowpvt). MAE-tuned on the full "
        "32-predictor DEFAULT_LEVEL set without outlier exclusion so "
        "importance rankings reflect the full range of outcomes. "
        "Feature-selection variants to follow. See "
        "notes/202604171240-lrp04-feature-selection.md."
    )
