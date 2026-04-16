# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP02: Predictors of word-reading level.

``LRP02`` is the exploratory model for word-reading level (``ewrswr``).
It is MAE-tuned with no outlier exclusion, designed to identify the
most important influences on reading level across the full range of
outcomes.

The predictor set starts from :attr:`Predictors.DEFAULT_LEVEL` and is
reduced by iterative importance-based feature selection under the
MAE-tuned params (see ``notes/202604161949-lrp02-feature-selection.md``).

``LRP02Select02`` is a selection variant that restores two features
(``yarcsi`` and ``b1exto``) dropped at Select03 and uses the earlier
17-predictor MAE-tuned hyperparameters. It holds the best CV metrics
of any LRP02 configuration and is retained as a reference point.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Documents the 32 → 15 feature-selection history under MAE-tuned params
# with no outlier exclusion (n=210).
# See notes/202604161949-lrp02-feature-selection.md for the full rationale.

_SELECTION_STEPS = [
    SelectionStep(
        removed=[
            V.TROG, V.TIME, V.B1RETO, V.BLENDING,
            V.BEHAV, V.AREA, V.GROUP, V.VISION,
        ],
        notes=(
            "Remove 8 features with permutation importance < 0.002 in the "
            "MAE-tuned baseline (test config, 10-fold GroupKFold, n=210). "
            "Low signal only — redundancy-driven drops deferred to later steps."
        ),
        date="2026-04-16",
        metrics_before={"cv_mae_mean": 6.980},
        metrics_after={"cv_mae_mean": 6.782},
    ),
    SelectionStep(
        removed=[
            V.ROWPVT, V.DEAPPIN, V.DEAPPFI, V.ERBNW,
            V.APTGRAM, V.DADEDUPOST16, V.EARINF,
        ],
        notes=(
            "Remove 7 features that are redundant with a higher-importance "
            "sibling or have dropped below 0.002 importance under select01: "
            "rowpvt (dcorr 0.72 with eowpvt), deappin/deappfi (mutually "
            "dcorr 0.77, both low importance), erbnw (dcorr 0.84 with "
            "erbword), aptgram (dcorr 0.77 with aptinfo), dadedupost16 "
            "(dcorr 0.61 with mumedupost16), earinf (importance 0.001)."
        ),
        date="2026-04-16",
        metrics_before={"cv_mae_mean": 6.782},
        metrics_after={"cv_mae_mean": 6.598},
    ),
    SelectionStep(
        removed=[V.YARCSI, V.B1EXTO],
        notes=(
            "Remove 2 features highly redundant with a higher-importance "
            "sibling: yarcsi (dcorr 0.75 with spphon, Spearman 0.72) and "
            "b1exto (dcorr 0.81 with eowpvt and 0.81 with aptinfo, and "
            "lower importance than both)."
        ),
        date="2026-04-16",
        metrics_before={"cv_mae_mean": 5.800},
        metrics_after={"cv_mae_mean": 6.031},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 15-predictor Select03 set, no outlier exclusion (Optuna
# 50 trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Best trial #32, CV MAE 6.5176 ± 2.0444. n=210.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 83,
    "learning_rate": 0.04935451138430074,
    "num_leaves": 33,
    "max_depth": 3,
    "min_child_samples": 6,
    "subsample": 0.9269822414293808,
    "subsample_freq": 1,
    "colsample_bytree": 0.6374248909658503,
    "reg_alpha": 0.25157549566014903,
    "reg_lambda": 0.00600868983112977,
    "n_jobs": 16,
    "verbosity": -1,
}

# MAE-tuned on the 17-predictor Select02 set (Optuna 50 trials, 10-split
# GroupKFold, seed 47, scoring=mae, lgbm_objective=mae). Best trial #35,
# CV MAE 6.4134 ± 1.9311. Preserved for the ``lrp02_select02`` variant,
# which achieved the best CV metrics of any LRP02 configuration.
_LGBM_MAE_PARAMS_SELECT02: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 156,
    "learning_rate": 0.02325963289635613,
    "num_leaves": 26,
    "max_depth": 7,
    "min_child_samples": 6,
    "subsample": 0.6291125666385646,
    "subsample_freq": 1,
    "colsample_bytree": 0.6831652991793304,
    "reg_alpha": 0.0010143700221596228,
    "reg_lambda": 0.002220144406178582,
    "n_jobs": 16,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRP02(LevelModel):
    """Word-reading level predictors — exploratory model (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_LEVEL` predictor set with
    MAE-tuned hyperparameters and no outlier exclusion. Serves as the
    starting point for feature selection on the level-prediction task.
    """

    model_id = "lrp02"
    target_var = V.EWRSWR
    description = (
        "LightGBM — word-reading level predictors "
        "(15 predictors, MAE-tuned, no outlier exclusion)"
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
        "Exploratory model for identifying important predictors of word-reading "
        "level (ewrswr). MAE-tuned on the current 15-predictor set with no "
        "outlier exclusion so importance rankings reflect the full range of "
        "outcomes. See notes/202604161949-lrp02-feature-selection.md."
    )


# ── selection variant (17 predictors, best CV) ──────────────────────────


class LRP02Select02(LRP02):
    """Word-reading level predictors — 17-predictor retuned variant.

    Restores ``yarcsi`` and ``b1exto`` — dropped by the primary model at
    Select03 — and pins to the hyperparameters tuned on that 17-predictor
    set. Holds the best CV MAE (5.800) of any LRP02 configuration; kept as
    a reference point for comparison with the more parsimonious primary
    model (15 predictors, CV MAE 6.023).
    """

    model_id = "lrp02_select02"
    variant_of = "lrp02"
    description = (
        "LightGBM — word-reading level predictors "
        "(17 predictors, MAE-tuned on 17 — best CV)"
    )
    params = _LGBM_MAE_PARAMS_SELECT02
    selection_steps = [
        SelectionStep(
            added=[V.YARCSI, V.B1EXTO],
            notes=(
                "Restore yarcsi and b1exto (removed by the primary model at "
                "Select03) and pin to the MAE-tuned hyperparameters from the "
                "17-predictor retune (Optuna trial #35)."
            ),
            date="2026-04-16",
            metrics_after={"cv_mae_mean": 5.865},
        ),
    ]
    notes = (
        "17-predictor retuned variant preserving the best CV performance of "
        "any LRP02 configuration so far (CV MAE 5.865 ± 1.537, CV R² 0.365). "
        "yarcsi (dcorr 0.75 with spphon) and b1exto (dcorr 0.81 with eowpvt) "
        "were dropped by the primary model on redundancy grounds, but "
        "carried enough unique signal that dropping them cost CV performance "
        "even after retuning. Retained as a reference point."
    )
