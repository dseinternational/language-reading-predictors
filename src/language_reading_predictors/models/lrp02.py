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

``LRP02Select02`` is a selection variant that restores four features
(``yarcsi``, ``b1exto``, ``hearing``, ``celf``) dropped after Select02
and uses the earlier 17-predictor MAE-tuned hyperparameters. It holds
the best CV metrics of any LRP02 configuration and is retained as a
reference point.

``LRP02Log`` is a target-transform variant that fits the same
13-predictor set on ``log1p(ewrswr)``. The target is heavily
right-skewed (floor at 0, tail to 64); the log transform homogenises
errors across quartiles. Predictions are inverse-transformed so all
reported metrics remain in the original ``ewrswr`` units.

``LRP02Select02Log`` crosses the 17-predictor Select02 feature set
with the log1p target transform and its own log-space MAE tune —
testing whether restoring the four features the primary dropped
(``yarcsi``, ``b1exto``, ``hearing``, ``celf``) still helps once the
target is compressed.

``LRP02Prediction`` and ``LRP02LogPrediction`` are RMSE-tuned
prediction variants of ``LRP02`` and ``LRP02Log`` respectively —
same 13-predictor feature set but with hyperparameters tuned for
squared-error loss rather than MAE. Their purpose is parity with the
``lrp01`` family's primary/prediction split and to supply a model
optimised for absolute prediction accuracy at the cost of
importance-ranking robustness.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_log_pipeline import LGBMLogPipeline
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Documents the 32 → 13 feature-selection history under MAE-tuned params
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
    SelectionStep(
        removed=[V.HEARING, V.CELF],
        notes=(
            "Remove 2 features: hearing (importance 0.003 under the "
            "15-predictor retune — below the 0.005 noise floor) and celf "
            "(importance 0.012, the weakest member of language cluster 9; "
            "dcorr 0.67 with eowpvt (0.050) and 0.63 with aptinfo (0.046), "
            "both of which rank higher on importance and cover the same "
            "language construct space)."
        ),
        date="2026-04-16",
        metrics_before={"cv_mae_mean": 6.023},
        metrics_after={"cv_mae_mean": 6.007},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 13-predictor Select04 set, no outlier exclusion (Optuna
# 150 trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Best trial #145, CV MAE 6.3065 ± 2.3839 (tuner inner). n=210.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 53,
    "learning_rate": 0.1030282991867544,
    "num_leaves": 49,
    "max_depth": 6,
    "min_child_samples": 16,
    "subsample": 0.9880905470494415,
    "subsample_freq": 1,
    "colsample_bytree": 0.6997497470635539,
    "reg_alpha": 0.001383118016594975,
    "reg_lambda": 0.0012224991529179404,
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

# MAE-tuned in log1p(y) space on the 13-predictor Select04 set (Optuna
# 150 trials, 10-split GroupKFold, seed 47, scoring=mae tuned on
# inverse-transformed predictions so tuner-inner CV MAE remains in
# original ewrswr units). Best trial #147, CV MAE 6.0010 ± 2.8026.
# n=210. Pinned on the ``lrp02_log`` variant.
_LGBM_MAE_PARAMS_LOG: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 358,
    "learning_rate": 0.013445072655254693,
    "num_leaves": 36,
    "max_depth": 8,
    "min_child_samples": 4,
    "subsample": 0.6931709324463539,
    "subsample_freq": 1,
    "colsample_bytree": 0.6354537611535218,
    "reg_alpha": 0.010804887107264944,
    "reg_lambda": 0.04796741692608438,
    "n_jobs": 16,
    "verbosity": -1,
}

# RMSE-tuned on the 13-predictor Select04 set, no outlier exclusion
# (Optuna 150 trials, 10-split GroupKFold, seed 47, scoring=rmse,
# lgbm_objective=regression). Tuner-inner CV RMSE 8.7517 ± 2.8635.
# n=210. Pinned on ``lrp02_prediction``.
_LGBM_RMSE_PARAMS: dict[str, float | int | str] = {
    "objective": "regression",
    "n_estimators": 72,
    "learning_rate": 0.07481599571185549,
    "num_leaves": 31,
    "max_depth": 3,
    "min_child_samples": 4,
    "subsample": 0.6084742628892493,
    "subsample_freq": 1,
    "colsample_bytree": 0.6132093303284623,
    "reg_alpha": 0.003422478748080231,
    "reg_lambda": 0.12984738927699754,
    "n_jobs": 16,
    "verbosity": -1,
}

# RMSE-tuned in log1p(y) space on the 13-predictor Select04 set (Optuna
# 150 trials, 10-split GroupKFold, seed 47, scoring=rmse,
# lgbm_objective=regression, target_transform=log1p). Tuner-inner CV
# RMSE 8.4455 ± 5.0674. n=210. Pinned on ``lrp02_log_prediction``.
_LGBM_RMSE_PARAMS_LOG: dict[str, float | int | str] = {
    "objective": "regression",
    "n_estimators": 28,
    "learning_rate": 0.14065753393164226,
    "num_leaves": 11,
    "max_depth": 11,
    "min_child_samples": 22,
    "subsample": 0.7793091074697025,
    "subsample_freq": 1,
    "colsample_bytree": 0.7795376336231212,
    "reg_alpha": 0.06759227929609403,
    "reg_lambda": 0.06485655423178838,
    "n_jobs": 16,
    "verbosity": -1,
}

# MAE-tuned in log1p(y) space on the 17-predictor Select02 set (Optuna
# 150 trials, 10-split GroupKFold, seed 47, scoring=mae,
# lgbm_objective=mae, target_transform=log1p). Best trial #102, CV MAE
# 5.7725 ± 2.5038. n=210. Pinned on ``lrp02_select02_log``.
_LGBM_MAE_PARAMS_SELECT02_LOG: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 156,
    "learning_rate": 0.04309025733167006,
    "num_leaves": 44,
    "max_depth": 6,
    "min_child_samples": 4,
    "subsample": 0.9405679124655911,
    "subsample_freq": 1,
    "colsample_bytree": 0.6098925462576346,
    "reg_alpha": 0.28851697467399,
    "reg_lambda": 0.008417559747075195,
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
        "(13 predictors, MAE-tuned, no outlier exclusion)"
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
            added=[V.YARCSI, V.B1EXTO, V.HEARING, V.CELF],
            notes=(
                "Restore features removed by the primary model after "
                "Select02: yarcsi and b1exto (dropped at Select03) plus "
                "hearing and celf (dropped at Select04). Lands at the "
                "17-predictor feature set that was MAE-tuned in Optuna "
                "trial #35 (the hyperparameters pinned here)."
            ),
            date="2026-04-16",
            metrics_after={"cv_mae_mean": 5.908},
        ),
    ]
    notes = (
        "17-predictor retuned variant preserving the best CV performance of "
        "any LRP02 configuration so far (CV MAE 5.908 ± 1.623, CV R² 0.390). "
        "The primary model progressively dropped yarcsi and b1exto (Select03, "
        "redundancy) and hearing and celf (Select04, noise + redundancy); "
        "this variant restores all four to land back at the Select02 feature "
        "set, where the 17-predictor tune applies. Retained as a reference "
        "point for the correlation-driven pruning trade-off."
    )


# ── log-transform variant ───────────────────────────────────────────────


class LRP02Log(LRP02):
    """Word-reading level predictors — log1p(ewrswr) target variant.

    Same 13-predictor feature set and MAE hyperparameters as the primary
    ``lrp02``, but fits ``log1p(ewrswr)`` via
    :class:`sklearn.compose.TransformedTargetRegressor`. Predictions are
    inverse-transformed so all reported CV / evaluation metrics remain
    in the original ``ewrswr`` units and are directly comparable with
    the primary.

    Motivation: ``ewrswr`` is heavily right-skewed (min 0, median 6.5,
    max 64) with a hard floor at 0. In-sample MAE on the primary model
    scales 6× across quartiles (0.45 at the floor vs 2.72 in the top
    quartile). The log transform compresses the tail and may let the
    model allocate capacity more evenly across the outcome range.
    """

    model_id = "lrp02_log"
    variant_of = "lrp02"
    description = (
        "LightGBM — word-reading level predictors "
        "(13 predictors, MAE-tuned in log1p space)"
    )
    pipeline_cls = LGBMLogPipeline
    params = _LGBM_MAE_PARAMS_LOG
    # Inherits selection_steps from LRP02 (same 13-predictor set).
    notes = (
        "Log-transform variant of lrp02. Fits log1p(ewrswr) via "
        "TransformedTargetRegressor so predictions are inverse-transformed "
        "back to the original scale — CV/evaluation metrics are comparable "
        "with the primary. Uses the same 13-predictor feature set and "
        "MAE-tuned hyperparameters tuned *in log space* (Optuna trial #147, "
        "tuner-inner CV MAE 6.0010)."
    )


# ── log-transform + 17-predictor variant ───────────────────────────────


class LRP02Select02Log(LRP02Select02):
    """Word-reading level predictors — log1p target on the 17-predictor set.

    Inherits `LRP02Select02`'s restoration of ``yarcsi``, ``b1exto``,
    ``hearing``, and ``celf``, but swaps the pipeline to
    :class:`LGBMLogPipeline` and pins hyperparameters tuned in log
    space on the 17-predictor feature set.

    Tests whether the four features the primary dropped retain value
    once the target is compressed. The `lrp02_log` tuning regime (many
    slow, shallow-ish trees with moderate regularisation) may not be
    directly portable to the larger predictor set, so it gets its own
    Optuna run.
    """

    model_id = "lrp02_select02_log"
    variant_of = "lrp02"
    description = (
        "LightGBM — word-reading level predictors "
        "(17 predictors, MAE-tuned in log1p space)"
    )
    pipeline_cls = LGBMLogPipeline
    params = _LGBM_MAE_PARAMS_SELECT02_LOG
    # Inherits selection_steps from LRP02Select02 (lands at 17 predictors).
    notes = (
        "Log-transform variant on the 17-predictor Select02 feature set. "
        "Restores yarcsi, b1exto, hearing, and celf on top of the primary's "
        "selection chain (inherited from LRP02Select02), fits log1p(ewrswr), "
        "and pins hyperparameters tuned on this exact configuration in log "
        "space (Optuna trial #102, tuner-inner CV MAE 5.7725)."
    )


# ── RMSE-tuned prediction variants ─────────────────────────────────────


class LRP02Prediction(LRP02):
    """Word-reading level predictors — prediction-focused (RMSE-tuned).

    Same 13-predictor feature set as the primary but with hyperparameters
    tuned for squared-error loss. Optimised for prediction accuracy at
    the cost of the importance-ranking robustness of the MAE-tuned
    exploratory model. Mirrors the ``lrp01`` / ``lrp01_prediction``
    split.
    """

    model_id = "lrp02_prediction"
    variant_of = "lrp02"
    description = (
        "LightGBM — word-reading level predictors "
        "(13 predictors, RMSE-tuned, no outlier exclusion)"
    )
    params = _LGBM_RMSE_PARAMS
    notes = (
        "Prediction-focused variant. Same 13-predictor feature set and "
        "selection history as the primary lrp02, but RMSE-tuned via "
        "Optuna (scoring=rmse, lgbm_objective=regression). Use this "
        "variant when absolute prediction accuracy on typical cases is "
        "the priority; use the MAE-tuned primary for importance ranking."
    )


class LRP02LogPrediction(LRP02Log):
    """Word-reading level predictors — log1p + RMSE-tuned.

    Same 13-predictor feature set as ``LRP02Log`` but with
    hyperparameters tuned for RMSE (not MAE) in log1p space. Combines
    the log-transform's benefits on typical-case prediction with an
    objective that explicitly penalises large squared errors.
    """

    model_id = "lrp02_log_prediction"
    variant_of = "lrp02"
    description = (
        "LightGBM — word-reading level predictors "
        "(13 predictors, RMSE-tuned in log1p space)"
    )
    params = _LGBM_RMSE_PARAMS_LOG
    notes = (
        "Prediction-focused log-transform variant. Same 13-predictor "
        "feature set and selection history as lrp02_log, but RMSE-tuned "
        "in log space via Optuna (scoring=rmse, lgbm_objective=regression, "
        "target_transform=log1p). Predictions inverse-transformed so CV "
        "metrics remain in original ewrswr units."
    )
