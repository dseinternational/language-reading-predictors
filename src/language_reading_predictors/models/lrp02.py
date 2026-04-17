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

``LRP02LogSelect`` diverges from the primary's selection chain to
apply feature selection using log-space importance and redundancy
criteria. Starts at the 17-predictor Select02 log-space state
(inherits from ``LRP02Select02Log``) and prunes toward a
log-space-preferred feature set.

``LRP02Quantile`` and ``LRP02LogQuantile`` are objective variants that
train with LightGBM's ``quantile`` loss at ``alpha=0.5`` — the pinball
loss at the median. The conditional-median objective is the direct
MedAE-minimising loss; the MAE variants (``objective="mae"``) optimise
the same *point estimator* in theory but LightGBM's concrete gradient
handling differs slightly between `mae` and `quantile(0.5)`, so it is
worth checking empirically whether the explicit median objective gives
a better CV MedAE.
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

# Quantile-tuned (α=0.5) on the 13-predictor Select04 set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=medae,
# lgbm_objective=quantile, alpha=0.5). Tuner-inner CV MedAE
# 3.5689 ± 0.7607. n=210. Pinned on ``lrp02_quantile``.
_LGBM_QUANTILE_PARAMS: dict[str, float | int | str] = {
    "objective": "quantile",
    "alpha": 0.5,
    "n_estimators": 243,
    "learning_rate": 0.01326170990571767,
    "num_leaves": 57,
    "max_depth": 12,
    "min_child_samples": 11,
    "subsample": 0.9916011129877821,
    "subsample_freq": 1,
    "colsample_bytree": 0.8440054964789133,
    "reg_alpha": 0.0012997855434450734,
    "reg_lambda": 0.07869980269664073,
    "n_jobs": 16,
    "verbosity": -1,
}

# Quantile-tuned (α=0.5) in log1p(y) space on the 13-predictor
# Select04 set (Optuna 150 trials, 10-split GroupKFold, seed 47,
# scoring=medae, lgbm_objective=quantile, alpha=0.5,
# target_transform=log1p). Tuner-inner CV MedAE 2.8961 ± 0.7713.
# n=210. Pinned on ``lrp02_log_quantile``.
_LGBM_QUANTILE_PARAMS_LOG: dict[str, float | int | str] = {
    "objective": "quantile",
    "alpha": 0.5,
    "n_estimators": 322,
    "learning_rate": 0.016693202308250114,
    "num_leaves": 32,
    "max_depth": 7,
    "min_child_samples": 6,
    "subsample": 0.9595524946260237,
    "subsample_freq": 1,
    "colsample_bytree": 0.8127580991616771,
    "reg_alpha": 0.010598834462173275,
    "reg_lambda": 0.08639165584266031,
    "n_jobs": 16,
    "verbosity": -1,
}

# MAE-tuned in log1p(y) space on the 13-predictor log-space-first
# Select set (Optuna 150 trials, 10-split GroupKFold, seed 47,
# scoring=mae, lgbm_objective=mae, target_transform=log1p).
# Tuner-inner CV MAE 5.9096 ± 2.4182. n=210. Pinned on
# ``lrp02_log_select``. An earlier 15-predictor tune
# (tuner-inner 5.8169) was superseded by this one after the second
# log-space selection step.
_LGBM_MAE_PARAMS_LOG_SELECT: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 159,
    "learning_rate": 0.0228666627320957,
    "num_leaves": 12,
    "max_depth": 12,
    "min_child_samples": 5,
    "subsample": 0.653550724374776,
    "subsample_freq": 1,
    "colsample_bytree": 0.6293455916115821,
    "reg_alpha": 0.007594621812491086,
    "reg_lambda": 0.20355563772815835,
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


# ── log-space feature-selection variant ────────────────────────────────


class LRP02LogSelect(LRP02Select02Log):
    """Word-reading level predictors — log-space-first feature selection.

    Starts from the 17-predictor Select02 log-space state (inherited
    from :class:`LRP02Select02Log`) and applies a fresh chain of
    selection steps based on **log-space** permutation importance and
    distance-correlation evidence. The earlier primary-space selection
    chain (32 → 13) dropped `b1exto`, `yarcsi`, `hearing`, and `celf` —
    but under the log transform `b1exto` (0.058), `yarcsi` (0.045), and
    especially `celf` (0.061) showed non-trivial importance, suggesting
    the primary-space selection is not invariant to the target
    transform.

    This variant investigates what a purely log-space-driven pruning
    path looks like. The primary `lrp02_log` stays unchanged as a
    comparability model (same 13 features as `lrp02`).
    """

    model_id = "lrp02_log_select"
    variant_of = "lrp02"
    description = (
        "LightGBM — word-reading level predictors "
        "(log-space first-principles selection, MAE-tuned in log1p space)"
    )
    params = _LGBM_MAE_PARAMS_LOG_SELECT
    selection_steps = [
        SelectionStep(
            removed=[V.HEARING, V.GENDER],
            notes=(
                "Log-space Step 1: remove 2 features at or below the 0.005 "
                "informal noise floor in the 17-predictor log-tuned model: "
                "hearing (0.003) and gender (0.006). Matches the primary's "
                "importance-only cut philosophy but uses log-space rankings."
            ),
            date="2026-04-17",
            metrics_before={"cv_mae_mean": 5.735},
            metrics_after={"cv_mae_mean": 5.840},
        ),
        SelectionStep(
            removed=[V.NUMCHIL, V.CELF],
            notes=(
                "Log-space Step 2: remove the two lowest-importance "
                "features from the 15-predictor log-tuned model: numchil "
                "(0.015) and celf (0.020). Note that celf fell from 0.061 "
                "at 17 predictors to 0.020 after Step 1 — its apparent "
                "log-space importance in the 17-predictor model was "
                "partly absorbing the dropped hearing/gender signal. "
                "Brings the feature count to 13, matching lrp02_log for "
                "direct comparison."
            ),
            date="2026-04-17",
            metrics_before={"cv_mae_mean": 5.840},
            metrics_after={"cv_mae_mean": 5.749},
        ),
    ]
    notes = (
        "Log-space feature-selection variant. Starts from the 17-predictor "
        "Select02 log-space state and applies its own selection chain "
        "driven by log-space importance rankings. Two selection steps "
        "(17 → 15 → 13); each retuned. Final 13-predictor feature set "
        "differs from lrp02_log by carrying yarcsi + b1exto instead of "
        "gender + numchil. Paired-fold tests show this log-first set is "
        "not better than lrp02_log's primary-space features at equal "
        "count: 3-4/10 folds win on each metric, all p > 0.1. Useful "
        "negative result — the log transform is the bigger lever; the "
        "feature-selection philosophy is not."
    )


# ── quantile-objective variants (median prediction) ───────────────────


class LRP02Quantile(LRP02):
    """Word-reading level predictors — quantile α=0.5 objective (raw target).

    Same 13-predictor feature set as the primary `LRP02`; trains with
    LightGBM's ``objective="quantile"`` at ``alpha=0.5`` instead of
    ``"mae"``. Both objectives converge on the conditional median in
    theory, but LightGBM's gradient handling is slightly different
    (pinball-loss gradient magnitudes are ±α / ±(1−α) rather than
    ``mae``'s uniform ±1). Tests whether the explicit median objective
    yields a different — possibly tighter — CV MedAE in practice.
    """

    model_id = "lrp02_quantile"
    variant_of = "lrp02"
    description = (
        "LightGBM — word-reading level predictors "
        "(13 predictors, quantile α=0.5, no outlier exclusion)"
    )
    params = _LGBM_QUANTILE_PARAMS
    notes = (
        "Median-objective variant of lrp02. Uses LightGBM's "
        "quantile loss at α=0.5 — the direct MedAE-minimising "
        "objective. Hyperparameters to be tuned specifically for the "
        "quantile objective via Optuna."
    )


class LRP02LogQuantile(LRP02Log):
    """Word-reading level predictors — quantile α=0.5 in log1p space.

    Same 13-predictor feature set as `LRP02Log` and same log1p target
    transform; trains with ``objective="quantile"`` at ``alpha=0.5``
    instead of ``"mae"``. Directly optimises the conditional median
    of ``log1p(ewrswr)``; predictions are inverse-transformed via
    `LGBMLogPipeline` so all reported metrics stay in original units.
    """

    model_id = "lrp02_log_quantile"
    variant_of = "lrp02"
    description = (
        "LightGBM — word-reading level predictors "
        "(13 predictors, quantile α=0.5 in log1p space)"
    )
    params = _LGBM_QUANTILE_PARAMS_LOG
    notes = (
        "Median-objective log-transform variant. Same 13-predictor "
        "feature set and log1p target transform as lrp02_log, but "
        "trains with quantile α=0.5 (pinball loss at the median). "
        "Directly MedAE-minimising. Tuned in log space."
    )
