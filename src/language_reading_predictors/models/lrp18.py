# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP18: Predictors of expressive-grammar (APT) level.

``LRP18`` is the baseline exploratory model for expressive-grammar
level (``aptgram``). ``aptgram`` is the grammar raw score from the
Action Picture Test (Renfrew, 1997) — the child is shown pictures
and asked to describe them, with scoring of the grammatical
structure of the response.

The target is **right-skewed** (``aptgram`` min 0, max 28,
median 6, mean 7.63, std 6.34, skewness 1.23, with ~9% at zero,
n ≈ 211) — comparable in skew magnitude to LRP02's ``ewrswr``
baseline, and a heavier floor than the receptive-grammar target
``trog`` (LRP12, skew 0.29).

``aptgram`` is the expressive-grammar parallel to ``trog``
(LRP12 receptive grammar) — the pair addresses the expressive vs
receptive grammar asymmetry that is a live question in DS
language research. The right-skew motivates a later log-transform
variant (mirroring LRP02's ``lrp02_log``).

Uniform feature selection (2026-06-21): reduced from the full 32-predictor set to 4 predictors via a distance-correlation redundancy filter plus an importance noise-floor cut, then re-tuned. See the SelectionStep below.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS, SelectionStep
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Feature selection (2026-06-21 uniform): distance-correlation
# redundancy filter + importance noise-floor cut; see the SelectionStep.

_SELECTION_STEPS: list[SelectionStep] = [
    SelectionStep(
        removed=[
            V.B1RETO, V.CELF, V.AGEBOOKS, V.B1EXTO, V.EOWPVT, V.BLENDING, V.EARINF,
            V.DADEDUPOST16, V.TIME, V.HEARING, V.GROUP, V.GENDER, V.AGESPEAK,
            V.VISION, V.NUMCHIL, V.AREA, V.BEHAV, V.MUMEDUPOST16, V.SPPHON, V.AGE,
            V.EWRSWR, V.DEAPPVO, V.ROWPVT, V.DEAPPFI, V.YARCLET, V.TROG, V.DEAPPIN,
            V.ERBWORD
        ],
        notes=(
            "Uniform feature selection (2026-06-21): from the full 32-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). Reduces to 4 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Applied uniformly across all GB models."
        ),
        date="2026-06-21",
        metrics_before={"cv_mae_mean": 2.5569},
        metrics_after={"cv_mae_mean": 2.3251},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 4-predictor uniform-selected set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 2.3251.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 290,
    "learning_rate": 0.041412982334710226,
    "num_leaves": 48,
    "max_depth": 4,
    "min_child_samples": 4,
    "subsample": 0.7867626106611368,
    "subsample_freq": 1,
    "colsample_bytree": 0.6621049966005043,
    "reg_alpha": 8.554800291256814,
    "reg_lambda": 0.02745428375418847,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP18(LevelModel):
    """APT expressive-grammar level predictors — baseline (all data, MAE-tuned).

    Uses a feature-selected subset of :attr:`Predictors.DEFAULT_LEVEL`
    (minus the target ``aptgram``) with MAE-tuned hyperparameters
    and no outlier exclusion. Feature selection was applied (2026-06-21 uniform); see the SelectionStep and the module docstring.
    """

    model_id = "lrp18"
    target_var = V.APTGRAM
    description = (
        "LightGBM — APT expressive-grammar level predictors "
        "(4 predictors, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    selection_steps = _SELECTION_STEPS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for aptgram (level). Uniform feature selection (2026-06-21) from the full 32-predictor DEFAULT_LEVEL set to 4 predictors (distance-correlation redundancy filter + importance noise-floor cut; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 2.557 -> 2.325). Treat the reduced ranking as exploratory."
    )


# Same-skill variant: MAE-tuned on the 3-predictor set after dropping
# aptinfo — the APT information score, taken from the same elicited
# picture descriptions as the target aptgram. Tuner-inner CV MAE 3.3181.
_LGBM_MAE_PARAMS_NOCONSTRUCT: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 313,
    "learning_rate": 0.016317971819433903,
    "num_leaves": 11,
    "max_depth": 4,
    "min_child_samples": 19,
    "subsample": 0.8624240982946859,
    "subsample_freq": 1,
    "colsample_bytree": 0.9090336123302216,
    "reg_alpha": 0.6633687828289669,
    "reg_lambda": 0.10568060745026951,
    "n_jobs": -1,
    "verbosity": -1,
}


class LRP18NoConstruct(LRP18):
    """aptgram — same-skill reduced (APT same-sample sibling aptinfo dropped)."""

    model_id = "lrp18_noconstruct"
    variant_of = "lrp18"
    description = (
        "LightGBM — aptgram predictors "
        "(3 predictors, same-skill reduced: APT sibling dropped)"
    )
    params = _LGBM_MAE_PARAMS_NOCONSTRUCT
    selection_steps = [
        SelectionStep(
            removed=[V.APTINFO],
            notes=(
                "Same-skill variant of lrp18: drops aptinfo — the Action Picture Test information score, scored from the same elicited picture descriptions as the target aptgram — so the model is not 'predicting' expressive grammar from a parallel scoring of the same sample. Receptive grammar (trog) and other constructs are kept deliberately, to be seen independently. Pooled CV falls accordingly; re-tuned on the reduced set. "
            ),
            date="2026-06-21",
            metrics_after={"cv_mae_mean": 3.3181},
        ),
    ]
    notes = (
        "Same-skill variant of lrp18: drops aptinfo (APT information, scored from the same picture descriptions as aptgram) to ask what predicts expressive grammar beyond a parallel scoring of the same test. Receptive grammar (trog) and other constructs kept visible. Re-tuned on the reduced set. "
    )
