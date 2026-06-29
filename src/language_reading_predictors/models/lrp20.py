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

Uniform feature selection (2026-06-21): reduced from the full 32-predictor set to 8 predictors via a distance-correlation redundancy filter plus an importance noise-floor cut, then re-tuned. See the SelectionStep below and notes/202606211200-uniform-gb-fs.md.
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
            V.EOWPVT, V.B1RETO, V.DEAPPFI, V.TROG, V.AGESPEAK, V.DEAPPIN, V.SPPHON,
            V.BEHAV, V.GENDER, V.AGEBOOKS, V.EARINF, V.NUMCHIL, V.AREA, V.GROUP,
            V.VISION, V.HEARING, V.MUMEDUPOST16, V.DADEDUPOST16, V.YARCLET, V.TIME,
            V.YARCSI, V.CELF, V.ERBWORD, V.NONWORD
        ],
        notes=(
            "Uniform feature selection (2026-06-21): from the full 32-predictor set, a distance-correlation redundancy filter (dcor >= 0.70, keep the highest out-of-fold permutation-importance representative) plus an importance noise-floor cut (<= 0.005). Reduces to 8 predictors with no dcor >= 0.70 pairs remaining; re-tuned on the reduced set (Optuna 150-trial MAE, 10-fold GroupKFold, seed 47). Applied uniformly across all GB models; see notes/202606211200-uniform-gb-fs.md."
        ),
        date="2026-06-21",
        metrics_before={"cv_mae_mean": 2.7114},
        metrics_after={"cv_mae_mean": 2.7061},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 8-predictor uniform-selected set (Optuna 150
# trials, 10-split GroupKFold, seed 47, scoring=mae, lgbm_objective=mae).
# Tuner-inner CV MAE 2.7061.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 276,
    "learning_rate": 0.06362816681666994,
    "num_leaves": 47,
    "max_depth": 6,
    "min_child_samples": 7,
    "subsample": 0.9174036085656114,
    "subsample_freq": 1,
    "colsample_bytree": 0.6656142066855005,
    "reg_alpha": 0.06215178250629114,
    "reg_lambda": 0.04469345927561462,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, MAE-tuned) ─────────────────────────────────


class LRP20(LevelModel):
    """APT expressive-information level predictors — baseline (all data, MAE-tuned).

    Uses a feature-selected subset of :attr:`Predictors.DEFAULT_LEVEL`
    (minus the target ``aptinfo``) with MAE-tuned hyperparameters
    and no outlier exclusion. Feature selection was applied (2026-06-21 uniform); see the SelectionStep and the module docstring.
    """

    model_id = "lrp20"
    target_var = V.APTINFO
    description = (
        "LightGBM — APT expressive-information level predictors "
        "(8 predictors, MAE-tuned, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    selection_steps = _SELECTION_STEPS
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for aptinfo (level). Uniform feature selection (2026-06-21) from the full 32-predictor DEFAULT_LEVEL set to 8 predictors (distance-correlation redundancy filter + importance noise-floor cut; no dcor >= 0.70 pairs remain), re-tuned on the reduced set (tuner-inner CV MAE 2.711 -> 2.706). Treat the reduced ranking as exploratory. See notes/202606211200-uniform-gb-fs.md."
    )


# Same-skill variant: MAE-tuned on the 7-predictor set after dropping
# aptgram — the APT grammar score, taken from the same elicited
# picture descriptions as the target aptinfo. Tuner-inner CV MAE 3.0591.
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
    """aptinfo — same-skill reduced (APT same-sample sibling aptgram dropped)."""

    model_id = "lrp20_noconstruct"
    variant_of = "lrp20"
    description = (
        "LightGBM — aptinfo predictors "
        "(7 predictors, same-skill reduced: APT sibling dropped)"
    )
    params = _LGBM_MAE_PARAMS_NOCONSTRUCT
    selection_steps = [
        SelectionStep(
            removed=[V.APTGRAM],
            notes=(
                "Same-skill variant of lrp20: drops aptgram — the Action Picture Test grammar score, scored from the same elicited picture descriptions as the target aptinfo — so the model is not 'predicting' expressive information from a parallel scoring of the same sample. Receptive grammar (trog) and other constructs are kept deliberately, to be seen independently. Pooled CV falls accordingly; re-tuned on the reduced set. See notes/202606210930-lrp-same-skill-variants.md."
            ),
            date="2026-06-21",
            metrics_after={"cv_mae_mean": 3.0591},
        ),
    ]
    notes = (
        "Same-skill variant of lrp20: drops aptgram (APT grammar, scored from the same picture descriptions as aptinfo) to ask what predicts expressive information beyond a parallel scoring of the same test. Receptive grammar (trog) and other constructs kept visible. Re-tuned on the reduced set. See notes/202606210930-lrp-same-skill-variants.md."
    )
