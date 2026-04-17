# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP03: Predictors of expressive-vocabulary gains.

``LRP03`` is the exploratory model for expressive-vocabulary gains
(``eowpvt_gain``). It is MAE-tuned on the full 34-predictor
:attr:`Predictors.DEFAULT_GAIN` set (plus the auto-included base
variable ``eowpvt``), with no outlier exclusion, designed to
identify the most important influences on expressive-vocabulary
gains.

The target is signed (``eowpvt_gain`` min ≈ −13, max ≈ 28, median 3,
skewness 0.32, with ~25% negative observations and n ≈ 161). That's
much milder skew than LRP01's ``ewrswr_gain`` and nearly symmetric —
a log / signed-log transform may or may not help and is a question
for future investigation.

The predictor set will be reduced by iterative importance-based
feature selection under the MAE-tuned params (see
``notes/202604171127-lpr03-feature-selection.md``). This is the
initial tuned baseline; no feature-selection steps yet.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# Documents the 34 → 11 feature-selection history under MAE-tuned params
# with no outlier exclusion (n=161).
# See notes/202604171127-lpr03-feature-selection.md for the full rationale.

_SELECTION_STEPS = [
    SelectionStep(
        removed=[
            # Tier A — exactly 0.000 importance in the 34-predictor tune
            V.YARCSI, V.SPPHON, V.NONWORD,
            V.HEARING, V.EARINF,
            V.NUMCHIL, V.GENDER,
            # Tier B — 0.001–0.005 importance (below the noise floor)
            V.BEHAV, V.ATTEND,
            V.ERBWORD, V.ERBNW,
            V.MUMEDUPOST16, V.DADEDUPOST16,
            V.AREA, V.VISION, V.BLENDING,
            V.TIME, V.GROUP,
        ],
        notes=(
            "Remove 18 features with importance ≤ 0.005 in the full "
            "34-predictor MAE-tuned baseline. Tier A (7 features, "
            "importance 0.000): yarcsi, spphon, nonword — all redundant "
            "with ewrswr (dcorr 0.55–0.80) and individually "
            "non-contributing; hearing, earinf, numchil, gender — "
            "weak across the board. Tier B (11 features, 0.001–0.005): "
            "behav, attend, erbword + erbnw (dcorr 0.84 pair), "
            "mumedupost16 + dadedupost16 (dcorr 0.56 pair), area, "
            "vision, blending (dcorr 0.53 with eowpvt), time, group. "
            "One-shot aggressive cut because the 16-tree tuned model "
            "had essentially no capacity for these features."
        ),
        date="2026-04-17",
        metrics_before={"cv_mae_mean": 5.277},
        metrics_after={"cv_mae_mean": 5.154},
    ),
    SelectionStep(
        removed=[
            V.AGESPEAK, V.YARCLET, V.EWRSWR, V.DEAPPFI, V.CELF,
        ],
        notes=(
            "Remove 5 features at the noise floor (importance ≤ 0.004) "
            "under the Select01 retune: agespeak (0.002, weakest and "
            "no strong partner); yarclet (0.003, dcorr 0.65 with "
            "ewrswr and 0.64 with b1exto — redundant in the reading "
            "cluster); ewrswr (0.004, dcorr 0.66 with b1exto and 0.65 "
            "with eowpvt — redundant with retained language features); "
            "deappfi (0.004, dcorr 0.76 with retained deappin — "
            "articulation pair redundancy); celf (0.004, dcorr 0.63–"
            "0.64 with eowpvt/b1exto/b1reto/rowpvt — sits inside the "
            "language cluster). agebooks (0.006) retained as the "
            "borderline case — above the 0.005 floor with only weak "
            "correlations (max 0.33 with aptgram) and a distinct "
            "early-life construct."
        ),
        date="2026-04-17",
        metrics_before={"cv_mae_mean": 5.084},
        metrics_after={"cv_mae_mean": 4.979},
    ),
]


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the 11-predictor Select02 set, no outlier exclusion
# (Optuna 150 trials, 10-split GroupKFold, seed 47, scoring=mae,
# lgbm_objective=mae). Tuner-inner CV MAE 5.0542 ± 0.6829. n=161.
# Supersedes earlier tunings:
#   34-predictor (tuner-inner 5.0256)
#   16-predictor Select01 (tuner-inner 5.0936)
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 25,
    "learning_rate": 0.06478548258507148,
    "num_leaves": 48,
    "max_depth": 11,
    "min_child_samples": 10,
    "subsample": 0.9835210793717761,
    "subsample_freq": 1,
    "colsample_bytree": 0.9203322386497722,
    "reg_alpha": 0.036771486040166265,
    "reg_lambda": 0.00745726685285877,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (exploratory, MAE-tuned) ──────────────────────────────


class LRP03(GainModel):
    """Expressive-vocabulary gain predictors — exploratory (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_GAIN` predictor set plus
    the base variable ``eowpvt`` (auto-included via :class:`GainModel`)
    with MAE-tuned hyperparameters and no outlier exclusion. The
    starting point for feature selection on the expressive-vocabulary
    gain-prediction task.
    """

    model_id = "lrp03"
    target_var = V.EOWPVT_GAIN
    description = (
        "LightGBM — expressive-vocabulary gain predictors "
        "(11 predictors, MAE-tuned, no outlier exclusion)"
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
        "expressive-vocabulary gains (eowpvt_gain). MAE-tuned on the "
        "full 34-predictor set (DEFAULT_GAIN + eowpvt) without outlier "
        "exclusion so importance rankings reflect the full range of "
        "outcomes. Feature-selection variants to follow. See "
        "notes/202604171127-lpr03-feature-selection.md."
    )
