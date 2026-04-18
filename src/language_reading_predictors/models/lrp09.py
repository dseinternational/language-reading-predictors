# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP09: Predictors of receptive-grammar gains (CELF).

``LRP09`` is the exploratory model for receptive-grammar gains
(``celf_gain``). It is MAE-tuned on the full 34-predictor
:attr:`Predictors.DEFAULT_GAIN` set (which already includes ``celf``
as a level predictor — the GainModel's auto-include is a no-op
here), with no outlier exclusion, designed to identify the most
important influences on receptive-grammar gains.

The target is **mildly right-skewed** (``celf_gain`` min ≈ −8,
max ≈ 10, median 1, mean 1.14, std 3.20, skewness 0.14, with ~26%
negative and ~17% zero observations, n ≈ 160). The zero pile-up
is heavier than in LRP07 / LRP05 gains (17% vs 3%/12%) —
consistent with the coarser 0-18 CELF raw-score scale.

The predictor set will be reduced by iterative importance-based
feature selection under the MAE-tuned params (see
``notes/202604181400-lrp09-feature-selection.md``). This is the
initial tuned baseline; no feature-selection steps yet.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP09 has not yet been through iterative feature selection. When
# selection variants are introduced, record their rationale here as
# ``SelectionStep`` entries and chain from ``LRP09`` the same way
# ``lrp01.py`` / ``lrp03.py`` / ``lrp05.py`` / ``lrp07.py`` does.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# MAE-tuned on the full 34-predictor set (DEFAULT_GAIN, which already
# includes celf), no outlier exclusion (Optuna 150 trials, 10-split
# GroupKFold, seed 47, scoring=mae, lgbm_objective=mae). Tuner-inner
# CV MAE 2.2244 ± 0.3661. n=160.
_LGBM_MAE_PARAMS: dict[str, float | int | str] = {
    "objective": "mae",
    "n_estimators": 194,
    "learning_rate": 0.012955887984432111,
    "num_leaves": 44,
    "max_depth": 12,
    "min_child_samples": 5,
    "subsample": 0.6522788933655064,
    "subsample_freq": 1,
    "colsample_bytree": 0.968420503348316,
    "reg_alpha": 0.0013680996955281002,
    "reg_lambda": 2.240185955527313,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, untuned) ───────────────────────────────────


class LRP09(GainModel):
    """CELF receptive-grammar gain predictors — exploratory (MAE-tuned, all data).

    Uses the full :attr:`Predictors.DEFAULT_GAIN` predictor set
    (``celf`` is already a member, so the GainModel auto-include is
    a no-op) with MAE-tuned hyperparameters and no outlier exclusion.
    The starting point for feature selection on the CELF gain-
    prediction task.
    """

    model_id = "lrp09"
    target_var = V.CELF_GAIN
    description = (
        "LightGBM — CELF (receptive grammar) gain predictors "
        "(34 predictors, MAE-tuned, no outlier exclusion)"
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
        "CELF receptive-grammar gains (celf_gain). MAE-tuned on the "
        "full 34-predictor DEFAULT_GAIN set (celf is already a "
        "member, so the GainModel auto-include is a no-op here) "
        "without outlier exclusion so importance rankings reflect "
        "the full range of outcomes. Target is mildly right-skewed "
        "(skew 0.14) with heavier zero pile-up than other gain "
        "targets (17% zero, vs 3-12% in LRP05/LRP07). "
        "Feature-selection variants to follow. See "
        "notes/202604181400-lrp09-feature-selection.md."
    )
