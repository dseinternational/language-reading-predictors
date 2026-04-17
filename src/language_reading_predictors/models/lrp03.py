# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP03: Predictors of expressive-vocabulary gains.

``LRP03`` is the baseline exploratory model for expressive-vocabulary
gains (``eowpvt_gain``). It uses the full :attr:`Predictors.DEFAULT_GAIN`
set with no outlier exclusion so the starting picture is unfiltered.
The base variable (``eowpvt``, the level) is auto-included on top of
the default gain predictors by :class:`GainModel`.

The target is signed (``eowpvt_gain`` min ≈ −13, max ≈ 28, median 3,
skewness 0.32, with ~25% negative observations and n ≈ 161). That's
much milder skew than LRP01's ``ewrswr_gain`` and nearly symmetric —
a log / signed-log transform may or may not help and is a question for
future investigation.

No tuning has been run for LRP03 yet — it runs on a reasonable
``_LGBM_BASELINE_PARAMS`` dict so later feature-selection variants
have a documented starting point.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import SelectionStep, ShapScatterSpec
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── predictor selection steps (shared by all variants) ───────────────────
#
# LRP03 has not yet been through iterative feature selection. When
# selection variants are introduced, record their rationale here as
# ``SelectionStep`` entries and chain from ``LRP03`` the same way
# ``lrp01.py`` does.

_SELECTION_STEPS: list[SelectionStep] = []


# ── hyperparameter sets ─────────────────────────────────────────────────

# Baseline — no tuning has been run for LRP03 yet. Reasonable defaults
# give the feature-selection work a reproducible starting point. Use
# ``python scripts/tune_model.py lrp03`` to produce a tuned set and
# replace this dict.
_LGBM_BASELINE_PARAMS: dict[str, float | int | str] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 6,
    "min_child_samples": 16,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_jobs": -1,
    "verbosity": -1,
}


# ── primary model (baseline, untuned) ───────────────────────────────────


class LRP03(GainModel):
    """Expressive-vocabulary gain predictors — baseline (all data, untuned).

    Uses the full :attr:`Predictors.DEFAULT_GAIN` predictor set plus
    the base variable ``eowpvt`` (auto-included via :class:`GainModel`)
    and a reasonable ``_LGBM_BASELINE_PARAMS`` set. Serves as the
    starting point for feature-selection and tuning work on the
    expressive-vocabulary gain-prediction task.
    """

    model_id = "lrp03"
    target_var = V.EOWPVT_GAIN
    description = (
        "LightGBM — expressive-vocabulary gain predictors "
        "(baseline, DEFAULT_GAIN set + eowpvt, untuned)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_BASELINE_PARAMS
    cv_splits = 51
    outlier_threshold = None
    selection_steps = _SELECTION_STEPS
    shap_scatter_specs = [
        ShapScatterSpec(description="All predictors, SHAP auto-colouring"),
    ]
    notes = (
        "Baseline exploratory model for expressive-vocabulary gains "
        "(eowpvt_gain). Uses the full default gain predictor set plus "
        "the base level variable eowpvt (auto-included by GainModel), "
        "without outlier exclusion, and a reasonable "
        "_LGBM_BASELINE_PARAMS starting point — no feature selection "
        "or hyperparameter tuning has been applied yet."
    )
