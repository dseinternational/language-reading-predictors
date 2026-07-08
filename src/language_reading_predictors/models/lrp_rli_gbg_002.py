# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRPGBG02: Predictors of taught-vocabulary gains.

``LRPGBG02`` is the exploratory model for *taught* expressive-vocabulary gains
(``b1extau_gain`` — change in the Block 1 directly-taught expressive vocabulary
score). It is the taught-vocabulary analogue of :mod:`lrp_rli_gbg_006` (standardised
expressive-vocabulary gains, ``eowpvt_gain``), added so the predictor analysis
covers taught vocabulary too, not only the standardised tests.

The target is signed and mildly skewed (``b1extau_gain`` min ≈ −6, max ≈ 12,
median 1, mean 1.83, skewness 0.71, ~20% negative, n ≈ 161) — comparable to
``eowpvt_gain``, so the plain MAE LightGBM pipeline (no target transform, no
outlier exclusion) used by LRPGBG06 carries over.

Predictor set: :attr:`Predictors.DEFAULT_GAIN` plus the auto-included baseline
``b1extau`` (via :class:`GainModel`), **minus** ``b1exto``. The Block 1 expressive
*total* ``b1exto`` equals taught + not-taught (``b1extau + b1exnt``) so it contains
the target/baseline construct directly; keeping it would make the model a
between-tests calibration of the same instrument rather than an identification of
substantive predictors. This is the only deviation from the LRPGBG06 predictor set.

Status: initial exploratory baseline. Hyperparameters were seeded from the
LRPGBG06 standardised analogue as a reasonable starting point and are a frozen
snapshot — they are not kept in sync as LRPGBG06 is retuned. A target-specific
Optuna tune (``scripts/tune_model.py lrpgbg02``) is a follow-up, exactly as for
LRPGBG06.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import GainModel
from language_reading_predictors.models.common import DEFAULT_SHAP_SCATTER_SPECS
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


# ── hyperparameter set ───────────────────────────────────────────────────
#
# Seeded from LRPGBG06 (MAE-tuned on eowpvt_gain) as a starting baseline for the
# closely-matched taught-vocabulary gain target — a frozen snapshot, not kept in
# sync with LRPGBG06 (which has since been retuned). Retune for this target before
# any quantitative reporting (scripts/tune_model.py lrpgbg02); importance rankings
# — the purpose of this exploratory model — are robust to reasonable params.
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


class LRPGBG02(GainModel):
    """Taught expressive-vocabulary gain predictors — exploratory (MAE, all data).

    Uses :attr:`Predictors.DEFAULT_GAIN` plus the auto-included baseline
    ``b1extau`` and minus the tautological total ``b1exto`` (see module
    docstring), with no outlier exclusion.
    """

    model_id = "lrp-rli-gbg-002"
    target_var = V.B1EXTAU_GAIN
    description = (
        "LightGBM — taught expressive-vocabulary gain predictors "
        "(DEFAULT_GAIN minus b1exto, MAE, no outlier exclusion)"
    )
    pipeline_cls = LGBMPipeline
    params = _LGBM_MAE_PARAMS
    exclude = (V.B1EXTO,)
    shap_scatter_specs = DEFAULT_SHAP_SCATTER_SPECS
    notes = (
        "Exploratory model for predictors of taught expressive-vocabulary gains "
        "(b1extau_gain), the taught-vocabulary analogue of lrpgbg06. b1exto (Block 1 "
        "expressive total = taught + not-taught) is excluded to avoid target "
        "leakage. Hyperparameters borrowed from lrpgbg06 pending a target-specific "
        "tune."
    )
