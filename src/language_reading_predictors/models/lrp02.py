# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP02 — word-reading level predictors.

Holds the final ``lrp02`` model, its LightGBM sibling ``lrp02_lgbm``, and
any historical selection variants. Variants carry ``variant_of="lrp02"``
so that ``fit_model.py all`` skips them unless ``--include-variants`` is
passed.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline
from language_reading_predictors.models.registry import (
    DEFAULT_LGBM_PARAMS,
    _level_model,
)

# ── final model ─────────────────────────────────────────────────────────

_level_model(
    "lrp02",
    V.EWRSWR,
    description="Random Forest — word-reading level predictors",
    include=[V.GROUP, V.AREA],
    cv_splits=51,
)

# ── LightGBM sibling ────────────────────────────────────────────────────

_level_model(
    "lrp02_lgbm",
    V.EWRSWR,
    description="LightGBM — word-reading level predictors",
    include=[V.GROUP, V.AREA],
    cv_splits=51,
    pipeline_cls=LGBMPipeline,
    params=DEFAULT_LGBM_PARAMS,
)
