# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LRP02 — word-reading level predictors.

Holds the final ``lrp02`` model and any historical selection variants.
Variants carry ``variant_of="lrp02"`` so that ``fit_model.py all`` skips
them unless ``--include-variants`` is passed.
"""

from language_reading_predictors.data_variables import Variables as V
from language_reading_predictors.models.base_model import LevelModel
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline
from language_reading_predictors.models.registry import DEFAULT_LGBM_PARAMS


class LRP02(LevelModel):
    """Word-reading level predictors — all default level predictors."""

    model_id = "lrp02"
    target_var = V.EWRSWR
    description = "LightGBM — word-reading level predictors"
    include = [V.GROUP, V.AREA]
    pipeline_cls = LGBMPipeline
    params = DEFAULT_LGBM_PARAMS
    cv_splits = 51
