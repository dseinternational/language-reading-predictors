# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model LRP02: Random Forest regression predicting word-reading level (ewrswr).
"""

from language_reading_predictors.models.common import ModelConfig, ModelFitContext
from language_reading_predictors.models import rf_pipeline

CONFIG = ModelConfig(
    model_id="lrp02",
    description="Random Forest — word-reading level predictors",
    target_var="ewrswr",
    predictor_vars=[
        "time",
        "group",
        "age",
        "area",
        "attend",
        "hearing",
        "vision",
        "behav",
        "rowpvt",
        "eowpvt",
        "yarclet",
        "spphon",
        "blending",
        "nonword",
        "trog",
        "aptgram",
        "aptinfo",
        "b1exto",
        "b1reto",
        "b2exto",
        "b2reto",
        "celf",
        "deappin",
        "deappvo",
        "deappfi",
        "deappav",
    ],
    rf_params=dict(
        n_estimators=1200,
        max_depth=8,
        min_samples_leaf=16,
        min_samples_split=4,
        max_features=0.5,
        bootstrap=False,
        criterion="squared_error",
        n_jobs=16,
    ),
    cv_splits=51,
)


def fit(config: str = "reporting") -> ModelFitContext:
    """Run the full LRP02 pipeline."""
    return rf_pipeline.fit(CONFIG, run_config=config)
