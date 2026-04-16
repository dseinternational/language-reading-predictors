# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LightGBM pipeline with log-transformed target.

Wraps the underlying ``LGBMRegressor`` in a
:class:`sklearn.compose.TransformedTargetRegressor` that applies
``log1p`` to ``y`` before fitting and ``expm1`` to predictions. All
downstream metrics (CV, evaluation, permutation importance) are
computed in the original target units because
``TransformedTargetRegressor.predict`` inverse-transforms before
returning.

SHAP still operates on the underlying tree model (fitted on
log-space), so SHAP values are in log units — which is conceptually
the right view for "which features pull the prediction up or down in
log space" but note that the base value and scale are not directly
comparable to SHAP values from :class:`LGBMPipeline`.
"""

import numpy as np
from rich import print
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from language_reading_predictors.models.base_pipeline import (
    ESTIMATOR_STEP,
    _section,
)
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


class LGBMLogPipeline(LGBMPipeline):
    """LightGBM regression pipeline with log1p target transformation."""

    def configure_model(self) -> None:
        from lightgbm import LGBMRegressor

        _section("Configure model")

        context = self.context
        cfg = context.config
        run = context.run_config

        lgbm_params = dict(cfg.model_params)
        if run.n_estimators is not None:
            lgbm_params["n_estimators"] = run.n_estimators

        lgbm = LGBMRegressor(
            **lgbm_params,
            random_state=cfg.random_seed,
        )
        wrapped = TransformedTargetRegressor(
            regressor=lgbm,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )
        context.pipeline = Pipeline([(ESTIMATOR_STEP, wrapped)])

        print(f"  LGBMRegressor (log1p-wrapped): {lgbm_params}")

        cv_splits = run.cv_splits if run.cv_splits is not None else cfg.cv_splits
        print(f"  CV: GroupKFold(n_splits={cv_splits})")

    def shap_analysis(self) -> None:
        """Re-aim SHAP at the underlying LGBM inside the TTR wrapper.

        The base-class ``shap_analysis`` looks up
        ``pipeline.named_steps[ESTIMATOR_STEP]`` and passes it directly
        to ``shap.TreeExplainer``. With a ``TransformedTargetRegressor``
        in that slot, we need to expose its fitted inner regressor
        (``.regressor_``) so ``TreeExplainer`` can introspect the
        booster. We temporarily swap the pipeline step, run the parent
        SHAP analysis, then restore.
        """
        context = self.context
        wrapper = context.pipeline.named_steps[ESTIMATOR_STEP]
        inner = wrapper.regressor_  # fitted LGBMRegressor

        original_steps = context.pipeline.steps
        context.pipeline.steps = [(ESTIMATOR_STEP, inner)]
        try:
            super().shap_analysis()
        finally:
            context.pipeline.steps = original_steps
