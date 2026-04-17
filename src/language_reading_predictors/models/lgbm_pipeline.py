# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LightGBM estimator pipeline.

Thin subclass of :class:`EstimatorPipeline` that plugs an
``LGBMRegressor`` into the shared step name ``"est"``. Everything
else — cross-validation, SHAP (``TreeExplainer`` supports LightGBM natively),
permutation importance, partial dependence, config saving, report
generation — is inherited unchanged.
"""

from rich import print
from sklearn.pipeline import Pipeline

from language_reading_predictors.models.base_pipeline import (
    ESTIMATOR_STEP,
    EstimatorPipeline,
)


class LGBMPipeline(EstimatorPipeline):
    """LightGBM regression pipeline."""

    def configure_model(self) -> None:
        from lightgbm import LGBMRegressor

        from language_reading_predictors.models.base_pipeline import _section

        _section("Configure model")

        context = self.context
        cfg = context.config
        run = context.run_config

        lgbm_params = dict(cfg.model_params)
        if run.n_estimators is not None:
            # Treat run.n_estimators as an upper cap, not a replacement:
            # models with a tuned n_estimators below the cap should keep
            # their tune rather than being inflated to the cap value.
            tuned = lgbm_params.get("n_estimators", run.n_estimators)
            lgbm_params["n_estimators"] = min(tuned, run.n_estimators)

        lgbm = LGBMRegressor(
            **lgbm_params,
            random_state=cfg.random_seed,
        )
        context.pipeline = Pipeline([(ESTIMATOR_STEP, lgbm)])

        print(f"  LGBMRegressor: {lgbm_params}")

        cv_splits = run.cv_splits if run.cv_splits is not None else cfg.cv_splits
        print(f"  CV: GroupKFold(n_splits={cv_splits})")
