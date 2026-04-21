# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LightGBM estimator pipeline.

Thin subclass of :class:`EstimatorPipeline` that plugs an
``LGBMRegressor`` into the shared step name ``"est"``. Everything
else — cross-validation, SHAP (``TreeExplainer`` supports LightGBM natively),
permutation importance, partial dependence, config saving, report
generation — is inherited unchanged.

Subclasses that wrap the estimator (e.g. with a
:class:`~sklearn.compose.TransformedTargetRegressor`) override
:meth:`_wrap_estimator` and :attr:`_estimator_label`.
"""

from rich import print
from sklearn.pipeline import Pipeline

from language_reading_predictors.models._reporting import section_header
from language_reading_predictors.models.base_pipeline import (
    ESTIMATOR_STEP,
    EstimatorPipeline,
    _cap_n_estimators,
)


class LGBMPipeline(EstimatorPipeline):
    """LightGBM regression pipeline."""

    _estimator_label: str = "LGBMRegressor"

    def _wrap_estimator(self, lgbm):
        """Return the estimator to insert into the pipeline step.

        Subclasses override this to wrap ``lgbm`` (e.g. in a
        ``TransformedTargetRegressor``).
        """
        return lgbm

    def configure_model(self) -> None:
        from lightgbm import LGBMRegressor

        section_header("Configure model")

        context = self.context
        cfg = context.config
        run = context.run_config

        lgbm_params = _cap_n_estimators(cfg.model_params, run)

        lgbm = LGBMRegressor(
            **lgbm_params,
            random_state=cfg.random_seed,
        )
        context.pipeline = Pipeline([(ESTIMATOR_STEP, self._wrap_estimator(lgbm))])

        print(f"  {self._estimator_label}: {lgbm_params}")

        cv_splits = run.cv_splits if run.cv_splits is not None else cfg.cv_splits
        print(f"  CV: GroupKFold(n_splits={cv_splits})")
