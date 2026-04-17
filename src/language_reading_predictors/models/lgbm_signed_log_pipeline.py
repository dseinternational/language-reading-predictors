# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
LightGBM pipeline with signed-log target transform.

Wraps the underlying ``LGBMRegressor`` in a
:class:`sklearn.compose.TransformedTargetRegressor` that applies
``sign(y) * log1p(|y|)`` before fitting and ``sign(x) * expm1(|x|)`` to
predictions. Designed for signed targets (e.g. ``ewrswr_gain``, where
children can regress as well as improve) — `log1p` would be NaN on
negatives, whereas the signed variant preserves sign and compresses
both tails symmetrically around zero.

All downstream metrics (CV, evaluation, permutation importance) are
computed in the original target units because
``TransformedTargetRegressor.predict`` inverse-transforms before
returning.

SHAP operates on the underlying tree model fitted on signed-log space,
so SHAP values are in signed-log units — the interpretation is "how
features pull the prediction up or down in signed-log space", not
directly comparable to SHAP values from :class:`LGBMPipeline`.
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


def signed_log1p(y):
    """Signed log: ``sign(y) * log1p(|y|)``. Works on scalars and arrays."""
    y = np.asarray(y)
    return np.sign(y) * np.log1p(np.abs(y))


def signed_expm1(x):
    """Inverse of :func:`signed_log1p`: ``sign(x) * expm1(|x|)``."""
    x = np.asarray(x)
    return np.sign(x) * np.expm1(np.abs(x))


class LGBMSignedLogPipeline(LGBMPipeline):
    """LightGBM regression pipeline with signed-log target transformation."""

    def configure_model(self) -> None:
        from lightgbm import LGBMRegressor

        _section("Configure model")

        context = self.context
        cfg = context.config
        run = context.run_config

        lgbm_params = dict(cfg.model_params)
        if run.n_estimators is not None:
            # Cap, not replace — see LGBMPipeline.configure_model.
            tuned = lgbm_params.get("n_estimators", run.n_estimators)
            lgbm_params["n_estimators"] = min(tuned, run.n_estimators)

        lgbm = LGBMRegressor(
            **lgbm_params,
            random_state=cfg.random_seed,
        )
        wrapped = TransformedTargetRegressor(
            regressor=lgbm,
            func=signed_log1p,
            inverse_func=signed_expm1,
            check_inverse=False,
        )
        context.pipeline = Pipeline([(ESTIMATOR_STEP, wrapped)])

        print(f"  LGBMRegressor (signed-log-wrapped): {lgbm_params}")

        cv_splits = run.cv_splits if run.cv_splits is not None else cfg.cv_splits
        print(f"  CV: GroupKFold(n_splits={cv_splits})")

    def shap_analysis(self) -> None:
        """Re-aim SHAP at the underlying LGBM inside the TTR wrapper.

        Identical mechanism to
        :meth:`LGBMLogPipeline.shap_analysis` — temporarily swap the
        pipeline step from the TTR wrapper to its fitted inner
        regressor so ``shap.TreeExplainer`` can introspect the booster.
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
