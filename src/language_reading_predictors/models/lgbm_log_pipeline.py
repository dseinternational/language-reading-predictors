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
from sklearn.compose import TransformedTargetRegressor

from language_reading_predictors.models.base_pipeline import ESTIMATOR_STEP
from language_reading_predictors.models.lgbm_pipeline import LGBMPipeline


class LGBMLogPipeline(LGBMPipeline):
    """LightGBM regression pipeline with log1p target transformation."""

    _estimator_label: str = "LGBMRegressor (log1p-wrapped)"

    def _wrap_estimator(self, lgbm):
        return TransformedTargetRegressor(
            regressor=lgbm,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )

    def _tree_estimator(self):
        """Return the LGBM regressor inside the TTR wrapper so
        :class:`shap.TreeExplainer` can introspect its booster.
        """
        wrapper = self.context.pipeline.named_steps[ESTIMATOR_STEP]
        return wrapper.regressor_
