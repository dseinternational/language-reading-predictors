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
from sklearn.compose import TransformedTargetRegressor

from language_reading_predictors.models.base_pipeline import ESTIMATOR_STEP
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

    _estimator_label: str = "LGBMRegressor (signed-log-wrapped)"

    def _wrap_estimator(self, lgbm):
        return TransformedTargetRegressor(
            regressor=lgbm,
            func=signed_log1p,
            inverse_func=signed_expm1,
            check_inverse=False,
        )

    def _tree_estimator(self):
        """Return the LGBM regressor inside the TTR wrapper so
        :class:`shap.TreeExplainer` can introspect its booster.
        """
        wrapper = self.context.pipeline.named_steps[ESTIMATOR_STEP]
        return wrapper.regressor_
