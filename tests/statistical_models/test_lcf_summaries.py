# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Independent unit test for the isolated LCF summary computations (#394 pillar 6).

Exercises the public ``lcf_summaries`` API without a factory, PyMC model, output
directory or plotting session — confirming the family's summary calculations are
testable in isolation. The end-to-end factory-backed coverage (the full items-scale
translation and the directed #312 concurrent comparison) lives in
``test_factories.py``.
"""

from __future__ import annotations

import numpy as np

from language_reading_predictors.statistical_models.lcf_summaries import (
    observed_conditional_slope,
)


def test_observed_conditional_slope_matches_the_delta_method_by_hand():
    """The conditional measurement translation is the delta-method slope of the
    target indicator's item count per +1 item of the predictor indicator, at the
    pooled-mean operating point, conditioning the two domains on the third. With C
    the third domain, ``Cov(a, b | C) = 0.5 - 0.2 * 0.3 = 0.44`` and
    ``Var(b | C) = 1 - 0.3**2 = 0.91``."""
    corr = np.array([[[[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]]])
    loadings = np.array([[2.0, 0.8]])
    residual_sds = np.array([[0.4, 0.6]])

    slope = observed_conditional_slope(
        corr,
        loadings,
        residual_sds,
        target_domain_idx=0,
        predictor_domain_idx=1,
        target_indicator_idx=0,
        predictor_indicator_idx=1,
    )

    expected = 2.0 * 0.8 * 0.44 / (0.8**2 * 0.91 + 0.6**2)
    np.testing.assert_allclose(slope, expected, rtol=1e-12, atol=1e-12)
