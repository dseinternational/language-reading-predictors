# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Backwards-compatible re-exports.

The generic ML scaffolding (hyperparameter search, cross-validation scoring,
Gaussian-process kernels) now lives in the shared ``dse_research_utils.ml``
package. This module re-exports it under the original names so existing
imports — ``from language_reading_predictors.ml_utils import ...`` — keep working.
"""

from dse_research_utils.ml.cross_validation import (
    DEFAULT_REGRESSION_CRITERION,
    DEFAULT_REGRESSION_PERM_IMPORTANCE_REPEATS,
    DEFAULT_REGRESSION_SCORERS,
    DEFAULT_REGRESSION_SCORING,
    DEFAULT_REGRESSION_SEARCH_ITERATIONS,
    cross_validation_score_rows,
    report_cross_validation_scores,
)
from dse_research_utils.ml.kernels import (
    ornstein_uhlenbeck_kernel,
    periodic_kernel,
    quadratic_distance_kernel,
)
from dse_research_utils.ml.search import hyperparam_search_randomized

__all__ = [
    "DEFAULT_REGRESSION_CRITERION",
    "DEFAULT_REGRESSION_PERM_IMPORTANCE_REPEATS",
    "DEFAULT_REGRESSION_SCORERS",
    "DEFAULT_REGRESSION_SCORING",
    "DEFAULT_REGRESSION_SEARCH_ITERATIONS",
    "cross_validation_score_rows",
    "hyperparam_search_randomized",
    "ornstein_uhlenbeck_kernel",
    "periodic_kernel",
    "quadratic_distance_kernel",
    "report_cross_validation_scores",
]
