# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Backwards-compatible re-exports.

The generic statistics helpers (descriptive summaries, numeric transforms, and
feature-dependence / clustering matrices) now live in the shared
``dse_research_utils`` package. This module re-exports them under the original
names so existing imports — ``from language_reading_predictors.stats_utils
import ...`` — keep working.
"""

from dse_research_utils.math.constants import EPSILON
from dse_research_utils.ml.feature_dependence import (
    distance_corr_dissimilarity,
    distance_corr_dissimilarity_linkage,
    distance_corr_matrix,
    mutual_info_dissimilarity,
    spearman_distance_matrix,
)
from dse_research_utils.statistics.descriptive import (
    describe,
    describe_all,
    describe_all_grouped,
    differential_entropy_standardized,
)
from dse_research_utils.statistics.transforms import (
    convert_to_categorical,
    invlogit,
    logit,
    standardize,
)

__all__ = [
    "EPSILON",
    "convert_to_categorical",
    "describe",
    "describe_all",
    "describe_all_grouped",
    "differential_entropy_standardized",
    "distance_corr_dissimilarity",
    "distance_corr_dissimilarity_linkage",
    "distance_corr_matrix",
    "invlogit",
    "logit",
    "mutual_info_dissimilarity",
    "spearman_distance_matrix",
    "standardize",
]
