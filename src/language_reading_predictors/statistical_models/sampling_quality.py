# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Backwards-compatible re-exports.

The sampling-quality extractor — one correct way to read max R-hat, min ESS,
min per-chain BFMI, total divergences and the unassessable-parameter names off a
trace, unrounded, deciding nothing — moved to
:mod:`dse_research_utils.statistics.sampling_quality` in v0.12.0, where the other
research repositories can reach it too. This module re-exports it under the
original names so existing imports keep working.

Note for tests: the ``_bfmi_per_chain`` seam now lives in the shared module, so
patch it there rather than here.
"""

from dse_research_utils.statistics.sampling_quality import SamplingQuality, sampling_quality

__all__ = ["SamplingQuality", "sampling_quality"]
