# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Beta-binomial likelihood helper used by every factory.

The implementation now lives in the shared package; re-exported here so the
factories' ``from ...likelihood import beta_binomial_from_logit`` keeps working.
"""

from dse_research_utils.statistics.models.likelihood import beta_binomial_from_logit

__all__ = ["beta_binomial_from_logit"]
