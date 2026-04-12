# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Models package.

Imports ``registry`` first (helpers + empty ``MODELS`` dict), then the
per-problem modules so their top-level ``_gain_model`` / ``_level_model``
calls populate the registry at import time.
"""

from language_reading_predictors.models import registry  # noqa: F401
from language_reading_predictors.models import lrp01, lrp02  # noqa: F401
