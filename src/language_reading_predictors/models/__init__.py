# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Models package.

Imports ``base_model`` first (defines ``ModelDefinition`` and the global
``MODELS`` dict), then ``registry`` (shared defaults), then the
per-problem modules so their class definitions auto-register into
``MODELS`` at import time.
"""

from language_reading_predictors.models import base_model  # noqa: F401
from language_reading_predictors.models import registry  # noqa: F401
from language_reading_predictors.models import (  # noqa: F401
    lrpgbg01,
    lrpgbg02,
    lrpgbg03,
    lrpgbg04,
    lrpgbg05,
    lrpgbg06,
    lrpgbg07,
    lrpgbg08,
    lrpgbg09,
    lrpgbg10,
    lrpgbg11,
    lrpgbg12,
    lrpgbg13,
    lrpgbg14,
    lrpgbg15,
    lrpgbg16,
    lrpgbg17,
    lrpgbg18,
    lrpgbg19,
    lrpgbg20,
    lrpgbg21,
    lrpgbg22,
    lrpgbl01,
    lrpgbl02,
    lrpgbl03,
    lrpgbl04,
    lrpgbl05,
    lrpgbl06,
    lrpgbl07,
    lrpgbl08,
    lrpgbl09,
    lrpgbl10,
    lrpgbl11,
    lrpgbl12,
    lrpgbl13,
    lrpgbl14,
    lrpgbl15,
    lrpgbl16,
    lrpgbl17,
    lrpgbl18,
    lrpgbl19,
    lrpgbl20,
    lrpgbl21,
    lrpgbl22,
    lrpgbl23,
    lrpgbl24,
    lrpgbl25,
    lrpgbl26,
    lrpgbl27,
    lrpgbl28,
)
