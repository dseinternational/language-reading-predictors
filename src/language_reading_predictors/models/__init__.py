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
    lrp01,
    lrp02,
    lrp03,
    lrp04,
    lrp05,
    lrp06,
    lrp07,
    lrp08,
    lrp09,
    lrp10,
    lrp11,
    lrp12,
    lrp13,
    lrp14,
    lrp15,
    lrp16,
    lrp19,
    lrp20,
)
