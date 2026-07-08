# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Models package.

Imports ``base_model`` first (defines ``ModelDefinition`` and the global
``MODELS`` dict), then ``registry`` (shared defaults), then the
per-problem modules so their class definitions auto-register into
``MODELS`` at import time. Module files use the canonical #168 underscore
names (``lrp_rli_gbg_012``); each class still auto-registers under its
canonical ``model_id`` (``lrp-rli-gbg-012``).
"""

from language_reading_predictors.models import base_model  # noqa: F401
from language_reading_predictors.models import registry  # noqa: F401
from language_reading_predictors.models import (  # noqa: F401
    lrp_rli_gbg_001,
    lrp_rli_gbg_002,
    lrp_rli_gbg_003,
    lrp_rli_gbg_004,
    lrp_rli_gbg_005,
    lrp_rli_gbg_006,
    lrp_rli_gbg_007,
    lrp_rli_gbg_008,
    lrp_rli_gbg_009,
    lrp_rli_gbg_010,
    lrp_rli_gbg_011,
    lrp_rli_gbg_012,
    lrp_rli_gbg_013,
    lrp_rli_gbg_014,
    lrp_rli_gbg_015,
    lrp_rli_gbg_016,
    lrp_rli_gbg_017,
    lrp_rli_gbg_018,
    lrp_rli_gbg_019,
    lrp_rli_gbg_020,
    lrp_rli_gbg_021,
    lrp_rli_gbg_022,
    lrp_rli_gbl_001,
    lrp_rli_gbl_002,
    lrp_rli_gbl_003,
    lrp_rli_gbl_004,
    lrp_rli_gbl_005,
    lrp_rli_gbl_006,
    lrp_rli_gbl_007,
    lrp_rli_gbl_008,
    lrp_rli_gbl_009,
    lrp_rli_gbl_010,
    lrp_rli_gbl_011,
    lrp_rli_gbl_012,
    lrp_rli_gbl_013,
    lrp_rli_gbl_014,
    lrp_rli_gbl_015,
    lrp_rli_gbl_016,
    lrp_rli_gbl_017,
    lrp_rli_gbl_018,
    lrp_rli_gbl_019,
    lrp_rli_gbl_020,
    lrp_rli_gbl_021,
    lrp_rli_gbl_022,
    lrp_rli_gbl_023,
    lrp_rli_gbl_024,
    lrp_rli_gbl_025,
    lrp_rli_gbl_026,
    lrp_rli_gbl_027,
    lrp_rli_gbl_028,
)
