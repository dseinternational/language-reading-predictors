# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Paths for the statistical-models package.

Thin compatibility shim over :mod:`language_reading_predictors.paths` (issue
#180): the constants below reflect the ``DSE_LRP_OUTPUT_DIR`` environment
override at import time. Code that must *also* honour a CLI ``--output-dir`` set
after import (the fit / compare / upload scripts, via ``paths.set_output_root``)
calls the :mod:`~language_reading_predictors.paths` functions directly.
"""

from language_reading_predictors import paths as _paths

ROOT_DIR = str(_paths.ROOT_DIR)
DATA_DIR = str(_paths.DATA_DIR)
DOCS_DIR = str(_paths.DOCS_DIR)
OUTPUT_DIR = str(_paths.output_root())
STAT_OUTPUT_DIR = str(_paths.stat_dir())
