# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Paths used by the statistical models package."""

import os

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(os.path.dirname(_MODULE_DIR))

ROOT_DIR = os.path.dirname(_SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
STAT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "statistical_models")
DOCS_DIR = os.path.join(ROOT_DIR, "docs")
