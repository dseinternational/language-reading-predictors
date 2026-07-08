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


def init_plotting() -> None:
    """Apply the shared DSE matplotlib house style for figure generation.

    The CLI entry point (``scripts/fit_statistical_model.py``) applies this via
    ``dse_research_utils.environment.setup.init_script`` — mirroring the GB
    ``scripts/fit_model.py``. This idempotent helper is called at the top of each
    Bayesian ``fit()`` too, so figures are styled consistently even when a model
    is fitted / replotted from a notebook or test that bypasses the CLI. Imported
    lazily so importing this paths module stays cheap and free of import-time
    matplotlib side effects.
    """
    from dse_research_utils.plot.styles import set_matplotlib_default_style

    set_matplotlib_default_style()
