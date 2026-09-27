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


def init_plotting() -> None:
    """Apply the DSE matplotlib house style for figure generation.

    The shared ``dse_research_utils`` style plus this repository's font fallback
    (``figure_io.use_house_style``). The CLI entry points
    (``scripts/fit_statistical_model.py`` and the GB ``scripts/fit_model.py``)
    apply it too. This idempotent helper is called at the top of each Bayesian
    ``fit()``, so figures are styled consistently even when a model is fitted /
    replotted from a notebook or test that bypasses the CLI. Imported lazily so
    importing this paths module stays cheap and free of import-time matplotlib
    side effects.
    """
    from language_reading_predictors.figure_io import use_house_style

    use_house_style()
