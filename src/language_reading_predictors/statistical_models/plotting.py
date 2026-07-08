# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Figure-saving helpers for the statistical models.

The implementations are shared with the gradient-boosting system and live in
:mod:`language_reading_predictors.figure_io`; they are re-exported here so the
statistical_models call sites keep a local import path. See that module for the
report figure-artifact policy (issue #208: PNG + SVG sibling (<~2 MB) + data CSV).
"""

from __future__ import annotations

from language_reading_predictors.figure_io import (
    DPI_FILE,
    SVG_MAX_BYTES,
    save_plot_data,
    save_plotcollection,
    save_styled_figure,
)

__all__ = [
    "DPI_FILE",
    "SVG_MAX_BYTES",
    "save_plot_data",
    "save_plotcollection",
    "save_styled_figure",
]
