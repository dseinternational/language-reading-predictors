# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Figure-saving helpers for the statistical models.

Re-exports from :mod:`language_reading_predictors.figure_io` preserve the local
import path. That module defines the shared figure and data-file policy.
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
