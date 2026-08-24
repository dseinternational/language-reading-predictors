# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared figure-saving helpers for both model systems (GB and Bayesian).

These centralise the report figure-artifact policy (issue #208) so every plotting
call site behaves identically:

* always write ``<name>.png`` — the artifact the report templates reference
  (raster keeps the model-output pages quick to browse);
* also write ``<name>.svg`` unless it would exceed ~2 MB, in which case the vector
  sibling is dropped (very large SVGs are the ones that make the viewer slow —
  exactly what #208 wants to avoid);
* optionally write ``<name>.csv`` of the data behind the plot.

Consistent house style (fonts, colours, grid, DPI) comes from ``dse_research_utils``
(``set_matplotlib_default_style``), applied once at the fit entry points; these
helpers only standardise *saving* — PNG + SVG sibling + optional data CSV — and
closing the figure. Both matplotlib figures and ``arviz_plots`` ``PlotCollection``
objects route through here so a single change propagates to every model.

The save mechanics moved into ``dse_research_utils.plot.io`` in v0.12.0 (they were
one of five parallel implementations across the research repositories); this module
now applies this repository's policy to them — the SVG size cap is on by default
here, whereas the shared helpers leave it opt-in. :data:`SVG_MAX_BYTES` is read at
call time so it stays configurable (and monkeypatchable in tests).
"""

from __future__ import annotations

from typing import Any

import dse_research_utils.plot.io as plot_io
from dse_research_utils.plot.styles import DPI_FILE

# Issue #208: still emit SVGs, but skip very large ones (they are what make the
# report viewer slow). ~2 MB is comfortably above a typical vector figure and well
# below the multi-megabyte beeswarm/interaction grids we want to keep raster.
SVG_MAX_BYTES = plot_io.SVG_MAX_BYTES


def save_plot_data(output_dir: str, name: str, data: Any, *, index: bool = False) -> str:
    """Write the data behind a plot as ``<name>.csv`` (issue #208)."""
    return plot_io.save_plot_data(output_dir, name, data, index=index)


def save_styled_figure(
    output_dir: str,
    name: str,
    *,
    fig: Any | None = None,
    dpi: float = DPI_FILE,
    bbox_inches: str = "tight",
    close: bool = True,
    svg: bool = True,
    data: Any | None = None,
) -> str:
    """Save a matplotlib figure as PNG (+ SVG sibling, + optional data CSV).

    ``name`` may be a stem or carry a ``.png`` extension. Returns the PNG path.
    """
    return plot_io.save_styled_figure(
        output_dir,
        name,
        fig=fig,
        dpi=dpi,
        bbox_inches=bbox_inches,
        close=close,
        svg=svg,
        svg_max_bytes=SVG_MAX_BYTES,
        data=data,
    )


def save_plotcollection(
    pc: Any,
    output_dir: str,
    name: str,
    *,
    suptitle: str | None = None,
    dpi: float = DPI_FILE,
    svg: bool = True,
    data: Any | None = None,
) -> None:
    """Save an ``arviz_plots`` ``PlotCollection`` as PNG (+ SVG sibling).

    Adds a figure-level ``suptitle`` (ArviZ plots render untitled) and emits the
    SVG through ``pc.savefig`` so the collection lays out correctly.
    """
    plot_io.save_plotcollection(
        pc,
        output_dir,
        name,
        suptitle=suptitle,
        dpi=dpi,
        svg=svg,
        svg_max_bytes=SVG_MAX_BYTES,
        data=data,
    )


__all__ = [
    "DPI_FILE",
    "SVG_MAX_BYTES",
    "save_plot_data",
    "save_plotcollection",
    "save_styled_figure",
]
